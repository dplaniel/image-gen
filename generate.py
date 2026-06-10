#!/usr/bin/env python3
"""
FLUX.1-schnell / Stable Diffusion 3.5 Large Turbo image generator.
Tuned for RTX 3080 12 GB.

T5-XXL is loaded on GPU per-image for fast encoding (~0.14 s), then unloaded
to make room for the diffusion transformer (~6.7 GB FLUX / ~4.5 GB SD 3.5).
Both can't coexist on this GPU, so they time-share: ~2 s swap overhead per image.

Usage:
  python generate.py "a sunset over misty mountains"
  python generate.py "portrait in oil paint style" --model sd35
  python generate.py "macro photo of a dragonfly" --seed 42 --width 896 --height 1152
  python generate.py  # interactive mode
"""

import argparse
import gc
import os
import sys
import textwrap
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from PIL import Image, ImageDraw, ImageFont
from transformers import BitsAndBytesConfig, T5EncoderModel


FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SD35_MODEL_ID  = "stabilityai/stable-diffusion-3.5-large-turbo"

MODELS_DIR   = Path(__file__).parent / "models"
FLUX_TR_PATH = MODELS_DIR / "transformer"
SD35_TR_PATH = MODELS_DIR / "sd35_transformer"
T5_PATH      = MODELS_DIR / "text_encoder_2"  # same Google T5-v1.1-xxl for both models

DEFAULT_STEPS  = 4
DEFAULT_WIDTH  = 1024
DEFAULT_HEIGHT = 1024
OUTPUT_DIR     = Path("/data/big/flux_images")


def _nf4_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _cast_fp16_to_bf16(model) -> None:
    """Cast stray fp16 biases/norms to bfloat16 after NF4 loading."""
    import bitsandbytes as bnb
    for param in model.parameters():
        if not isinstance(param, bnb.nn.Params4bit) and param.dtype == torch.float16:
            param.data = param.data.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Transformer loaders
# ---------------------------------------------------------------------------

def _load_flux_transformer() -> FluxTransformer2DModel:
    tr = FluxTransformer2DModel.from_pretrained(FLUX_TR_PATH)
    _cast_fp16_to_bf16(tr)
    return tr


def _load_sd35_transformer() -> SD3Transformer2DModel:
    tr = SD3Transformer2DModel.from_pretrained(SD35_TR_PATH)
    _cast_fp16_to_bf16(tr)
    return tr


# ---------------------------------------------------------------------------
# GPU T5 swap
# ---------------------------------------------------------------------------

def _encode_t5_gpu_swap(pipe, prompt: str, max_seq: int) -> torch.Tensor:
    """
    Unload transformer → load T5 NF4 on GPU → encode → unload T5 → reload transformer.

    Returns T5 last-hidden-state [1, max_seq, 4096].
    T5 weights are shared between FLUX and SD 3.5 (same Google T5-v1.1-xxl).
    """
    t0 = time.perf_counter()

    # Free transformer VRAM
    old_tr = pipe.transformer
    pipe.transformer = None
    del old_tr
    gc.collect()
    torch.cuda.empty_cache()

    # Determine T5 source and tokenizer
    is_sd35 = isinstance(pipe, StableDiffusion3Pipeline)
    if T5_PATH.exists():
        t5_src, t5_kwargs = T5_PATH, {}
    elif is_sd35:
        t5_src = SD35_MODEL_ID
        t5_kwargs = {"subfolder": "text_encoder_3", "quantization_config": _nf4_config(), "torch_dtype": torch.bfloat16}
    else:
        t5_src = FLUX_MODEL_ID
        t5_kwargs = {"subfolder": "text_encoder_2", "quantization_config": _nf4_config(), "torch_dtype": torch.bfloat16}

    t5_tok = pipe.tokenizer_3 if is_sd35 else pipe.tokenizer_2

    t5 = T5EncoderModel.from_pretrained(t5_src, **t5_kwargs)
    ids = t5_tok(
        prompt,
        padding="max_length",
        max_length=max_seq,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to("cuda")
    with torch.no_grad():
        embeds = t5(ids)[0].to(dtype=torch.bfloat16)
    del t5
    gc.collect()
    torch.cuda.empty_cache()
    t_enc = time.perf_counter() - t0

    # Reload correct transformer
    pipe.transformer = _load_sd35_transformer() if is_sd35 else _load_flux_transformer()
    t_total = time.perf_counter() - t0
    print(f"  T5: {t_enc:.2f}s  transformer reload: {t_total - t_enc:.2f}s")

    return embeds


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------

def _encode_prompt_flux(
    pipe: FluxPipeline,
    prompt: str,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pipe.text_encoder_2 is None:
        # Fast path: T5 GPU swap (~0.14 s encode)
        prompt_embeds = _encode_t5_gpu_swap(pipe, prompt, max_sequence_length)
    else:
        # Slow path: T5 resident on CPU (~127 s) — run quantize.py to fix
        t5_device = pipe.text_encoder_2.device
        ids = pipe.tokenizer_2(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        ).input_ids.to(t5_device)
        with torch.no_grad():
            prompt_embeds = pipe.text_encoder_2(ids)[0].to(dtype=torch.bfloat16, device="cuda")

    ids = pipe.tokenizer(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to("cuda")
    with torch.no_grad():
        pooled = pipe.text_encoder(ids, output_hidden_states=False).pooler_output.to(dtype=torch.bfloat16)

    return prompt_embeds, pooled


def _encode_prompt_sd35(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # --- CLIP-L (text_encoder) ---
    ids_l = pipe.tokenizer(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to("cuda")
    with torch.no_grad():
        out_l = pipe.text_encoder(ids_l, output_hidden_states=True)
    clip_l_hidden = out_l.hidden_states[-2].to(dtype=torch.bfloat16)  # [1, 77, 768]
    clip_l_pooled = out_l.text_embeds.to(dtype=torch.bfloat16)        # [1, 768]

    # --- CLIP-G (text_encoder_2) ---
    ids_g = pipe.tokenizer_2(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to("cuda")
    with torch.no_grad():
        out_g = pipe.text_encoder_2(ids_g, output_hidden_states=True)
    clip_g_hidden = out_g.hidden_states[-2].to(dtype=torch.bfloat16)  # [1, 77, 1280]
    clip_g_pooled = out_g.text_embeds.to(dtype=torch.bfloat16)        # [1, 1280]

    # --- T5 (text_encoder_3) ---
    if pipe.text_encoder_3 is None:
        # Fast path: GPU swap
        t5_embeds = _encode_t5_gpu_swap(pipe, prompt, max_sequence_length)  # [1, seq, 4096]
    else:
        # Slow path: T5 resident on CPU
        t5_device = pipe.text_encoder_3.device
        ids_t5 = pipe.tokenizer_3(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        ).input_ids.to(t5_device)
        with torch.no_grad():
            t5_embeds = pipe.text_encoder_3(ids_t5)[0].to(dtype=torch.bfloat16, device="cuda")

    # Assemble: pad CLIP concat to T5 hidden dim, then cat along sequence axis
    # [1, 77, 768+1280=2048] → [1, 77, 4096] → cat with [1, seq, 4096] → [1, 77+seq, 4096]
    clip_concat = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)
    clip_padded = F.pad(clip_concat, (0, t5_embeds.shape[-1] - clip_concat.shape[-1]))
    prompt_embeds = torch.cat([clip_padded, t5_embeds], dim=1)

    pooled_prompt_embeds = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)  # [1, 2048]

    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt(
    pipe,
    prompt: str,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(pipe, StableDiffusion3Pipeline):
        return _encode_prompt_sd35(pipe, prompt, max_sequence_length)
    return _encode_prompt_flux(pipe, prompt, max_sequence_length)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(model: str = "flux") -> FluxPipeline | StableDiffusion3Pipeline:
    if model == "sd35":
        prebuilt = SD35_TR_PATH.exists()

        if prebuilt:
            print(f"Loading pre-quantized NF4 SD 3.5 transformer from {MODELS_DIR} …")
            print("  • SD 3.5 transformer  (NF4, GPU, ~4.5 GB VRAM)")
            transformer = _load_sd35_transformer()
            print("  • assembling pipeline  (T5 loaded per-image on GPU)")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                SD35_MODEL_ID,
                transformer=transformer,
                text_encoder_3=None,  # loaded/unloaded per generate() call
                torch_dtype=torch.bfloat16,
            )
        else:
            print(f"Loading {SD35_MODEL_ID} from HuggingFace Hub …")
            print("  (run quantize.py --model sd35 once for faster loading and generation)")
            print("  • SD 3.5 transformer  (NF4, GPU, ~4.5 GB VRAM)")
            transformer = SD3Transformer2DModel.from_pretrained(
                SD35_MODEL_ID, subfolder="transformer",
                quantization_config=_nf4_config(), torch_dtype=torch.bfloat16,
            )
            print("  • T5 text encoder  (fp16, CPU — slow; run quantize.py --model sd35 to fix)")
            text_encoder_3 = T5EncoderModel.from_pretrained(
                SD35_MODEL_ID, subfolder="text_encoder_3",
                torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
            )
            print("  • assembling pipeline …")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                SD35_MODEL_ID,
                transformer=transformer,
                text_encoder_3=text_encoder_3,
                torch_dtype=torch.bfloat16,
            )

        pipe.text_encoder.to("cuda")
        pipe.text_encoder_2.to("cuda")
        pipe.vae.to("cuda")
        pipe.vae.enable_tiling()

    else:  # flux
        prebuilt = FLUX_TR_PATH.exists()

        if prebuilt:
            print(f"Loading pre-quantized NF4 FLUX transformer from {MODELS_DIR} …")
            print("  • FLUX transformer  (NF4, GPU, ~6.7 GB VRAM)")
            transformer = _load_flux_transformer()
            print("  • assembling pipeline  (T5 loaded per-image on GPU)")
            pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID,
                transformer=transformer,
                text_encoder_2=None,  # loaded/unloaded per generate() call
                torch_dtype=torch.bfloat16,
            )
        else:
            print(f"Loading {FLUX_MODEL_ID} from HuggingFace Hub …")
            print("  (run quantize.py once for faster loading and generation)")
            print("  • FLUX transformer  (NF4, GPU, ~6.5 GB VRAM)")
            transformer = FluxTransformer2DModel.from_pretrained(
                FLUX_MODEL_ID, subfolder="transformer",
                quantization_config=_nf4_config(), torch_dtype=torch.bfloat16,
            )
            print("  • T5 text encoder  (fp16, CPU — slow; run quantize.py to fix)")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                FLUX_MODEL_ID, subfolder="text_encoder_2",
                torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
            )
            print("  • assembling pipeline …")
            pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.bfloat16,
            )

        pipe.text_encoder.to("cuda")
        pipe.vae.to("cuda")
        pipe.vae.enable_tiling()

    pipe.set_progress_bar_config(desc="  Denoising")
    return pipe


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def _add_caption(image: Image.Image, prompt: str) -> Image.Image:
    """Return a new image with a dark bar containing the prompt appended below."""
    font_size = 20
    padding = 14
    font = ImageFont.load_default(size=font_size)

    max_text_px = image.width - 2 * padding
    chars_per_line = max(1, int(max_text_px / (font_size * 0.55)))
    lines = textwrap.wrap(prompt, width=chars_per_line) or [prompt]

    line_height = font_size + 6
    bar_height = len(lines) * line_height + 2 * padding

    canvas = Image.new("RGB", (image.width, image.height + bar_height), (28, 28, 28))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (image.width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=(210, 210, 210), font=font)
        y += line_height

    return canvas


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    pipe,
    prompt: str,
    *,
    steps: int = DEFAULT_STEPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int | None = None,
    caption: bool = True,
    output: Path | None = None,
) -> Path:
    generator = (
        torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    )

    prompt_embeds, pooled_prompt_embeds = _encode_prompt(pipe, prompt)

    t0 = time.perf_counter()
    result = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=steps,
        guidance_scale=0.0,
        width=width,
        height=height,
        generator=generator,
        output_type="pil",
    )
    elapsed = time.perf_counter() - t0

    image = result.images[0]
    if caption:
        image = _add_caption(image, prompt)

    if output is None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stem = prompt[:60].replace(" ", "_").replace("/", "-")
        ts = int(time.time())
        output = OUTPUT_DIR / f"{stem}_{ts}.png"

    image.save(output)
    print(f"\nSaved: {output}  ({elapsed:.1f}s)")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images from text using FLUX or SD 3.5 Large Turbo (local, NF4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", help="Text prompt (omit for interactive mode)")
    parser.add_argument(
        "--model", choices=["flux", "sd35"], default="flux",
        help="Model: flux = FLUX.1-schnell, sd35 = Stable Diffusion 3.5 Large Turbo",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Inference steps (4 is optimal for both distilled models)")
    parser.add_argument("--width",  type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed",   type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--output", type=Path, default=None, help="Output file path (.png)")
    parser.add_argument(
        "--caption", default=True, action=argparse.BooleanOptionalAction,
        help="Append the prompt as a caption below the image",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A CUDA-capable GPU is required.", file=sys.stderr)
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({vram_gb:.1f} GB VRAM)")

    pipe = load_pipeline(args.model)

    if args.prompt:
        generate(
            pipe, args.prompt,
            steps=args.steps, width=args.width, height=args.height,
            seed=args.seed, caption=args.caption, output=args.output,
        )
    else:
        print("\nInteractive mode — enter prompts (empty line or Ctrl-C to quit):\n")
        try:
            while True:
                prompt = input("prompt> ").strip()
                if not prompt:
                    break
                generate(
                    pipe, prompt,
                    steps=args.steps, width=args.width, height=args.height,
                    seed=args.seed, caption=args.caption,
                )
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
