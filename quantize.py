#!/usr/bin/env python3
"""
Pre-quantize models to NF4 and save to disk (run once per model).

T5-XXL weights are shared between FLUX and SD 3.5 (same Google checkpoint);
only one copy is saved at models/text_encoder_2/.

Usage:
  python quantize.py              # FLUX only (default)
  python quantize.py --model sd35
  python quantize.py --model all
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
import torch
from pathlib import Path

from diffusers import FluxTransformer2DModel
from transformers import BitsAndBytesConfig, T5EncoderModel

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SD35_MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"

MODELS_DIR = Path(__file__).parent / "models"

NF4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def saved_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def quantize_t5(model_id: str = FLUX_MODEL_ID, subfolder: str = "text_encoder_2") -> None:
    """Quantize T5-XXL to NF4. Shared by FLUX and SD 3.5 — only saved once."""
    out = MODELS_DIR / "text_encoder_2"
    if out.exists():
        print(f"T5 already saved at {out}  ({saved_gb(out):.2f} GB) — skipping.")
        return

    print(f"Quantizing T5-v1.1-xxl → NF4 …  (loading from {model_id}/{subfolder})")
    vram_free = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated()) / 1e9
    print(f"  VRAM free: {vram_free:.1f} GB")

    t0 = time.perf_counter()
    model = T5EncoderModel.from_pretrained(
        model_id,
        subfolder=subfolder,
        quantization_config=NF4,
        torch_dtype=torch.bfloat16,
    )
    print(f"  quantized in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    model.save_pretrained(out)
    print(f"  saved in {time.perf_counter() - t0:.1f}s  →  {saved_gb(out):.2f} GB on disk")

    del model
    torch.cuda.empty_cache()


def quantize_flux_transformer() -> None:
    out = MODELS_DIR / "transformer"
    if out.exists():
        print(f"FLUX transformer already saved at {out}  ({saved_gb(out):.2f} GB) — skipping.")
        return

    print("Quantizing FLUX.1-schnell transformer → NF4 …")
    t0 = time.perf_counter()
    model = FluxTransformer2DModel.from_pretrained(
        FLUX_MODEL_ID,
        subfolder="transformer",
        quantization_config=NF4,
        torch_dtype=torch.bfloat16,
    )
    print(f"  quantized in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    model.save_pretrained(out)
    print(f"  saved in {time.perf_counter() - t0:.1f}s  →  {saved_gb(out):.2f} GB on disk")

    del model
    torch.cuda.empty_cache()


def quantize_sd35_transformer() -> None:
    from diffusers import SD3Transformer2DModel

    out = MODELS_DIR / "sd35_transformer"
    if out.exists():
        print(f"SD 3.5 transformer already saved at {out}  ({saved_gb(out):.2f} GB) — skipping.")
        return

    print("Quantizing SD 3.5 Large Turbo transformer → NF4 …")
    t0 = time.perf_counter()
    model = SD3Transformer2DModel.from_pretrained(
        SD35_MODEL_ID,
        subfolder="transformer",
        quantization_config=NF4,
        torch_dtype=torch.bfloat16,
    )
    print(f"  quantized in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    model.save_pretrained(out)
    print(f"  saved in {time.perf_counter() - t0:.1f}s  →  {saved_gb(out):.2f} GB on disk")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    parser = argparse.ArgumentParser(description="Pre-quantize models to NF4.")
    parser.add_argument(
        "--model", choices=["flux", "sd35", "all"], default="flux",
        help="Which model(s) to quantize (default: flux)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    if args.model in ("flux", "all"):
        # T5 must be quantized first while VRAM is mostly free
        quantize_t5(FLUX_MODEL_ID, "text_encoder_2")
        quantize_flux_transformer()

    if args.model in ("sd35", "all"):
        # Reuse the T5 save from FLUX (same Google T5-v1.1-xxl checkpoint)
        quantize_t5(SD35_MODEL_ID, "text_encoder_3")
        quantize_sd35_transformer()

    print("\nSummary:")
    for name in ("text_encoder_2", "transformer", "sd35_transformer"):
        p = MODELS_DIR / name
        if p.exists():
            print(f"  {name}: {saved_gb(p):.2f} GB")
    print("\ngenerate.py will use these models automatically on next run.")
