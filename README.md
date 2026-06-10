# Local Image Generation — FLUX & SD 3.5

Text-to-image generation running entirely locally on an RTX 3080 12 GB.
Supports two models, both distilled to 4 inference steps.

| Model | Params | NF4 VRAM | License |
|---|---|---|---|
| **FLUX.1-schnell** | 12B | ~6.7 GB | Apache 2.0 |
| **SD 3.5 Large Turbo** | 8B | ~4.5 GB | Stability AI Community |

Both require a HuggingFace account and license acceptance (see Setup).

---

## How it works

T5-XXL (the text encoder, 4.3 GB NF4) and the diffusion transformer can't coexist
on this GPU simultaneously. Each image generation time-shares the GPU:

1. Unload transformer → load T5 on GPU → encode prompt in ~0.14 s → unload T5
2. Reload transformer → denoise → VAE decode

Total per image: **~9 seconds** (vs ~127 s if T5 ran on CPU).

Model weights are pre-quantized to NF4 once via `quantize.py` and saved to disk
(~11 GB total for FLUX, ~9 GB for SD 3.5), cutting load-time disk reads by ~3×.
Outputs save to `/data/big/flux_images/`.

---

## Setup

**One-time:**

```bash
# 1. Accept model licenses on HuggingFace:
#    https://huggingface.co/black-forest-labs/FLUX.1-schnell
#    https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo

# 2. Create venv and install dependencies
bash setup.sh
source .venv/bin/activate

# 3. Authenticate with HuggingFace
hf auth login

# 4. Pre-quantize weights (downloads ~33 GB, saves ~11 GB NF4; takes ~10 min)
python quantize.py                  # FLUX
python quantize.py --model sd35     # SD 3.5 Large Turbo (~16 GB download)
```

Pre-quantized models are saved under `models/`:

```
models/
  text_encoder_2/   # T5-XXL NF4 (~4.3 GB) — shared by both models
  transformer/      # FLUX NF4 (~6.7 GB)
  sd35_transformer/ # SD 3.5 NF4 (~4.5 GB)
```

---

## Usage

```bash
source .venv/bin/activate

# FLUX (default)
python generate.py "a misty forest at dawn, cinematic lighting"

# SD 3.5 Large Turbo
python generate.py "portrait in oil paint style" --model sd35

# Options
python generate.py "prompt" --seed 42 --width 896 --height 1152
python generate.py "prompt" --steps 8
python generate.py "prompt" --no-caption   # omit prompt caption bar
python generate.py "prompt" --output ~/my_image.png

# Interactive mode (keeps model loaded between prompts)
python generate.py
python generate.py --model sd35
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--model` | `flux` | `flux` or `sd35` |
| `--steps` | `4` | Inference steps (4 is optimal for both distilled models) |
| `--width` | `1024` | Output width in pixels |
| `--height` | `1024` | Output height in pixels |
| `--seed` | random | Integer seed for reproducibility |
| `--output` | auto | Custom output path (`.png`) |
| `--caption` / `--no-caption` | on | Append prompt as caption below image |

---

## Hardware requirements

- **GPU:** CUDA-capable, 12 GB VRAM (tested on RTX 3080 12 GB)
- **RAM:** 8 GB+ (T5 is held in GPU memory during encoding, then freed)
- **Disk:** ~25 GB for pre-quantized models + HuggingFace cache (~50 GB peak during quantization)
- **CUDA:** 13.2 (PyTorch `cu132` wheels; bitsandbytes symlink handled by `setup.sh`)

Smaller GPUs with less VRAM are not supported without further quantization or model changes.

---

## File overview

| File | Purpose |
|---|---|
| `generate.py` | Main generation script |
| `quantize.py` | One-time offline NF4 quantization |
| `setup.sh` | Creates venv, installs deps, patches bitsandbytes for CUDA 13.2 |
| `requirements.txt` | pip dependencies |
