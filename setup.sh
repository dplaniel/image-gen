#!/usr/bin/env bash
# One-time setup: creates a venv and installs all dependencies.
# Run once: bash setup.sh
# Then activate: source .venv/bin/activate
set -euo pipefail

VENV=".venv"

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip

# PyTorch with CUDA 13.2 wheels
pip install torch==2.12.0+cu132 torchvision==0.27.0+cu132 \
    --index-url https://download.pytorch.org/whl/cu132

# Remaining dependencies
pip install -r requirements.txt

# bitsandbytes 0.49.x ships cu130 at most; cu132 is ABI-compatible
BNB_DIR="$VENV/lib/$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/bitsandbytes"
if [ -f "$BNB_DIR/libbitsandbytes_cuda130.so" ] && [ ! -f "$BNB_DIR/libbitsandbytes_cuda132.so" ]; then
    ln -sf "$BNB_DIR/libbitsandbytes_cuda130.so" "$BNB_DIR/libbitsandbytes_cuda132.so"
fi

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then generate an image with:"
echo "  python generate.py \"a sunset over misty mountains, photorealistic\""
