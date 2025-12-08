#!/bin/bash
# install_mini.sh - Only needed for training AutoGEO Mini
set -e

echo "=== Installing AutoGEO Mini Dependencies ==="
echo ""
echo "This will install:"
echo "  - vLLM (for inference)"
echo "  - Flash Attention (for efficient training)"
echo "  - open-r1 (reasoning framework)"
echo "  - LLaMA-Factory (training framework)"
echo ""
echo "Requirements:"
echo "  - CUDA 11.8+"
echo "  - ~50GB disk space"
echo "  - GPU with sufficient memory (A100 40GB+ recommended)"
echo ""

# Check if in autogeo environment
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "autogeo" ]]; then
    echo "Error: Please activate autogeo environment first"
    echo "Run: conda activate autogeo"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU training may not work."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# vLLM
echo "[1/7] Installing vLLM..."
pip install vllm==0.8.5.post1

# Flash Attention
echo "[2/7] Installing Flash Attention..."
pip install setuptools
pip install flash-attn --no-build-isolation

# open-r1
echo "[3/7] Installing open-r1..."
if [ ! -d "open-r1" ]; then
    echo "Error: open-r1 directory not found. Please ensure submodules are initialized."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi
cd open-r1
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
cd ..
echo "✓ open-r1 installed"

# Google Generative AI
echo "[4/7] Installing google-generativeai..."
pip install google-generativeai

# LLaMA-Factory
echo "[5/7] Installing LLaMA-Factory..."
if [ ! -d "LLaMA-Factory" ]; then
    echo "Error: LLaMA-Factory directory not found. Please ensure submodules are initialized."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
echo "✓ LLaMA-Factory installed"

# TRL
echo "[6/7] Installing TRL..."
pip install trl==0.22.0

# Additional dependencies
echo "[7/7] Installing additional dependencies..."
pip install -U antlr4-python3-runtime==4.13.0

echo ""
echo "=========================================="
echo "✓ AutoGEO Mini installation complete!"
echo "=========================================="
echo ""

