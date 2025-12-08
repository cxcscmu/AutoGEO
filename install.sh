#!/bin/bash
set -e  # Exit on error

echo "=== AutoGEO Environment Setup ==="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "Creating conda environment 'autogeo' with Python 3.11..."
conda create -n autogeo python=3.11 -y
echo "✓ Environment created"

# Get the conda base path
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate autogeo

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install basic dependencies
echo "Installing basic dependencies..."
pip install -r requirements.txt
echo "✓ Basic dependencies installed"

# Setup API keys
if [ ! -f "keys.env" ]; then
    echo "Creating keys.env from template..."
    cp keys.env.example keys.env
    echo "⚠️  Please edit keys.env with your API keys before using AutoGEO"
else
    echo "✓ keys.env already exists"
fi

echo ""
echo "=========================================="
echo "✓ Basic installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:  conda activate autogeo"
echo "  2. Edit API keys:         cp  keys.env.example keys.env"
echo ""
echo "Optional: For AutoGEO Mini training, run:"
echo "  bash install_mini.sh"
echo ""

