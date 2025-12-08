#!/bin/bash
#SBATCH --job-name=cold_start
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00                

# Usage: sh run_cold_start.sh [DATASET_NAME]
# Example: sh run_cold_start.sh Researchy-GEO
# Example: sh run_cold_start.sh E-commerce
# Example: sh run_cold_start.sh GEO-Bench

# Get parameters or use defaults
DATASET_NAME=${1:-"E-commerce"}

echo "========================================="
echo "Job started on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET_NAME"
echo "========================================="
echo ""

echo "Initializing Conda..."
eval "$(conda shell.bash hook)"

echo "Activating Conda environment: autogeo"
conda activate autogeo
echo "Conda environment activated."
echo ""

# Use SLURM_SUBMIT_DIR if running under SLURM, otherwise use script's directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/LLaMA-Factory"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/LLaMA-Factory" && pwd)"
fi
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo ""

echo "Starting LLaMA-Factory training..."
echo "Command: llamafactory-cli train examples/autogeo/cold_start.yaml"
echo ""

# Run training with dynamic dataset and save only last checkpoint
llamafactory-cli train examples/autogeo/cold_start.yaml \
    output_dir=outputs/${DATASET_NAME}/cold_start \
    learning_rate=5e-5 \
    dataset=$DATASET_NAME \
    save_steps=1000 \
    save_total_limit=1 \
    save_only_model=true \

echo ""
echo "========================================="
echo "Training finished."
echo "Job finished on: $(hostname)"
echo "========================================="

