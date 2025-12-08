#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=general
#SBATCH --gpus=2
#SBATCH --mem=256G
#SBATCH --time=48:00:00

echo "Initializing Conda..."
eval "$(conda shell.bash hook)"

echo "Activating Conda environment: autogeo"
conda activate autogeo
echo "Conda environment activated."
echo ""

DATASET_NAME=${1:-"E-commerce"} # E-commerce, GEO-Bench, Researchy-GEO
DATA_DIR="./data"
DATASET_PATH="${DATA_DIR}/${DATASET_NAME}/RL/grpo_input.json"

# Validate dataset existence
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    echo "Available datasets: E-commerce, GEO-Bench, Researchy-GEO"
    exit 1
fi

echo "Job started on node: $HOSTNAME"
echo "SLURM assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET_NAME"
echo "Dataset path: $DATASET_PATH"

# Use SLURM_SUBMIT_DIR if available (when running via sbatch), otherwise use script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Set paths based on dataset name
MODEL_PATH="outputs/${DATASET_NAME}/cold_start"
OUTPUT_DIR="outputs/${DATASET_NAME}/grpo"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"

# Auto-detect latest checkpoint
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
        echo "Found checkpoint in output directory: $CHECKPOINT_NAME"
        echo "Training will automatically resume from: $LATEST_CHECKPOINT"
    else
        echo "Output directory exists but no checkpoint found. Training will start from scratch."
    fi
else
    echo "No existing output directory. Training will start from scratch."
fi
echo ""

echo "Starting VLLM Server for Completions on GPU 1..."
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

trl vllm-serve \
    --model ${MODEL_PATH} \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 > logs/vllm_${DATASET_NAME}.log 2>&1 &

VLLM_COMP_PID=$! 
echo "VLLM Completions Server started with PID: $VLLM_COMP_PID"

echo "Waiting for servers to be ready..."
while ! nc -z localhost 8000; do
  echo "Waiting for server on port 8000..."
  sleep 5
done
echo "Server on port 8000 is ready."


echo "Starting GRPO Training Client on GPU 0..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

# Execute training command (will auto-detect checkpoint from output_dir)
accelerate launch --config_file open-r1/recipes/accelerate_configs/zero3.yaml \
    open-r1/src/grpo.py \
    --config open-r1/recipes/autogeo/grpo.yaml \
    --dataset_name ${DATASET_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}

echo "Training finished."

echo "Cleaning up server processes..."
kill -9 $VLLM_COMP_PID
echo "Cleanup complete. Job finished."
