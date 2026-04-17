#!/bin/bash

# Diffusion Policy Training Setup Script
# This script sets up the conda environment and runs the training

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Conda environment name
CONDA_ENV_NAME="diffusion-policy-ms"
PYTHON_VERSION="3.9"

# Get parameters from entrypoint script
EXP_NAME="${1}"
DEMO_PATH="${2}"
ENV_ID="${3}"
OBS_MODE="${4}"
CONTROL_MODE="${5}"
MAX_EPISODE_STEPS="${6}"
TOTAL_ITERS="${7}"
LR="${8}"
OBS_HORIZON="${9}"
ACT_HORIZON="${10}"
PRED_HORIZON="${11}"
BATCH_SIZE="${12}"
NUM_DEMOS="${13}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================================
# Setup and Activation
# ============================================================================

echo "=========================================="
echo "Diffusion Policy Training Setup"
echo "=========================================="
echo ""
echo "Experiment: $EXP_NAME"
echo "Environment: $ENV_ID"
echo "Demo Path: $DEMO_PATH"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install conda first: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
    exit 1
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Create the environment if it doesn't exist
if ! conda env list | grep -q "^${CONDA_ENV_NAME}"; then
    echo "Creating conda environment: ${CONDA_ENV_NAME}"
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

# Install dependencies in the environment
echo "Installing dependencies..."
cd "$SCRIPT_DIR"
pip install -e . > /dev/null 2>&1 || pip install -e .

# ============================================================================
# Training
# ============================================================================

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Build the training command
TRAINING_CMD="python train_rgbd.py \
    --exp_name $EXP_NAME \
    --track \
    --env_id $ENV_ID \
    --demo-path $DEMO_PATH \
    --total_iters $TOTAL_ITERS \
    --obs_mode $OBS_MODE \
    --max_episode_steps $MAX_EPISODE_STEPS \
    --control_mode $CONTROL_MODE \
    --demo_type motionplanning \
    --lr $LR \
    --obs_horizon $OBS_HORIZON \
    --act_horizon $ACT_HORIZON \
    --pred_horizon $PRED_HORIZON \
    --batch_size $BATCH_SIZE"

# Add optional num_demos parameter if provided
if [ -n "$NUM_DEMOS" ]; then
    TRAINING_CMD="$TRAINING_CMD \
    --num_demos $NUM_DEMOS"
fi

# Execute the training command
eval "$TRAINING_CMD"

# ============================================================================
# Cleanup
# ============================================================================

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="

