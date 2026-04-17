#!/bin/bash

# Diffusion Policy Training Entry Point Script
# This script sets up the training parameters and calls the setup script
# Modify the variables below to customize your training run

# ============================================================================
# TRAINING PARAMETERS - Modify these as needed
# ============================================================================

# Experiment name
EXP_NAME="pick_cube_diffusion"

# Path to demonstration data (IMPORTANT: Change this to your demo file path)
DEMO_PATH="/home/rakasheh/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"

# Environment ID
ENV_ID="PickCube-v1"

# Observation mode (rgb, state, etc.)
OBS_MODE="rgb"

# Control mode
CONTROL_MODE="pd_ee_delta_pose"

# Maximum episode steps
MAX_EPISODE_STEPS=100

# Total training iterations
TOTAL_ITERS=30000

# Number of demonstration trajectories to use (optional, comment out to use all)
# NUM_DEMOS=100

# ============================================================================
# DIFFUSION POLICY HYPERPARAMETERS
# ============================================================================

# Learning rate
LR=1e-4

# Observation horizon (number of past observations to condition on)
OBS_HORIZON=2

# Action horizon (number of actions to predict)
ACT_HORIZON=8

# Prediction horizon (number of diffusion steps)
PRED_HORIZON=16

# Batch size
BATCH_SIZE=256

# ============================================================================
# WANDB SETTINGS
# ============================================================================

# Set your Weights & Biases API key here (leave empty to skip wandb logging)
# Get your key from https://wandb.ai/authorize
WANDB_API_KEY=""

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set the wandb API key if provided
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
fi

# Call the setup script with all parameters
"$SCRIPT_DIR/train_diffusion_setup.sh" \
    "$EXP_NAME" \
    "$DEMO_PATH" \
    "$ENV_ID" \
    "$OBS_MODE" \
    "$CONTROL_MODE" \
    "$MAX_EPISODE_STEPS" \
    "$TOTAL_ITERS" \
    "$LR" \
    "$OBS_HORIZON" \
    "$ACT_HORIZON" \
    "$PRED_HORIZON" \
    "$BATCH_SIZE" \
    "${NUM_DEMOS:-}"

echo "Training script completed!"

