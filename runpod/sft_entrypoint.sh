#!/bin/bash
# Stick-Gen SFT (Supervised Fine-Tuning) Entrypoint for RunPod
# Gestura AI - https://gestura.ai
#
# This script orchestrates SFT training on RunPod GPU Pods:
# 1. Initializes model from pretrained checkpoint (weights only)
# 2. Runs SFT training with curated high-quality data
# 3. Optionally uses LoRA for efficient fine-tuning
# 4. Pushes best checkpoint to HuggingFace Hub
#
# Environment Variables:
#   MODEL_VARIANT          - Which SFT config to use: sft_small, sft_base, sft_large (default: sft_base)
#   DATA_PATH              - Path to training data (default: /runpod-volume/data)
#   TRAIN_DATA_PATH        - Path to SFT dataset (default: DATA_PATH/curated/sft_data_embedded.pt)
#   INIT_FROM_CHECKPOINT   - Path to pretrained checkpoint for weight initialization (required for warm-start)
#   CHECKPOINT_DIR         - Where to save checkpoints (default: /runpod-volume/checkpoints/sft)
#   USE_LORA               - Enable LoRA fine-tuning (default: false)
#   LORA_RANK              - LoRA rank (default: 8)
#   HF_TOKEN               - HuggingFace token for model uploads
#   HF_REPO_NAME           - Base HuggingFace repo name (default: GesturaAI/stick-gen-sft)
#   AUTO_PUSH              - Auto-push to HuggingFace after training (default: true)
#   VERSION                - Model version tag for HuggingFace (default: 1.0.0)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration with defaults
MODEL_VARIANT="${MODEL_VARIANT:-sft_base}"
DATA_PATH="${DATA_PATH:-/runpod-volume/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/runpod-volume/checkpoints/sft}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"
USE_LORA="${USE_LORA:-false}"
LORA_RANK="${LORA_RANK:-8}"
AUTO_PUSH="${AUTO_PUSH:-true}"
VERSION="${VERSION:-1.0.0}"
HF_REPO_NAME="${HF_REPO_NAME:-GesturaAI/stick-gen-sft}"

# Default to curated SFT dataset
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_PATH}/curated/sft_data_embedded.pt}"

# Workspace directory
WORKSPACE="/workspace"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Stick-Gen SFT Training - Gestura AI                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  MODEL_VARIANT:        ${MODEL_VARIANT}"
echo "  DATA_PATH:            ${DATA_PATH}"
echo "  TRAIN_DATA_PATH:      ${TRAIN_DATA_PATH}"
echo "  CHECKPOINT_DIR:       ${CHECKPOINT_DIR}"
echo "  INIT_FROM_CHECKPOINT: ${INIT_FROM_CHECKPOINT:-[not set - training from scratch]}"
echo "  USE_LORA:             ${USE_LORA}"
echo "  LORA_RANK:            ${LORA_RANK}"
echo "  AUTO_PUSH:            ${AUTO_PUSH}"
echo "  VERSION:              ${VERSION}"
echo "  HF_REPO_NAME:         ${HF_REPO_NAME}"
echo ""

# Validate environment
validate_environment() {
    echo -e "${YELLOW}[1/4] Validating environment...${NC}"

    # Check if SFT dataset exists
    if [ ! -f "${TRAIN_DATA_PATH}" ]; then
        echo -e "${RED}ERROR: SFT dataset not found at ${TRAIN_DATA_PATH}${NC}"
        echo "  Run data preparation with --curated flag first."
        exit 1
    fi

    # Check if init checkpoint exists (if specified)
    if [ -n "${INIT_FROM_CHECKPOINT}" ] && [ ! -f "${INIT_FROM_CHECKPOINT}" ]; then
        echo -e "${RED}ERROR: Init checkpoint not found at ${INIT_FROM_CHECKPOINT}${NC}"
        exit 1
    fi

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}  GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/    /'
    else
        echo -e "${RED}ERROR: No NVIDIA GPU detected${NC}"
        exit 1
    fi

    # Create checkpoint directory
    mkdir -p "${CHECKPOINT_DIR}"

    echo -e "${GREEN}  ✓ Environment validated${NC}"
}

# Run SFT training
run_sft_training() {
    echo ""
    echo -e "${YELLOW}[2/4] Running SFT training...${NC}"
    echo "  Config: configs/${MODEL_VARIANT}.yaml"
    echo "  Data:   ${TRAIN_DATA_PATH}"
    
    cd "${WORKSPACE}"

    # Build training command
    TRAIN_CMD="python -m src.train.train"
    TRAIN_CMD="${TRAIN_CMD} --config configs/${MODEL_VARIANT}.yaml"
    TRAIN_CMD="${TRAIN_CMD} --data_path ${TRAIN_DATA_PATH}"
    TRAIN_CMD="${TRAIN_CMD} --checkpoint_dir ${CHECKPOINT_DIR}"

    # Add init_from if specified
    if [ -n "${INIT_FROM_CHECKPOINT}" ]; then
        echo "  Initializing from: ${INIT_FROM_CHECKPOINT}"
        TRAIN_CMD="${TRAIN_CMD} --init_from ${INIT_FROM_CHECKPOINT}"
    fi

    # Run training
    echo ""
    echo -e "${CYAN}Executing: ${TRAIN_CMD}${NC}"
    eval "${TRAIN_CMD}" 2>&1 | tee "${CHECKPOINT_DIR}/sft_training.log"

    local train_exit_code=${PIPESTATUS[0]}
    if [ $train_exit_code -ne 0 ]; then
        echo -e "${RED}ERROR: SFT training failed with exit code ${train_exit_code}${NC}"
        return 1
    fi

    echo -e "${GREEN}  ✓ SFT training completed${NC}"
}

# Push to HuggingFace (reuse logic from train_entrypoint.sh)
push_to_huggingface() {
    echo ""
    echo -e "${YELLOW}[3/4] Pushing SFT model to HuggingFace...${NC}"

    local checkpoint="${CHECKPOINT_DIR}/model_checkpoint_best.pth"
    if [ ! -f "${checkpoint}" ]; then
        echo -e "${YELLOW}WARNING: Best checkpoint not found at ${checkpoint}${NC}"
        return 1
    fi

    cd "${WORKSPACE}"
    local variant_suffix="${MODEL_VARIANT#sft_}"  # Remove sft_ prefix
    local full_repo_name="${HF_REPO_NAME}-${variant_suffix}"

    python scripts/push_to_huggingface.py \
        --checkpoint "${checkpoint}" \
        --variant "${variant_suffix}" \
        --version "${VERSION}" \
        --token "${HF_TOKEN}" \
        --repo-name "${full_repo_name}" \
        --skip-validation || return 1

    echo -e "${GREEN}  ✓ Model pushed to: https://huggingface.co/${full_repo_name}${NC}"
}

# Main execution
main() {
    validate_environment
    run_sft_training || exit 1

    if [ "${AUTO_PUSH}" = "true" ] && [ -n "${HF_TOKEN}" ]; then
        push_to_huggingface || echo -e "${YELLOW}  ⚠️  HuggingFace push failed${NC}"
    fi

    echo ""
    echo -e "${GREEN}[4/4] SFT workflow completed!${NC}"
    echo -e "${YELLOW}REMINDER: Terminate this Pod to stop GPU charges!${NC}"
}

main "$@"

