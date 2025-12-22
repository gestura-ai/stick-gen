#!/bin/bash
# Stick-Gen Training Entrypoint for RunPod
# Gestura AI - https://gestura.ai
#
# This script orchestrates model training on RunPod GPU Pods:
# 1. Validates environment and data availability
# 2. Runs training for specified variant (small, medium, large)
# 3. Runs comprehensive evaluation on trained model
# 4. Optionally uploads dataset to HuggingFace Dataset Hub
# 5. Pushes trained model to HuggingFace Model Hub
#
# Environment Variables:
#   MODEL_VARIANT          - Which config to use: small, medium (base), large (default: medium)
#   DATA_PATH              - Path to data directory (default: /runpod-volume/data)
#   TRAIN_DATA_PATH        - Path to training data (default: DATA_PATH/curated/pretrain_data_embedded.pt)
#   CHECKPOINT_DIR         - Where to save checkpoints (default: /runpod-volume/checkpoints)
#   HF_TOKEN               - HuggingFace token for model/dataset uploads
#   HF_REPO_NAME           - Base HuggingFace repo name (default: GesturaAI/stick-gen)
#   HF_DATASET_REPO        - Dataset repo name (default: GesturaAI/stick-gen-dataset)
#   AUTO_PUSH              - Auto-push to HuggingFace after training (default: true)
#   UPLOAD_DATASET         - Upload dataset to HuggingFace (default: false)
#   VERSION                - Model version tag for HuggingFace (default: 1.0.0)
#   AUTO_CLEANUP           - Terminate pod after completion (default: false)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration with defaults
MODEL_VARIANT="${MODEL_VARIANT:-medium}"
DATA_PATH="${DATA_PATH:-/runpod-volume/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/runpod-volume/checkpoints}"
AUTO_PUSH="${AUTO_PUSH:-true}"
UPLOAD_DATASET="${UPLOAD_DATASET:-false}"
VERSION="${VERSION:-1.0.0}"
HF_REPO_NAME="${HF_REPO_NAME:-GesturaAI/stick-gen}"
HF_DATASET_REPO="${HF_DATASET_REPO:-GesturaAI/stick-gen-dataset}"
AUTO_CLEANUP="${AUTO_CLEANUP:-false}"

# Normalize variant name (base -> medium)
if [ "${MODEL_VARIANT}" = "base" ]; then
    MODEL_VARIANT="medium"
fi

# Default to curated pretrain dataset
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_PATH}/curated/pretrain_data_embedded.pt}"

# Workspace directory
WORKSPACE="/workspace"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Stick-Gen Training Pipeline - Gestura AI               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  MODEL_VARIANT:     ${MODEL_VARIANT}"
echo "  DATA_PATH:         ${DATA_PATH}"
echo "  TRAIN_DATA_PATH:   ${TRAIN_DATA_PATH}"
echo "  CHECKPOINT_DIR:    ${CHECKPOINT_DIR}"
echo "  AUTO_PUSH:         ${AUTO_PUSH}"
echo "  UPLOAD_DATASET:    ${UPLOAD_DATASET}"
echo "  VERSION:           ${VERSION}"
echo "  HF_REPO_NAME:      ${HF_REPO_NAME}"
echo "  HF_DATASET_REPO:   ${HF_DATASET_REPO}"
echo "  AUTO_CLEANUP:      ${AUTO_CLEANUP}"
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

validate_environment() {
    echo -e "${YELLOW}[1/5] Validating environment...${NC}"

    # Check for data prep failure marker first
    local failure_marker="${DATA_PATH}/.prep_failed"
    if [ -f "${failure_marker}" ]; then
        echo -e "${RED}ERROR: Data preparation failed!${NC}"
        echo ""
        echo "  Failure marker found: ${failure_marker}"
        echo "  Contents:"
        cat "${failure_marker}" | sed 's/^/    /'
        echo ""
        echo "  Check data prep logs at: ${DATA_PATH}/logs/"
        echo "  Re-run data preparation with FORCE_REGENERATE=true"
        exit 1
    fi

    # Check for completion marker
    local completion_marker="${DATA_PATH}/.prep_complete"
    if [ -f "${completion_marker}" ]; then
        echo -e "${GREEN}  ✓ Data prep completion marker found${NC}"
        # Try to extract output_path from marker (if jq available)
        if command -v jq &> /dev/null && [ -f "${completion_marker}" ]; then
            local marker_output=$(jq -r '.output_path // empty' "${completion_marker}" 2>/dev/null)
            if [ -n "${marker_output}" ] && [ -f "${marker_output}" ]; then
                echo "    Output path from marker: ${marker_output}"
            fi
        fi
    else
        echo -e "${YELLOW}  ⚠ No completion marker found at ${completion_marker}${NC}"
        echo "    Data preparation may not have completed successfully."
    fi

    # Check if training data exists
    if [ ! -f "${TRAIN_DATA_PATH}" ]; then
        echo -e "${RED}ERROR: Training data not found at ${TRAIN_DATA_PATH}${NC}"
        echo ""
        echo "  Troubleshooting:"
        echo "    1. Run data preparation first: ./runpod/deploy.sh prep-data"
        echo "    2. Check if USE_CURATED_DATA matches your data prep mode:"
        echo "       - If USE_CURATED_DATA=true was used: data at ${DATA_PATH}/curated/pretrain_data_embedded.pt"
        echo "       - If USE_CURATED_DATA=false was used: data at ${DATA_PATH}/train_data_final.pt"
        echo "    3. Set TRAIN_DATA_PATH explicitly to match your data location"
        echo "    4. Check data prep logs: ls -la ${DATA_PATH}/logs/"
        echo ""
        echo "  Available files in ${DATA_PATH}:"
        ls -la "${DATA_PATH}/" 2>/dev/null | head -20 || echo "    (directory not accessible)"
        echo ""
        if [ -d "${DATA_PATH}/curated" ]; then
            echo "  Files in ${DATA_PATH}/curated/:"
            ls -la "${DATA_PATH}/curated/" 2>/dev/null | head -10 || echo "    (empty or not accessible)"
        fi
        exit 1
    fi
    echo -e "${GREEN}  ✓ Training data found: ${TRAIN_DATA_PATH}${NC}"

    # Show training data size
    local data_size=$(du -sh "${TRAIN_DATA_PATH}" 2>/dev/null | cut -f1)
    echo "    Size: ${data_size}"

    # Check config file exists
    local config_file="configs/${MODEL_VARIANT}.yaml"
    if [ ! -f "${WORKSPACE}/${config_file}" ]; then
        echo -e "${RED}ERROR: Config not found at ${WORKSPACE}/${config_file}${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Config found: ${config_file}${NC}"

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}  ✓ GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/    /'
    else
        echo -e "${RED}ERROR: No NVIDIA GPU detected${NC}"
        exit 1
    fi

    # Create checkpoint directory
    mkdir -p "${CHECKPOINT_DIR}"

    echo -e "${GREEN}  ✓ Environment validated${NC}"
}

run_training() {
    local variant="$1"
    
    echo ""
    echo -e "${YELLOW}[2/5] Running training for ${variant}...${NC}"
    echo "  Config: configs/${variant}.yaml"
    echo "  Data:   ${TRAIN_DATA_PATH}"
    
    cd "${WORKSPACE}"

    # Build training command
    local train_cmd="python -m src.train.train"
    train_cmd="${train_cmd} --config configs/${variant}.yaml"
    train_cmd="${train_cmd} --data_path ${TRAIN_DATA_PATH}"
    train_cmd="${train_cmd} --checkpoint_dir ${CHECKPOINT_DIR}"

    # Run training
    echo ""
    echo -e "${CYAN}Executing: ${train_cmd}${NC}"
    eval "${train_cmd}" 2>&1 | tee "${CHECKPOINT_DIR}/training.log"

    local train_exit_code=${PIPESTATUS[0]}
    if [ $train_exit_code -ne 0 ]; then
        echo -e "${RED}ERROR: Training failed with exit code ${train_exit_code}${NC}"
        return 1
    fi

    echo -e "${GREEN}  ✓ Training completed${NC}"
}

run_evaluation() {
    local variant="$1"
    local checkpoint="${CHECKPOINT_DIR}/model_checkpoint_best.pth"
    local output_file="${CHECKPOINT_DIR}/evaluation_results.json"
    
    echo ""
    echo -e "${YELLOW}[3/5] Running comprehensive evaluation for ${variant}...${NC}"
    
    # Check for checkpoint
    if [ ! -f "${checkpoint}" ]; then
        # Try alternate name
        checkpoint="${CHECKPOINT_DIR}/best_model.pth"
        if [ ! -f "${checkpoint}" ]; then
            echo -e "${YELLOW}  WARNING: Checkpoint not found for evaluation${NC}"
            return 1
        fi
    fi
    
    cd "${WORKSPACE}"
    
    python scripts/run_comprehensive_eval.py \
        --checkpoint "${checkpoint}" \
        --output "${output_file}" \
        --num_samples 50
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Evaluation complete${NC}"
        echo -e "${CYAN}  Results:${NC}"
        cat "${output_file}" | head -20
    else
        echo -e "${YELLOW}  ⚠️ Evaluation failed (non-critical)${NC}"
    fi
}

upload_dataset() {
    if [ "${UPLOAD_DATASET}" != "true" ]; then
        echo -e "${YELLOW}[Optional] Dataset upload skipped (UPLOAD_DATASET=${UPLOAD_DATASET})${NC}"
        return 0
    fi
    
    echo ""
    echo -e "${YELLOW}[3.5/5] Uploading dataset to HuggingFace...${NC}"
    
    if [ -z "${HF_TOKEN:-}" ]; then
        echo -e "${YELLOW}  WARNING: HF_TOKEN not set - skipping dataset upload${NC}"
        return 0
    fi
    
    cd "${WORKSPACE}"
    
    python scripts/push_dataset_to_hub.py \
        --dataset-path "${TRAIN_DATA_PATH}" \
        --repo-id "${HF_DATASET_REPO}" \
        --token "${HF_TOKEN}"
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✅ Dataset uploaded: ${HF_DATASET_REPO}${NC}"
    else
        echo -e "${YELLOW}  ⚠️ Dataset upload failed (non-critical)${NC}"
    fi
}

push_to_huggingface() {
    local variant="$1"
    local checkpoint="${CHECKPOINT_DIR}/model_checkpoint_best.pth"
    local metrics_file="${CHECKPOINT_DIR}/evaluation_results.json"
    
    echo ""
    echo -e "${YELLOW}[4/5] Pushing model to HuggingFace...${NC}"

    # Check for checkpoint
    if [ ! -f "${checkpoint}" ]; then
        checkpoint="${CHECKPOINT_DIR}/best_model.pth"
        if [ ! -f "${checkpoint}" ]; then
            echo -e "${YELLOW}  WARNING: No checkpoint found to upload${NC}"
            return 1
        fi
    fi

    if [ -z "${HF_TOKEN:-}" ]; then
        echo -e "${YELLOW}  WARNING: HF_TOKEN not set - skipping HuggingFace push${NC}"
        echo "  Set HF_TOKEN environment variable to enable auto-push"
        return 0
    fi

    cd "${WORKSPACE}"
    
    local full_repo_name="${HF_REPO_NAME}-${variant}"
    
    # Build push command with optional metrics
    local push_cmd="python scripts/push_to_hub.py"
    push_cmd="${push_cmd} --checkpoint ${checkpoint}"
    push_cmd="${push_cmd} --variant ${variant}"
    push_cmd="${push_cmd} --version ${VERSION}"
    push_cmd="${push_cmd} --token ${HF_TOKEN}"
    push_cmd="${push_cmd} --repo-name ${full_repo_name}"
    
    # Add metrics if available
    if [ -f "${metrics_file}" ]; then
        push_cmd="${push_cmd} --metrics ${metrics_file}"
    fi

    echo -e "${CYAN}Executing: ${push_cmd}${NC}"
    eval "${push_cmd}"
    
    local push_exit_code=$?
    if [ $push_exit_code -eq 0 ]; then
        echo -e "${GREEN}  ✓ Model pushed to: https://huggingface.co/${full_repo_name}${NC}"
    else
        echo -e "${RED}  ❌ HuggingFace push failed${NC}"
        return 1
    fi
}

cleanup() {
    echo ""
    echo -e "${YELLOW}[5/5] Cleanup...${NC}"
    
    if [ "${AUTO_CLEANUP}" = "true" ]; then
        echo -e "${CYAN}  Auto-cleanup enabled - pod will terminate${NC}"
        echo -e "${GREEN}  Training artifacts saved to: ${CHECKPOINT_DIR}${NC}"
        # RunPod will auto-terminate when the script exits
    else
        echo -e "${GREEN}  Training complete!${NC}"
        echo -e "${YELLOW}  REMINDER: Terminate this Pod to stop GPU charges!${NC}"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local variant="${MODEL_VARIANT}"
    
    # Step 1: Validate environment
    validate_environment
    
    # Step 2: Run training
    run_training "${variant}" || exit 1
    
    # Step 3: Run evaluation
    run_evaluation "${variant}"
    
    # Step 3.5: Upload dataset (optional)
    upload_dataset
    
    # Step 4: Push to HuggingFace
    if [ "${AUTO_PUSH}" = "true" ]; then
        push_to_huggingface "${variant}" || echo -e "${YELLOW}  ⚠️ HuggingFace push failed${NC}"
    else
        echo -e "${YELLOW}[4/5] HuggingFace push skipped (AUTO_PUSH=${AUTO_PUSH})${NC}"
    fi
    
    # Step 5: Cleanup
    cleanup
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Training Pipeline Complete!                            ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Summary:"
    echo "  Variant:     ${variant}"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    if [ "${AUTO_PUSH}" = "true" ] && [ -n "${HF_TOKEN:-}" ]; then
        echo "  HuggingFace: https://huggingface.co/${HF_REPO_NAME}-${variant}"
    fi
    if [ "${UPLOAD_DATASET}" = "true" ] && [ -n "${HF_TOKEN:-}" ]; then
        echo "  Dataset:     https://huggingface.co/datasets/${HF_DATASET_REPO}"
    fi
}

main "$@"
