#!/bin/bash
# Stick-Gen Data Preparation Entrypoint for RunPod
# Gestura AI - https://gestura.ai
#
# This script runs automatically when a data preparation Pod starts.
# It orchestrates the complete data preparation pipeline:
#   1. Validates raw data exists on Network Volume
#   2. Converts all motion capture datasets to canonical format
#   3. Merges canonical datasets into unified training data
#   4. Generates synthetic motion-conditioned samples (augmented from real data)
#   5. Curates into pretrain/SFT splits with quality filtering
#   6. Optionally uploads dataset to HuggingFace
#   7. Writes completion marker for orchestration
#
# Pipeline Modes (in order of precedence):
#   USE_ASYNC_PIPELINE=true (default) - Async pipeline with motion-conditioned generation
#   USE_CURATED_DATA=true             - Legacy curated pipeline
#   Both false                        - Legacy synchronous prepare_data.py
#
# Environment Variables:
#   DATA_PATH                    - Path to raw/canonical data (default: /runpod-volume/data)
#   OUTPUT_PATH                  - Output path for final training dataset
#                                   Async mode default: $DATA_PATH/processed/curated/pretrain_data_embedded.pt
#                                   Legacy mode default: $DATA_PATH/train_data_final.pt
#                                   Curated mode default: $DATA_PATH/curated/pretrain_data_embedded.pt
#   USE_ASYNC_PIPELINE           - Use new async data pipeline (default: true)
#   ASYNC_MAX_CONCURRENT         - Max concurrent converter processes (default: 4)
#   ASYNC_SKIP_SYNTHETIC         - Skip synthetic generation (default: false)
#   ASYNC_RESUME                 - Resume from checkpoint if available (default: true)
#   USE_CURATED_DATA             - Use legacy curated pipeline (default: false)
#   CURATED_OUTPUT_DIR           - Directory for curated outputs (default: $DATA_PATH/curated)
#   CURATION_CANONICAL_DIR       - Directory containing canonical .pt inputs (default: $DATA_PATH/canonical)
#   CURATION_INPUTS              - Optional space-separated list of canonical .pt files
#   CURATION_MIN_QUALITY_PRETRAIN        - Min quality_score for pretrain split (default: 0.5)
#   CURATION_MIN_QUALITY_SFT            - Min quality_score for SFT split (default: 0.8)
#   CURATION_MIN_CAMERA_STABILITY_SFT   - Min camera stability score for SFT (default: 0.6)
#   CURATION_BALANCE_MAX_FRACTION       - Max fraction per action in SFT balancing (default: 0.3)
#   MERGE_BALANCE_SOURCES               - Enable source balancing in merge step (default: true)
#   MERGE_MAX_SOURCE_FRACTION           - Max fraction per source in merged dataset (default: 0.3)
#   MERGE_FILTER_ARTIFACTS              - Filter artifacts during merge (default: true)
#   MERGE_MAX_ARTIFACT_SCORE            - Max artifact score to keep during merge (default: 0.4)
#   MERGE_MIN_FRAMES                    - Min sequence length in frames (default: 25)
#   MERGE_MAX_FRAMES                    - Max sequence length in frames (default: 500)
#   SYNTHETIC_SAMPLES            - Number of synthetic samples to generate (default: 50000)
#   HF_TOKEN                     - HuggingFace token (required for dataset upload)
#   HF_PUSH_DATASET              - Whether to upload dataset to HuggingFace (default: true)
#   HF_ASYNC_UPLOAD              - Upload in background while training starts (default: true)
#   HF_DATASET_REPO              - HuggingFace dataset repository (default: GesturaAI/stick-gen-dataset)
#   GROK_API_KEY                 - X.AI Grok API key (for LLM-enhanced dataset generation)
#                                   Set in RunPod secrets as RUNPOD_SECRET_GROK_API_KEY
#                                   When set, real LLM is automatically used (no mock)
#   DISABLE_MOCK_LLM             - Set to "true" to force real LLM usage
#   FORCE_MOCK_LLM               - Set to "true" to force mock LLM even if GROK_API_KEY is set
#   FORCE_REGENERATE             - Set to "true" to overwrite previous training data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration with defaults
WORKSPACE="${WORKSPACE:-/workspace}"
DATA_PATH="${DATA_PATH:-/runpod-volume/data}"

# Pipeline mode: async (default), legacy, or curated
#
# USE_ASYNC_PIPELINE=true (default) will:
#   - Use the new AsyncDataPrepPipeline from src/data_gen/async_data_prep.py
#   - Run all dataset converters concurrently with proper dependency ordering
#   - Follow correct phase order: Convert → Merge → Synthetic → Curate
#   - Generate motion-conditioned synthetic samples (augmented from real data)
#   - Support checkpointing and resume for long-running jobs
#
# USE_CURATED_DATA=true (legacy curated mode) will:
#   - Load canonical .pt files (from CURATION_INPUTS or CURATION_CANONICAL_DIR)
#   - Run scripts/prepare_curated_datasets.py to create pretrain_data.pt/sft_data.pt
#   - Run scripts/build_dataset_for_training.py to add text embeddings
#   - Use the resulting pretrain dataset as OUTPUT_PATH for validation/upload
#
# If both are false, fall back to legacy prepare_data.py pipeline
USE_ASYNC_PIPELINE="${USE_ASYNC_PIPELINE:-true}"
USE_CURATED_DATA="${USE_CURATED_DATA:-false}"
CURATED_OUTPUT_DIR="${CURATED_OUTPUT_DIR:-${DATA_PATH}/curated}"
CURATION_CANONICAL_DIR="${CURATION_CANONICAL_DIR:-${DATA_PATH}/canonical}"

# Allow overriding OUTPUT_PATH; otherwise choose sensible default per mode
if [ -z "${OUTPUT_PATH:-}" ]; then
    if [ "${USE_ASYNC_PIPELINE}" = "true" ]; then
        OUTPUT_PATH="${DATA_PATH}/processed/curated/pretrain_data_embedded.pt"
    elif [ "${USE_CURATED_DATA}" = "true" ]; then
        OUTPUT_PATH="${CURATED_OUTPUT_DIR}/pretrain_data_embedded.pt"
    else
        OUTPUT_PATH="${DATA_PATH}/train_data_final.pt"
    fi
fi

# Thresholds for curated pipeline (match scripts/prepare_curated_datasets.py defaults)
CURATION_MIN_QUALITY_PRETRAIN="${CURATION_MIN_QUALITY_PRETRAIN:-0.5}"
CURATION_MIN_QUALITY_SFT="${CURATION_MIN_QUALITY_SFT:-0.8}"
CURATION_MIN_CAMERA_STABILITY_SFT="${CURATION_MIN_CAMERA_STABILITY_SFT:-0.6}"
CURATION_BALANCE_MAX_FRACTION="${CURATION_BALANCE_MAX_FRACTION:-0.3}"

# Merge settings (run merge_datasets.py before curation)
MERGE_BALANCE_SOURCES="${MERGE_BALANCE_SOURCES:-true}"
MERGE_MAX_SOURCE_FRACTION="${MERGE_MAX_SOURCE_FRACTION:-0.3}"
MERGE_FILTER_ARTIFACTS="${MERGE_FILTER_ARTIFACTS:-true}"
MERGE_MAX_ARTIFACT_SCORE="${MERGE_MAX_ARTIFACT_SCORE:-0.4}"
MERGE_MIN_FRAMES="${MERGE_MIN_FRAMES:-25}"
MERGE_MAX_FRAMES="${MERGE_MAX_FRAMES:-500}"

SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-50000}"
HF_PUSH_DATASET="${HF_PUSH_DATASET:-true}"
HF_ASYNC_UPLOAD="${HF_ASYNC_UPLOAD:-true}"
HF_DATASET_REPO="${HF_DATASET_REPO:-GesturaAI/stick-gen-dataset}"
COMPLETION_MARKER="${DATA_PATH}/.prep_complete"

# Async pipeline settings (when USE_ASYNC_PIPELINE=true)
ASYNC_MAX_CONCURRENT="${ASYNC_MAX_CONCURRENT:-4}"       # Max concurrent converter processes
ASYNC_SKIP_SYNTHETIC="${ASYNC_SKIP_SYNTHETIC:-false}"   # Skip synthetic generation if true
ASYNC_RESUME="${ASYNC_RESUME:-true}"                    # Resume from checkpoint if available

# Persistent log directory on network volume for debugging
LOG_DIR="${DATA_PATH}/logs"
LOG_FILE="${LOG_DIR}/data_prep_$(date +%Y%m%d_%H%M%S).log"

# Force regeneration (overwrite existing data)
# Set FORCE_REGENERATE=true to overwrite previous training data
FORCE_REGENERATE="${FORCE_REGENERATE:-false}"

# Ensure data and log directories exist before logging starts
mkdir -p "${DATA_PATH}"
mkdir -p "${LOG_DIR}"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          Stick-Gen Data Preparation Pipeline                   ║${NC}"
echo -e "${CYAN}║                    by Gestura AI                               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Log to both console and file (persisted to network volume for debugging)
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

echo "Started at: $(date)"
echo ""

# ============================================================================
# Phase 1: Environment Validation
# ============================================================================
validate_environment() {
    echo -e "${YELLOW}[1/5] Validating environment...${NC}"

    # Check GPU availability (optional for data prep, but useful)
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/    /'
    else
        echo -e "  ${YELLOW}No GPU detected - data preparation will use CPU${NC}"
    fi

    # Check data path exists
    if [ ! -d "${DATA_PATH}" ]; then
        echo -e "${RED}ERROR: Data path does not exist: ${DATA_PATH}${NC}"
        echo "  Please ensure Network Volume is mounted and data is uploaded."
        exit 1
    fi

    # Check for raw data files
    local txt_count=$(find "${DATA_PATH}" -name "*.txt" -type f 2>/dev/null | wc -l)
    local total_size=$(du -sh "${DATA_PATH}" 2>/dev/null | cut -f1)

    echo "  Data path: ${DATA_PATH}"
    echo "  Total size: ${total_size}"
    echo "  .txt files found: ${txt_count}"

    if [ "${txt_count}" -eq 0 ]; then
        echo -e "${YELLOW}WARNING: No .txt files found in ${DATA_PATH}${NC}"
        echo "  Will proceed with synthetic data generation only."
    fi

    # Check workspace
    if [ ! -d "${WORKSPACE}" ]; then
        echo -e "${RED}ERROR: Workspace not found: ${WORKSPACE}${NC}"
        exit 1
    fi

	    # Check for required scripts based on pipeline mode
	    if [ "${USE_ASYNC_PIPELINE}" = "true" ]; then
	        if [ ! -f "${WORKSPACE}/src/data_gen/async_data_prep.py" ]; then
	            echo -e "${RED}ERROR: async_data_prep.py not found in src/data_gen/${NC}"
	            exit 1
	        fi
	        if [ ! -f "${WORKSPACE}/scripts/build_dataset_for_training.py" ]; then
	            echo -e "${RED}ERROR: build_dataset_for_training.py not found in scripts/${NC}"
	            exit 1
	        fi
	    elif [ "${USE_CURATED_DATA}" = "true" ]; then
	        if [ ! -f "${WORKSPACE}/scripts/prepare_curated_datasets.py" ]; then
	            echo -e "${RED}ERROR: prepare_curated_datasets.py not found in scripts/${NC}"
	            exit 1
	        fi
	        if [ ! -f "${WORKSPACE}/scripts/build_dataset_for_training.py" ]; then
	            echo -e "${RED}ERROR: build_dataset_for_training.py not found in scripts/${NC}"
	            exit 1
	        fi
	    else
	        if [ ! -f "${WORKSPACE}/scripts/prepare_data.py" ]; then
	            echo -e "${RED}ERROR: prepare_data.py not found in scripts/${NC}"
	            exit 1
	        fi
	    fi

	    echo -e "${GREEN}  ✓ Environment validated${NC}"
}

# ============================================================================
# Phase 2: Generate and Process Data
# ============================================================================
prepare_data() {
    echo ""
    echo -e "${YELLOW}[2/5] Running data preparation pipeline...${NC}"
    echo "  Synthetic samples: ${SYNTHETIC_SAMPLES}"
    echo "  Output path: ${OUTPUT_PATH}"
    echo "  Data path: ${DATA_PATH}"
    echo "  Force regenerate: ${FORCE_REGENERATE}"
    echo ""

    cd "${WORKSPACE}"

    # If force regenerate, remove existing output and completion marker
    if [ "${FORCE_REGENERATE}" = "true" ]; then
        echo -e "${YELLOW}  Force mode: removing previous data...${NC}"
        rm -f "${OUTPUT_PATH}"
        rm -f "${COMPLETION_MARKER}"
        rm -f "${DATA_PATH}/synthetic_data.pt"
        echo -e "${GREEN}  ✓ Previous data removed${NC}"
    fi

    # Create output directory
    mkdir -p "$(dirname "${OUTPUT_PATH}")"

    # Determine data directories based on DATA_PATH
    # DATA_PATH is typically /runpod-volume/data on RunPod
    local STYLE_DIR="${DATA_PATH}/100Style"
    local AMASS_DIR="${DATA_PATH}/amass"
    local SMPL_DIR="${DATA_PATH}/smpl_models"

    echo "  100STYLE dir: ${STYLE_DIR}"
    echo "  AMASS dir: ${AMASS_DIR}"
    echo "  SMPL dir: ${SMPL_DIR}"
    echo ""

    # Build command with optional --force flag
    local FORCE_FLAG=""
    if [ "${FORCE_REGENERATE}" = "true" ]; then
        FORCE_FLAG="--force"
    fi

    # Run the prepare_data.py script with proper paths
    python scripts/prepare_data.py \
        --100style-dir "${STYLE_DIR}" \
        --amass-dir "${AMASS_DIR}" \
        --smpl-dir "${SMPL_DIR}" \
        --synthetic-samples "${SYNTHETIC_SAMPLES}" \
        --output "${OUTPUT_PATH}" \
        --target-frames 250 \
        ${FORCE_FLAG}

    # Verify output was created
    if [ ! -f "${OUTPUT_PATH}" ]; then
        echo -e "${RED}ERROR: Output file was not created: ${OUTPUT_PATH}${NC}"
        exit 1
    fi

    local output_size=$(du -sh "${OUTPUT_PATH}" | cut -f1)
    echo -e "${GREEN}  ✓ Dataset created: ${OUTPUT_PATH} (${output_size})${NC}"
}


# Curated pipeline: canonical (.pt) -> merge -> curated splits -> embedded pretrain/SFT
prepare_curated_data() {
    echo ""
    echo -e "${YELLOW}[2/5] Running curated data preparation pipeline...${NC}"
    echo "  Canonical dir: ${CURATION_CANONICAL_DIR}"
    echo "  Curated output dir: ${CURATED_OUTPUT_DIR}"
    echo "  Output (training dataset): ${OUTPUT_PATH}"
    echo "  Merge settings: balance_sources=${MERGE_BALANCE_SOURCES}, max_source_fraction=${MERGE_MAX_SOURCE_FRACTION}"
    echo "  Curation thresholds: quality_pretrain >= ${CURATION_MIN_QUALITY_PRETRAIN}, quality_sft >= ${CURATION_MIN_QUALITY_SFT}"
    echo ""

    cd "${WORKSPACE}"

    # Build list of canonical .pt inputs
    local inputs=()
    if [ -n "${CURATION_INPUTS:-}" ]; then
        # Space-separated list of paths provided via env
        for p in ${CURATION_INPUTS}; do
            inputs+=("${p}")
        done
    else
        if [ -d "${CURATION_CANONICAL_DIR}" ]; then
            while IFS= read -r -d '' f; do
                inputs+=("${f}")
            done < <(find "${CURATION_CANONICAL_DIR}" -maxdepth 1 -type f -name "*.pt" -print0)
        fi
    fi

    if [ ${#inputs[@]} -eq 0 ]; then
        echo -e "${RED}ERROR: No canonical .pt files found for curation.${NC}"
        echo "  Set CURATION_INPUTS to a space-separated list of .pt files,"
        echo "  or place canonical .pt files under: ${CURATION_CANONICAL_DIR}"
        exit 1
    fi

    echo "  Using canonical inputs:"
    for p in "${inputs[@]}"; do
        echo "    - ${p}"
    done
    echo ""

    # Ensure output directory exists
    mkdir -p "${CURATED_OUTPUT_DIR}"

    # Phase 2a: Merge datasets with source balancing and filtering
    echo -e "${YELLOW}[2a/5] Merging datasets with source balancing...${NC}"
    local merged_dataset="${CURATED_OUTPUT_DIR}/merged_canonical.pt"

    local merge_args=("--inputs" "${inputs[@]}" "--output" "${merged_dataset}")

    if [ "${MERGE_BALANCE_SOURCES}" = "true" ]; then
        merge_args+=("--balance-sources")
        merge_args+=("--max-source-fraction" "${MERGE_MAX_SOURCE_FRACTION}")
    fi

    if [ "${MERGE_FILTER_ARTIFACTS}" = "true" ]; then
        merge_args+=("--filter-artifacts")
        merge_args+=("--max-artifact-score" "${MERGE_MAX_ARTIFACT_SCORE}")
    fi

    merge_args+=("--min-frames" "${MERGE_MIN_FRAMES}")
    merge_args+=("--max-frames" "${MERGE_MAX_FRAMES}")
    merge_args+=("--compute-stats")

    python -m scripts.merge_datasets "${merge_args[@]}"

    if [ ! -f "${merged_dataset}" ]; then
        echo -e "${RED}ERROR: Merged dataset was not created: ${merged_dataset}${NC}"
        exit 1
    fi

    local merge_size=$(du -sh "${merged_dataset}" | cut -f1)
    echo -e "${GREEN}  ✓ Merged dataset created: ${merged_dataset} (${merge_size})${NC}"
    echo ""

    # Phase 2b: run curation to produce pretrain_data.pt, sft_data.pt and stats
    echo -e "${YELLOW}[2b/5] Curating into pretrain/SFT splits...${NC}"
    python scripts/prepare_curated_datasets.py \
        --inputs "${merged_dataset}" \
        --output-dir "${CURATED_OUTPUT_DIR}" \
        --min-quality-pretrain "${CURATION_MIN_QUALITY_PRETRAIN}" \
        --min-quality-sft "${CURATION_MIN_QUALITY_SFT}" \
        --min-camera-stability-sft "${CURATION_MIN_CAMERA_STABILITY_SFT}" \
        --balance-max-fraction "${CURATION_BALANCE_MAX_FRACTION}"

    local pretrain_canonical="${CURATED_OUTPUT_DIR}/pretrain_data.pt"
    local sft_canonical="${CURATED_OUTPUT_DIR}/sft_data.pt"

    if [ ! -f "${pretrain_canonical}" ]; then
        echo -e "${RED}ERROR: Curated pretraining file not found: ${pretrain_canonical}${NC}"
        exit 1
    fi

    # Phase 2c: add embeddings to pretrain (and optionally SFT) datasets
    echo ""
    echo -e "${YELLOW}[2c/5] Adding text embeddings to curated datasets...${NC}"
    python scripts/build_dataset_for_training.py \
        --canonical "${pretrain_canonical}" \
        --output "${OUTPUT_PATH}"

    # Also embed SFT split if present (for future SFT/LoRA runs)
    if [ -f "${sft_canonical}" ]; then
        local sft_embedded="${CURATED_OUTPUT_DIR}/sft_data_embedded.pt"
        python scripts/build_dataset_for_training.py \
            --canonical "${sft_canonical}" \
            --output "${sft_embedded}"
    fi

    # Verify training dataset exists at OUTPUT_PATH
    if [ ! -f "${OUTPUT_PATH}" ]; then
        echo -e "${RED}ERROR: Embedded training dataset was not created: ${OUTPUT_PATH}${NC}"
        exit 1
    fi

    local output_size=$(du -sh "${OUTPUT_PATH}" | cut -f1)
    echo -e "${GREEN}  ✓ Curated training dataset created: ${OUTPUT_PATH} (${output_size})${NC}"
}

# Async pipeline: Convert → Merge → Synthetic (motion-conditioned) → Curate
# Uses AsyncDataPrepPipeline from src/data_gen/async_data_prep.py
prepare_data_async() {
    echo ""
    echo -e "${YELLOW}[2/5] Running async data preparation pipeline...${NC}"
    echo "  Data root: ${DATA_PATH}"
    echo "  Output root: ${DATA_PATH}/processed"
    echo "  Max concurrent: ${ASYNC_MAX_CONCURRENT}"
    echo "  Synthetic samples: ${SYNTHETIC_SAMPLES}"
    echo "  Skip synthetic: ${ASYNC_SKIP_SYNTHETIC}"
    echo "  Resume from checkpoint: ${ASYNC_RESUME}"
    echo ""

    cd "${WORKSPACE}"

    # Build command arguments
    local async_args=(
        "--data-root" "${DATA_PATH}"
        "--output-root" "${DATA_PATH}/processed"
        "--smpl-path" "${DATA_PATH}/smpl_models"
        "--max-concurrent" "${ASYNC_MAX_CONCURRENT}"
        "--synthetic-samples" "${SYNTHETIC_SAMPLES}"
    )

    if [ "${ASYNC_SKIP_SYNTHETIC}" = "true" ]; then
        async_args+=("--skip-synthetic")
    fi

    if [ "${ASYNC_RESUME}" != "true" ]; then
        async_args+=("--no-resume")
    fi

    echo "  Running: python -m src.data_gen.async_data_prep ${async_args[*]}"
    echo ""

    python -m src.data_gen.async_data_prep "${async_args[@]}"

    # The async pipeline outputs to DATA_PATH/processed/curated/
    # We need to build the final embedded training dataset
    local curated_dir="${DATA_PATH}/processed/curated"
    local pretrain_canonical="${curated_dir}/pretrain_data.pt"

    if [ ! -f "${pretrain_canonical}" ]; then
        echo -e "${RED}ERROR: Async pipeline did not produce pretrain_data.pt${NC}"
        echo "  Expected: ${pretrain_canonical}"
        exit 1
    fi

    echo ""
    echo -e "${YELLOW}[2b/5] Adding text embeddings to curated datasets...${NC}"
    python scripts/build_dataset_for_training.py \
        --canonical "${pretrain_canonical}" \
        --output "${OUTPUT_PATH}"

    # Also embed SFT split if present
    local sft_canonical="${curated_dir}/sft_data.pt"
    if [ -f "${sft_canonical}" ]; then
        local sft_embedded="${curated_dir}/sft_data_embedded.pt"
        python scripts/build_dataset_for_training.py \
            --canonical "${sft_canonical}" \
            --output "${sft_embedded}"
    fi

    # Verify training dataset exists at OUTPUT_PATH
    if [ ! -f "${OUTPUT_PATH}" ]; then
        echo -e "${RED}ERROR: Embedded training dataset was not created: ${OUTPUT_PATH}${NC}"
        exit 1
    fi

    local output_size=$(du -sh "${OUTPUT_PATH}" | cut -f1)
    echo -e "${GREEN}  ✓ Async pipeline complete: ${OUTPUT_PATH} (${output_size})${NC}"
}

# ============================================================================
# Phase 3: Validate Dataset
# ============================================================================
validate_dataset() {
    echo ""
    echo -e "${YELLOW}[3/5] Validating dataset...${NC}"

    cd "${WORKSPACE}"

    python -c "
import torch
import sys

data = torch.load('${OUTPUT_PATH}')
print(f'  Dataset type: {type(data)}')
print(f'  Total samples: {len(data)}')

if len(data) > 0:
    sample = data[0]
    print(f'  Sample keys: {list(sample.keys()) if isinstance(sample, dict) else \"list format\"}')
    if isinstance(sample, dict) and 'motion' in sample:
        motion = sample['motion']
        print(f'  Motion shape: {motion.shape if hasattr(motion, \"shape\") else len(motion)}')
    print('  ✓ Dataset validation passed')
else:
    print('  ⚠ Dataset is empty!')
    sys.exit(1)
"
    echo -e "${GREEN}  ✓ Dataset validated${NC}"
}

# ============================================================================
# Phase 4: Upload Dataset to HuggingFace (with async support)
# ============================================================================

# Background upload log file
UPLOAD_LOG_FILE="${DATA_PATH}/dataset_upload.log"
UPLOAD_PID_FILE="${DATA_PATH}/.upload_pid"

# Async upload function - runs in background
upload_dataset_async() {
    local dataset_path="$1"
    local repo="$2"
    
    echo "[$(date)] Starting async dataset upload to ${repo}" >> "${UPLOAD_LOG_FILE}"
    
    cd "${WORKSPACE}"
    
    # Upload pretrain dataset
    if [ -f "${dataset_path}" ]; then
        echo "[$(date)] Uploading pretrain dataset: ${dataset_path}" >> "${UPLOAD_LOG_FILE}"
        python scripts/push_dataset_to_hub.py \
            --dataset-path "${dataset_path}" \
            --repo-id "${repo}" \
            --token "${HF_TOKEN}" >> "${UPLOAD_LOG_FILE}" 2>&1
    fi
    
    # Also upload SFT dataset if it exists
    local sft_embedded="${CURATED_OUTPUT_DIR}/sft_data_embedded.pt"
    if [ -f "${sft_embedded}" ]; then
        echo "[$(date)] Uploading SFT dataset: ${sft_embedded}" >> "${UPLOAD_LOG_FILE}"
        python scripts/push_dataset_to_hub.py \
            --dataset-path "${sft_embedded}" \
            --repo-id "${repo}-sft" \
            --token "${HF_TOKEN}" >> "${UPLOAD_LOG_FILE}" 2>&1
    fi
    
    echo "[$(date)] Dataset upload complete" >> "${UPLOAD_LOG_FILE}"
    rm -f "${UPLOAD_PID_FILE}"
}

# Start async upload in background
start_async_upload() {
    echo ""
    echo -e "${YELLOW}[ASYNC] Starting background dataset upload...${NC}"
    
    if [ "${HF_PUSH_DATASET}" != "true" ]; then
        echo "  Skipping (HF_PUSH_DATASET != true)"
        return 0
    fi
    
    if [ -z "${HF_TOKEN:-}" ]; then
        echo -e "${YELLOW}  WARNING: HF_TOKEN not set - skipping dataset upload${NC}"
        return 0
    fi
    
    echo "  Uploading to: ${HF_DATASET_REPO}"
    echo "  Mode: ASYNC (running in background)"
    echo "  Log file: ${UPLOAD_LOG_FILE}"
    
    # Run in background
    upload_dataset_async "${OUTPUT_PATH}" "${HF_DATASET_REPO}" &
    local upload_pid=$!
    echo "${upload_pid}" > "${UPLOAD_PID_FILE}"
    
    echo -e "${GREEN}  ✓ Background upload started (PID: ${upload_pid})${NC}"
}

# Synchronous upload (fallback)
upload_dataset() {
    echo ""
    echo -e "${YELLOW}[4/5] HuggingFace dataset upload...${NC}"

    if [ "${HF_PUSH_DATASET}" != "true" ]; then
        echo "  Skipping (HF_PUSH_DATASET != true)"
        return 0
    fi

    if [ -z "${HF_TOKEN:-}" ]; then
        echo -e "${YELLOW}  WARNING: HF_TOKEN not set - skipping dataset upload${NC}"
        echo "  To upload, set HF_TOKEN and HF_PUSH_DATASET=true"
        return 0
    fi

    echo "  Uploading to: ${HF_DATASET_REPO}"
    echo "  Mode: SYNC (waiting for completion)"

    cd "${WORKSPACE}"

    # Upload pretrain dataset
    python scripts/push_dataset_to_hub.py \
        --dataset-path "${OUTPUT_PATH}" \
        --repo-id "${HF_DATASET_REPO}" \
        --token "${HF_TOKEN}"
    
    # Also upload SFT dataset if exists
    local sft_embedded="${CURATED_OUTPUT_DIR}/sft_data_embedded.pt"
    if [ -f "${sft_embedded}" ]; then
        python scripts/push_dataset_to_hub.py \
            --dataset-path "${sft_embedded}" \
            --repo-id "${HF_DATASET_REPO}-sft" \
            --token "${HF_TOKEN}"
    fi

    echo -e "${GREEN}  ✓ Dataset uploaded to HuggingFace${NC}"
}

# Wait for async upload to complete (called at end)
wait_for_upload() {
    if [ -f "${UPLOAD_PID_FILE}" ]; then
        local upload_pid=$(cat "${UPLOAD_PID_FILE}")
        if kill -0 "${upload_pid}" 2>/dev/null; then
            echo ""
            echo -e "${YELLOW}Waiting for background upload to complete (PID: ${upload_pid})...${NC}"
            wait "${upload_pid}" 2>/dev/null || true
            echo -e "${GREEN}  ✓ Background upload finished${NC}"
        fi
        rm -f "${UPLOAD_PID_FILE}"
    fi
}

# ============================================================================
# Phase 5: Final Validation and Write Completion Marker
# ============================================================================
write_completion_marker() {
    echo ""
    echo -e "${YELLOW}[5/5] Final validation and completion marker...${NC}"

    # CRITICAL: Final validation before marking success
    # This ensures training pods don't start with missing/invalid data

    # Check 1: Output file exists
    if [ ! -f "${OUTPUT_PATH}" ]; then
        echo -e "${RED}FATAL: Output file does not exist: ${OUTPUT_PATH}${NC}"
        echo -e "${RED}Data preparation FAILED - completion marker NOT written${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check logs at: ${LOG_FILE}"
        echo "  2. Check if disk is full: df -h"
        echo "  3. Check error messages above"
        exit 1
    fi

    # Check 2: Output file has non-zero size
    local file_size=$(stat -c%s "${OUTPUT_PATH}" 2>/dev/null || stat -f%z "${OUTPUT_PATH}" 2>/dev/null)
    if [ -z "$file_size" ] || [ "$file_size" -eq 0 ]; then
        echo -e "${RED}FATAL: Output file is empty: ${OUTPUT_PATH}${NC}"
        echo -e "${RED}Data preparation FAILED - completion marker NOT written${NC}"
        exit 1
    fi

    local file_size_human=$(du -sh "${OUTPUT_PATH}" | cut -f1)
    echo -e "${GREEN}  ✓ Output file validated: ${OUTPUT_PATH} (${file_size_human})${NC}"

    # Check 3: Validate file is loadable (quick sanity check)
    echo "  Performing final data integrity check..."
    python -c "
import torch
import sys
try:
    data = torch.load('${OUTPUT_PATH}', weights_only=False)
    if len(data) == 0:
        print('  ✗ Dataset is empty!')
        sys.exit(1)
    print(f'  ✓ Dataset loadable: {len(data)} samples')
except Exception as e:
    print(f'  ✗ Failed to load dataset: {e}')
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        echo -e "${RED}FATAL: Output file is corrupted or unloadable${NC}"
        echo -e "${RED}Data preparation FAILED - completion marker NOT written${NC}"
        exit 1
    fi

    echo -e "${GREEN}  ✓ All validations passed${NC}"
    echo ""

    # Now safe to write completion marker with proper JSON format
    cat > "${COMPLETION_MARKER}" << EOF
{
    "status": "complete",
    "timestamp": "$(date -Iseconds)",
    "output_path": "${OUTPUT_PATH}",
    "output_size_bytes": ${file_size},
    "use_curated_data": ${USE_CURATED_DATA},
    "curated_output_dir": "${CURATED_OUTPUT_DIR}",
    "curation_canonical_dir": "${CURATION_CANONICAL_DIR}",
    "synthetic_samples": ${SYNTHETIC_SAMPLES},
    "dataset_uploaded": ${HF_PUSH_DATASET},
    "dataset_repo": "${HF_DATASET_REPO}",
    "log_file": "${LOG_FILE}"
}
EOF

    echo "  Marker written: ${COMPLETION_MARKER}"
    echo -e "${GREEN}  ✓ Completion marker written - data preparation SUCCESS${NC}"
}

# ============================================================================
# Failure Handler - Write failure marker for orchestration debugging
# ============================================================================
write_failure_marker() {
    local error_msg="${1:-Unknown error}"
    local failure_marker="${DATA_PATH}/.prep_failed"

    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              Data Preparation FAILED                           ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Error: ${error_msg}"
    echo "  Log file: ${LOG_FILE}"
    echo "  Failed at: $(date)"
    echo ""

    # Write failure marker for orchestration to detect
    cat > "${failure_marker}" << EOF
{
    "status": "failed",
    "timestamp": "$(date -Iseconds)",
    "error": "${error_msg}",
    "output_path": "${OUTPUT_PATH}",
    "log_file": "${LOG_FILE}"
}
EOF

    echo "  Failure marker written: ${failure_marker}"
    echo ""
    echo -e "${YELLOW}Debug steps:${NC}"
    echo "  1. Check log file: cat ${LOG_FILE}"
    echo "  2. Check disk space: df -h"
    echo "  3. Check memory: free -h"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    local start_time=$(date +%s)

    # Set up error trap to write failure marker on any error
    trap 'write_failure_marker "Script exited with error on line $LINENO"' ERR

    # Run all phases
    validate_environment
    if [ "${USE_ASYNC_PIPELINE}" = "true" ]; then
        prepare_data_async
    elif [ "${USE_CURATED_DATA}" = "true" ]; then
        prepare_curated_data
    else
        prepare_data
    fi

    # Start async upload immediately after data is ready (saves time!)
    if [ "${HF_ASYNC_UPLOAD}" = "true" ]; then
        start_async_upload
    fi

    validate_dataset

    # If not async, do sync upload here
    if [ "${HF_ASYNC_UPLOAD}" != "true" ]; then
        upload_dataset
    fi

    write_completion_marker

    # Clear error trap after success
    trap - ERR

    # Wait for async upload to complete before exiting
    wait_for_upload

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Data Preparation Complete!                        ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Output: ${OUTPUT_PATH}"
    echo "  Log: ${LOG_FILE}"
    echo "  Duration: ${hours}h ${minutes}m ${seconds}s"
    echo "  Completed at: $(date)"
    echo ""
    echo -e "${CYAN}The data preparation Pod can now be terminated.${NC}"
    echo -e "${CYAN}Training Pods can use the prepared data at: ${OUTPUT_PATH}${NC}"
    echo ""

    # Keep container running briefly to allow log collection
    echo "Container will exit in 60 seconds..."
    sleep 60
}

# Run main function
main "$@"
