#!/bin/bash
# Resume Stick-Gen Data Upload (Chunked)
# Usage: ./runpod/resume_upload.sh --volume-id VOLUME_ID --datacenter ID

set -e

# Default values
VOLUME_ID=""
RUNPOD_DATACENTER="EU-CZ-1"
DATA_DIR="./data"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --volume-id)
            VOLUME_ID="$2"
            shift 2
            ;;
        --datacenter)
            RUNPOD_DATACENTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VOLUME_ID" ]; then
    echo "Error: --volume-id is required"
    exit 1
fi

# Configuration
DATACENTER_LOWER=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
S3_ENDPOINT="https://s3api-${DATACENTER_LOWER}.runpod.io"

echo "Starting Chunked Resume Upload..."
echo "  Volume: ${VOLUME_ID}"
echo "  Endpoint: ${S3_ENDPOINT}"

# Check for credentials
if [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_KEY}" ]; then
    echo "Error: RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY must be set."
    exit 1
fi

# Unset potential conflicting AWS env vars (e.g. from SSO)
unset AWS_SESSION_TOKEN
unset AWS_PROFILE

export AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_KEY}"

# Optimize settings
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 8MB
aws configure set default.s3.max_concurrent_requests 4
export AWS_RETRY_MODE=adaptive
export AWS_MAX_ATTEMPTS=5

# List of subdirectories to sync individually
SUBDIRS=(
    "smpl_models"
    "amass"
    "100Style"
    "HumanML3D"
    "InterHuman Dataset"
    "KIT-ML"
    "NTU_RGB_D"
    "aist_plusplus"
    "lsmb19-mocap"
    "motions_processed"
)

# Sync each subdirectory
for subdir in "${SUBDIRS[@]}"; do
    if [ -d "${DATA_DIR}/${subdir}" ]; then
        echo ""
        echo "=== Syncing ${subdir} ==="
        echo "Source: ${DATA_DIR}/${subdir}"
        echo "Dest:   s3://${VOLUME_ID}/data/${subdir}"
        
        # Switching back to 'sync' to enable resume capability (incremental upload).
        # Previous issues with ContinuationToken might be mitigated by chunked subdir syncing.
        aws s3 sync "${DATA_DIR}/${subdir}" "s3://${VOLUME_ID}/data/${subdir}/" \
            --endpoint-url "${S3_ENDPOINT}" \
            --region "${DATACENTER_LOWER}" \
            --no-progress \
            --cli-read-timeout 300 \
            --cli-connect-timeout 60
            
        echo "✓ ${subdir} synced."
    else
        echo "Skipping ${subdir} (not found locally)"
    fi
done

echo ""
echo "=== Finalizing Root Files ==="
# Sync root files of data/ (excluding directories)
aws s3 sync "${DATA_DIR}/" "s3://${VOLUME_ID}/data/" \
    --endpoint-url "${S3_ENDPOINT}" \
    --region "${DATACENTER_LOWER}" \
    --exclude "*" --include "*.*" \
    --no-progress

echo ""
echo "Upload Resume Complete!"
