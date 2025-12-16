#!/bin/bash
# Resume Stick-Gen Data Upload (Chunked)
# Usage: ./runpod/resume_upload.sh --volume-id VOLUME_ID --datacenter ID

set -e

# Default values
VOLUME_ID=""
RUNPOD_DATACENTER="EU-CZ-1"
DATA_DIR="./data"
SKIP_DIRS=""

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
        --skip)
            SKIP_DIRS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --volume-id VOLUME_ID [--datacenter DATACENTER] [--skip 'dir1,dir2']"
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

# Function to sync using smaller page size and prefix-based listing
sync_with_retry() {
    local source=$1
    local dest=$2
    local max_retries=3
    local retry_count=0

    # Extract bucket and prefix from dest (format: s3://bucket/prefix/)
    local bucket=$(echo "$dest" | sed 's|s3://||' | cut -d'/' -f1)
    local prefix=$(echo "$dest" | sed 's|s3://||' | cut -d'/' -f2-)

    echo "  Strategy: Using very small page-size (10) to avoid pagination bug"

    while [ $retry_count -lt $max_retries ]; do
        # Use extremely small page-size to avoid pagination issues
        # Also use --size-only to skip timestamp checks which require more listing
        if aws s3 sync "${source}" "${dest}" \
            --endpoint-url "${S3_ENDPOINT}" \
            --region "${DATACENTER_LOWER}" \
            --no-progress \
            --page-size 10 \
            --size-only \
            --cli-read-timeout 300 \
            --cli-connect-timeout 60 2>&1 | grep -v "Completed"; then
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "  ⚠️  Sync failed, retrying ($retry_count/$max_retries)..."
                sleep 5
            else
                echo "  ❌ Sync failed after $max_retries attempts"
                return 1
            fi
        fi
    done
}

# Convert skip list to array
IFS=',' read -ra SKIP_ARRAY <<< "$SKIP_DIRS"

# Sync each subdirectory
for subdir in "${SUBDIRS[@]}"; do
    # Check if this directory should be skipped
    skip=false
    for skip_dir in "${SKIP_ARRAY[@]}"; do
        if [ "$subdir" = "$skip_dir" ]; then
            skip=true
            break
        fi
    done

    if [ "$skip" = true ]; then
        echo ""
        echo "=== Skipping ${subdir} (--skip flag) ==="
        continue
    fi

    if [ -d "${DATA_DIR}/${subdir}" ]; then
        echo ""
        echo "=== Syncing ${subdir} ==="
        echo "Source: ${DATA_DIR}/${subdir}"
        echo "Dest:   s3://${VOLUME_ID}/data/${subdir}"

        if sync_with_retry "${DATA_DIR}/${subdir}" "s3://${VOLUME_ID}/data/${subdir}/"; then
            echo "✓ ${subdir} synced."
        else
            echo "⚠️  ${subdir} sync failed, continuing with next directory..."
            # Don't exit, continue with other directories
        fi
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
    --page-size 100 \
    --no-progress

echo ""
echo "Upload Resume Complete!"
