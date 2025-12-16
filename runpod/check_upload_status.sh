#!/bin/bash
# Check Upload Status for RunPod Volume
# Usage: ./runpod/check_upload_status.sh --volume-id VOLUME_ID

set -e

# Default values
VOLUME_ID=""
RUNPOD_DATACENTER="EU-CZ-1"

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
            echo "Usage: $0 --volume-id VOLUME_ID [--datacenter DATACENTER]"
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

echo "Checking Upload Status..."
echo "  Volume: ${VOLUME_ID}"
echo "  Endpoint: ${S3_ENDPOINT}"
echo ""

# Check for credentials
if [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_KEY}" ]; then
    echo "Error: RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY must be set."
    exit 1
fi

# Unset potential conflicting AWS env vars
unset AWS_SESSION_TOKEN
unset AWS_PROFILE

export AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_KEY}"

# List of subdirectories to check
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

echo "=== Upload Status by Directory ==="
echo ""

for subdir in "${SUBDIRS[@]}"; do
    echo -n "Checking ${subdir}... "
    
    # Count files in S3
    s3_count=$(aws s3 ls "s3://${VOLUME_ID}/data/${subdir}/" \
        --endpoint-url "${S3_ENDPOINT}" \
        --region "${DATACENTER_LOWER}" \
        --recursive \
        --page-size 1000 \
        2>/dev/null | wc -l || echo "0")
    
    # Count local files
    if [ -d "./data/${subdir}" ]; then
        local_count=$(find "./data/${subdir}" -type f | wc -l)
        
        if [ "$s3_count" -eq "$local_count" ]; then
            echo "✓ Complete (${s3_count} files)"
        elif [ "$s3_count" -eq 0 ]; then
            echo "❌ Not uploaded (0/${local_count} files)"
        else
            echo "⚠️  Partial (${s3_count}/${local_count} files)"
        fi
    else
        echo "⊘ Not found locally"
    fi
done

echo ""
echo "=== Summary ==="
echo "To skip already uploaded directories, use:"
echo "  ./runpod/resume_upload.sh --volume-id ${VOLUME_ID} --skip 'dir1,dir2,dir3'"
echo ""
echo "Example (skip completed directories):"
echo "  ./runpod/resume_upload.sh --volume-id ${VOLUME_ID} --skip 'amass,100Style'"

