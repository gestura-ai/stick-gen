#!/bin/bash
# Resume Stick-Gen Data Upload using rclone (more reliable than aws-cli for large datasets)
# Usage: ./runpod/resume_upload_rclone.sh --volume-id VOLUME_ID --datacenter ID

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

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "Error: rclone is not installed."
    echo ""
    echo "Install rclone:"
    echo "  macOS:   brew install rclone"
    echo "  Linux:   curl https://rclone.org/install.sh | sudo bash"
    echo ""
    exit 1
fi

# Configuration
DATACENTER_LOWER=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
S3_ENDPOINT="s3api-${DATACENTER_LOWER}.runpod.io"

echo "Starting Upload with rclone..."
echo "  Volume: ${VOLUME_ID}"
echo "  Endpoint: ${S3_ENDPOINT}"

# Check for credentials
if [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_KEY}" ]; then
    echo "Error: RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY must be set."
    exit 1
fi

# Create rclone config on-the-fly
RCLONE_CONFIG=$(mktemp)
trap "rm -f ${RCLONE_CONFIG}" EXIT

cat > "${RCLONE_CONFIG}" << EOF
[runpod]
type = s3
provider = Other
access_key_id = ${RUNPOD_S3_ACCESS_KEY}
secret_access_key = ${RUNPOD_S3_SECRET_KEY}
endpoint = https://${S3_ENDPOINT}
region = ${DATACENTER_LOWER}
bucket_acl = private
force_path_style = true
EOF

echo "✓ rclone configured"
echo "  Bucket: ${VOLUME_ID}"

# List of subdirectories to sync
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
        echo "Dest:   runpod:${VOLUME_ID}/data/${subdir}"

        # rclone sync with progress and retries
        # Note: RunPod S3 uses volume ID as bucket name
        # Path format: runpod:BUCKET/path/to/files
        # --fast-list: Use recursive listing (faster for large directories)
        # --transfers: Number of parallel transfers
        # --checkers: Number of parallel checksum checkers
        # --s3-upload-concurrency: Parallel chunk uploads
        rclone sync "${DATA_DIR}/${subdir}" "runpod:${VOLUME_ID}/data/${subdir}" \
            --config="${RCLONE_CONFIG}" \
            --progress \
            --transfers 4 \
            --checkers 8 \
            --retries 3 \
            --low-level-retries 10 \
            --stats 30s \
            --stats-one-line \
            --s3-upload-concurrency 2

        echo "✓ ${subdir} synced."
    else
        echo "Skipping ${subdir} (not found locally)"
    fi
done

echo ""
echo "Upload Complete!"

