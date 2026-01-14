#!/bin/bash
# Upload all data directories file-by-file to avoid S3 pagination bug
# Usage: ./runpod/upload_all_by_file.sh --volume-id VOLUME_ID [--skip 'dir1,dir2']

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

# Check for RUNPOD_API_KEY to query volume datacenter
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "Warning: RUNPOD_API_KEY not set - using datacenter: ${RUNPOD_DATACENTER}"
else
    # Query the volume's datacenter from API
    echo "Querying volume datacenter from API..."
    VOLUME_DATACENTER=$(curl -s -X POST "https://api.runpod.io/graphql" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -d '{"query": "query { myself { networkVolumes { id dataCenterId } } }"}' \
        | grep -o "\"id\":\"${VOLUME_ID}\"[^}]*\"dataCenterId\":\"[^\"]*\"" \
        | grep -o '"dataCenterId":"[^"]*"' \
        | cut -d'"' -f4)

    if [ -n "$VOLUME_DATACENTER" ]; then
        if [ "$VOLUME_DATACENTER" != "$RUNPOD_DATACENTER" ]; then
            echo "  Volume is in ${VOLUME_DATACENTER}, updating datacenter setting"
        fi
        RUNPOD_DATACENTER="$VOLUME_DATACENTER"
    else
        echo "  Could not detect volume datacenter, using: ${RUNPOD_DATACENTER}"
    fi
fi

echo ""
echo "========================================="
echo "  RunPod Data Upload (File-by-File)"
echo "========================================="
echo "  Volume: ${VOLUME_ID}"
echo "  Datacenter: ${RUNPOD_DATACENTER}"
echo ""
echo "This script uploads files individually to avoid S3 pagination bugs."
echo "Already-uploaded files will be skipped (size comparison)."
echo ""

# List of subdirectories to upload
SUBDIRS=(
    "smpl_models"
    "amass"
    "100Style"
    "BEAT"
    "HumanML3D"
    "InterHuman Dataset"
    "KIT-ML"
    "NTU_RGB_D"
    "aist_plusplus"
    "lsmb19-mocap"
    "babel"
)

# Convert skip list to array
IFS=',' read -ra SKIP_ARRAY <<< "$SKIP_DIRS"

# Track overall stats
total_dirs=0
success_dirs=0
skip_dirs=0
failed_dirs=0

# Upload each subdirectory
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
        echo "========================================="
        echo "  Skipping: ${subdir} (--skip flag)"
        echo "========================================="
        skip_dirs=$((skip_dirs + 1))
        continue
    fi
    
    if [ ! -d "${DATA_DIR}/${subdir}" ]; then
        echo ""
        echo "========================================="
        echo "  Skipping: ${subdir} (not found locally)"
        echo "========================================="
        skip_dirs=$((skip_dirs + 1))
        continue
    fi
    
    total_dirs=$((total_dirs + 1))
    
    echo ""
    echo "========================================="
    echo "  Directory $total_dirs: ${subdir}"
    echo "========================================="
    
    # Upload this directory
    if ./runpod/upload_by_file.sh \
        --volume-id "${VOLUME_ID}" \
        --datacenter "${RUNPOD_DATACENTER}" \
        --dir "${subdir}"; then
        success_dirs=$((success_dirs + 1))
        echo "✓ ${subdir} completed successfully"
    else
        failed_dirs=$((failed_dirs + 1))
        echo "⚠️  ${subdir} had errors (continuing with next directory)"
    fi
done

echo ""
echo "========================================="
echo "  Upload Summary"
echo "========================================="
echo "Total directories: $((total_dirs + skip_dirs))"
echo "Uploaded successfully: ${success_dirs}"
echo "Skipped: ${skip_dirs}"
echo "Failed: ${failed_dirs}"
echo ""

if [ $failed_dirs -gt 0 ]; then
    echo "⚠️  Some directories had errors. Review the output above."
    exit 1
else
    echo "✓ All uploads completed successfully!"
    exit 0
fi

