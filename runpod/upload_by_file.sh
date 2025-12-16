#!/bin/bash
# Upload files individually to avoid S3 pagination bug
# Usage: ./runpod/upload_by_file.sh --volume-id VOLUME_ID --dir DIRECTORY

set -e

# Default values
VOLUME_ID=""
RUNPOD_DATACENTER="EU-CZ-1"
DATA_DIR="./data"
TARGET_DIR=""

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
        --dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --volume-id VOLUME_ID --dir DIRECTORY"
            exit 1
            ;;
    esac
done

if [ -z "$VOLUME_ID" ] || [ -z "$TARGET_DIR" ]; then
    echo "Error: --volume-id and --dir are required"
    echo "Usage: $0 --volume-id VOLUME_ID --dir DIRECTORY"
    exit 1
fi

# Configuration
DATACENTER_LOWER=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
S3_ENDPOINT="https://s3api-${DATACENTER_LOWER}.runpod.io"

echo "Uploading ${TARGET_DIR} file-by-file..."
echo "  Volume: ${VOLUME_ID}"
echo "  Endpoint: ${S3_ENDPOINT}"

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

SOURCE_PATH="${DATA_DIR}/${TARGET_DIR}"

if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Directory not found: $SOURCE_PATH"
    exit 1
fi

echo ""
echo "=== Uploading ${TARGET_DIR} ==="
echo "Source: ${SOURCE_PATH}"
echo "Dest:   s3://${VOLUME_ID}/data/${TARGET_DIR}"
echo ""

# Count total files
total_files=$(find "$SOURCE_PATH" -type f | wc -l | tr -d ' ')
echo "Total files to upload: ${total_files}"
echo ""

# Upload files individually
file_count=0
error_count=0
skip_count=0

while IFS= read -r -d '' file; do
    # Get relative path
    rel_path="${file#${SOURCE_PATH}/}"
    s3_path="s3://${VOLUME_ID}/data/${TARGET_DIR}/${rel_path}"
    
    # Check if file already exists with same size
    local_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    s3_size=$(aws s3api head-object \
        --bucket "${VOLUME_ID}" \
        --key "data/${TARGET_DIR}/${rel_path}" \
        --endpoint-url "${S3_ENDPOINT}" \
        --region "${DATACENTER_LOWER}" \
        --query 'ContentLength' \
        --output text 2>/dev/null || echo "0")
    
    if [ "$local_size" = "$s3_size" ]; then
        skip_count=$((skip_count + 1))
        file_count=$((file_count + 1))
        
        # Progress indicator every 100 files
        if [ $((file_count % 100)) -eq 0 ]; then
            echo "Progress: ${file_count}/${total_files} (${skip_count} skipped, ${error_count} errors)"
        fi
        continue
    fi
    
    # Upload individual file
    if aws s3 cp "${file}" "${s3_path}" \
        --endpoint-url "${S3_ENDPOINT}" \
        --region "${DATACENTER_LOWER}" \
        --only-show-errors \
        --cli-read-timeout 300 \
        --cli-connect-timeout 60 2>/dev/null; then
        :  # Success
    else
        error_count=$((error_count + 1))
        echo "  ⚠️  Failed: ${rel_path}"
    fi
    
    file_count=$((file_count + 1))
    
    # Progress indicator every 100 files
    if [ $((file_count % 100)) -eq 0 ]; then
        echo "Progress: ${file_count}/${total_files} (${skip_count} skipped, ${error_count} errors)"
    fi
done < <(find "${SOURCE_PATH}" -type f -print0)

echo ""
echo "=== Upload Complete ==="
echo "Total files: ${total_files}"
echo "Uploaded: $((file_count - skip_count - error_count))"
echo "Skipped (already uploaded): ${skip_count}"
echo "Errors: ${error_count}"

if [ $error_count -gt 0 ]; then
    exit 1
fi

