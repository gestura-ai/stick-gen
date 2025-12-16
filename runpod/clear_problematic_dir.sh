#!/bin/bash
# Clear problematic directory from S3 to fix pagination issues
# Usage: ./runpod/clear_problematic_dir.sh --volume-id VOLUME_ID --dir HumanML3D

set -e

# Default values
VOLUME_ID=""
RUNPOD_DATACENTER="EU-CZ-1"
DIR_TO_CLEAR=""

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
            DIR_TO_CLEAR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --volume-id VOLUME_ID --dir DIRECTORY"
            exit 1
            ;;
    esac
done

if [ -z "$VOLUME_ID" ] || [ -z "$DIR_TO_CLEAR" ]; then
    echo "Error: --volume-id and --dir are required"
    echo "Usage: $0 --volume-id VOLUME_ID --dir DIRECTORY"
    exit 1
fi

# Configuration
DATACENTER_LOWER=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
S3_ENDPOINT="https://s3api-${DATACENTER_LOWER}.runpod.io"

echo "⚠️  WARNING: This will DELETE all files in s3://${VOLUME_ID}/data/${DIR_TO_CLEAR}/"
echo "This is necessary to fix S3 pagination bugs that prevent re-uploading."
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

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

echo ""
echo "Clearing s3://${VOLUME_ID}/data/${DIR_TO_CLEAR}/ ..."

# Delete all files in the directory
aws s3 rm "s3://${VOLUME_ID}/data/${DIR_TO_CLEAR}/" \
    --recursive \
    --endpoint-url "${S3_ENDPOINT}" \
    --region "${DATACENTER_LOWER}" \
    --page-size 100

echo ""
echo "✓ Directory cleared. You can now re-upload it with:"
echo "  make resume-upload VOLUME_ID=${VOLUME_ID}"

