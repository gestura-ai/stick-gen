#!/bin/bash
# Clear problematic directory from S3 to fix pagination issues
# This script intentionally avoids S3 ListObjects pagination by using a
# local or mounted directory for file discovery (similar to upload_by_file.sh).
#
# Usage:
#   # From laptop, using ./data as local mirror of S3 contents
#   ./runpod/clear_problematic_dir.sh --volume-id VOLUME_ID --dir HumanML3D
#
#   # From a Runpod pod, using the mounted network volume as source of truth
#   ./runpod/clear_problematic_dir.sh --volume-id VOLUME_ID --dir HumanML3D \
#       --data-dir /runpod-volume/data

set -e

# Default values
VOLUME_ID=""
RUNPOD_DATACENTER="EU-CZ-1"
DIR_TO_CLEAR=""
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
	    --dir)
	        DIR_TO_CLEAR="$2"
	        shift 2
	        ;;
	    --data-dir)
	        DATA_DIR="$2"
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

SOURCE_PATH="${DATA_DIR}/${DIR_TO_CLEAR}"

if [ -d "${SOURCE_PATH}" ]; then
	echo "Using local directory for file discovery: ${SOURCE_PATH}"
	total_files=$(find "${SOURCE_PATH}" -type f | wc -l | tr -d ' ')
	echo "Total local files to delete remotely: ${total_files}"
	echo "(This avoids S3 ListObjects pagination by issuing per-file deletes.)"

	file_count=0
	error_count=0

	while IFS= read -r -d '' file; do
		# Relative path within the directory
		rel_path="${file#${SOURCE_PATH}/}"
		s3_path="s3://${VOLUME_ID}/data/${DIR_TO_CLEAR}/${rel_path}"

		if aws s3 rm "${s3_path}" \
			--endpoint-url "${S3_ENDPOINT}" \
			--region "${DATACENTER_LOWER}" \
			--only-show-errors \
			--cli-read-timeout 300 \
			--cli-connect-timeout 60 2>/dev/null; then
			:
		else
			error_count=$((error_count + 1))
			echo "  ⚠️  Failed to delete: ${rel_path}"
		fi

		file_count=$((file_count + 1))
		if [ $((file_count % 100)) -eq 0 ]; then
			echo "Progress: ${file_count}/${total_files} (${error_count} errors)"
		fi
	done < <(find "${SOURCE_PATH}" -type f -print0)

	echo ""
	echo "Per-file deletion complete. Attempted to delete: ${file_count} files."
	if [ "${error_count}" -gt 0 ]; then
		echo "Warnings: ${error_count} delete errors encountered."
		exit 1
	fi
else
		echo "Local directory not found: ${SOURCE_PATH}"
		echo "No deletions were performed because we intentionally avoid S3 listing."
		echo "Provide a matching local or mounted directory via --data-dir (for example,"
		echo "  --data-dir ./data               # laptop mirror of S3 data"
		echo "  --data-dir /runpod-volume/data  # inside pod, mounted volume)"
		exit 1
fi

echo ""
echo "✓ Clear operation finished. You can now re-upload with:"
echo "  make resume-upload VOLUME_ID=${VOLUME_ID}"

