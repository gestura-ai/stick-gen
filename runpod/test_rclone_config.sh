#!/bin/bash
# Test rclone configuration for RunPod S3
# Usage: ./runpod/test_rclone_config.sh --volume-id VOLUME_ID

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
    exit 1
fi

# Configuration
DATACENTER_LOWER=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
S3_ENDPOINT="s3api-${DATACENTER_LOWER}.runpod.io"

echo "Testing rclone configuration..."
echo "  Volume: ${VOLUME_ID}"
echo "  Endpoint: https://${S3_ENDPOINT}"

# Check for credentials
if [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_KEY}" ]; then
    echo "Error: RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY must be set."
    exit 1
fi

# Create rclone config on-the-fly
RCLONE_CONFIG=$(mktemp)
trap "rm -f ${RCLONE_CONFIG}" EXIT

echo ""
echo "Testing different rclone configurations..."
echo ""

# Test 1: Standard S3 config
echo "=== Test 1: Standard S3 config ==="
cat > "${RCLONE_CONFIG}" << EOF
[runpod]
type = s3
provider = Other
access_key_id = ${RUNPOD_S3_ACCESS_KEY}
secret_access_key = ${RUNPOD_S3_SECRET_KEY}
endpoint = https://${S3_ENDPOINT}
region = ${DATACENTER_LOWER}
EOF

echo "Trying: rclone lsd runpod:${VOLUME_ID}"
rclone lsd "runpod:${VOLUME_ID}" --config="${RCLONE_CONFIG}" -vv 2>&1 | head -20 || true

echo ""
echo "=== Test 2: With force_path_style ==="
cat > "${RCLONE_CONFIG}" << EOF
[runpod]
type = s3
provider = Other
access_key_id = ${RUNPOD_S3_ACCESS_KEY}
secret_access_key = ${RUNPOD_S3_SECRET_KEY}
endpoint = https://${S3_ENDPOINT}
region = ${DATACENTER_LOWER}
force_path_style = true
EOF

echo "Trying: rclone lsd runpod:${VOLUME_ID}"
rclone lsd "runpod:${VOLUME_ID}" --config="${RCLONE_CONFIG}" -vv 2>&1 | head -20 || true

echo ""
echo "=== Test 3: Without https:// prefix ==="
cat > "${RCLONE_CONFIG}" << EOF
[runpod]
type = s3
provider = Other
access_key_id = ${RUNPOD_S3_ACCESS_KEY}
secret_access_key = ${RUNPOD_S3_SECRET_KEY}
endpoint = ${S3_ENDPOINT}
region = ${DATACENTER_LOWER}
force_path_style = true
EOF

echo "Trying: rclone lsd runpod:${VOLUME_ID}"
rclone lsd "runpod:${VOLUME_ID}" --config="${RCLONE_CONFIG}" -vv 2>&1 | head -20 || true

echo ""
echo "=== Test 4: Comparing with AWS CLI ==="
export AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_KEY}"
unset AWS_SESSION_TOKEN
unset AWS_PROFILE

echo "AWS CLI command:"
echo "aws s3 ls s3://${VOLUME_ID}/ --endpoint-url https://${S3_ENDPOINT} --region ${DATACENTER_LOWER}"
aws s3 ls "s3://${VOLUME_ID}/" --endpoint-url "https://${S3_ENDPOINT}" --region "${DATACENTER_LOWER}" 2>&1 | head -10 || true

