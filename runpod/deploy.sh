#!/bin/bash
# Stick-Gen RunPod Deployment Script
# Gestura AI - https://gestura.ai
#
# Usage: ./deploy.sh [action] [options]
#
# Actions:
#   build        - Build Docker image locally
#   push         - Push image to GitHub Container Registry (ghcr.io)
#   deploy       - Deploy to RunPod Serverless endpoint
#   login        - Authenticate with GitHub Container Registry
#   all          - Build, push, and deploy (full workflow)
#
#   === Training Infrastructure (Pods) ===
#   create-volume  - Create a RunPod Network Volume for training data
#   upload-data    - Upload ./data directory to Network Volume via rsync
#   upload-s3      - Upload ./data directory via S3-compatible API (automated)
#   create-pod     - Create a training Pod with Network Volume attached
#   train-setup    - Full training setup (create-volume + upload-data + create-pod)
#   list-volumes   - List existing Network Volumes
#   list-pods      - List running Pods
#
#   === Automated Training (End-to-End) ===
#   prep-data      - Create a data preparation Pod (generates train_data_final.pt)
#   train-small    - Train small model (5.6M params) - single command automation
#   train-base     - Train base model (22M params) - single command automation
#   train-large    - Train large model (88M params) - single command automation
#   auto-train-all - Full automated pipeline: data prep + train all variants
#   full-deploy    - Complete pipeline: build image + create volume + upload data + train all
#
#   === Inference Infrastructure (Serverless) ===
#   create-endpoint  - Create a Serverless endpoint for inference
#   list-endpoints   - List existing Serverless endpoints
#   submit-job       - Submit an inference job to a Serverless endpoint
#   delete-endpoint  - Delete a Serverless endpoint
#
# Options:
#   --variant small|medium|large  - Model variant for single-model commands (default: medium)
#   --models MODELS               - Models to train: small,medium | large | all (default: all)
#                                   Examples: --models small,medium  --models large  --models all
#   --version VERSION             - Image version tag (default: latest)
#   --volume-id ID                - Network Volume ID (for upload-data, create-pod, create-endpoint)
#   --volume-size GB              - Network Volume size in GB (default: 200)
#   --datacenter ID               - RunPod data center ID (default: EU-CZ-1)
#                                   Options: US-TX-3, US-CA-1, EU-NL-1, EU-CZ-1, etc.
#                                   NOTE: Pods MUST be in same datacenter as Network Volume!
#   --gpu GPU_TYPE                - GPU type for training/inference (default: NVIDIA RTX A4000)
#   --pod-id ID                   - Pod ID (for SSH connection info)
#   --endpoint-id ID              - Serverless endpoint ID (for submit-job, delete-endpoint)
#   --workers-min N               - Minimum workers for serverless (default: 0)
#   --workers-max N               - Maximum workers for serverless (default: 3)
#   --curated                     - Use curated data prep pipeline (USE_CURATED_DATA=true)
#
# GitHub Container Registry (ghcr.io) Authentication:
#   1. Create a Personal Access Token (PAT) at: https://github.com/settings/tokens
#      - Required scopes: read:packages, write:packages, delete:packages
#   2. Login: echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
#
# RunPod API Authentication:
#   1. Get API key from: https://www.runpod.io/console/user/settings
#   2. Export: export RUNPOD_API_KEY=your_api_key
#
# Environment Variables:
#   GITHUB_USERNAME  - Your GitHub username (default: gestura-ai)
#   GITHUB_TOKEN     - Your GitHub Personal Access Token (for docker login)
#   RUNPOD_API_KEY   - Your RunPod API key (for volume/pod management)
#   HF_TOKEN         - HuggingFace API token (for model/dataset uploads)
#   GROK_API_KEY     - X.AI Grok API key (for LLM-enhanced dataset generation)

set -e

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
GITHUB_USERNAME="${GITHUB_USERNAME:-gestura-ai}"
IMAGE_NAME="stick-gen"
VERSION="${VERSION:-latest}"
MODEL_VARIANT="${MODEL_VARIANT:-base}"
USE_CURATED_DATA="${USE_CURATED_DATA:-false}"

# RunPod Configuration
RUNPOD_API_URL="https://api.runpod.io/graphql"
VOLUME_NAME="${VOLUME_NAME:-stick-gen-training-data}"
VOLUME_SIZE="${VOLUME_SIZE:-200}"  # GB (200GB recommended for full pipeline)
# Model selection: comma-separated list of variants to train
# Options: small, medium (or base), large, all
# Default: all (trains small, medium, large)
TRAIN_MODELS="${TRAIN_MODELS:-all}"
# RunPod data center ID (use specific IDs like US-TX-3, EU-NL-1, etc.)
# IMPORTANT: Network Volumes are region-locked - Pods MUST be in the same datacenter as the volume
# Common US options: US-TX-3, US-CA-1, US-GA-1, US-OR-1
# Common EU options: EU-NL-1, EU-SE-1, EU-RO-1, EU-CZ-1
# Default: EU-CZ-1 (good GPU availability)
RUNPOD_DATACENTER="${RUNPOD_DATACENTER:-EU-CZ-1}"
GPU_TYPE="${GPU_TYPE:-NVIDIA RTX A4000}"
DATA_DIR="${DATA_DIR:-./data}"
# Volume mount path inside the container (REQUIRED when using networkVolumeId)
VOLUME_MOUNT_PATH="${VOLUME_MOUNT_PATH:-/runpod-volume}"

# S3-compatible API configuration for RunPod Network Volumes
# IMPORTANT: RunPod S3 API requires SEPARATE credentials from the API key!
# Generate S3 credentials from: https://www.runpod.io/console/user/settings -> Storage -> S3 Access Keys
# Endpoint format: https://s3api-{datacenter}.runpod.io (e.g., https://s3api-eu-cz-1.runpod.io)
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"  # Will be auto-generated from datacenter
RUNPOD_S3_ACCESS_KEY="${RUNPOD_S3_ACCESS_KEY:-}"  # Set via environment variable
RUNPOD_S3_SECRET_KEY="${RUNPOD_S3_SECRET_KEY:-}"  # Set via environment variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
ACTION=""
VOLUME_ID=""
POD_ID=""
ENDPOINT_ID=""
WORKERS_MIN="${WORKERS_MIN:-0}"
WORKERS_MAX="${WORKERS_MAX:-3}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            MODEL_VARIANT="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --volume-id)
            VOLUME_ID="$2"
            shift 2
            ;;
        --volume-size)
            VOLUME_SIZE="$2"
            shift 2
            ;;
        --region|--datacenter)
            RUNPOD_DATACENTER="$2"
            shift 2
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --pod-id)
            POD_ID="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --endpoint-id)
            ENDPOINT_ID="$2"
            shift 2
            ;;
        --workers-min)
            WORKERS_MIN="$2"
            shift 2
            ;;
        --workers-max)
            WORKERS_MAX="$2"
            shift 2
            ;;
	        --curated|--use-curated-data)
	            USE_CURATED_DATA="true"
	            shift
	            ;;
        --models)
            TRAIN_MODELS="$2"
            shift 2
            ;;
        --help|-h)
            ACTION="help"
            shift
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run './runpod/deploy.sh help' for usage information."
            exit 1
            ;;
        *)
            # Non-option argument is the action
            if [[ -z "$ACTION" ]]; then
                ACTION="$1"
            else
                echo -e "${RED}Multiple actions specified: $ACTION and $1${NC}"
                echo "Run './runpod/deploy.sh help' for usage information."
                exit 1
            fi
            shift
            ;;
    esac
done

# If no action specified but options were provided, use intelligent defaults
if [[ -z "$ACTION" ]]; then
    # If --datacenter and --volume-size are provided, assume full-deploy
    if [[ -n "$RUNPOD_DATACENTER" ]] && [[ -n "$VOLUME_SIZE" ]]; then
        ACTION="full-deploy"
        echo -e "${CYAN}No action specified. Detected --datacenter and --volume-size.${NC}"
        echo -e "${CYAN}Defaulting to action: full-deploy${NC}"
        echo ""
    # If --volume-id is provided, assume auto-train-all
    elif [[ -n "$VOLUME_ID" ]]; then
        ACTION="auto-train-all"
        echo -e "${CYAN}No action specified. Detected --volume-id.${NC}"
        echo -e "${CYAN}Defaulting to action: auto-train-all${NC}"
        echo ""
    else
        ACTION="help"
    fi
fi

FULL_IMAGE="${DOCKER_REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
VARIANT_IMAGE="${DOCKER_REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${MODEL_VARIANT}-${VERSION}"

# Navigate to project root
cd "$(dirname "$0")/.."

# ============================================================================
# Helper Functions
# ============================================================================

check_runpod_api_key() {
    if [ -z "${RUNPOD_API_KEY}" ]; then
        echo -e "${RED}Error: RUNPOD_API_KEY environment variable is not set${NC}"
        echo ""
        echo "To get your API key:"
        echo "  1. Go to: https://www.runpod.io/console/user/settings"
        echo "  2. Create an API key"
        echo "  3. Export: export RUNPOD_API_KEY=your_api_key"
        exit 1
    fi
}

# Make GraphQL request to RunPod API
runpod_graphql() {
    local query="$1"
    curl -s -X POST "${RUNPOD_API_URL}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -d "{\"query\": \"${query}\"}"
}

# Check if jq is installed
check_jq() {
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}Warning: jq not found. Install for better output formatting:${NC}"
        echo "  brew install jq  # macOS"
        echo "  apt-get install jq  # Ubuntu/Debian"
        return 1
    fi
    return 0
}

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           Stick-Gen RunPod Deployment                      ║${NC}"
    echo -e "${BLUE}║           by Gestura AI                                    ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  Image:   ${FULL_IMAGE}"
    echo "  Variant: ${MODEL_VARIANT}"
    echo "  Action:  ${ACTION}"
    echo ""
}

# ============================================================================
# Network Volume Management
# ============================================================================

create_network_volume() {
    check_runpod_api_key
    print_header

    echo -e "${GREEN}Creating Network Volume...${NC}"
    echo "  Name: ${VOLUME_NAME}"
    echo "  Size: ${VOLUME_SIZE}GB"
    echo "  Data Center: ${RUNPOD_DATACENTER}"
    echo ""

    # GraphQL mutation to create network volume
    local query="mutation { createNetworkVolume(input: { name: \\\"${VOLUME_NAME}\\\", size: ${VOLUME_SIZE}, dataCenterId: \\\"${RUNPOD_DATACENTER}\\\" }) { id name size dataCenterId } }"

    local response
    response=$(runpod_graphql "$query")

    if check_jq; then
        echo "$response" | jq .
        VOLUME_ID=$(echo "$response" | jq -r '.data.createNetworkVolume.id // empty')
    else
        echo "$response"
        VOLUME_ID=$(echo "$response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -n "$VOLUME_ID" ] && [ "$VOLUME_ID" != "null" ]; then
        echo ""
        echo -e "${GREEN}✓ Network Volume created successfully!${NC}"
        echo -e "${CYAN}Volume ID: ${VOLUME_ID}${NC}"
        echo ""
        echo "Save this ID for later use:"
        echo "  export VOLUME_ID=${VOLUME_ID}"
        echo ""
        echo "Next step - upload data:"
        echo "  ./runpod/deploy.sh upload-data --volume-id ${VOLUME_ID}"
    else
        echo -e "${RED}Failed to create Network Volume${NC}"
        echo "Response: $response"
        exit 1
    fi
}

list_network_volumes() {
    check_runpod_api_key
    print_header

    echo -e "${GREEN}Listing Network Volumes...${NC}"
    echo ""

    local query="query { myself { networkVolumes { id name size dataCenterId } } }"
    local response
    response=$(runpod_graphql "$query")

    if check_jq; then
        echo "$response" | jq '.data.myself.networkVolumes[] | {id, name, size: "\(.size)GB", region: .dataCenterId}'
    else
        echo "$response"
    fi
}

upload_data_to_volume() {
    check_runpod_api_key
    print_header

    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}Error: --volume-id is required${NC}"
        echo ""
        echo "Usage: ./runpod/deploy.sh upload-data --volume-id YOUR_VOLUME_ID"
        echo ""
        echo "To find your volume ID:"
        echo "  ./runpod/deploy.sh list-volumes"
        exit 1
    fi

    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}Error: Data directory not found: ${DATA_DIR}${NC}"
        exit 1
    fi

    local data_size
    data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    echo -e "${GREEN}Uploading training data to Network Volume...${NC}"
    echo "  Source: ${DATA_DIR} (${data_size})"
    echo "  Volume ID: ${VOLUME_ID}"
    echo ""

    # Step 1: Create a temporary Pod with the volume attached
    echo -e "${CYAN}Step 1/3: Creating temporary transfer Pod...${NC}"

    # Use gpuTypeIdList with cloudType: COMMUNITY for best availability
    # This approach uses Community Cloud which has better availability than Secure Cloud
    local gpu_list='[\"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 3080\", \"NVIDIA GeForce RTX 4090\", \"NVIDIA GeForce RTX 4080\", \"NVIDIA RTX A4000\", \"NVIDIA GeForce RTX 4070 Ti\", \"NVIDIA GeForce RTX 3080 Ti\", \"NVIDIA L4\"]'
    local transfer_pod_id=""
    local pod_response=""

    # Try Community Cloud On-Demand with multiple GPU options (best availability)
    # CRITICAL: volumeMountPath is REQUIRED when using networkVolumeId
    # CRITICAL: ports: "22/tcp" is REQUIRED to expose SSH for direct TCP access (SCP/SFTP support)
    # Use our pre-built Docker image for consistency
    echo -e "${CYAN}  Creating transfer Pod (Community Cloud, multiple GPU options)...${NC}"
    echo -e "${CYAN}  Using Docker image: ${FULL_IMAGE}${NC}"
    local pod_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-transfer\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: COMMUNITY, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 20, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true }) { id machine { podHostId gpuDisplayName } } }"

    pod_response=$(runpod_graphql "$pod_query")
    echo "  API Response: $pod_response"

    if check_jq; then
        transfer_pod_id=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
        local gpu_used=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "unknown"')
    else
        transfer_pod_id=$(echo "$pod_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        gpu_used="unknown"
    fi

    if [ -n "$transfer_pod_id" ] && [ "$transfer_pod_id" != "null" ]; then
        echo -e "${GREEN}  ✓ Got Pod with GPU: ${gpu_used}${NC}"
    fi

    # If Community Cloud failed, try Secure Cloud
    if [ -z "$transfer_pod_id" ] || [ "$transfer_pod_id" == "null" ]; then
        echo -e "${CYAN}  Community Cloud unavailable, trying Secure Cloud...${NC}"
        local secure_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-transfer\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: SECURE, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 20, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true }) { id machine { podHostId gpuDisplayName } } }"

        pod_response=$(runpod_graphql "$secure_query")
        echo "  API Response: $pod_response"

        if check_jq; then
            transfer_pod_id=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
            gpu_used=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "unknown"')
        else
            transfer_pod_id=$(echo "$pod_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        fi

        if [ -n "$transfer_pod_id" ] && [ "$transfer_pod_id" != "null" ]; then
            echo -e "${GREEN}  ✓ Got Secure Cloud Pod with GPU: ${gpu_used}${NC}"
        fi
    fi

    # If both failed, try ALL cloud types
    if [ -z "$transfer_pod_id" ] || [ "$transfer_pod_id" == "null" ]; then
        echo -e "${CYAN}  Trying ALL cloud types...${NC}"
        local all_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-transfer\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: ALL, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 20, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true }) { id machine { podHostId gpuDisplayName } } }"

        pod_response=$(runpod_graphql "$all_query")
        echo "  API Response: $pod_response"

        if check_jq; then
            transfer_pod_id=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
            gpu_used=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "unknown"')
        else
            transfer_pod_id=$(echo "$pod_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        fi

        if [ -n "$transfer_pod_id" ] && [ "$transfer_pod_id" != "null" ]; then
            echo -e "${GREEN}  ✓ Got Pod with GPU: ${gpu_used}${NC}"
        fi
    fi

    if [ -z "$transfer_pod_id" ] || [ "$transfer_pod_id" == "null" ]; then
        echo -e "${YELLOW}No GPUs available in the volume's data center.${NC}"
        echo ""
        echo -e "${CYAN}Using runpodctl for peer-to-peer transfer instead...${NC}"
        echo ""

        if ! command -v runpodctl &> /dev/null; then
            echo -e "${RED}runpodctl not found. Install with:${NC}"
            echo "  brew install runpod/runpodctl/runpodctl  # macOS"
            echo "  # or"
            echo "  pip install runpodctl"
            exit 1
        fi

        # Use runpodctl for data transfer - this requires manual steps
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║      Manual Data Transfer Required (runpodctl)             ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Since no GPUs are available in the volume's data center,"
        echo "you'll need to complete the transfer in two steps:"
        echo ""
        echo -e "${CYAN}Step 1: Create a Pod manually in RunPod Console${NC}"
        echo "  1. Go to: https://www.runpod.io/console/pods"
        echo "  2. Click 'Deploy'"
        echo "  3. Select any available GPU"
        echo "  4. Attach Network Volume: ${VOLUME_ID}"
        echo "  5. Use image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        echo ""
        echo -e "${CYAN}Step 2: Transfer data using runpodctl${NC}"
        echo "  On your local machine, run:"
        echo -e "    ${GREEN}runpodctl send ${DATA_DIR}${NC}"
        echo ""
        echo "  Copy the receive code that appears, then on the RunPod Pod terminal:"
        echo -e "    ${GREEN}cd /runpod-volume && runpodctl receive <CODE>${NC}"
        echo ""
        echo -e "${CYAN}Step 3: Terminate the transfer Pod when done${NC}"
        echo ""
        exit 0
    fi

    echo -e "${GREEN}✓ Transfer Pod created: ${transfer_pod_id}${NC}"

    # Step 2: Wait for Pod to be ready and get SSH info
    echo -e "${CYAN}Step 2/3: Waiting for Pod to be ready...${NC}"

    local max_attempts=30
    local attempt=0
    local ssh_host=""
    local ssh_port=""

    while [ $attempt -lt $max_attempts ]; do
        sleep 10
        attempt=$((attempt + 1))
        echo "  Checking Pod status (attempt ${attempt}/${max_attempts})..."

        local status_query="query { pod(input: { podId: \\\"${transfer_pod_id}\\\" }) { id desiredStatus runtime { ports { privatePort publicPort ip } } } }"
        local status_response
        status_response=$(runpod_graphql "$status_query")

        if check_jq; then
            local runtime_status
            runtime_status=$(echo "$status_response" | jq -r '.data.pod.runtime // empty')
            if [ -n "$runtime_status" ] && [ "$runtime_status" != "null" ]; then
                ssh_host=$(echo "$status_response" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .ip // empty')
                ssh_port=$(echo "$status_response" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .publicPort // empty')
            fi
        fi

        if [ -n "$ssh_host" ] && [ -n "$ssh_port" ]; then
            break
        fi
    done

    if [ -z "$ssh_host" ] || [ -z "$ssh_port" ]; then
        echo -e "${YELLOW}Pod is starting but SSH info not yet available.${NC}"
        echo ""
        echo "Please complete the transfer manually:"
        echo "  1. Go to: https://www.runpod.io/console/pods"
        echo "  2. Find Pod: ${transfer_pod_id}"
        echo "  3. Click 'Connect' to get SSH command"
        echo "  4. Run: rsync -avz --progress ${DATA_DIR}/ root@HOST:/runpod-volume/data/"
        echo "  5. Terminate the transfer Pod when done"
        echo ""
        echo -e "${CYAN}Transfer Pod ID: ${transfer_pod_id}${NC}"
        exit 0
    fi

    echo -e "${GREEN}✓ Pod ready: ${ssh_host}:${ssh_port}${NC}"

    # Step 3: Transfer data via rsync
    echo -e "${CYAN}Step 3/3: Transferring data via rsync...${NC}"
    echo "  This may take a while for large datasets..."
    echo ""

    rsync -avz --progress \
        -e "ssh -p ${ssh_port} -o StrictHostKeyChecking=no" \
        "${DATA_DIR}/" \
        "root@${ssh_host}:/runpod-volume/data/"

    echo ""
    echo -e "${GREEN}✓ Data transfer complete!${NC}"
    echo ""

    # Cleanup: Terminate the transfer Pod
    echo -e "${CYAN}Cleaning up: Terminating transfer Pod...${NC}"
    local terminate_query="mutation { podTerminate(input: { podId: \\\"${transfer_pod_id}\\\" }) }"
    runpod_graphql "$terminate_query" > /dev/null
    echo -e "${GREEN}✓ Transfer Pod terminated${NC}"
    echo ""
    echo "Your data is now available at /runpod-volume/data/ on any Pod attached to volume ${VOLUME_ID}"
}

create_training_pod() {
    check_runpod_api_key
    print_header

    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}Error: --volume-id is required${NC}"
        echo ""
        echo "Usage: ./runpod/deploy.sh create-pod --volume-id YOUR_VOLUME_ID [--gpu GPU_TYPE]"
        echo ""
        echo "To find your volume ID:"
        echo "  ./runpod/deploy.sh list-volumes"
        exit 1
    fi

    echo -e "${GREEN}Creating Training Pod...${NC}"
    echo "  GPU: ${GPU_TYPE}"
    echo "  Image: ${FULL_IMAGE}"
    echo "  Volume: ${VOLUME_ID}"
    echo ""

    # Build GPU list - if specific GPU requested, try that first, otherwise use multiple options
    local gpu_list
    if [ "$GPU_TYPE" == "NVIDIA RTX A4000" ]; then
        # Default - try multiple GPUs for better availability
        gpu_list='[\"NVIDIA RTX A4000\", \"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 4090\", \"NVIDIA GeForce RTX 4080\", \"NVIDIA GeForce RTX 3080\"]'
    else
        # Specific GPU requested - still include fallbacks
        gpu_list="[\\\"${GPU_TYPE}\\\", \\\"NVIDIA GeForce RTX 3090\\\", \\\"NVIDIA RTX A4000\\\", \\\"NVIDIA GeForce RTX 4090\\\"]"
    fi

    # Use cloudType: COMMUNITY with gpuTypeIdList for best availability
    # CRITICAL: volumeMountPath is REQUIRED when using networkVolumeId
    # CRITICAL: ports: "22/tcp" is REQUIRED to expose SSH for direct TCP access (SCP/SFTP support)
    local query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-training\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: COMMUNITY, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 20, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, env: [ { key: \\\"MODEL_VARIANT\\\", value: \\\"${MODEL_VARIANT}\\\" }, { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"DEVICE\\\", value: \\\"cuda\\\" } ] }) { id machine { podHostId gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } }"

    local response
    response=$(runpod_graphql "$query")

    local pod_id
    local pod_host_id
    local ssh_ip
    local ssh_port
    if check_jq; then
        echo "$response" | jq .
        pod_id=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
        pod_host_id=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.machine.podHostId // empty')
        ssh_ip=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.runtime.ports[] | select(.privatePort == 22) | .ip // empty' 2>/dev/null || echo "")
        ssh_port=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.runtime.ports[] | select(.privatePort == 22) | .publicPort // empty' 2>/dev/null || echo "")
    else
        echo "$response"
        pod_id=$(echo "$response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -n "$pod_id" ] && [ "$pod_id" != "null" ]; then
        echo ""
        echo -e "${GREEN}✓ Training Pod created successfully!${NC}"
        echo -e "${CYAN}Pod ID: ${pod_id}${NC}"
        echo ""
        echo -e "${YELLOW}SSH Connection Methods:${NC}"
        echo ""
        echo "  1. SSH via RunPod Proxy (always works):"
        if [ -n "$pod_host_id" ]; then
            echo "     ssh ${pod_id}-${pod_host_id}@ssh.runpod.io -i ~/.ssh/id_ed25519"
        else
            echo "     ssh <pod_id>-<host_id>@ssh.runpod.io -i ~/.ssh/id_ed25519"
        fi
        echo ""
        echo "  2. SSH over TCP (supports SCP & SFTP - preferred for file transfers):"
        if [ -n "$ssh_ip" ] && [ -n "$ssh_port" ]; then
            echo "     ssh root@${ssh_ip} -p ${ssh_port} -i ~/.ssh/id_ed25519"
        else
            echo "     (Wait for Pod to start, then check RunPod console for IP:port)"
        fi
        echo ""
        echo "Training data available at: ${VOLUME_MOUNT_PATH}/data/"
        echo ""
        echo "To start training:"
        echo "  python train.py --config configs/${MODEL_VARIANT}.yaml --data_path ${VOLUME_MOUNT_PATH}/data"
    else
        echo -e "${YELLOW}Could not create Training Pod via API (no GPUs available)${NC}"
        echo ""
        echo -e "${CYAN}Please create the Pod manually via the RunPod Console:${NC}"
        echo ""
        echo "  1. Go to: https://www.runpod.io/console/pods"
        echo "  2. Click 'Deploy' or '+ GPU Pod'"
        echo "  3. Select a GPU with availability (look for green indicators)"
        echo "     IMPORTANT: Select a GPU in the SAME datacenter as your Network Volume!"
        echo "  4. Configure:"
        echo "     - Container Image: ${FULL_IMAGE}"
        echo "     - Attach Network Volume: ${VOLUME_ID}"
        echo "     - Volume Mount Path: ${VOLUME_MOUNT_PATH}"
        echo "     - Container Disk: 50 GB"
        echo "     - Expose TCP Ports: 22/tcp (for SSH/SCP/SFTP access)"
        echo "     - Enable SSH: Yes"
        echo "  5. Add Environment Variables:"
        echo "     - MODEL_VARIANT=${MODEL_VARIANT}"
        echo "     - DATA_PATH=${VOLUME_MOUNT_PATH}/data"
        echo "     - DEVICE=cuda"
        echo "  6. Click 'Deploy'"
        echo ""
        echo "Once deployed, your training data will be at: ${VOLUME_MOUNT_PATH}/data/"
    fi
}

list_pods() {
    check_runpod_api_key
    print_header

    echo -e "${GREEN}Listing Pods...${NC}"
    echo ""

    local query="query { myself { pods { id name desiredStatus runtime { gpus { displayName } } } } }"
    local response
    response=$(runpod_graphql "$query")

    if check_jq; then
        echo "$response" | jq '.data.myself.pods[] | {id, name, status: .desiredStatus, gpu: .runtime.gpus[0].displayName}'
    else
        echo "$response"
    fi
}

send_data_runpodctl() {
    print_header

    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}Error: Data directory not found: ${DATA_DIR}${NC}"
        exit 1
    fi

    if ! command -v runpodctl &> /dev/null; then
        echo -e "${RED}Error: runpodctl not found${NC}"
        echo ""
        echo "Install runpodctl with:"
        echo "  brew install runpod/runpodctl/runpodctl  # macOS"
        echo "  # or"
        echo "  pip install runpodctl"
        exit 1
    fi

    local data_size
    data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)

    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Peer-to-Peer Data Transfer with runpodctl              ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "This will initiate a peer-to-peer data transfer."
    echo "  Source: ${DATA_DIR} (${data_size})"
    echo ""
    echo -e "${YELLOW}Prerequisites:${NC}"
    echo "  1. Create a Pod in RunPod Console with your Network Volume attached"
    echo "  2. SSH into the Pod and navigate to /runpod-volume/"
    echo "  3. Run 'runpodctl receive <CODE>' with the code shown below"
    echo ""
    echo -e "${CYAN}Starting runpodctl send...${NC}"
    echo ""

    # Run runpodctl send
    runpodctl send "${DATA_DIR}"
}

# ============================================================================
# S3-Compatible API Data Upload
# ============================================================================

upload_data_s3() {
    check_runpod_api_key
    print_header

    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}Error: --volume-id is required${NC}"
        echo ""
        echo "Usage: ./runpod/deploy.sh upload-s3 --volume-id YOUR_VOLUME_ID"
        echo ""
        echo "To find your volume ID:"
        echo "  ./runpod/deploy.sh list-volumes"
        exit 1
    fi

    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}Error: Data directory not found: ${DATA_DIR}${NC}"
        exit 1
    fi

    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI not found${NC}"
        echo ""
        echo "Install AWS CLI with:"
        echo "  brew install awscli  # macOS"
        echo "  pip install awscli   # Python"
        exit 1
    fi

    local data_size
    data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)

    # Generate S3 endpoint URL from datacenter
    local datacenter_lower
    datacenter_lower=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
    local s3_endpoint="https://s3api-${datacenter_lower}.runpod.io"

    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     S3-Compatible API Data Upload                          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Source: ${DATA_DIR} (${data_size})"
    echo "  Destination: s3://${VOLUME_ID}/data/"
    echo "  Endpoint: ${s3_endpoint}"
    echo "  Region: ${datacenter_lower}"
    echo ""
    echo -e "${YELLOW}Note: Large uploads (>10GB) may take a long time.${NC}"
    echo -e "${YELLOW}The RunPod S3 API has limited 'sync' support for large directories.${NC}"
    echo ""

    # Check for S3 credentials (separate from API key!)
    if [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_KEY}" ]; then
        echo -e "${RED}Error: RunPod S3 credentials not set${NC}"
        echo ""
        echo "The RunPod S3 API requires SEPARATE credentials from the API key."
        echo ""
        echo "To generate S3 credentials:"
        echo "  1. Go to: https://www.runpod.io/console/user/settings"
        echo "  2. Click 'Storage' in the left menu"
        echo "  3. Under 'S3 Access Keys', click 'Generate Key'"
        echo "  4. Set the environment variables:"
        echo ""
        echo "     export RUNPOD_S3_ACCESS_KEY='user_xxxxx'"
        echo "     export RUNPOD_S3_SECRET_KEY='rps_xxxxx'"
        echo ""
        exit 1
    fi

    export AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_KEY}"

    echo -e "${CYAN}Starting S3 upload...${NC}"
    echo ""

    # Configure AWS CLI for RunPod S3 compatibility
    # - Reduce multipart threshold to avoid "InvalidPart" errors
    # - Use smaller chunk sizes for more reliable uploads
    # - Increase retries for network resilience
    echo "  Configuring S3 transfer settings for RunPod compatibility..."
    aws configure set default.s3.multipart_threshold 64MB
    aws configure set default.s3.multipart_chunksize 16MB
    aws configure set default.s3.max_concurrent_requests 4
    aws configure set default.s3.max_queue_size 100

    # Set retry configuration
    export AWS_RETRY_MODE=adaptive
    export AWS_MAX_ATTEMPTS=5

    # Use aws s3 sync for better reliability (skips existing files on retry)
    # Add timeouts and disable unnecessary signature checks
    echo "  Starting upload with optimized settings..."
    echo ""

    local retry_count=0
    local max_retries=3
    local exit_code=1

    while [ $retry_count -lt $max_retries ] && [ $exit_code -ne 0 ]; do
        if [ $retry_count -gt 0 ]; then
            echo ""
            echo -e "${YELLOW}  Retry attempt ${retry_count}/${max_retries}...${NC}"
            sleep 5
        fi

        aws s3 sync "${DATA_DIR}" "s3://${VOLUME_ID}/data/" \
            --endpoint-url "${s3_endpoint}" \
            --region "${datacenter_lower}" \
            --cli-read-timeout 300 \
            --cli-connect-timeout 60 \
            --no-progress 2>&1 | while read -r line; do
                echo "  $line"
            done

        exit_code=${PIPESTATUS[0]}
        retry_count=$((retry_count + 1))
    done

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Data upload complete!${NC}"
        echo ""
        echo "Your data is now available at /data/ on Network Volume ${VOLUME_ID}"
    else
        echo ""
        echo -e "${YELLOW}S3 upload encountered issues after ${max_retries} attempts (exit code: ${exit_code})${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. The upload uses 'sync' - re-run this command to resume from where it left off"
        echo "  2. Check your network connection stability"
        echo "  3. Try reducing chunk size further:"
        echo "     aws configure set default.s3.multipart_chunksize 8MB"
        echo "     ./runpod/deploy.sh upload-s3 --volume-id ${VOLUME_ID}"
        echo ""
        echo "Alternative methods:"
        echo "  1. Use SSH/SCP with an existing Pod:"
        echo "     ./runpod/deploy.sh upload-data --volume-id ${VOLUME_ID}"
        echo ""
        echo "  2. Use runpodctl peer-to-peer transfer:"
        echo "     ./runpod/deploy.sh send-data"
    fi
}

# ============================================================================
# End-to-End Training Automation
# ============================================================================

# Check S3 credentials and provide helpful error message if missing
check_s3_credentials() {
    if [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_KEY}" ]; then
        echo -e "${YELLOW}S3 credentials not set. Data verification will be skipped.${NC}"
        echo ""
        echo "To enable S3 data verification, set:"
        echo "  export RUNPOD_S3_ACCESS_KEY='user_xxxxx'"
        echo "  export RUNPOD_S3_SECRET_KEY='rps_xxxxx'"
        echo ""
        echo "Generate credentials at: https://www.runpod.io/console/user/settings -> Storage -> S3 Access Keys"
        echo ""
        return 1
    fi
    return 0
}

# Verify training data exists on volume via S3 API
verify_data_on_volume() {
    local volume_id="$1"
    local datacenter_lower
    datacenter_lower=$(echo "${RUNPOD_DATACENTER}" | tr '[:upper:]' '[:lower:]')
    local s3_endpoint="https://s3api-${datacenter_lower}.runpod.io"

    if ! check_s3_credentials; then
        return 2  # Credentials not available, skip check
    fi

    if ! command -v aws &> /dev/null; then
        echo -e "${YELLOW}AWS CLI not installed. Skipping data verification.${NC}"
        return 2
    fi

    echo -e "${CYAN}Verifying training data on volume...${NC}"

    # Use the separate S3 credentials (NOT the API key!)
    export AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_KEY}"

    local s3_check
    s3_check=$(aws s3 ls "s3://${volume_id}/data/" \
        --endpoint-url "${s3_endpoint}" \
        --region "${datacenter_lower}" \
        --recursive --summarize 2>&1 || true)

    # Check for expected data directories
    if echo "$s3_check" | grep -q -E "(100Style|amass|smpl_models)"; then
        local total_size
        total_size=$(echo "$s3_check" | grep "Total Size:" | awk '{print $3}')
        local total_objects
        total_objects=$(echo "$s3_check" | grep "Total Objects:" | awk '{print $3}')
        echo -e "${GREEN}  ✓ Training data found: ${total_objects} files (${total_size} bytes)${NC}"
        return 0
    else
        echo -e "${YELLOW}  Training data not found or incomplete on volume${NC}"
        return 1
    fi
}

train_model() {
    check_runpod_api_key
    print_header

    local model_variant="${MODEL_VARIANT}"
    local variant_display=$(echo "$model_variant" | tr '[:lower:]' '[:upper:]')

    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Automated Training: ${variant_display} Model                         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}Error: --volume-id is required${NC}"
        echo ""
        echo "Usage: ./runpod/deploy.sh train-${model_variant} --volume-id YOUR_VOLUME_ID"
        echo ""
        echo "To find your volume ID:"
        echo "  ./runpod/deploy.sh list-volumes"
        echo ""
        echo "Complete workflow example:"
        echo "  1. Create volume:    ./runpod/deploy.sh create-volume --datacenter EU-CZ-1"
        echo "  2. Upload data:      ./runpod/deploy.sh upload-s3 --volume-id YOUR_ID"
        echo "  3. Start training:   ./runpod/deploy.sh train-${model_variant} --volume-id YOUR_ID"
        exit 1
    fi

    echo -e "${CYAN}Configuration:${NC}"
    echo "  Model Variant: ${model_variant}"
    echo "  Volume ID:     ${VOLUME_ID}"
    echo "  Datacenter:    ${RUNPOD_DATACENTER}"
    echo "  Data Path:     ${VOLUME_MOUNT_PATH}/data"
    echo ""

    # Step 1: Verify data exists on volume
    echo -e "${CYAN}Step 1/4: Verifying training data on volume...${NC}"

    local data_status
    verify_data_on_volume "$VOLUME_ID"
    data_status=$?

    if [ $data_status -eq 1 ]; then
        echo ""
        echo -e "${YELLOW}Training data not found on volume.${NC}"
        echo ""
        echo "Please upload data first using:"
        echo "  export RUNPOD_S3_ACCESS_KEY='user_xxxxx'"
        echo "  export RUNPOD_S3_SECRET_KEY='rps_xxxxx'"
        echo "  ./runpod/deploy.sh upload-s3 --volume-id ${VOLUME_ID}"
        echo ""
        echo "Or use the direct AWS S3 command:"
        echo "  AWS_ACCESS_KEY_ID='...' AWS_SECRET_ACCESS_KEY='...' \\"
        echo "  aws s3 sync ./data s3://${VOLUME_ID}/data/ \\"
        echo "    --endpoint-url https://s3api-$(echo ${RUNPOD_DATACENTER} | tr '[:upper:]' '[:lower:]').runpod.io \\"
        echo "    --region $(echo ${RUNPOD_DATACENTER} | tr '[:upper:]' '[:lower:]')"
        exit 1
    elif [ $data_status -eq 2 ]; then
        echo -e "${YELLOW}  ⚠ Data verification skipped (credentials not available)${NC}"
        echo -e "${YELLOW}  Proceeding with Pod creation - ensure data is uploaded!${NC}"
    fi

    # Step 2: Check for existing training Pods (avoid duplicate costs)
    echo ""
    echo -e "${CYAN}Step 2/4: Checking for existing training Pods...${NC}"

    local existing_pods
    existing_pods=$(runpod_graphql "query { myself { pods { id name desiredStatus } } }")

    if check_jq; then
        local training_pod
        training_pod=$(echo "$existing_pods" | jq -r ".data.myself.pods[] | select(.name | contains(\"stick-gen-train\")) | .id" 2>/dev/null | head -1)
        if [ -n "$training_pod" ] && [ "$training_pod" != "null" ]; then
            echo -e "${YELLOW}  ⚠ Found existing training Pod: ${training_pod}${NC}"
            echo ""
            echo "To avoid duplicate costs, you can:"
            echo "  1. Use the existing Pod (check RunPod console for SSH info)"
            echo "  2. Terminate it first: curl -X POST ... podTerminate"
            echo ""
            echo -e "${YELLOW}Press Enter to create a new Pod anyway, or Ctrl+C to cancel...${NC}"
            read -r
        else
            echo -e "${GREEN}  ✓ No existing training Pods found${NC}"
        fi
    fi

    # Step 3: Create training Pod
    echo ""
    echo -e "${CYAN}Step 3/4: Creating training Pod...${NC}"

    # Prioritize RTX 3090 for cost-effectiveness, then other good training GPUs
    local gpu_list='[\"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 4090\", \"NVIDIA GeForce RTX 4080\", \"NVIDIA RTX A4000\", \"NVIDIA GeForce RTX 3080 Ti\", \"NVIDIA GeForce RTX 3080\", \"NVIDIA L4\", \"NVIDIA GeForce RTX 4070 Ti\"]'

    # Use our pre-built Docker image from GitHub Container Registry
    # This has all dependencies pre-installed and code baked in at /workspace
    local training_image="${FULL_IMAGE}"
    echo -e "${CYAN}  Using Docker image: ${training_image}${NC}"

    # Build environment variables list for the Pod
    # HF_TOKEN is optional but enables automatic push to HuggingFace
    # GROK_API_KEY is optional but enables LLM-enhanced dataset generation
    local hf_token_env=""
    if [ -n "${HF_TOKEN:-}" ]; then
        hf_token_env=", { key: \\\"HF_TOKEN\\\", value: \\\"${HF_TOKEN}\\\" }"
        echo -e "${GREEN}  HF_TOKEN detected - auto-push to HuggingFace enabled${NC}"
    else
        echo -e "${YELLOW}  HF_TOKEN not set - auto-push to HuggingFace disabled${NC}"
        echo -e "${YELLOW}  Set HF_TOKEN environment variable to enable auto-push${NC}"
    fi

    local grok_api_key_env=""
    if [ -n "${GROK_API_KEY:-}" ]; then
        grok_api_key_env=", { key: \\\"GROK_API_KEY\\\", value: \\\"${GROK_API_KEY}\\\" }"
        echo -e "${GREEN}  GROK_API_KEY detected - LLM dataset enhancement enabled${NC}"
    fi

    local pod_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-train-${model_variant}\\\", imageName: \\\"${training_image}\\\", cloudType: COMMUNITY, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, env: [ { key: \\\"MODEL_VARIANT\\\", value: \\\"${model_variant}\\\" }, { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"CHECKPOINT_DIR\\\", value: \\\"${VOLUME_MOUNT_PATH}/checkpoints\\\" }, { key: \\\"DEVICE\\\", value: \\\"cuda\\\" }, { key: \\\"PYTHONPATH\\\", value: \\\"/workspace\\\" }, { key: \\\"AUTO_PUSH\\\", value: \\\"true\\\" }, { key: \\\"AUTO_CLEANUP\\\", value: \\\"true\\\" }, { key: \\\"VERSION\\\", value: \\\"${VERSION:-1.0.0}\\\" }${hf_token_env}${grok_api_key_env} ] }) { id machine { podHostId gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } }"

    local pod_response
    pod_response=$(runpod_graphql "$pod_query")

    local pod_id=""
    local pod_host_id=""
    local gpu_used=""
    local ssh_ip=""
    local ssh_port=""

    if check_jq; then
        pod_id=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
        pod_host_id=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.machine.podHostId // empty')
        gpu_used=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "unknown"')

        # Check for errors
        local error_msg
        error_msg=$(echo "$pod_response" | jq -r '.errors[0].message // empty')
        if [ -n "$error_msg" ] && [ "$error_msg" != "null" ]; then
            echo -e "${RED}API Error: ${error_msg}${NC}"
        fi
    else
        pod_id=$(echo "$pod_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -z "$pod_id" ] || [ "$pod_id" == "null" ]; then
        echo -e "${RED}Failed to create training Pod${NC}"
        echo ""
        echo "Possible causes:"
        echo "  1. No GPUs available in ${RUNPOD_DATACENTER} datacenter"
        echo "  2. Network Volume ${VOLUME_ID} is in a different datacenter"
        echo "  3. API rate limit or authentication issue"
        echo ""
        echo "Please create the Pod manually via the RunPod Console:"
        echo "  https://www.runpod.io/console/pods"
        echo ""
        echo "Required settings:"
        echo "  - Network Volume: ${VOLUME_ID}"
        echo "  - Volume Mount Path: ${VOLUME_MOUNT_PATH}"
        echo "  - Expose Ports: 22/tcp"
        echo "  - Enable SSH: Yes"
        exit 1
    fi

    echo -e "${GREEN}  ✓ Training Pod created: ${pod_id}${NC}"
    echo -e "${GREEN}  ✓ GPU: ${gpu_used}${NC}"

    # Step 4: Wait for Pod to be ready and get SSH info
    echo ""
    echo -e "${CYAN}Step 4/4: Waiting for Pod to be ready...${NC}"

    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        sleep 10
        attempt=$((attempt + 1))
        echo "  Checking Pod status (attempt ${attempt}/${max_attempts})..."

        local status_query="query { pod(input: { podId: \\\"${pod_id}\\\" }) { id desiredStatus runtime { ports { privatePort publicPort ip } uptimeInSeconds } } }"
        local status_response
        status_response=$(runpod_graphql "$status_query")

        if check_jq; then
            local runtime_status
            runtime_status=$(echo "$status_response" | jq -r '.data.pod.runtime // empty')
            if [ -n "$runtime_status" ] && [ "$runtime_status" != "null" ]; then
                ssh_ip=$(echo "$status_response" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .ip // empty' 2>/dev/null || echo "")
                ssh_port=$(echo "$status_response" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .publicPort // empty' 2>/dev/null || echo "")
            fi
        fi

        if [ -n "$ssh_ip" ] && [ -n "$ssh_port" ]; then
            echo -e "${GREEN}  ✓ Pod is ready!${NC}"
            break
        fi
    done

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Training Pod Ready!                              ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Pod ID: ${pod_id}${NC}"
    echo -e "${CYAN}GPU:    ${gpu_used}${NC}"
    echo ""

    # Display SSH connection info
    echo -e "${YELLOW}═══ SSH Connection ═══${NC}"
    if [ -n "$ssh_ip" ] && [ -n "$ssh_port" ]; then
        echo ""
        echo "  Direct TCP (preferred for SCP/SFTP):"
        echo -e "    ${GREEN}ssh root@${ssh_ip} -p ${ssh_port}${NC}"
        echo ""
    fi
    if [ -n "$pod_host_id" ]; then
        echo "  RunPod Proxy (always works):"
        echo -e "    ${GREEN}ssh ${pod_id}-${pod_host_id}@ssh.runpod.io${NC}"
        echo ""
    fi

    # Display automated training info
    echo -e "${YELLOW}═══ Automated Training ═══${NC}"
    echo ""
    echo -e "  ${GREEN}✓ Training starts automatically when the Pod boots!${NC}"
    echo "  The Docker image runs the training entrypoint script automatically."
    echo ""
    echo "  Training workflow:"
    echo "    1. Validates training data and GPU"
    echo "    2. Trains the ${model_variant} model"
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "    3. Pushes best checkpoint to HuggingFace (GesturaAI/stick-gen-${model_variant})"
    else
        echo "    3. [SKIPPED] Push to HuggingFace (HF_TOKEN not set)"
    fi
    echo "    4. Cleans up intermediate checkpoints"
    echo ""

    # Display monitoring info
    echo -e "${YELLOW}═══ Monitor Training ═══${NC}"
    echo ""
    echo "  SSH into the Pod to monitor progress:"
    echo ""
    echo -e "  ${CYAN}# View training logs${NC}"
    echo "  tail -f ${VOLUME_MOUNT_PATH}/checkpoints/training_${model_variant}.log"
    echo ""
    echo -e "  ${CYAN}# Check GPU utilization${NC}"
    echo "  nvidia-smi -l 1"
    echo ""
    echo -e "  ${CYAN}# Manually run training (if needed)${NC}"
    echo "  cd /workspace && /workspace/runpod/train_entrypoint.sh"
    echo ""
    echo "  - RunPod Console: https://www.runpod.io/console/pods"
    echo ""

    # Cost reminder
    echo -e "${YELLOW}═══ Cost Optimization ═══${NC}"
    echo ""
    echo "  ⚠ Remember to TERMINATE the Pod when training is complete!"
    echo "  GPU costs ~\$0.20-0.50/hour depending on the GPU type."
    echo ""
    echo "  To terminate via API:"
    echo "  curl -X POST https://api.runpod.io/graphql \\"
    echo "    -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"query\": \"mutation { podTerminate(input: { podId: \\\"${pod_id}\\\" }) }\"}'"
    echo ""
    echo -e "${GREEN}Training infrastructure is ready!${NC}"
}

train_setup() {
    print_header

    echo -e "${GREEN}Full Training Infrastructure Setup${NC}"
    echo "This will:"
    echo "  1. Create a Network Volume (${VOLUME_SIZE}GB)"
    echo "  2. Provide instructions to upload training data"
    echo "  3. Create a training Pod with GPU (if available)"
    echo ""
    echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
    read -r

    # Step 1: Create volume
    create_network_volume

    # VOLUME_ID should be set by create_network_volume
    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}Failed to get Volume ID${NC}"
        exit 1
    fi

    # Step 2: Try to upload data (may require manual steps)
    upload_data_to_volume

    # Check if upload was successful (the function exits 0 even if it just provides instructions)
    # Continue to Pod creation

    # Step 3: Create training Pod
    create_training_pod

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Training Infrastructure Setup Complete!          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Volume ID: ${VOLUME_ID}${NC}"
    echo ""
    echo "If data transfer requires manual steps, use:"
    echo "  ./runpod/deploy.sh send-data"
    echo ""
    echo "Then on your RunPod Pod:"
    echo "  cd /runpod-volume && runpodctl receive <CODE>"
}

# ============================================================================
# Fully Automated Training Pipeline (Data Prep + All Variants)
# ============================================================================

# Create a data preparation Pod (standalone command)
create_data_prep_pod() {
    check_runpod_api_key
    print_header

    echo -e "${YELLOW}Creating data preparation Pod...${NC}"

    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}ERROR: --volume-id is required${NC}"
        echo "  Usage: ./runpod/deploy.sh prep-data --volume-id <VOLUME_ID>"
        exit 1
    fi

	    # Use Secure Cloud with full GPU type IDs for better availability
	    # Priority: RTX A4000 ($0.25), RTX A4500 ($0.25), RTX 4000 Ada ($0.26), RTX A5000 ($0.27), L4 ($0.39)
	    local gpu_list='[\"NVIDIA RTX A4000\", \"NVIDIA RTX A4500\", \"NVIDIA RTX 4000 Ada Generation\", \"NVIDIA RTX A5000\", \"NVIDIA L4\", \"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 4090\"]'
	    local pod_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-data-prep\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: SECURE, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, dockerArgs: \\\"/workspace/runpod/data_prep_entrypoint.sh\\\", env: [ { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"OUTPUT_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data/train_data_final.pt\\\" }, { key: \\\"USE_CURATED_DATA\\\", value: \\\"${USE_CURATED_DATA}\\\" }, { key: \\\"SYNTHETIC_SAMPLES\\\", value: \\\"${SYNTHETIC_SAMPLES:-50000}\\\" }, { key: \\\"HF_PUSH_DATASET\\\", value: \\\"${HF_PUSH_DATASET:-false}\\\" }, { key: \\\"HF_DATASET_REPO\\\", value: \\\"${HF_DATASET_REPO:-GesturaAI/stick-gen-dataset}\\\" }, { key: \\\"HF_TOKEN\\\", value: \\\"${HF_TOKEN:-}\\\" }, { key: \\\"GROK_API_KEY\\\", value: \\\"${GROK_API_KEY:-}\\\" } ] }) { id machine { podHostId gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } }"

    local response
    response=$(runpod_graphql "$pod_query")

    local pod_id
    if check_jq; then
        echo "$response" | jq .
        pod_id=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
    else
        echo "$response"
        pod_id=$(echo "$response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -n "$pod_id" ]; then
        echo ""
        echo -e "${GREEN}✓ Data preparation Pod created: ${pod_id}${NC}"
        echo ""
        echo "The Pod will automatically:"
        echo "  1. Generate synthetic motion data"
        echo "  2. Convert and merge data into train_data_final.pt"
        echo "  3. Write completion marker at ${VOLUME_MOUNT_PATH}/data/.prep_complete"
        echo ""
        echo "Monitor progress with:"
        echo "  ./runpod/deploy.sh list-pods"
    else
        echo -e "${RED}ERROR: Failed to create data preparation Pod${NC}"
        exit 1
    fi
}

auto_train_all() {
    check_runpod_api_key
    print_header

    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Fully Automated Training Pipeline                      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "This will orchestrate the complete training pipeline:"
    echo ""
    echo "  Phase 1: Data Preparation"
    echo "    - Create a data preparation Pod"
    echo "    - Generate and process training data"
    echo "    - Wait for completion"
    echo "    - Terminate data prep Pod"
    echo ""
    echo "  Phase 2: Sequential Training"
    echo "    - Train small model (5.6M params)"
    echo "    - Train base model (15.8M params)"
    echo "    - Train large model (28M params)"
    echo "    - Push each model to HuggingFace"
    echo "    - Clean up between variants"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Volume ID:     ${VOLUME_ID:-[REQUIRED]}"
    echo "  HF_TOKEN:      ${HF_TOKEN:+[SET]}${HF_TOKEN:-[NOT SET - models will not be pushed]}"
    echo "  HF_REPO_NAME:  ${HF_REPO_NAME:-GesturaAI/stick-gen}"
    echo ""

    if [ -z "$VOLUME_ID" ]; then
        echo -e "${RED}ERROR: --volume-id is required${NC}"
        echo "  Usage: ./runpod/deploy.sh auto-train-all --volume-id <VOLUME_ID>"
        exit 1
    fi

    echo -e "${YELLOW}Press Enter to start the automated pipeline, or Ctrl+C to cancel...${NC}"
    read -r

    local start_time=$(date +%s)
    local data_prep_pod_id=""
    local training_pod_id=""

    # =========================================================================
    # Phase 1: Data Preparation
    # =========================================================================
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Phase 1: Data Preparation                              ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${YELLOW}[1/8] Creating data preparation Pod...${NC}"

    # Use Secure Cloud with full GPU type IDs for better availability
    # Priority: RTX A4000 ($0.25), RTX A4500 ($0.25), RTX 4000 Ada ($0.26), RTX A5000 ($0.27), L4 ($0.39)
    local gpu_list='[\"NVIDIA RTX A4000\", \"NVIDIA RTX A4500\", \"NVIDIA RTX 4000 Ada Generation\", \"NVIDIA RTX A5000\", \"NVIDIA L4\", \"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 4090\"]'
    local data_prep_image="${FULL_IMAGE}"

	    # Override CMD to run data_prep_entrypoint.sh instead of train_entrypoint.sh
	    local pod_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-data-prep\\\", imageName: \\\"${data_prep_image}\\\", cloudType: SECURE, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, dockerArgs: \\\"/workspace/runpod/data_prep_entrypoint.sh\\\", env: [ { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"OUTPUT_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data/train_data_final.pt\\\" }, { key: \\\"USE_CURATED_DATA\\\", value: \\\"${USE_CURATED_DATA}\\\" }, { key: \\\"SYNTHETIC_SAMPLES\\\", value: \\\"50000\\\" }, { key: \\\"HF_PUSH_DATASET\\\", value: \\\"${HF_PUSH_DATASET:-false}\\\" }, { key: \\\"HF_DATASET_REPO\\\", value: \\\"${HF_DATASET_REPO:-GesturaAI/stick-gen-dataset}\\\" }, { key: \\\"HF_TOKEN\\\", value: \\\"${HF_TOKEN:-}\\\" }, { key: \\\"GROK_API_KEY\\\", value: \\\"${GROK_API_KEY:-}\\\" } ] }) { id machine { podHostId gpuDisplayName } } }"

    local pod_response
    pod_response=$(runpod_graphql "$pod_query")

    if check_jq; then
        data_prep_pod_id=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
        local gpu_used=$(echo "$pod_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "Unknown"')
    else
        data_prep_pod_id=$(echo "$pod_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        gpu_used="Unknown"
    fi

    if [ -z "$data_prep_pod_id" ]; then
        echo -e "${RED}ERROR: Failed to create data preparation Pod${NC}"
        echo "$pod_response"
        exit 1
    fi

    echo -e "${GREEN}  ✓ Data prep Pod created: ${data_prep_pod_id}${NC}"
    echo "  GPU: ${gpu_used}"
    echo ""

    # =========================================================================
    # Poll for Data Prep Completion
    # =========================================================================
    echo -e "${YELLOW}[2/8] Waiting for data preparation to complete...${NC}"
    echo "  Polling for completion marker: ${VOLUME_MOUNT_PATH}/data/.prep_complete"
    echo "  This may take 30-60 minutes depending on dataset size."
    echo ""

    local poll_interval=60  # Poll every 60 seconds
    local max_wait=7200     # Max 2 hours
    local elapsed=0
    local completion_marker="${VOLUME_MOUNT_PATH}/data/.prep_complete"

    while [ $elapsed -lt $max_wait ]; do
        # Check Pod status first
        local status_query="query { pod(input: { podId: \\\"${data_prep_pod_id}\\\" }) { desiredStatus runtime { uptimeInSeconds } } }"
        local status_response=$(runpod_graphql "$status_query")

        local pod_status
        if check_jq; then
            pod_status=$(echo "$status_response" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
        else
            pod_status=$(echo "$status_response" | grep -o '"desiredStatus":"[^"]*"' | cut -d'"' -f4)
        fi

        # If Pod terminated/exited, check if it succeeded
        if [ "$pod_status" = "EXITED" ] || [ "$pod_status" = "TERMINATED" ]; then
            echo ""
            echo -e "${GREEN}  ✓ Data preparation Pod has exited${NC}"
            break
        fi

        # Show progress
        local mins=$((elapsed / 60))
        printf "\r  Elapsed: %d minutes | Status: %s | Checking..." "$mins" "$pod_status"

        sleep $poll_interval
        elapsed=$((elapsed + poll_interval))
    done
    echo ""

    if [ $elapsed -ge $max_wait ]; then
        echo -e "${YELLOW}WARNING: Data preparation timed out after ${max_wait}s${NC}"
        echo "  The Pod may still be running. Check manually."
    fi

    # =========================================================================
    # Terminate Data Prep Pod
    # =========================================================================
    echo -e "${YELLOW}[3/8] Terminating data preparation Pod...${NC}"
    local term_query="mutation { podTerminate(input: { podId: \\\"${data_prep_pod_id}\\\" }) }"
    runpod_graphql "$term_query" > /dev/null 2>&1
    echo -e "${GREEN}  ✓ Data prep Pod terminated${NC}"
    echo ""

    # =========================================================================
    # Phase 2: Sequential Training
    # =========================================================================
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Phase 2: Sequential Model Training                     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Build variants array based on TRAIN_MODELS setting
    local variants=()
    case "$TRAIN_MODELS" in
        all)
            variants=("small" "medium" "large")
            ;;
        small,medium|medium,small)
            variants=("small" "medium")
            ;;
        small)
            variants=("small")
            ;;
        medium|base)
            variants=("medium")
            ;;
        large)
            variants=("large")
            ;;
        *)
            # Parse comma-separated list
            IFS=',' read -ra variants <<< "$TRAIN_MODELS"
            # Normalize "base" to "medium"
            for i in "${!variants[@]}"; do
                if [[ "${variants[$i]}" == "base" ]]; then
                    variants[$i]="medium"
                fi
            done
            ;;
    esac
    local step=4

    for variant in "${variants[@]}"; do
        echo -e "${YELLOW}[${step}/8] Training ${variant} model...${NC}"

        # Create training Pod for this variant
        MODEL_VARIANT="${variant}"

        # Use Secure Cloud with full GPU type IDs for better availability
        local train_gpu_list='[\"NVIDIA RTX A4000\", \"NVIDIA RTX A4500\", \"NVIDIA RTX 4000 Ada Generation\", \"NVIDIA RTX A5000\", \"NVIDIA L4\", \"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 4090\"]'
        local train_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-train-${variant}\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: SECURE, gpuTypeIdList: ${train_gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, env: [ { key: \\\"MODEL_VARIANT\\\", value: \\\"${variant}\\\" }, { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"CHECKPOINT_DIR\\\", value: \\\"${VOLUME_MOUNT_PATH}/checkpoints\\\" }, { key: \\\"HF_TOKEN\\\", value: \\\"${HF_TOKEN:-}\\\" }, { key: \\\"GROK_API_KEY\\\", value: \\\"${GROK_API_KEY:-}\\\" }, { key: \\\"HF_REPO_NAME\\\", value: \\\"${HF_REPO_NAME:-GesturaAI/stick-gen}\\\" }, { key: \\\"AUTO_PUSH\\\", value: \\\"true\\\" }, { key: \\\"AUTO_CLEANUP\\\", value: \\\"true\\\" }, { key: \\\"VERSION\\\", value: \\\"${VERSION:-1.0.0}\\\" } ] }) { id machine { gpuDisplayName } } }"

        local train_response=$(runpod_graphql "$train_query")

        if check_jq; then
            training_pod_id=$(echo "$train_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
            gpu_used=$(echo "$train_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "Unknown"')
        else
            training_pod_id=$(echo "$train_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
            gpu_used="Unknown"
        fi

        if [ -z "$training_pod_id" ]; then
            echo -e "${RED}  ERROR: Failed to create training Pod for ${variant}${NC}"
            continue
        fi

        echo -e "${GREEN}  ✓ Training Pod created: ${training_pod_id} (GPU: ${gpu_used})${NC}"

        # Wait for training to complete (Pod will exit when done)
        echo "  Waiting for training to complete..."
        local train_elapsed=0
        local train_max_wait=28800  # 8 hours max per variant

        while [ $train_elapsed -lt $train_max_wait ]; do
            local status_query="query { pod(input: { podId: \\\"${training_pod_id}\\\" }) { desiredStatus runtime { uptimeInSeconds } } }"
            local status_response=$(runpod_graphql "$status_query")

            local pod_status
            if check_jq; then
                pod_status=$(echo "$status_response" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
            else
                pod_status=$(echo "$status_response" | grep -o '"desiredStatus":"[^"]*"' | cut -d'"' -f4)
            fi

            if [ "$pod_status" = "EXITED" ] || [ "$pod_status" = "TERMINATED" ]; then
                echo ""
                echo -e "${GREEN}  ✓ Training complete for ${variant}${NC}"
                break
            fi

            local train_mins=$((train_elapsed / 60))
            printf "\r  Training %s: %d minutes | Status: %s" "$variant" "$train_mins" "$pod_status"

            sleep 120  # Check every 2 minutes
            train_elapsed=$((train_elapsed + 120))
        done
        echo ""

        # Terminate training Pod (in case it's still running)
        local term_query="mutation { podTerminate(input: { podId: \\\"${training_pod_id}\\\" }) }"
        runpod_graphql "$term_query" > /dev/null 2>&1
        echo -e "${GREEN}  ✓ Training Pod terminated${NC}"
        echo ""

        step=$((step + 1))
    done

    # =========================================================================
    # Summary
    # =========================================================================
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Automated Training Pipeline Complete!                  ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Total Duration: ${hours}h ${minutes}m"
    echo ""
    echo "  Models trained and pushed to HuggingFace:"
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "    - ${HF_REPO_NAME:-GesturaAI/stick-gen}-small"
        echo "    - ${HF_REPO_NAME:-GesturaAI/stick-gen}-base"
        echo "    - ${HF_REPO_NAME:-GesturaAI/stick-gen}-large"
    else
        echo "    [Models not pushed - HF_TOKEN was not set]"
    fi
    echo ""
    echo "  Checkpoints saved to: ${VOLUME_MOUNT_PATH}/checkpoints/"
    echo ""
}

# ============================================================================
# Full Deployment Pipeline (Everything from scratch)
# ============================================================================

full_deploy() {
    check_runpod_api_key
    print_header

    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Full Deployment Pipeline                               ║${NC}"
    echo -e "${BLUE}║     Build → Upload → Prep → Train → Push to HuggingFace    ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "This will execute the complete pipeline from scratch:"
    echo ""
    echo "  Phase 1: Docker Image"
    echo "    [1/9] Build Docker image from docker/Dockerfile"
    echo "    [2/9] Push image to GitHub Container Registry (ghcr.io)"
    echo ""
    echo "  Phase 2: RunPod Infrastructure"
    echo "    [3] Create Network Volume (${VOLUME_SIZE}GB in ${RUNPOD_DATACENTER})"
    echo "    [4] Upload training data via S3 API (~90GB)"
    echo ""
    echo "  Phase 3: Data Preparation"
    echo "    [5] Create data preparation Pod"
    echo "    [6] Wait for data processing to complete"
    echo ""
    echo "  Phase 4: Model Training (${TRAIN_MODELS})"
    case "$TRAIN_MODELS" in
        all)
            echo "    [7] Train small model (5.6M params)"
            echo "    [8] Train medium model (15.8M params)"
            echo "    [9] Train large model (28M params)"
            ;;
        small,medium|medium,small)
            echo "    [7] Train small model (5.6M params)"
            echo "    [8] Train medium model (15.8M params)"
            ;;
        small)
            echo "    [7] Train small model (5.6M params)"
            ;;
        medium|base)
            echo "    [7] Train medium model (15.8M params)"
            ;;
        large)
            echo "    [7] Train large model (28M params)"
            ;;
        *)
            echo "    [7+] Train: ${TRAIN_MODELS}"
            ;;
    esac
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Datacenter:     ${RUNPOD_DATACENTER}"
    echo "  Volume Size:    ${VOLUME_SIZE}GB"
    echo "  Models:         ${TRAIN_MODELS}"
    echo "  Data Directory: ${DATA_DIR}"
    echo "  Docker Image:   ${FULL_IMAGE}"
    echo "  HF_TOKEN:       ${HF_TOKEN:+[SET]}${HF_TOKEN:-[NOT SET - models will not be pushed]}"
    echo "  HF_REPO_NAME:   ${HF_REPO_NAME:-GesturaAI/stick-gen}"
    echo ""

    # Validate prerequisites
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}ERROR: Data directory not found: ${DATA_DIR}${NC}"
        echo "  Please ensure your training data is in the ./data directory"
        exit 1
    fi

    local data_size
    data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    echo "  Data Size:      ${data_size}"
    echo ""

    # Check for required credentials
    if [ -z "${GITHUB_TOKEN:-}" ]; then
        echo -e "${YELLOW}WARNING: GITHUB_TOKEN not set - Docker push may fail${NC}"
        echo "  Set with: export GITHUB_TOKEN='ghp_xxxxx'"
        echo ""
    fi

    if [ -z "${RUNPOD_S3_ACCESS_KEY:-}" ] || [ -z "${RUNPOD_S3_SECRET_KEY:-}" ]; then
        echo -e "${YELLOW}WARNING: RunPod S3 credentials not set - data upload may fail${NC}"
        echo "  Set with:"
        echo "    export RUNPOD_S3_ACCESS_KEY='user_xxxxx'"
        echo "    export RUNPOD_S3_SECRET_KEY='rps_xxxxx'"
        echo "  Get credentials at: https://www.runpod.io/console/user/settings → Storage → S3"
        echo ""
    fi

    echo -e "${YELLOW}This pipeline may take 6-12 hours depending on data size and GPU availability.${NC}"
    echo -e "${YELLOW}Estimated cost: \$50-100 (GPU hours for data prep + 3x training)${NC}"
    echo ""
    echo -e "${YELLOW}Press Enter to start the full deployment, or Ctrl+C to cancel...${NC}"
    read -r

    local pipeline_start_time=$(date +%s)
    local step_times=()
    local created_volume_id=""

    # =========================================================================
    # Phase 1: Docker Image
    # =========================================================================
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Phase 1: Docker Image Build & Push                     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    local step_start=$(date +%s)
    echo -e "${YELLOW}[1/9] Building Docker image...${NC}"
    echo "  Image: ${FULL_IMAGE}"
    echo ""

    if ! docker build -t "${FULL_IMAGE}" -f docker/Dockerfile .; then
        echo -e "${RED}ERROR: Docker build failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Docker image built successfully${NC}"
    step_times+=("Build: $(($(date +%s) - step_start))s")

    step_start=$(date +%s)
    echo ""
    echo -e "${YELLOW}[2/9] Pushing Docker image to ghcr.io...${NC}"

    # Login to ghcr.io if token is available
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_USERNAME}" --password-stdin
    fi

    if ! docker push "${FULL_IMAGE}"; then
        echo -e "${RED}ERROR: Docker push failed${NC}"
        echo "  Make sure you are logged in: docker login ghcr.io"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Docker image pushed to ghcr.io${NC}"
    step_times+=("Push: $(($(date +%s) - step_start))s")

    # =========================================================================
    # Phase 2: RunPod Infrastructure
    # =========================================================================
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Phase 2: RunPod Infrastructure Setup                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    step_start=$(date +%s)
    echo -e "${YELLOW}[3/9] Creating Network Volume...${NC}"
    echo "  Size: ${VOLUME_SIZE}GB"
    echo "  Datacenter: ${RUNPOD_DATACENTER}"
    echo ""

    local volume_query="mutation { createNetworkVolume(input: { name: \\\"stick-gen-training-data\\\", size: ${VOLUME_SIZE}, dataCenterId: \\\"${RUNPOD_DATACENTER}\\\" }) { id name size dataCenterId } }"
    local volume_response
    volume_response=$(runpod_graphql "$volume_query")

    if check_jq; then
        created_volume_id=$(echo "$volume_response" | jq -r '.data.createNetworkVolume.id // empty')
        echo "$volume_response" | jq .
    else
        created_volume_id=$(echo "$volume_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        echo "$volume_response"
    fi

    if [ -z "$created_volume_id" ]; then
        echo -e "${RED}ERROR: Failed to create Network Volume${NC}"
        exit 1
    fi

    VOLUME_ID="$created_volume_id"
    echo ""
    echo -e "${GREEN}  ✓ Network Volume created: ${VOLUME_ID}${NC}"
    step_times+=("Volume: $(($(date +%s) - step_start))s")

    # Wait for volume to be ready
    echo "  Waiting for volume to be ready..."
    sleep 10

    step_start=$(date +%s)
    echo ""
    echo -e "${YELLOW}[4/9] Uploading training data via S3 API...${NC}"
    echo "  Source: ${DATA_DIR} (${data_size})"
    echo "  Destination: s3://${VOLUME_ID}/data/"
    echo ""

    if [ -z "${RUNPOD_S3_ACCESS_KEY:-}" ] || [ -z "${RUNPOD_S3_SECRET_KEY:-}" ]; then
        echo -e "${RED}ERROR: S3 credentials not set${NC}"
        echo "  Set RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY"
        echo ""
        echo "  Volume ID: ${VOLUME_ID}"
        echo "  You can upload data manually with:"
        echo "    ./runpod/deploy.sh upload-s3 --volume-id ${VOLUME_ID}"
        exit 1
    fi

    local s3_endpoint="https://s3api-$(echo ${RUNPOD_DATACENTER} | tr '[:upper:]' '[:lower:]').runpod.io"
    local s3_region=$(echo ${RUNPOD_DATACENTER} | tr '[:upper:]' '[:lower:]')

    echo "  S3 Endpoint: ${s3_endpoint}"
    echo "  Region: ${s3_region}"
    echo ""

    # Configure AWS CLI for RunPod S3 compatibility
    # Reduce multipart threshold/chunksize to avoid "InvalidPart" errors
    echo "  Configuring S3 transfer settings..."
    aws configure set default.s3.multipart_threshold 64MB
    aws configure set default.s3.multipart_chunksize 16MB
    aws configure set default.s3.max_concurrent_requests 4
    aws configure set default.s3.max_queue_size 100

    export AWS_RETRY_MODE=adaptive
    export AWS_MAX_ATTEMPTS=5

    # Run S3 upload with retries
    local upload_retry=0
    local upload_max_retries=3
    local upload_success=false

    while [ $upload_retry -lt $upload_max_retries ] && [ "$upload_success" = "false" ]; do
        if [ $upload_retry -gt 0 ]; then
            echo -e "${YELLOW}  Retry attempt ${upload_retry}/${upload_max_retries}...${NC}"
            sleep 5
        fi

        if AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}" \
             AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_KEY}" \
             aws s3 sync "${DATA_DIR}" "s3://${VOLUME_ID}/data/" \
             --endpoint-url "${s3_endpoint}" \
             --region "${s3_region}" \
             --cli-read-timeout 300 \
             --cli-connect-timeout 60 \
             --no-progress; then
            upload_success=true
        else
            upload_retry=$((upload_retry + 1))
        fi
    done

    if [ "$upload_success" = "false" ]; then
        echo -e "${RED}ERROR: S3 upload failed after ${upload_max_retries} attempts${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Re-run this command - 'sync' will resume from where it left off"
        echo "  2. Try smaller chunks: aws configure set default.s3.multipart_chunksize 8MB"
        echo "  3. Check network stability"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}  ✓ Training data uploaded successfully${NC}"
    step_times+=("Upload: $(($(date +%s) - step_start))s")

    # =========================================================================
    # Phase 3: Data Preparation
    # =========================================================================
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Phase 3: Data Preparation                              ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    step_start=$(date +%s)
    echo -e "${YELLOW}[5/9] Creating data preparation Pod...${NC}"

	    local gpu_list='[\"NVIDIA GeForce RTX 3090\", \"NVIDIA GeForce RTX 4090\", \"NVIDIA GeForce RTX 4080\", \"NVIDIA RTX A4000\", \"NVIDIA L4\"]'
	    local data_prep_pod_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-data-prep\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: COMMUNITY, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, dockerArgs: \\\"/workspace/runpod/data_prep_entrypoint.sh\\\", env: [ { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"OUTPUT_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data/train_data_final.pt\\\" }, { key: \\\"USE_CURATED_DATA\\\", value: \\\"${USE_CURATED_DATA}\\\" }, { key: \\\"SYNTHETIC_SAMPLES\\\", value: \\\"50000\\\" }, { key: \\\"HF_PUSH_DATASET\\\", value: \\\"${HF_PUSH_DATASET:-false}\\\" }, { key: \\\"HF_DATASET_REPO\\\", value: \\\"${HF_DATASET_REPO:-GesturaAI/stick-gen-dataset}\\\" }, { key: \\\"HF_TOKEN\\\", value: \\\"${HF_TOKEN:-}\\\" }, { key: \\\"GROK_API_KEY\\\", value: \\\"${GROK_API_KEY:-}\\\" } ] }) { id machine { podHostId gpuDisplayName } } }"

    local data_prep_response
    data_prep_response=$(runpod_graphql "$data_prep_pod_query")

    local data_prep_pod_id
    local gpu_used
    if check_jq; then
        data_prep_pod_id=$(echo "$data_prep_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
        gpu_used=$(echo "$data_prep_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "Unknown"')
    else
        data_prep_pod_id=$(echo "$data_prep_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        gpu_used="Unknown"
    fi

    if [ -z "$data_prep_pod_id" ]; then
        echo -e "${RED}ERROR: Failed to create data preparation Pod${NC}"
        echo "$data_prep_response"
        exit 1
    fi

    echo -e "${GREEN}  ✓ Data prep Pod created: ${data_prep_pod_id} (GPU: ${gpu_used})${NC}"

    step_start=$(date +%s)
    echo ""
    echo -e "${YELLOW}[6/9] Waiting for data preparation to complete...${NC}"
    echo "  This may take 30-60 minutes..."
    echo ""

    local prep_elapsed=0
    local prep_max_wait=7200  # 2 hours max

    while [ $prep_elapsed -lt $prep_max_wait ]; do
        local status_query="query { pod(input: { podId: \\\"${data_prep_pod_id}\\\" }) { desiredStatus runtime { uptimeInSeconds } } }"
        local status_response
        status_response=$(runpod_graphql "$status_query")

        local pod_status
        if check_jq; then
            pod_status=$(echo "$status_response" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
        else
            pod_status=$(echo "$status_response" | grep -o '"desiredStatus":"[^"]*"' | cut -d'"' -f4)
        fi

        if [ "$pod_status" = "EXITED" ] || [ "$pod_status" = "TERMINATED" ]; then
            echo ""
            echo -e "${GREEN}  ✓ Data preparation complete${NC}"
            break
        fi

        local prep_mins=$((prep_elapsed / 60))
        printf "\r  Data prep running: %d minutes | Status: %s     " "$prep_mins" "$pod_status"

        sleep 60
        prep_elapsed=$((prep_elapsed + 60))
    done

    # Terminate data prep Pod
    local term_query="mutation { podTerminate(input: { podId: \\\"${data_prep_pod_id}\\\" }) }"
    runpod_graphql "$term_query" > /dev/null 2>&1
    echo -e "${GREEN}  ✓ Data prep Pod terminated${NC}"
    step_times+=("DataPrep: $(($(date +%s) - step_start))s")

    # =========================================================================
    # Phase 4: Model Training
    # =========================================================================
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Phase 4: Sequential Model Training                     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Build variants array based on TRAIN_MODELS setting
    local variants=()
    case "$TRAIN_MODELS" in
        all)
            variants=("small" "medium" "large")
            ;;
        small,medium|medium,small)
            variants=("small" "medium")
            ;;
        small)
            variants=("small")
            ;;
        medium|base)
            variants=("medium")
            ;;
        large)
            variants=("large")
            ;;
        *)
            # Parse comma-separated list
            IFS=',' read -ra variants <<< "$TRAIN_MODELS"
            # Normalize "base" to "medium"
            for i in "${!variants[@]}"; do
                if [[ "${variants[$i]}" == "base" ]]; then
                    variants[$i]="medium"
                fi
            done
            ;;
    esac

    echo "  Training models: ${variants[*]}"
    echo ""

    local variant_step=7
    local total_steps=$((6 + ${#variants[@]}))

    for variant in "${variants[@]}"; do
        step_start=$(date +%s)
        echo -e "${YELLOW}[${variant_step}/${total_steps}] Training ${variant} model...${NC}"

        local train_query="mutation { podFindAndDeployOnDemand(input: { name: \\\"stick-gen-train-${variant}\\\", imageName: \\\"${FULL_IMAGE}\\\", cloudType: COMMUNITY, gpuTypeIdList: ${gpu_list}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50, networkVolumeId: \\\"${VOLUME_ID}\\\", volumeMountPath: \\\"${VOLUME_MOUNT_PATH}\\\", ports: \\\"22/tcp\\\", startSsh: true, env: [ { key: \\\"MODEL_VARIANT\\\", value: \\\"${variant}\\\" }, { key: \\\"DATA_PATH\\\", value: \\\"${VOLUME_MOUNT_PATH}/data\\\" }, { key: \\\"CHECKPOINT_DIR\\\", value: \\\"${VOLUME_MOUNT_PATH}/checkpoints\\\" }, { key: \\\"HF_TOKEN\\\", value: \\\"${HF_TOKEN:-}\\\" }, { key: \\\"GROK_API_KEY\\\", value: \\\"${GROK_API_KEY:-}\\\" }, { key: \\\"HF_REPO_NAME\\\", value: \\\"${HF_REPO_NAME:-GesturaAI/stick-gen}\\\" }, { key: \\\"AUTO_PUSH\\\", value: \\\"true\\\" }, { key: \\\"AUTO_CLEANUP\\\", value: \\\"true\\\" }, { key: \\\"VERSION\\\", value: \\\"${VERSION:-1.0.0}\\\" } ] }) { id machine { gpuDisplayName } } }"

        local train_response
        train_response=$(runpod_graphql "$train_query")

        local training_pod_id
        if check_jq; then
            training_pod_id=$(echo "$train_response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
            gpu_used=$(echo "$train_response" | jq -r '.data.podFindAndDeployOnDemand.machine.gpuDisplayName // "Unknown"')
        else
            training_pod_id=$(echo "$train_response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
            gpu_used="Unknown"
        fi

        if [ -z "$training_pod_id" ]; then
            echo -e "${RED}  ERROR: Failed to create training Pod for ${variant}${NC}"
            continue
        fi

        echo -e "${GREEN}  ✓ Training Pod created: ${training_pod_id} (GPU: ${gpu_used})${NC}"

        # Wait for training to complete
        echo "  Waiting for training to complete..."
        local train_elapsed=0
        local train_max_wait=28800  # 8 hours max per variant

        while [ $train_elapsed -lt $train_max_wait ]; do
            local status_query="query { pod(input: { podId: \\\"${training_pod_id}\\\" }) { desiredStatus runtime { uptimeInSeconds } } }"
            local status_response
            status_response=$(runpod_graphql "$status_query")

            local pod_status
            if check_jq; then
                pod_status=$(echo "$status_response" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
            else
                pod_status=$(echo "$status_response" | grep -o '"desiredStatus":"[^"]*"' | cut -d'"' -f4)
            fi

            if [ "$pod_status" = "EXITED" ] || [ "$pod_status" = "TERMINATED" ]; then
                echo ""
                echo -e "${GREEN}  ✓ Training complete for ${variant}${NC}"
                break
            fi

            local train_mins=$((train_elapsed / 60))
            printf "\r  Training %s: %d minutes | Status: %s     " "$variant" "$train_mins" "$pod_status"

            sleep 120
            train_elapsed=$((train_elapsed + 120))
        done

        # Terminate training Pod
        local term_query="mutation { podTerminate(input: { podId: \\\"${training_pod_id}\\\" }) }"
        runpod_graphql "$term_query" > /dev/null 2>&1
        echo -e "${GREEN}  ✓ Training Pod terminated${NC}"
        step_times+=("Train-${variant}: $(($(date +%s) - step_start))s")
        echo ""

        variant_step=$((variant_step + 1))
    done

    # =========================================================================
    # Summary
    # =========================================================================
    local pipeline_end_time=$(date +%s)
    local total_duration=$((pipeline_end_time - pipeline_start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Full Deployment Pipeline Complete!                     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Total Duration: ${hours}h ${minutes}m"
    echo ""
    echo "  Step Durations:"
    for step_time in "${step_times[@]}"; do
        echo "    - ${step_time}"
    done
    echo ""
    echo "  Infrastructure:"
    echo "    Network Volume: ${VOLUME_ID} (${VOLUME_SIZE}GB in ${RUNPOD_DATACENTER})"
    echo "    Docker Image:   ${FULL_IMAGE}"
    echo ""
    echo "  Models pushed to HuggingFace:"
    if [ -n "${HF_TOKEN:-}" ]; then
        local hf_base="${HF_REPO_NAME:-GesturaAI/stick-gen}"
        echo "    - https://huggingface.co/${hf_base}-small"
        echo "    - https://huggingface.co/${hf_base}-base"
        echo "    - https://huggingface.co/${hf_base}-large"
    else
        echo "    [Models not pushed - HF_TOKEN was not set]"
    fi
    echo ""
    echo "  Estimated Cost:"
    local gpu_hours=$(echo "scale=1; ${total_duration} / 3600" | bc 2>/dev/null || echo "N/A")
    echo "    - GPU Hours: ~${gpu_hours}"
    echo "    - Estimated: \$50-100 (varies by GPU type and availability)"
    echo ""
    echo "  Next Steps:"
    echo "    - Verify models on HuggingFace"
    echo "    - Create inference endpoint: ./runpod/deploy.sh create-endpoint --volume-id ${VOLUME_ID}"
    echo "    - Network Volume retained for inference (delete manually when done)"
    echo ""
}

# ============================================================================
# Serverless Endpoint Management (for Inference)
# ============================================================================

create_serverless_endpoint() {
    check_runpod_api_key
    print_header

    echo -e "${GREEN}Creating Serverless Endpoint for Inference...${NC}"
    echo "  Name: stick-gen-${MODEL_VARIANT}-inference"
    echo "  Image: ${FULL_IMAGE}"
    echo "  GPU: ${GPU_TYPE}"
    echo "  Workers: ${WORKERS_MIN} - ${WORKERS_MAX}"
    if [ -n "$VOLUME_ID" ]; then
        echo "  Network Volume: ${VOLUME_ID}"
    fi
    echo ""

    # Map GPU type to RunPod GPU ID format
    # Common mappings: AMPERE_16 (RTX 3000/4000 series), AMPERE_24 (A10), ADA_24 (L4)
    local gpu_ids="AMPERE_16"
    case "$GPU_TYPE" in
        *"A100"*) gpu_ids="AMPERE_80" ;;
        *"A40"*) gpu_ids="AMPERE_48" ;;
        *"A10"*|*"RTX A4000"*) gpu_ids="AMPERE_16" ;;
        *"L4"*) gpu_ids="ADA_24" ;;
        *"4090"*|*"4080"*) gpu_ids="ADA_24" ;;
        *"3090"*|*"3080"*) gpu_ids="AMPERE_24" ;;
        *) gpu_ids="AMPERE_16" ;;
    esac

    # Build the mutation with optional network volume
    local volume_param=""
    if [ -n "$VOLUME_ID" ]; then
        volume_param=", networkVolumeId: \\\"${VOLUME_ID}\\\""
    fi

    local query="mutation { saveEndpoint(input: { name: \\\"stick-gen-${MODEL_VARIANT}-inference\\\", gpuIds: \\\"${gpu_ids}\\\", idleTimeout: 5, scalerType: \\\"QUEUE_DELAY\\\", scalerValue: 4, workersMin: ${WORKERS_MIN}, workersMax: ${WORKERS_MAX}, templateId: null, dockerArgs: \\\"\\\", imageName: \\\"${FULL_IMAGE}\\\", env: [ { key: \\\"MODEL_VARIANT\\\", value: \\\"${MODEL_VARIANT}\\\" }, { key: \\\"DEVICE\\\", value: \\\"cuda\\\" }, { key: \\\"DATA_PATH\\\", value: \\\"/runpod-volume/data\\\" } ]${volume_param} }) { id name gpuIds workersMin workersMax } }"

    local response
    response=$(runpod_graphql "$query")

    local endpoint_id
    if check_jq; then
        echo "$response" | jq .
        endpoint_id=$(echo "$response" | jq -r '.data.saveEndpoint.id // empty')
    else
        echo "$response"
        endpoint_id=$(echo "$response" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -n "$endpoint_id" ] && [ "$endpoint_id" != "null" ]; then
        echo ""
        echo -e "${GREEN}✓ Serverless Endpoint created successfully!${NC}"
        echo -e "${CYAN}Endpoint ID: ${endpoint_id}${NC}"
        echo ""
        echo "Your endpoint URL:"
        echo "  https://api.runpod.ai/v2/${endpoint_id}/runsync"
        echo ""
        echo "Submit inference jobs with:"
        echo "  ./runpod/deploy.sh submit-job --endpoint-id ${endpoint_id}"
        echo ""
        echo "Or via curl:"
        echo "  curl -X POST https://api.runpod.ai/v2/${endpoint_id}/runsync \\"
        echo "    -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"input\": {\"prompt\": \"A person walking forward\"}}'"
    else
        echo -e "${RED}Failed to create Serverless Endpoint${NC}"
        echo "Response: $response"
        exit 1
    fi
}

list_serverless_endpoints() {
    check_runpod_api_key
    print_header

    echo -e "${GREEN}Listing Serverless Endpoints...${NC}"
    echo ""

    local query="query { myself { endpoints { id name gpuIds workersMin workersMax networkVolumeId } } }"
    local response
    response=$(runpod_graphql "$query")

    if check_jq; then
        echo "$response" | jq '.data.myself.endpoints[] | {id, name, gpuIds, workers: "\(.workersMin)-\(.workersMax)", networkVolumeId}'
    else
        echo "$response"
    fi
}

submit_inference_job() {
    check_runpod_api_key
    print_header

    if [ -z "$ENDPOINT_ID" ]; then
        echo -e "${RED}Error: --endpoint-id is required${NC}"
        echo ""
        echo "Usage: ./runpod/deploy.sh submit-job --endpoint-id YOUR_ENDPOINT_ID"
        echo ""
        echo "To find your endpoint ID:"
        echo "  ./runpod/deploy.sh list-endpoints"
        exit 1
    fi

    echo -e "${GREEN}Submitting Inference Job...${NC}"
    echo "  Endpoint: ${ENDPOINT_ID}"
    echo ""

    # Example inference request
    local prompt="${PROMPT:-A person walking forward}"
    echo "  Prompt: ${prompt}"
    echo ""

    local response
    response=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"input\": {\"prompt\": \"${prompt}\", \"model_variant\": \"${MODEL_VARIANT}\"}}")

    if check_jq; then
        echo "$response" | jq .
    else
        echo "$response"
    fi
}

delete_serverless_endpoint() {
    check_runpod_api_key
    print_header

    if [ -z "$ENDPOINT_ID" ]; then
        echo -e "${RED}Error: --endpoint-id is required${NC}"
        echo ""
        echo "Usage: ./runpod/deploy.sh delete-endpoint --endpoint-id YOUR_ENDPOINT_ID"
        echo ""
        echo "To find your endpoint ID:"
        echo "  ./runpod/deploy.sh list-endpoints"
        exit 1
    fi

    echo -e "${YELLOW}Deleting Serverless Endpoint: ${ENDPOINT_ID}${NC}"
    echo ""
    echo -e "${RED}Warning: This action cannot be undone!${NC}"
    echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
    read -r

    local query="mutation { deleteEndpoint(id: \\\"${ENDPOINT_ID}\\\") }"
    local response
    response=$(runpod_graphql "$query")

    if check_jq; then
        echo "$response" | jq .
    else
        echo "$response"
    fi

    echo ""
    echo -e "${GREEN}✓ Endpoint deleted${NC}"
}

# ============================================================================
# Docker Build & Deploy Functions
# ============================================================================

build_image() {
    echo -e "${GREEN}[1/3] Building Docker image...${NC}"
    docker build \
        --platform linux/amd64 \
        --build-arg MODEL_VARIANT="${MODEL_VARIANT}" \
        -t "${FULL_IMAGE}" \
        -t "${VARIANT_IMAGE}" \
        -f docker/Dockerfile \
        .
    echo -e "${GREEN}✓ Build complete${NC}"
}

push_image() {
    echo -e "${GREEN}[2/3] Pushing to GitHub Container Registry...${NC}"

    # Check if logged in to ghcr.io
    if ! docker info 2>/dev/null | grep -q "ghcr.io"; then
        echo -e "${YELLOW}Tip: If push fails, authenticate with:${NC}"
        echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u ${GITHUB_USERNAME} --password-stdin"
        echo ""
    fi

    docker push "${FULL_IMAGE}"
    docker push "${VARIANT_IMAGE}"

    echo -e "${GREEN}✓ Push complete${NC}"
    echo ""
    echo -e "${YELLOW}After first push, make the image public at:${NC}"
    echo "  https://github.com/users/${GITHUB_USERNAME}/packages/container/${IMAGE_NAME}/settings"
    echo "  (Or for orgs: https://github.com/orgs/${GITHUB_USERNAME}/packages/container/${IMAGE_NAME}/settings)"
}

deploy_endpoint() {
    echo -e "${GREEN}[3/3] Deploying to RunPod...${NC}"
    
    if ! command -v runpodctl &> /dev/null; then
        echo -e "${YELLOW}runpodctl not found. Install with:${NC}"
        echo "  pip install runpodctl"
        echo ""
        echo -e "${YELLOW}Manual deployment steps:${NC}"
        echo "  1. Go to https://www.runpod.io/console/serverless"
        echo "  2. Create new endpoint"
        echo "  3. Use image: ${FULL_IMAGE}"
        echo "  4. Set environment variables from runpod/config.yaml"
        return 0
    fi
    
    echo "Creating RunPod serverless endpoint..."
    runpodctl create endpoint \
        --name "stick-gen-${MODEL_VARIANT}" \
        --image "${FULL_IMAGE}" \
        --gpu "NVIDIA RTX A4000" \
        --env "MODEL_VARIANT=${MODEL_VARIANT}" \
        --env "DEVICE=cuda" \
        --env "MODEL_PATH=/workspace/models/model_checkpoint.pth" \
        --env "CONFIG_PATH=/workspace/configs/${MODEL_VARIANT}.yaml" \
        --env "GROK_API_KEY={{ RUNPOD_SECRET_GROK_API_KEY }}"

    echo -e "${GREEN}✓ Deployment complete${NC}"
}

# Execute based on action
case $ACTION in
    # === Docker Build & Deploy ===
    build)
        print_header
        build_image
        ;;
    push)
        print_header
        push_image
        ;;
    deploy)
        print_header
        deploy_endpoint
        ;;
    all)
        print_header
        build_image
        push_image
        deploy_endpoint
        ;;
    login)
        print_header
        echo -e "${GREEN}Logging in to GitHub Container Registry...${NC}"
        if [ -z "${GITHUB_TOKEN}" ]; then
            echo -e "${RED}Error: GITHUB_TOKEN environment variable is not set${NC}"
            echo ""
            echo "To authenticate with ghcr.io:"
            echo "  1. Create a PAT at: https://github.com/settings/tokens"
            echo "     Required scopes: read:packages, write:packages, delete:packages"
            echo "  2. Export your token: export GITHUB_TOKEN=ghp_xxxxxxxxxxxx"
            echo "  3. Run this command again: ./deploy.sh login"
            exit 1
        fi
        echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GITHUB_USERNAME}" --password-stdin
        echo -e "${GREEN}✓ Login successful${NC}"
        exit 0
        ;;

    # === Training Infrastructure ===
    create-volume)
        create_network_volume
        ;;
    list-volumes)
        list_network_volumes
        ;;
    upload-data)
        upload_data_to_volume
        ;;
    create-pod)
        create_training_pod
        ;;
    list-pods)
        list_pods
        ;;
    train-setup)
        train_setup
        ;;
    send-data)
        send_data_runpodctl
        ;;
    upload-s3)
        upload_data_s3
        ;;
    train-small)
        MODEL_VARIANT="small"
        train_model
        ;;
    train-base)
        MODEL_VARIANT="base"
        train_model
        ;;
    train-large)
        MODEL_VARIANT="large"
        train_model
        ;;
    auto-train-all)
        auto_train_all
        ;;
    prep-data)
        # Create a data preparation Pod only (without training)
        create_data_prep_pod
        ;;
    full-deploy)
        # Complete pipeline: build, upload, prep, train all
        full_deploy
        ;;

    # === Serverless Inference Infrastructure ===
    create-endpoint)
        create_serverless_endpoint
        ;;
    list-endpoints)
        list_serverless_endpoints
        ;;
    submit-job)
        submit_inference_job
        ;;
    delete-endpoint)
        delete_serverless_endpoint
        ;;

    # === Help ===
    help|--help|-h)
        echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║           Stick-Gen RunPod Deployment                      ║${NC}"
        echo -e "${BLUE}║           by Gestura AI                                    ║${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Usage:${NC} ./deploy.sh [action] [options]"
        echo ""
        echo -e "${GREEN}=== Docker Build & Deploy ===${NC}"
        echo "  build          Build Docker image locally"
        echo "  push           Push image to GitHub Container Registry (ghcr.io)"
        echo "  deploy         Deploy to RunPod Serverless endpoint"
        echo "  login          Authenticate with GitHub Container Registry"
        echo "  all            Build, push, and deploy (full workflow)"
        echo ""
        echo -e "${GREEN}=== Training Infrastructure (Pods) ===${NC}"
        echo "  create-volume  Create a RunPod Network Volume for training data"
        echo "  list-volumes   List existing Network Volumes"
        echo "  upload-data    Upload ./data directory to Network Volume (via SSH/rsync)"
        echo "  upload-s3      Upload ./data directory via S3-compatible API (automated)"
        echo "  send-data      Send data via runpodctl (peer-to-peer transfer)"
        echo "  create-pod     Create a training Pod with Network Volume attached"
        echo "  list-pods      List running Pods"
        echo "  train-setup    Full training setup (volume + upload + pod)"
        echo ""
        echo -e "${GREEN}=== Automated Training (End-to-End) ===${NC}"
        echo "  prep-data      Create a data preparation Pod (generates train_data_final.pt)"
        echo "  train-small    Train small model variant (5.6M params, ~30 epochs)"
        echo "  train-base     Train medium model variant (15.8M params)"
        echo "  train-large    Train large model variant (28M params)"
        echo "  auto-train-all Full automated pipeline: data prep + train selected models"
        echo "  full-deploy    Complete end-to-end pipeline from scratch (RECOMMENDED)"
        echo ""
        echo -e "${GREEN}=== Inference Infrastructure (Serverless) ===${NC}"
        echo "  create-endpoint   Create a Serverless endpoint for inference"
        echo "  list-endpoints    List existing Serverless endpoints"
        echo "  submit-job        Submit an inference job to a Serverless endpoint"
        echo "  delete-endpoint   Delete a Serverless endpoint"
        echo ""
        echo -e "${YELLOW}Options:${NC}"
        echo "  --models MODELS              Models to train (default: all)"
        echo "                               Options: small,medium | large | all"
        echo "  --variant small|medium|large Model variant for single commands (default: medium)"
        echo "  --version VERSION            Image version tag (default: latest)"
        echo "  --volume-id ID               Network Volume ID"
        echo "  --volume-size GB             Volume size in GB (default: 200)"
        echo "  --datacenter ID              Data center ID (default: EU-CZ-1)"
        echo "                               Options: US-TX-3, US-CA-1, EU-NL-1, EU-CZ-1, etc."
        echo "                               NOTE: Pods MUST be in same datacenter as Network Volume!"
        echo "  --gpu GPU_TYPE               GPU type (default: NVIDIA RTX A4000)"
        echo "  --data-dir PATH              Data directory (default: ./data)"
        echo "  --curated                    Use curated data prep pipeline"
        echo "  --endpoint-id ID             Serverless endpoint ID"
        echo "  --workers-min N              Min workers for serverless (default: 0)"
        echo "  --workers-max N              Max workers for serverless (default: 3)"
        echo ""
        echo -e "${YELLOW}Environment Variables:${NC}"
        echo "  GITHUB_USERNAME   GitHub username (default: gestura-ai)"
        echo "  GITHUB_TOKEN      GitHub PAT for ghcr.io authentication"
        echo "  RUNPOD_API_KEY    RunPod API key (REQUIRED)"
        echo "  RUNPOD_S3_ACCESS_KEY  RunPod S3 access key (for data upload)"
        echo "  RUNPOD_S3_SECRET_KEY  RunPod S3 secret key (for data upload)"
        echo "  HF_TOKEN          HuggingFace API token (for model uploads)"
        echo "  GROK_API_KEY      X.AI Grok API key (optional, for LLM dataset enhancement)"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo ""
        echo "  # Train small + medium models (recommended for first-time, ~\$50)"
        echo "  ./runpod/deploy.sh --datacenter EU-CZ-1 --models small,medium"
        echo ""
        echo "  # Train large model only (requires A100, ~\$165)"
        echo "  ./runpod/deploy.sh --datacenter EU-CZ-1 --models large --gpu 'NVIDIA A100 PCIe'"
        echo ""
        echo "  # Train all models (small, medium, large, ~\$215)"
        echo "  ./runpod/deploy.sh --datacenter EU-CZ-1 --models all"
        echo ""
        echo "  # With existing volume (skip volume creation + data upload)"
        echo "  ./runpod/deploy.sh auto-train-all --volume-id vol_xxxxx --models small,medium"
        echo ""
        echo "  # Explicit full-deploy action"
        echo "  ./runpod/deploy.sh full-deploy --datacenter EU-CZ-1 --volume-size 200 --models all"
        exit 0
        ;;

    *)
        echo -e "${RED}Unknown action: ${ACTION}${NC}"
        echo ""
        echo "Run './runpod/deploy.sh help' for usage information."
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Operation Complete!                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

