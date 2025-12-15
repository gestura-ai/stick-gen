# Stick-Gen Docker Image for RunPod Training & Inference
# Gestura AI - https://gestura.ai
#
# Build: docker build --platform linux/amd64 -t ghcr.io/gestura-ai/stick-gen:latest -f docker/Dockerfile .
# Push:  docker push ghcr.io/gestura-ai/stick-gen:latest
#
# For automated training on RunPod:
#   - Mount Network Volume at /runpod-volume with training data at /runpod-volume/data
#   - Set environment variables: MODEL_VARIANT, HF_TOKEN, AUTO_PUSH, AUTO_CLEANUP
#   - The container will automatically start training when it boots

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

LABEL maintainer="Gestura AI <dev@gestura.ai>"
LABEL description="Stick-Gen: Text-to-Animation Transformer Model"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    WORKSPACE_DIR=/workspace \
    MODEL_DIR=/workspace/models \
    DATA_DIR=/workspace/data \
    HF_HOME=/workspace/.cache/huggingface \
    # Training environment variables (can be overridden at runtime)
    MODEL_VARIANT=small \
    DATA_PATH=/runpod-volume/data \
    CHECKPOINT_DIR=/runpod-volume/checkpoints \
    AUTO_PUSH=true \
    AUTO_CLEANUP=true \
    TRAIN_ALL=false \
    VERSION=1.0.0

# Install system dependencies (including build tools for pycairo and manimpango)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    openssh-server \
    nginx \
    build-essential \
    gcc \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create workspace directories
RUN mkdir -p ${WORKSPACE_DIR} ${MODEL_DIR} ${DATA_DIR} ${HF_HOME}

WORKDIR ${WORKSPACE_DIR}

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: chumpy has a broken setup.py that imports pip directly, so we install it with --no-build-isolation
RUN pip install --upgrade pip setuptools wheel && \
    pip install chumpy --no-build-isolation && \
    pip install -r requirements.txt && \
    pip install runpod>=1.6.0 huggingface_hub>=0.20.0

# Pre-download the text embedding model for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Copy source code
COPY src/ ${WORKSPACE_DIR}/src/
COPY configs/ ${WORKSPACE_DIR}/configs/
COPY scripts/ ${WORKSPACE_DIR}/scripts/
COPY runpod/ ${WORKSPACE_DIR}/runpod/

# Copy model cards for HuggingFace push
COPY model_cards/ ${WORKSPACE_DIR}/model_cards/

# Copy additional files needed for HuggingFace push
COPY requirements.txt ${WORKSPACE_DIR}/
COPY LICENSE ${WORKSPACE_DIR}/
COPY CITATIONS.md ${WORKSPACE_DIR}/

# Make entrypoint scripts executable
RUN chmod +x ${WORKSPACE_DIR}/runpod/train_entrypoint.sh \
    && chmod +x ${WORKSPACE_DIR}/runpod/data_prep_entrypoint.sh

# Set Python path
ENV PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

# Expose SSH port (for RunPod SSH access)
EXPOSE 22

# Default command - automated training workflow
# Override with: docker run ... python -u runpod/handler.py (for serverless inference)
# Or SSH into the container and run training manually
CMD ["/workspace/runpod/train_entrypoint.sh"]
