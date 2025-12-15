# RunPod Deployment Guide

This guide covers deploying Stick-Gen training infrastructure on [RunPod](https://runpod.io?ref=z71ozsfc).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start - Automated Training](#quick-start---automated-training)
  - [Environment Variables for Training](#environment-variables-for-training)
  - [Environment Variables for Data Preparation](#environment-variables-for-data-preparation)
  - [Complete Pipeline from Scratch](#example-complete-pipeline-from-scratch-recommended)
  - [Train with Existing Volume](#example-train-with-existing-volume)
  - [Custom HuggingFace Repository](#custom-huggingface-repository)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Cost Optimization](#cost-optimization)
- [SSH Connection Methods](#ssh-connection-methods)
- [Critical API Parameters](#critical-api-parameters)
- [Troubleshooting](#troubleshooting)
- [Monitoring Training](#monitoring-training)

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io?ref=z71ozsfc)
2. **API Key**: Get your API key from [RunPod Settings](https://www.runpod.io/console/user/settings)
3. **S3 Credentials** (for data upload without GPU): Generate at [Storage Settings](https://www.runpod.io/console/user/settings) → Storage → S3 Access Keys
4. **SSH Key**: Generate an SSH key pair if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
5. **Add SSH Key to RunPod**: Go to [SSH Keys](https://www.runpod.io/console/user/settings) and add your public key
6. **AWS CLI**: Install for S3 uploads: `brew install awscli` (macOS) or `pip install awscli`

## Quick Start - Automated Training

The Docker image supports **fully automated "fire and forget" training**. When a Pod starts, it automatically:
1. Validates training data exists on the Network Volume
2. Trains the specified model variant
3. Pushes the best checkpoint to HuggingFace Hub
4. Cleans up intermediate checkpoints to free space

### Environment Variables for Training

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VARIANT` | `small` | Which model to train: `small`, `medium`, or `large` |
| `DATA_PATH` | `/runpod-volume/data` | Path to training data |
| `TRAIN_DATA_PATH` | `$DATA_PATH/train_data_final.pt` | Path to training dataset file (can point at curated pretraining set) |
| `CHECKPOINT_DIR` | `/runpod-volume/checkpoints` | Where to save checkpoints |
| `RESUME_FROM_CHECKPOINT` | (none) | Optional absolute path on the Network Volume to a checkpoint file. If set, `runpod/train_entrypoint.sh` forwards `--resume_from` to `src.train.train` for continued pretraining/resume. |
| `HF_TOKEN` | (none) | HuggingFace token for auto-push (required for auto-push) |
| `GROK_API_KEY` | (none) | **NEW:** X.AI Grok API key for LLM-enhanced dataset generation |
| `HF_REPO_NAME` | `GesturaAI/stick-gen` | Base HuggingFace repo name (variant suffix appended automatically) |
| `AUTO_PUSH` | `true` | Auto-push to HuggingFace after training (with 3 retry attempts) |
| `AUTO_CLEANUP` | `true` | Clean up intermediate checkpoints after successful push |
| `TRAIN_ALL` | `false` | Train all variants sequentially (small → medium → large) |
| `VERSION` | `1.0.0` | Model version tag for HuggingFace |

### Environment Variables for Data Preparation

These control the behavior of `runpod/data_prep_entrypoint.sh`, which now supports **two modes**:

- **Legacy pipeline**: raw text/100STYLE + synthetic → `train_data_final.pt` (with embeddings)
- **Curated pipeline**: canonical `.pt` files → curated splits → embedded pretraining/SFT datasets

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `/runpod-volume/data` | Path to raw **or canonical** data on the Network Volume |
| `OUTPUT_PATH` | `/runpod-volume/data/train_data_final.pt` | Output path for the final training dataset. In legacy mode this is the combined dataset; in curated mode it typically points at the curated pretraining dataset (with embeddings). |
| `USE_CURATED_DATA` | `false` | If `true`, run the curated canonical → curated → embedded pipeline instead of the legacy `prepare_data.py` pipeline. |
| `CURATED_OUTPUT_DIR` | `$DATA_PATH/curated` | Directory where curated outputs are written: `pretrain_data.pt`, `sft_data.pt`, `curation_stats.json`, and `sft_data_embedded.pt`. |
| `CURATION_CANONICAL_DIR` | `$DATA_PATH/canonical` | Directory to search for canonical `.pt` inputs when `CURATION_INPUTS` is not set. |
| `CURATION_INPUTS` | (none) | Optional **space-separated** list of canonical `.pt` files. If set, this overrides scanning `CURATION_CANONICAL_DIR`. |
| `CURATION_MIN_QUALITY_PRETRAIN` | `0.5` | Minimum `quality_score` for the pretraining split. |
| `CURATION_MIN_QUALITY_SFT` | `0.8` | Minimum `quality_score` for the SFT split. |
| `CURATION_MIN_CAMERA_STABILITY_SFT` | `0.6` | Minimum camera stability score for SFT samples. |
| `CURATION_BALANCE_MAX_FRACTION` | `0.3` | Maximum fraction of SFT samples assigned to any single dominant action label (for balancing). |
| `SYNTHETIC_SAMPLES` | `50000` | Number of synthetic samples to generate (**legacy pipeline only**). Ignored in curated mode. |
| `HF_PUSH_DATASET` | `false` | Upload processed dataset to HuggingFace. |
| `HF_DATASET_REPO` | `GesturaAI/stick-gen-dataset` | HuggingFace dataset repository. |
| `HF_TOKEN` | (none) | HuggingFace token for dataset upload. |
| `GROK_API_KEY` | (none) | X.AI Grok API key for LLM story generation (optional). |

#### Curated data prep mode (canonical → curated → embedded)

To use the curated pipeline on RunPod:

1. **Upload canonical `.pt` files** to your Network Volume, e.g. under:

   ```bash
   # Example: upload canonical datasets from ./data/canonical on your machine
   aws s3 sync ./data/canonical s3://YOUR_VOLUME_ID/data/canonical/ \
     --endpoint-url https://s3api-eu-cz-1.runpod.io --region eu-cz-1
   ```

2. **Run data prep in curated mode** using the `--curated` flag (shorthand for `USE_CURATED_DATA=true`):

   ```bash
   # Curated data preparation only (no training yet)
   ./runpod/deploy.sh prep-data --volume-id YOUR_VOLUME_ID --curated
   ```

   This will:

   - Load canonical `.pt` files from `$DATA_PATH/canonical` (or `CURATION_INPUTS`).
   - Run `scripts/prepare_curated_datasets.py` to produce:
     - `$DATA_PATH/curated/pretrain_data.pt`
     - `$DATA_PATH/curated/sft_data.pt`
     - `$DATA_PATH/curated/curation_stats.json`
   - Run `scripts/build_dataset_for_training.py` to attach text embeddings and write the **training dataset** to `OUTPUT_PATH` (by default `/runpod-volume/data/train_data_final.pt` when using `deploy.sh`).
   - Optionally embed the SFT split as `$DATA_PATH/curated/sft_data_embedded.pt`.

3. **Train on the curated pretraining dataset** by pointing `TRAIN_DATA_PATH` at the prepared file (see next section), or by using the default `/runpod-volume/data/train_data_final.pt` when you let `deploy.sh` manage both prep and training.

### Example: Complete Pipeline from Scratch (Recommended)

The `full-deploy` command handles everything from building the Docker image to training all models:

```bash
# Set all credentials
export RUNPOD_API_KEY='your_api_key'
export HF_TOKEN='hf_xxxxx'  # Get from https://huggingface.co/settings/tokens
export RUNPOD_S3_ACCESS_KEY='user_xxxxx'  # Get from RunPod Storage Settings
export RUNPOD_S3_SECRET_KEY='rps_xxxxx'
export GROK_API_KEY='xai-xxxxx'  # Optional: for LLM-enhanced datasets

# Single command that does everything (legacy data prep pipeline)
./runpod/deploy.sh full-deploy --datacenter EU-CZ-1 --volume-size 120

# To use the **curated** data prep pipeline as part of full-deploy
# (requires canonical .pt files under ./data/canonical before upload):
# ./runpod/deploy.sh full-deploy --datacenter EU-CZ-1 --volume-size 120 --curated
```

This will:
1. ✅ Build Docker image from `docker/Dockerfile`
2. ✅ Push image to GitHub Container Registry (ghcr.io)
3. ✅ Create a Network Volume (120GB in EU-CZ-1)
4. ✅ Upload training data via S3 API (~87GB)
5. ✅ Create data preparation Pod and wait for completion
6. ✅ Train small → medium → large models sequentially
7. ✅ Push each model to HuggingFace (with 3 retry attempts and exponential backoff)
8. ✅ Display comprehensive summary with success/failure status for each model

**New Features:**
- 🤖 **LLM Enhancement**: Set `GROK_API_KEY` to use Grok API for AI-generated story scripts
- 🔄 **Retry Logic**: Automatic retry (3 attempts) for HuggingFace uploads with exponential backoff
- 💾 **Checkpoint Preservation**: Failed model pushes preserve checkpoints and continue to next variant
- 📊 **Pipeline Summary**: Detailed report showing which models succeeded/failed with HuggingFace URLs

**Estimated time:** 6-12 hours | **Estimated cost:** $50-100

### Example: Train with Existing Volume

If you already have a Network Volume with data uploaded, use `auto-train-all`:

```bash
export RUNPOD_API_KEY='your_api_key'
export HF_TOKEN='hf_xxxxx'

# Run legacy data prep + train all variants
./runpod/deploy.sh auto-train-all --volume-id gol2v1emhp

# Or run the **curated** pipeline (requires canonical .pt files on the volume)
./runpod/deploy.sh auto-train-all --volume-id gol2v1emhp --curated
```

The legacy command will:
1. ✅ Create a data preparation Pod
2. ✅ Generate and process training data into `train_data_final.pt`
3. ✅ Wait for data prep to complete, then terminate the Pod
4. ✅ Sequentially train small → medium → large models
5. ✅ Push each model to HuggingFace (e.g., `GesturaAI/stick-gen-small`)
6. ✅ Clean up checkpoints between variants

### Example: Individual Training Commands

```bash
# Train small model with auto-push to HuggingFace (legacy data prep already run)
./runpod/deploy.sh train-small --volume-id gol2v1emhp

# Or train other variants
./runpod/deploy.sh train-medium --volume-id gol2v1emhp   # 15.8M params
./runpod/deploy.sh train-large --volume-id gol2v1emhp    # 28M params

# Data preparation only (without training) - legacy pipeline
./runpod/deploy.sh prep-data --volume-id gol2v1emhp

# Data preparation only, using the curated pipeline (requires canonical .pt files)
./runpod/deploy.sh prep-data --volume-id gol2v1emhp --curated
```

### Checkpoint resume and continued pretraining on RunPod

Once you have trained a model and saved checkpoints under `CHECKPOINT_DIR`
(`runpod/train_entrypoint.sh` defaults to `/runpod-volume/checkpoints`), you can
launch a **follow-up training Pod** that resumes from an existing checkpoint.

Common checkpoint locations are:

- `/runpod-volume/checkpoints/model_checkpoint_best.pth`
- `/runpod-volume/checkpoints/checkpoint_epoch_50.pth`

To resume training from one of these:

1. Note the exact path on the Network Volume, e.g.

   ```bash
   /runpod-volume/checkpoints/model_checkpoint_best.pth
   ```

2. When launching the training Pod (via `deploy.sh` or directly in the RunPod
   UI/API), set the environment variable:

   ```bash
   export RESUME_FROM_CHECKPOINT=/runpod-volume/checkpoints/model_checkpoint_best.pth
   ```

3. `runpod/train_entrypoint.sh` will detect `RESUME_FROM_CHECKPOINT` and call:

   ```bash
   python -m src.train.train \
     --config "configs/${MODEL_VARIANT}.yaml" \
     --data_path "${TRAIN_DATA_PATH}" \
     --checkpoint_dir "${CHECKPOINT_DIR}" \
     --resume_from "${RESUME_FROM_CHECKPOINT}"
   ```

4. Inside `src/train/train.py`, the checkpoint is loaded and training resumes
   from the next epoch, preserving optimizer/scheduler state and `best_val_loss`.

This mechanism works for both **legacy** and **curated** pipelines; only the
contents of `TRAIN_DATA_PATH` differ.

### SFT (Supervised Fine-Tuning) on RunPod

For SFT workflows, use the dedicated `runpod/sft_entrypoint.sh` script which
supports:

- **init_from**: Load pretrained weights only (fresh optimizer for SFT)
- **LoRA**: Efficient fine-tuning with frozen pretrained model

#### Environment Variables for SFT

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VARIANT` | `sft_medium` | SFT config: `sft_small`, `sft_medium`, or `sft_large` |
| `INIT_FROM_CHECKPOINT` | (none) | Path to pretrained checkpoint for weight initialization |
| `USE_LORA` | `false` | Enable LoRA fine-tuning |
| `LORA_RANK` | `8` | LoRA rank (when USE_LORA=true) |
| `TRAIN_DATA_PATH` | `$DATA_PATH/curated/sft_data_embedded.pt` | Path to SFT dataset |

#### Example: SFT from Pretrained Checkpoint

```bash
# After pretraining completes, run SFT on the same volume
export MODEL_VARIANT=sft_medium
export INIT_FROM_CHECKPOINT=/runpod-volume/checkpoints/pretrain/model_checkpoint_best.pth
export TRAIN_DATA_PATH=/runpod-volume/data/curated/sft_data_embedded.pt
export CHECKPOINT_DIR=/runpod-volume/checkpoints/sft

./runpod/sft_entrypoint.sh
```

#### Example: SFT with LoRA

```bash
export MODEL_VARIANT=sft_medium
export INIT_FROM_CHECKPOINT=/runpod-volume/checkpoints/pretrain/model_checkpoint_best.pth
export USE_LORA=true
export LORA_RANK=8

./runpod/sft_entrypoint.sh
```

With LoRA enabled, only ~1-5% of parameters are trainable, significantly
reducing memory usage and training time while preserving pretrained knowledge.

### Deployment Readiness Checklist (Curated Continued Pretraining)

Before starting long curated continued pretraining runs on RunPod, ensure:

- [ ] A Network Volume of at least the recommended size from `RUNPOD_SYSTEM_REQUIREMENTS.md` (typically ≥160 GB for curated runs) is created.
- [ ] Canonical `.pt` datasets have been uploaded under `$DATA_PATH/canonical` (or paths are provided via `CURATION_INPUTS`).
- [ ] `prep-data --curated` has completed successfully on that volume and `train_data_final.pt` (curated embedded dataset) exists under `$DATA_PATH`.
- [ ] Training configs (`configs/small.yaml`, `configs/medium.yaml`, `configs/large.yaml`) point at the desired curated dataset (or `TRAIN_DATA_PATH` is set accordingly).
- [ ] Evaluation/e2e scripts are configured to read checkpoints from `CHECKPOINT_DIR` and produce the usual JSON/HTML reports.

Once all items are checked, you can safely use `auto-train-all --curated` or individual `train-*` commands for curated continued pretraining.

### Custom HuggingFace Repository

To push models to a different HuggingFace account or organization:

```bash
# Push to your personal account
export HF_REPO_NAME='your-username/stick-gen'
./runpod/deploy.sh train-small --volume-id gol2v1emhp
# → Pushes to: your-username/stick-gen-small

# Push to an organization
export HF_REPO_NAME='my-org/custom-model-name'
./runpod/deploy.sh train-medium --volume-id gol2v1emhp
# → Pushes to: my-org/custom-model-name-medium
```

**Note:** Remember to terminate Pods after training completes to save costs!

## Step-by-Step Deployment

### 1. Create a Network Volume

Network Volumes provide persistent storage that survives Pod restarts.

```bash
export RUNPOD_API_KEY='your_api_key'
./runpod/deploy.sh create-volume --volume-size 100 --datacenter EU-CZ-1
```

**Important**: Note the Volume ID returned (e.g., `gol2v1emhp`). You'll need this for subsequent commands.

### 2. Upload Training Data

#### Option A: S3-Compatible API (Recommended - No GPU Required!)

**This is the most cost-effective method.** It uploads directly to the Network Volume without needing a running Pod, saving GPU costs.

First, generate S3 credentials from the RunPod console:
1. Go to: https://www.runpod.io/console/user/settings
2. Click 'Storage' in the left menu
3. Under 'S3 Access Keys', click 'Generate Key'

Then set the environment variables and run:

```bash
export RUNPOD_S3_ACCESS_KEY='user_xxxxx'
export RUNPOD_S3_SECRET_KEY='rps_xxxxx'
./runpod/deploy.sh upload-s3 --volume-id YOUR_VOLUME_ID
```

Or use AWS CLI directly (for large uploads with progress monitoring):

```bash
AWS_ACCESS_KEY_ID='user_xxxxx' \
AWS_SECRET_ACCESS_KEY='rps_xxxxx' \
aws s3 sync ./data s3://YOUR_VOLUME_ID/data/ \
  --endpoint-url https://s3api-eu-cz-1.runpod.io \
  --region eu-cz-1 \
  --cli-read-timeout 7200
```

**Note**: The S3 credentials are DIFFERENT from your RunPod API key!

#### Option B: SSH/rsync Upload (Requires GPU Pod)

This creates a temporary Pod, uploads your `./data` directory via rsync, then terminates the Pod.

```bash
./runpod/deploy.sh upload-data --volume-id YOUR_VOLUME_ID
```

**Warning**: This method requires a running GPU Pod during the upload, incurring ~$0.20-0.50/hour costs.

### 3. Start Training

```bash
./runpod/deploy.sh train-small --volume-id YOUR_VOLUME_ID
```

Or create a Pod manually and start training:

```bash
./runpod/deploy.sh create-pod --volume-id YOUR_VOLUME_ID
```

## Retry Logic and Error Handling

### HuggingFace Upload Retry Logic

All HuggingFace uploads (models and datasets) now include automatic retry logic:

- **3 retry attempts** with exponential backoff
- **Delays**: 5 seconds → 10 seconds → 20 seconds
- **Applies to**: Repository creation and file uploads
- **Detailed logging**: Shows each attempt and failure reason

**Example Output:**
```
[2/5] Pushing model to HuggingFace...
  Repository: GesturaAI/stick-gen-small
  Retry policy: 3 attempts with exponential backoff

  Attempt 1/3: Creating repository...
  ⚠️  Repository creation failed (attempt 1/3): Connection timeout
  ⏳ Retrying in 5.0 seconds...

  Attempt 2/3: Creating repository...
  ✓ Repository created successfully

  Attempt 1/3: Uploading files...
  ✓ Upload complete

  ✅ Model pushed successfully
```

### Graceful Error Handling

The pipeline handles failures intelligently:

| Scenario | Behavior |
|----------|----------|
| **Training fails** | ❌ Stop pipeline immediately - do not continue to next variant |
| **Model push succeeds** | ✅ Clean up intermediate checkpoints, continue to next variant |
| **Model push fails (after retries)** | ⚠️ Preserve checkpoint, show manual push instructions, **continue to next variant** |
| **Dataset push fails (after retries)** | ❌ Stop pipeline - dataset is critical for training |

### Checkpoint Preservation

When a model push fails after all retry attempts:

1. **Checkpoint is preserved** at `/runpod-volume/checkpoints/`
2. **Backup marker created**: `.${variant}_checkpoint_backup`
3. **Manual push instructions** displayed in logs
4. **Pipeline continues** to train next variant (small → medium → large)

**Manual Recovery:**
```bash
# If medium model push failed, you can manually push later:
python scripts/push_to_huggingface.py \
  --checkpoint /runpod-volume/checkpoints/model_checkpoint_best.pth \
  --variant medium \
  --version 1.0.0 \
  --token $HF_TOKEN \
  --repo-name GesturaAI/stick-gen-medium
```

### Pipeline Summary Report

At the end of training all variants, you'll see a comprehensive summary:

```
╔════════════════════════════════════════════════════════════╗
║              Training Pipeline Summary                     ║
╚════════════════════════════════════════════════════════════╝

  small model:
    Training: ✅ Success
    HF Push:  ✅ Success - https://huggingface.co/GesturaAI/stick-gen-small

  medium model:
    Training: ✅ Success
    HF Push:  ❌ Failed - Checkpoint preserved at /runpod-volume/checkpoints

  large model:
    Training: ✅ Success
    HF Push:  ✅ Success - https://huggingface.co/GesturaAI/stick-gen-large

⚠️  Some models failed to push to HuggingFace
   Checkpoints are preserved on the volume for manual upload
   See instructions above for manual push commands
```

This ensures you never lose trained models due to temporary API issues!

## Cost Optimization

### Key Principles

1. **S3 Upload Does NOT Require a GPU Pod**
   - Use the S3 API to upload data directly to Network Volumes
   - This saves the entire cost of running a GPU during data transfer
   - For 87GB of data at 10MB/s upload speed (~2.5 hours), you save ~$0.50-$1.25

2. **Terminate Pods When Not Training**
   - GPU Pods cost ~$0.20-0.50/hour depending on GPU type
   - Always terminate Pods after training completes
   - Use `screen` or `tmux` to keep training running after SSH disconnect

3. **Choose Cost-Effective GPUs**
   | GPU | Approx. Cost/Hour | Training Speed |
   |-----|-------------------|----------------|
   | RTX 3090 | ~$0.22 | Good |
   | RTX 3080 | ~$0.18 | Moderate |
   | RTX A4000 | ~$0.20 | Good |
   | RTX 4090 | ~$0.44 | Excellent |
   | L4 | ~$0.30 | Good |

4. **Network Volumes Are Persistent**
   - You only pay for storage ($0.10/GB/month)
   - Data survives Pod terminations
   - Don't re-upload data for each training run

### Terminate Pods to Save Costs

```bash
# Via API
curl -X POST "https://api.runpod.io/graphql" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { podTerminate(input: { podId: \"YOUR_POD_ID\" }) }"}'

# Check if Pod is terminated
curl -s -X POST "https://api.runpod.io/graphql" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query": "query { myself { pods { id name desiredStatus } } }"}' | jq .
```

## SSH Connection Methods

Once your Pod is running, you can connect via SSH using two methods:

### Method 1: RunPod Proxy (Always Works)

```bash
ssh <pod_id>-<host_id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Example:
```bash
ssh p4wapk7sjja2qw-64410b2f@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Method 2: Direct TCP (Preferred for File Transfers)

This method supports SCP and SFTP for efficient file transfers:

```bash
ssh root@<public_ip> -p <port> -i ~/.ssh/id_ed25519
```

Example:
```bash
ssh root@213.192.2.88 -p 40137 -i ~/.ssh/id_ed25519
```

**File Transfer Examples:**
```bash
# Upload file
scp -P 40137 -i ~/.ssh/id_ed25519 local_file.py root@213.192.2.88:/runpod-volume/

# Download file
scp -P 40137 -i ~/.ssh/id_ed25519 root@213.192.2.88:/runpod-volume/model.pth ./

# Sync directory
rsync -avz -e "ssh -p 40137 -i ~/.ssh/id_ed25519" ./data/ root@213.192.2.88:/runpod-volume/data/
```

## Docker Image

All training and inference Pods use our pre-built Docker image from GitHub Container Registry:

```
ghcr.io/gestura-ai/stick-gen:latest
```

### What's in the Docker Image

- **Base**: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`
- **Python dependencies**: All requirements pre-installed (no `pip install` needed)
- **Code**: Baked in at `/workspace` (no `git clone` needed)
- **Text embedder**: BAAI/bge-large-en-v1.5 pre-downloaded for fast startup
- **Configs**: `/workspace/configs/small.yaml`, `medium.yaml`, `large.yaml`

### Building the Docker Image

```bash
# Build locally
./runpod/deploy.sh build

# Push to GitHub Container Registry
./runpod/deploy.sh login
./runpod/deploy.sh push

# Or do all at once
./runpod/deploy.sh all
```

## Critical API Parameters

When creating Pods programmatically, these parameters are **required**:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `imageName` | Our pre-built Docker image | `"ghcr.io/gestura-ai/stick-gen:latest"` |
| `networkVolumeId` | ID of the Network Volume to attach | `"gol2v1emhp"` |
| `volumeMountPath` | **REQUIRED** - Mount path inside container | `"/runpod-volume"` |
| `ports` | **REQUIRED** for SSH - TCP ports to expose | `"22/tcp"` |
| `startSsh` | Enable SSH service inside container | `true` |
| `gpuTypeIdList` | List of acceptable GPU types | `["NVIDIA GeForce RTX 3090", ...]` |

### Example GraphQL Mutation

```graphql
mutation {
  podFindAndDeployOnDemand(input: {
    name: "stick-gen-training",
    imageName: "ghcr.io/gestura-ai/stick-gen:latest",
    gpuTypeIdList: ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"],
    gpuCount: 1,
    volumeInGb: 0,
    containerDiskInGb: 20,
    networkVolumeId: "YOUR_VOLUME_ID",
    volumeMountPath: "/runpod-volume",
    ports: "22/tcp",
    startSsh: true
  }) {
    id
    machine { podHostId gpuDisplayName }
    runtime { ports { ip publicPort privatePort } }
  }
}
```

## Troubleshooting

### Error: "invalid mount config for type bind: field Target must not be empty"

**Cause**: Missing `volumeMountPath` parameter when using `networkVolumeId`.

**Solution**: Always include `volumeMountPath: "/runpod-volume"` in your Pod creation request.

### Error: "No instances available"

**Cause**: No GPUs available in the specified datacenter, or trying to create a Pod in a different datacenter than the Network Volume.

**Solution**:
1. Network Volumes are **region-locked** - Pods MUST be in the same datacenter
2. Use `gpuTypeIdList` with multiple GPU types for better availability
3. Try different datacenters (EU-CZ-1 often has good availability)

### SSH Connection Refused

**Cause**: SSH port not exposed externally.

**Solution**: Include both `startSsh: true` AND `ports: "22/tcp"` in your Pod creation request.
- `startSsh: true` only enables the SSH service inside the container
- `ports: "22/tcp"` exposes port 22 externally via TCP

### Pod Created but No SSH Info

**Cause**: Pod is still starting up.

**Solution**: Wait 30-60 seconds for the Pod to fully initialize, then query the Pod status:
```bash
curl -s -X POST "https://api.runpod.io/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{"query": "query { pod(input: { podId: \"YOUR_POD_ID\" }) { runtime { ports { ip publicPort privatePort } } } }"}'
```

### S3 Upload SignatureDoesNotMatch Error

**Cause**: Using the wrong credentials for S3 API.

**Solution**: RunPod S3 API requires SEPARATE credentials from the API key:
```bash
# WRONG - These are API keys, not S3 keys
export AWS_ACCESS_KEY_ID="${RUNPOD_API_KEY}"  # ❌

# CORRECT - Use S3-specific credentials
export AWS_ACCESS_KEY_ID="user_xxxxx"         # ✅
export AWS_SECRET_ACCESS_KEY="rps_xxxxx"      # ✅
```

Generate S3 credentials at: https://www.runpod.io/console/user/settings → Storage → S3 Access Keys

### SSH Passphrase Prompt Blocking Automation

**Cause**: SSH key has a passphrase but ssh-agent doesn't have the key loaded.

**Solution**: Add your SSH key to ssh-agent:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Training Data Not Found

**Cause**: Data was uploaded to wrong path or volume.

**Solution**: Verify data location:
```bash
# Check via S3 API
AWS_ACCESS_KEY_ID='user_xxxxx' \
AWS_SECRET_ACCESS_KEY='rps_xxxxx' \
aws s3 ls s3://YOUR_VOLUME_ID/data/ \
  --endpoint-url https://s3api-eu-cz-1.runpod.io \
  --region eu-cz-1 --recursive --summarize
```

Expected structure:
```
/data/
├── 100Style/
├── amass/
└── smpl_models/
```

## Monitoring Training

### Via SSH

```bash
# Connect to Pod
ssh root@<ip> -p <port>

# Check GPU utilization
nvidia-smi -l 1

# Check training logs
tail -f /workspace/stick-gen/outputs/*.log

# Check running processes
ps aux | grep python

# Monitor memory usage
watch -n 5 free -h
```

### Using Screen/Tmux for Long Training

Always use screen or tmux for training to prevent interruption:

```bash
# Start a screen session
screen -S training

# Run training
python -m src.train.train --config configs/small.yaml --data_path /runpod-volume/data

# Detach: Ctrl+A, then D

# Reattach later
screen -r training

# List sessions
screen -ls
```

### Check Training Progress

```bash
# From inside the Pod (code is pre-installed at /workspace)
ls -la /workspace/outputs/
cat /workspace/outputs/training_log.txt

# Check checkpoints
ls -la /runpod-volume/checkpoints/
```

## Current Resources

| Resource | ID | Details |
|----------|-----|---------|
| Network Volume | `gol2v1emhp` | 100GB in EU-CZ-1 |
| Docker Image | `ghcr.io/gestura-ai/stick-gen:latest` | Pre-built with all dependencies |

## Complete Workflow Summary

```bash
# 1. Set credentials (one-time)
export RUNPOD_API_KEY='your_api_key'
export RUNPOD_S3_ACCESS_KEY='user_xxxxx'
export RUNPOD_S3_SECRET_KEY='rps_xxxxx'

# 2. Build and push Docker image (one-time, or when dependencies change)
./runpod/deploy.sh login   # Authenticate with ghcr.io
./runpod/deploy.sh build   # Build Docker image
./runpod/deploy.sh push    # Push to GitHub Container Registry

# 3. Create volume (one-time)
./runpod/deploy.sh create-volume --volume-size 100 --datacenter EU-CZ-1
# Note the VOLUME_ID returned

# 4. Upload data (one-time, ~2-3 hours for 87GB)
./runpod/deploy.sh upload-s3 --volume-id YOUR_VOLUME_ID

# 5. Start training (each training run)
./runpod/deploy.sh train-small --volume-id YOUR_VOLUME_ID

# 6. SSH into Pod and start training
# The Docker image has all code at /workspace - no git clone or pip install needed!
ssh root@<ip> -p <port>
screen -S training
cd /workspace && python -m src.train.train --config configs/small.yaml --data_path /runpod-volume/data

# 7. IMPORTANT: Terminate Pod when done to save costs!
curl -X POST "https://api.runpod.io/graphql" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { podTerminate(input: { podId: \"YOUR_POD_ID\" }) }"}'
```
