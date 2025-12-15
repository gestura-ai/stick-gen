# RunPod Training Commands for Stick-Gen

**Last Updated**: December 2024

Complete end-to-end guide for training Stick-Gen models on RunPod, starting from zero infrastructure.

---

## TL;DR - Single Command Pipelines

### Small + Medium Models (~$50, ~98 GPU-hours)
```bash
# Set credentials once
export RUNPOD_API_KEY="rpa_xxx" HF_TOKEN="hf_xxx"
export RUNPOD_S3_ACCESS_KEY="user_xxx" RUNPOD_S3_SECRET_KEY="rps_xxx"

# Run complete pipeline
./runpod/deploy.sh --datacenter EU-CZ-1 --models small,medium
```

### Large Model Only (~$165, ~145 GPU-hours)
```bash
./runpod/deploy.sh --datacenter EU-CZ-1 --models large --gpu "NVIDIA A100 PCIe"
```

### All Models (~$215, ~243 GPU-hours)
```bash
./runpod/deploy.sh --datacenter EU-CZ-1 --models all
```

These commands handle everything automatically:
1. ✅ Create 200GB network volume
2. ✅ Upload training data via S3
3. ✅ Run data preparation (curated pipeline)
4. ✅ Train pretrain → SFT → LoRA for each model
5. ✅ Push models to HuggingFace
6. ✅ Terminate pods when done

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pipeline 1: Small + Medium Models](#pipeline-1-small--medium-models-recommended)
3. [Pipeline 2: Large Model Only](#pipeline-2-large-model-only)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Quick Reference](#quick-reference)

---

## Prerequisites

### Step 0.1: Create Accounts and Get API Keys

Before starting, you need:

| Service | URL | What You Need |
|---------|-----|---------------|
| RunPod | https://www.runpod.io/console/user/settings | API Key |
| RunPod S3 | Same page → Storage → S3 Access Keys | Access Key + Secret Key |
| HuggingFace | https://huggingface.co/settings/tokens | Write token (for model upload) |
| GitHub | https://github.com/settings/tokens | PAT with `read:packages` (for Docker image) |

### Step 0.2: Set Environment Variables

Create a file `~/.stick-gen-env` (DO NOT commit this file):

```bash
# RunPod API key (REQUIRED)
export RUNPOD_API_KEY="rpa_XXXXXXXXXXXXXXXXXXXX"

# RunPod S3 credentials (REQUIRED for data upload)
export RUNPOD_S3_ACCESS_KEY="user_XXXXX"
export RUNPOD_S3_SECRET_KEY="rps_XXXXXXXXXXXXX"

# HuggingFace token (REQUIRED for auto-push models)
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXX"

# Optional: GitHub token for Docker image push
export GITHUB_TOKEN="ghp_XXXXXXXXXXXXXXXXXXXX"

# Optional: Grok API for LLM-enhanced data generation
export GROK_API_KEY="xai-XXXXXXXXXXXXXXXXXXXX"
```

Then source it:
```bash
source ~/.stick-gen-env
```

### Step 0.3: Install Local Dependencies

```bash
# macOS
brew install runpod/runpodctl/runpodctl
brew install awscli
brew install jq

# Linux
pip install runpodctl awscli
sudo apt-get install jq

# Verify installations
runpodctl version
aws --version
jq --version
```

### Step 0.4: Verify Local Data Directory

Ensure you have the training data locally:

```bash
# Check data directory exists and has content
ls -la data/

# Expected structure (minimum):
# data/
# ├── 100Style/          (~2 GB)
# ├── amass/             (~45 GB)
# ├── smpl_models/       (~1 GB)
# └── (other datasets as needed)

# Check total size (should be ~90+ GB for full pipeline)
du -sh data/
```

**If you don't have the data**: See [Data Acquisition Guide](./DATA_ACQUISITION.md) for download instructions.

---

## Pipeline 1: Small + Medium Models (Recommended)

**Best for**: First-time setup, testing, cost-effective training
**GPU**: RTX A5000 (24GB VRAM)
**Total Time**: ~98 GPU-hours
**Total Cost**: ~$50 (including 200GB storage for 1 month)

### Option A: Single Command (Recommended)

```bash
# This handles everything: volume creation, data upload, training, HuggingFace push
./runpod/deploy.sh --datacenter EU-CZ-1 --models small,medium
```

The script will:
1. Build and push Docker image to ghcr.io
2. Create a 200GB network volume in EU-CZ-1
3. Upload your local `./data` directory via S3
4. Run data preparation (curated pipeline)
5. Train small model (pretrain → SFT → LoRA)
6. Train medium model (pretrain → SFT → LoRA)
7. Push all models to HuggingFace
8. Terminate pods and display summary

**Monitor progress**: The script shows live status updates. Full pipeline takes ~12-24 hours.

---

### Option B: Step-by-Step (For Troubleshooting)

If the single command fails at any stage, use these individual steps:

#### Stage 1.1: Create Network Volume

```bash
DATACENTER="EU-CZ-1"

./runpod/deploy.sh create-volume \
  --volume-size 200 \
  --datacenter $DATACENTER
```

**Save the Volume ID:**
```bash
export VOLUME_ID="vol_abc123xyz"  # Use your actual ID
```

#### ❌ If Volume Creation Fails:

| Error | Cause | Solution |
|-------|-------|----------|
| "Insufficient quota" | Account limit reached | Contact RunPod support or delete unused volumes |
| "Datacenter not available" | Region offline | Try different datacenter: `--datacenter EU-NL-1` |
| "Authentication failed" | Invalid API key | Regenerate key at RunPod settings |

#### Stage 1.2: Upload Training Data

```bash
./runpod/deploy.sh upload-s3 \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --data-dir ./data
```

#### ❌ If Data Upload Fails:

| Error | Cause | Solution |
|-------|-------|----------|
| "S3 credentials not set" | Missing env vars | Set `RUNPOD_S3_ACCESS_KEY` and `RUNPOD_S3_SECRET_KEY` |
| "Access Denied" | Wrong credentials | Regenerate S3 keys at RunPod settings → Storage |
| "Connection timeout" | Network issues | Retry - script uses `sync` which resumes automatically |
| "Bucket not found" | Volume in different region | Ensure datacenter matches volume location |
| "InvalidPart" / "Part size mismatch" | Multipart upload issue | See fix below ↓ |

**Fix for "InvalidPart" / "Part size mismatch" errors:**
```bash
# The script auto-configures this, but if issues persist, reduce chunk size further:
aws configure set default.s3.multipart_chunksize 8MB
aws configure set default.s3.multipart_threshold 32MB

# Then re-run upload (sync will skip already-uploaded files)
./runpod/deploy.sh upload-s3 --volume-id $VOLUME_ID --datacenter $DATACENTER
```

**Verify data uploaded:**
```bash
aws s3 ls s3://$VOLUME_ID/data/ \
  --endpoint-url https://s3api-$(echo $DATACENTER | tr '[:upper:]' '[:lower:]').runpod.io \
  --recursive --summarize

# Expected: Total Objects: 10000+, Total Size: 90+ GB
```

### Stage 1.3: Run Data Preparation

```bash
# Create data prep pod and run curated pipeline
./runpod/deploy.sh prep-data \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --curated
```

**Expected Duration:** ~4-6 hours
**Expected Cost:** ~$1.50

#### ❌ If Data Prep Fails:

| Error | Cause | Solution |
|-------|-------|----------|
| "No GPUs available" | Datacenter busy | Wait 10-30 min and retry, or try different datacenter |
| "Data not found at /runpod-volume/data" | Upload incomplete | Re-run upload step |
| "SMPL models not found" | Missing smpl_models/ | Download SMPL models and re-upload |
| Pod terminates unexpectedly | Spot instance preempted | Re-run; data is preserved on volume |

**Verify data prep completed:**
```bash
# SSH into any pod with the volume attached, then:
ls -la /runpod-volume/data/curated/
# Should contain:
#   pretrain_data_embedded.pt  (~8 GB)
#   sft_data_embedded.pt       (~2 GB)
#   curation_stats.json

# Or check completion marker:
cat /runpod-volume/data/.prep_complete
```

### Stage 1.4: Train Small Model (Pretrain → SFT → LoRA)

```bash
# Train small model (~25 GPU-hours total)
./runpod/deploy.sh train-small \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER
```

**Expected Duration:** ~12 hours (pretrain) + ~8 hours (SFT) + ~5 hours (LoRA)
**Expected Cost:** ~$6-8

**Monitor training progress:**
```bash
# Get pod SSH info
./runpod/deploy.sh list-pods

# SSH into pod
ssh root@<IP> -p <PORT>

# Watch training logs
tail -f /runpod-volume/checkpoints/training_small.log

# Check GPU utilization
nvidia-smi -l 5
```

#### ❌ If Training Fails:

| Error | Cause | Solution |
|-------|-------|----------|
| "CUDA out of memory" | Batch size too large | Reduce batch_size in configs/small.yaml |
| "Data file not found" | Wrong TRAIN_DATA_PATH | Set `TRAIN_DATA_PATH=/runpod-volume/data/curated/pretrain_data_embedded.pt` |
| Training stalls at epoch X | GPU throttling/crash | Check `nvidia-smi`, restart pod if needed |
| "Checkpoint not found" for SFT | Pretrain didn't complete | Resume pretrain with `--resume` flag |

**Resume from checkpoint (if interrupted):**
```bash
# SSH into pod, then:
RESUME_FROM_CHECKPOINT=/runpod-volume/checkpoints/checkpoint_epoch_XX.pth \
MODEL_VARIANT=small \
./runpod/train_entrypoint.sh
```

**Verify training completed:**
```bash
ls -la /runpod-volume/checkpoints/
# Should contain:
#   model_checkpoint_best.pth   (best pretrain checkpoint)
#   sft_model_best.pth          (best SFT checkpoint)
#   lora_adapters.pth           (LoRA weights)
```

### Stage 1.5: Train Medium Model (Pretrain → SFT → LoRA)

```bash
# Train medium model (~73 GPU-hours total)
./runpod/deploy.sh train-base \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --gpu "NVIDIA RTX A5000"
```

**Expected Duration:** ~50 hours (pretrain) + ~15 hours (SFT) + ~8 hours (LoRA)
**Expected Cost:** ~$26-30

Same monitoring and troubleshooting as Stage 1.4.

### Stage 1.6: Verify Models and Cleanup

```bash
# Verify models were pushed to HuggingFace
# Check: https://huggingface.co/GesturaAI/stick-gen-small
# Check: https://huggingface.co/GesturaAI/stick-gen-medium

# If auto-push failed, manually retrieve checkpoints:
scp -P <PORT> root@<IP>:/runpod-volume/checkpoints/model_checkpoint_best.pth ./small_model.pth
scp -P <PORT> root@<IP>:/runpod-volume/checkpoints/sft_model_best.pth ./small_sft_model.pth

# IMPORTANT: Terminate all pods to stop billing!
./runpod/deploy.sh list-pods
# For each pod:
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { podTerminate(input: { podId: \"POD_ID\" }) }"}'
```

---

## Pipeline 2: Large Model Only

**Best for**: Maximum quality, production deployment
**GPU**: A100 PCIe 40GB or 80GB (REQUIRED - large model needs 16GB+ VRAM)
**Total Time**: ~145 GPU-hours
**Total Cost**: ~$165 (including 200GB storage for 1 month)

### Option A: Single Command (Recommended)

```bash
# Complete pipeline with A100 GPU for large model
./runpod/deploy.sh --datacenter EU-CZ-1 --models large --gpu "NVIDIA A100 PCIe"
```

The script will:
1. Build and push Docker image
2. Create 200GB network volume
3. Upload training data via S3
4. Run data preparation
5. Train large model (pretrain → SFT → LoRA)
6. Push model to HuggingFace
7. Terminate pods

**⚠️ IMPORTANT**: Large model training requires A100 or better (16GB+ VRAM).

---

### Option B: Step-by-Step (For Troubleshooting)

#### Stage 2.1: Create Volume + Upload Data

```bash
DATACENTER="EU-CZ-1"

./runpod/deploy.sh create-volume --volume-size 200 --datacenter $DATACENTER
export VOLUME_ID="vol_abc123xyz"

./runpod/deploy.sh upload-s3 --volume-id $VOLUME_ID --datacenter $DATACENTER
```

#### Stage 2.2: Data Preparation

```bash
./runpod/deploy.sh prep-data \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --curated
```

#### Stage 2.3: Train Large Model

```bash
./runpod/deploy.sh train-large \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --gpu "NVIDIA A100 PCIe"
```

**Expected Duration:** ~100 hours (pretrain) + ~30 hours (SFT) + ~15 hours (LoRA)

#### ❌ If A100 Not Available:

```bash
# Try different datacenter
./runpod/deploy.sh --datacenter EU-NL-1 --models large --gpu "NVIDIA A100 PCIe"

# Or use A100 80GB SXM (often more available)
./runpod/deploy.sh --datacenter EU-CZ-1 --models large --gpu "NVIDIA A100-SXM4-80GB"

# Or use H100 (fastest, ~$2.50/hr)
./runpod/deploy.sh --datacenter EU-CZ-1 --models large --gpu "NVIDIA H100 PCIe"
```

### Stage 2.5: Manual SFT and LoRA (if automated fails)

If the automated pipeline doesn't run SFT/LoRA, do it manually:

```bash
# SSH into the training pod
ssh root@<IP> -p <PORT>

# Run SFT (~30 GPU-hours)
MODEL_VARIANT=sft_large \
INIT_FROM_CHECKPOINT=/runpod-volume/checkpoints/model_checkpoint_best.pth \
CHECKPOINT_DIR=/runpod-volume/checkpoints/sft_large \
./runpod/sft_entrypoint.sh

# Run LoRA (~15 GPU-hours)
MODEL_VARIANT=sft_large \
USE_LORA=true \
LORA_RANK=16 \
INIT_FROM_CHECKPOINT=/runpod-volume/checkpoints/model_checkpoint_best.pth \
CHECKPOINT_DIR=/runpod-volume/checkpoints/lora_large \
./runpod/sft_entrypoint.sh
```

### Stage 2.6: Verify and Cleanup

Same as Pipeline 1, Stage 1.6.

---

## Troubleshooting Guide

### Issue: S3 Upload "InvalidPart" / "Part size mismatch"

**Symptom:** `An error occurred (InvalidPart) when calling the UploadPart operation: part 188 Part size mismatch`

**Cause:** RunPod's S3 API has stricter multipart upload requirements than AWS S3. Default AWS CLI chunk sizes can cause issues.

**Solution:**
```bash
# The script now auto-configures this, but if you still see errors:

# 1. Reduce chunk size further
aws configure set default.s3.multipart_chunksize 8MB
aws configure set default.s3.multipart_threshold 32MB
aws configure set default.s3.max_concurrent_requests 2

# 2. Re-run upload (sync skips already-uploaded files)
./runpod/deploy.sh upload-s3 --volume-id $VOLUME_ID --datacenter $DATACENTER

# 3. If still failing, try even smaller chunks
aws configure set default.s3.multipart_chunksize 5MB
```

**Alternative - Upload via SSH/SCP (more reliable for unstable networks):**
```bash
# Create a minimal pod to upload data
./runpod/deploy.sh upload-data --volume-id $VOLUME_ID --datacenter $DATACENTER
```

### Issue: "Volume and Pod in Different Datacenters"

**Symptom:** Pod fails to start or can't see data on volume

**Solution:**
```bash
# Check where your volume is located
./runpod/deploy.sh list-volumes
# Note the dataCenterId

# Create pod in the SAME datacenter
./runpod/deploy.sh create-pod \
  --volume-id $VOLUME_ID \
  --datacenter <SAME_DATACENTER_AS_VOLUME>
```

### Issue: "Checkpoint Corruption / Can't Load Model"

**Symptom:** `RuntimeError: Error(s) in loading state_dict`

**Solution:**
```bash
# SSH into pod
ssh root@<IP> -p <PORT>

# Check checkpoint file integrity
python -c "import torch; torch.load('/runpod-volume/checkpoints/model_checkpoint_best.pth')"

# If corrupted, find last good checkpoint
ls -la /runpod-volume/checkpoints/checkpoint_epoch_*.pth

# Resume from last good checkpoint
RESUME_FROM_CHECKPOINT=/runpod-volume/checkpoints/checkpoint_epoch_45.pth \
MODEL_VARIANT=medium \
./runpod/train_entrypoint.sh
```

### Issue: "HuggingFace Push Failed"

**Symptom:** Training completes but model not on HuggingFace

**Solution:**
```bash
# SSH into pod
ssh root@<IP> -p <PORT>

# Manually push to HuggingFace
cd /workspace
python scripts/push_to_huggingface.py \
  --checkpoint /runpod-volume/checkpoints/model_checkpoint_best.pth \
  --variant medium \
  --token $HF_TOKEN \
  --repo-name GesturaAI/stick-gen-medium

# Or copy checkpoints locally first
exit  # Exit SSH
scp -P <PORT> root@<IP>:/runpod-volume/checkpoints/*.pth ./checkpoints/
```

### Issue: "OOM (Out of Memory) During Training"

**Symptom:** `CUDA out of memory` error

**Solution:**
```bash
# Option 1: Reduce batch size
# Edit configs/medium.yaml or configs/large.yaml:
#   training:
#     batch_size: 8  # Reduce from 16 to 8

# Option 2: Increase gradient accumulation
# Edit config:
#   training:
#     grad_accum_steps: 8  # Increase from 4 to 8

# Option 3: Use a GPU with more VRAM
./runpod/deploy.sh create-pod \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --gpu "NVIDIA A100-SXM4-80GB"
```

### Issue: "Training Stuck / No Progress"

**Symptom:** Training log shows no new epochs for hours

**Solution:**
```bash
# SSH into pod
ssh root@<IP> -p <PORT>

# Check if Python process is running
ps aux | grep python

# Check GPU utilization
nvidia-smi

# If GPU shows 0% utilization, restart training
pkill -f "python -m src.train"

# Resume from last checkpoint
RESUME_FROM_CHECKPOINT=/runpod-volume/checkpoints/checkpoint_epoch_XX.pth \
MODEL_VARIANT=medium \
./runpod/train_entrypoint.sh
```

### Issue: "Data Not Found on Volume"

**Symptom:** `FileNotFoundError: /runpod-volume/data/...`

**Solution:**
```bash
# SSH into pod and check what's on the volume
ssh root@<IP> -p <PORT>
ls -la /runpod-volume/
ls -la /runpod-volume/data/

# If empty, data upload failed - re-upload
exit
./runpod/deploy.sh upload-s3 \
  --volume-id $VOLUME_ID \
  --datacenter $DATACENTER \
  --data-dir ./data
```

---

## Quick Reference

### GPU Selection by Model

| Variant | Min VRAM | Recommended GPU | RunPod Cost/hr |
|---------|----------|-----------------|----------------|
| Small | 4 GB | RTX A4000 | ~$0.25 |
| Medium | 8 GB | RTX A5000 | ~$0.35 |
| Large | 16 GB | A100 PCIe 40GB | ~$1.00 |

### Cost Summary

| Pipeline | GPU Hours | GPU Cost | Storage (1mo) | Total |
|----------|-----------|----------|---------------|-------|
| Small+Medium (all stages) | ~98 hrs | ~$30 | $20 | **~$50** |
| Large only (all stages) | ~145 hrs | ~$145 | $20 | **~$165** |
| Full pipeline (all 9 models) | ~243 hrs | ~$175 | $40 | **~$215** |

### Datacenter Options

| Region | ID | Notes |
|--------|-----|-------|
| Texas, US | US-TX-3 | Good A100 availability |
| California, US | US-CA-1 | Good RTX availability |
| Netherlands, EU | EU-NL-1 | Good overall availability |
| Czech Republic, EU | EU-CZ-1 | Budget option |
| Sweden, EU | EU-SE-1 | Good H100 availability |

### Key Paths on RunPod Volume

| Path | Contents |
|------|----------|
| `/runpod-volume/data/` | Raw uploaded training data |
| `/runpod-volume/data/curated/` | Processed datasets (after data prep) |
| `/runpod-volume/checkpoints/` | Model checkpoints during training |
| `/runpod-volume/logs/` | Training logs |

### Essential Commands

```bash
# ============================================
# SINGLE-COMMAND PIPELINES (RECOMMENDED)
# ============================================

# Small + Medium models (~$50)
./runpod/deploy.sh --datacenter EU-CZ-1 --models small,medium

# Large model only (~$165)
./runpod/deploy.sh --datacenter EU-CZ-1 --models large --gpu "NVIDIA A100 PCIe"

# All models (~$215)
./runpod/deploy.sh --datacenter EU-CZ-1 --models all

# ============================================
# STEP-BY-STEP COMMANDS (for troubleshooting)
# ============================================

# List resources
./runpod/deploy.sh list-volumes
./runpod/deploy.sh list-pods

# Create volume
./runpod/deploy.sh create-volume --volume-size 200 --datacenter EU-CZ-1

# Upload data
./runpod/deploy.sh upload-s3 --volume-id $VOLUME_ID --datacenter EU-CZ-1

# Train with existing volume
./runpod/deploy.sh auto-train-all --volume-id $VOLUME_ID --models small,medium

# Terminate pod (STOP BILLING!)
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { podTerminate(input: { podId: \"POD_ID\" }) }"}'
```

