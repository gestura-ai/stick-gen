# Deployment Pipeline Fix Summary

## Problem Fixed

**Original Issue**: Running `./runpod/deploy.sh --datacenter EU-CZ-1 --volume-size 200` resulted in error:
```
unknown action: --datacenter
```

**Root Cause**: The script expected the action to be the **first** argument, but options were provided before the action.

**Solution**: Modified the argument parser to:
1. Accept options in any order (before or after the action)
2. Provide intelligent defaults when no action is specified:
   - If `--datacenter` and `--volume-size` are provided → defaults to `full-deploy`
   - If `--volume-id` is provided → defaults to `auto-train-all`
   - Otherwise → shows help

## How to Use the Fixed Script

### Option 1: Complete Pipeline from Scratch (RECOMMENDED)

This is what you wanted - the complete end-to-end workflow:

```bash
# Set required environment variables
export RUNPOD_API_KEY='your_runpod_api_key'
export RUNPOD_S3_ACCESS_KEY='user_xxxxx'  # Get from RunPod Console → Settings → Storage → S3
export RUNPOD_S3_SECRET_KEY='rps_xxxxx'
export HF_TOKEN='hf_xxxxx'  # Optional but recommended for auto-push to HuggingFace

# Run the complete pipeline (now works with your original command!)
./runpod/deploy.sh --datacenter EU-CZ-1 --volume-size 200

# Or explicitly specify the action:
./runpod/deploy.sh full-deploy --datacenter EU-CZ-1 --volume-size 200
```

**What this does** (9 automated steps):
1. ✅ Builds Docker image from `docker/Dockerfile`
2. ✅ Pushes image to GitHub Container Registry (ghcr.io)
3. ✅ Creates a Network Volume (200GB in EU-CZ-1 datacenter) with S3 connectivity
4. ✅ Uploads `./data` folder contents to the volume via S3 API (~87GB)
5. ✅ Creates a data preparation Pod that:
   - Generates synthetic motion data
   - Converts datasets in the data folder
   - Merges everything into `train_data_final.pt`
6. ✅ Waits for data prep to complete, then terminates the Pod
7. ✅ Trains small model (5.6M params) → pushes to HuggingFace
8. ✅ Trains base model (15.8M params) → pushes to HuggingFace
9. ✅ Trains large model (28M params) → pushes to HuggingFace

**Estimated time**: 6-12 hours
**Estimated cost**: $50-100 (GPU hours)

### Option 2: Train with Existing Volume

If you already have a volume with data uploaded:

```bash
# This now works automatically (detects --volume-id and defaults to auto-train-all)
./runpod/deploy.sh --volume-id YOUR_VOLUME_ID

# Or explicitly:
./runpod/deploy.sh auto-train-all --volume-id YOUR_VOLUME_ID
```

### Option 3: Individual Steps

```bash
# Create volume only
./runpod/deploy.sh create-volume --datacenter EU-CZ-1 --volume-size 200

# Upload data to existing volume
./runpod/deploy.sh upload-s3 --volume-id YOUR_VOLUME_ID

# Data preparation only
./runpod/deploy.sh prep-data --volume-id YOUR_VOLUME_ID

# Train individual model variant
./runpod/deploy.sh train-small --volume-id YOUR_VOLUME_ID
./runpod/deploy.sh train-base --volume-id YOUR_VOLUME_ID
./runpod/deploy.sh train-large --volume-id YOUR_VOLUME_ID
```

## HuggingFace Resources

### What Gets Created Automatically

The pipeline will **automatically create** these HuggingFace repositories if they don't exist:

1. **Model Repositories** (3 repos):
   - `GesturaAI/stick-gen-small` (5.6M params)
   - `GesturaAI/stick-gen-base` (15.8M params)
   - `GesturaAI/stick-gen-large` (28M params)

2. **Dataset Repository** (optional, if `HF_PUSH_DATASET=true`):
   - `GesturaAI/stick-gen-dataset`

### What You Need to Do

**Nothing!** The repositories are created automatically via the `push_to_huggingface.py` script using the `create_repo()` function with `exist_ok=True`.

However, you **must** set the `HF_TOKEN` environment variable:

```bash
# Get token from: https://huggingface.co/settings/tokens
# Create a token with "Write" permissions
export HF_TOKEN='hf_xxxxx'
```

### Custom HuggingFace Account/Organization

To push to your own account instead of `GesturaAI`:

```bash
export HF_REPO_NAME='your-username/stick-gen'
# This will create: your-username/stick-gen-small, your-username/stick-gen-base, your-username/stick-gen-large
```

## Environment Variables Reference

### Required for Full Pipeline

| Variable | Where to Get It | Purpose |
|----------|----------------|---------|
| `RUNPOD_API_KEY` | [RunPod Console → Settings](https://www.runpod.io/console/user/settings) | Create/manage volumes and pods |
| `RUNPOD_S3_ACCESS_KEY` | RunPod Console → Settings → Storage → S3 Access Keys | Upload data via S3 |
| `RUNPOD_S3_SECRET_KEY` | RunPod Console → Settings → Storage → S3 Access Keys | Upload data via S3 |

### Optional but Recommended

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | (none) | Auto-push models to HuggingFace after training |
| `HF_REPO_NAME` | `GesturaAI/stick-gen` | Base repository name (variant suffix added automatically) |
| `GITHUB_TOKEN` | (none) | Push Docker image to ghcr.io |

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Docker Image                                        │
│  • Build from docker/Dockerfile                              │
│  • Push to ghcr.io/gestura-ai/stick-gen:latest              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Infrastructure                                      │
│  • Create Network Volume (with S3 connectivity)              │
│  • Upload ./data via S3 API                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Data Preparation Pod                                │
│  • Generate synthetic motion data (50k samples)              │
│  • Convert AMASS, 100Style datasets                          │
│  • Merge into train_data_final.pt                            │
│  • Auto-terminates when complete                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Training Pods (Sequential)                          │
│  • Train small → push to HF → cleanup                        │
│  • Train base → push to HF → cleanup                         │
│  • Train large → push to HF → cleanup                        │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Set environment variables** (see above)
2. **Run the deployment**:
   ```bash
   ./runpod/deploy.sh --datacenter EU-CZ-1 --volume-size 200
   ```
3. **Monitor progress** via RunPod Console: https://www.runpod.io/console/pods
4. **Remember to terminate pods** when training completes to save costs!

## Cost Optimization Tips

- Training pods auto-terminate after pushing to HuggingFace
- Data prep pod auto-terminates after completion
- Network Volume persists (costs ~$0.10/GB/month) - delete when done
- Estimated GPU costs: $0.20-0.50/hour depending on GPU type

