# RunPod Deployment Documentation

Complete guide for deploying Stick-Gen training infrastructure on [RunPod](https://runpod.io?ref=z71ozsfc).

## 📚 Documentation Index

### Quick Start Guides
- **[QUICK_START.md](QUICK_START.md)** - Get started in 5 minutes with the automated pipeline
- **[RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md)** - Comprehensive deployment guide with all options

### Recent Improvements
- **[RUNPOD_PIPELINE_IMPROVEMENTS.md](RUNPOD_PIPELINE_IMPROVEMENTS.md)** - Latest enhancements (GROK_API_KEY, retry logic, graceful error handling)
- **[DEPLOYMENT_FIX_SUMMARY.md](DEPLOYMENT_FIX_SUMMARY.md)** - Deployment script fixes and improvements

### LLM Integration
- **[ENABLE_GROK_API.md](ENABLE_GROK_API.md)** - How to enable Grok API for LLM-enhanced dataset generation
- **[GROK_API_FIX_SUMMARY.md](GROK_API_FIX_SUMMARY.md)** - Grok API integration details
- **[GROK_INVESTIGATION_REPORT.md](GROK_INVESTIGATION_REPORT.md)** - Technical investigation of Grok API integration

## 🚀 Quick Start

### Complete Pipeline (Recommended)

```bash
# Set environment variables
export RUNPOD_API_KEY='your_api_key'
export RUNPOD_S3_ACCESS_KEY='user_xxxxx'
export RUNPOD_S3_SECRET_KEY='rps_xxxxx'
export HF_TOKEN='hf_xxxxx'
export GROK_API_KEY='xai-xxxxx'  # Optional: for LLM-enhanced datasets

# Run complete pipeline
./runpod/deploy.sh full-deploy --datacenter EU-CZ-1 --volume-size 200
```

This single command:
1. ✅ Builds and pushes Docker image
2. ✅ Creates RunPod volume with S3 connectivity
3. ✅ Uploads training data
4. ✅ Generates and processes datasets (with optional LLM enhancement)
5. ✅ Trains all model variants (small → base → large)
6. ✅ Pushes models to HuggingFace with retry logic
7. ✅ Provides comprehensive summary

**Estimated time:** 6-12 hours | **Estimated cost:** $50-100

### Train with Existing Volume

```bash
export RUNPOD_API_KEY='your_api_key'
export HF_TOKEN='hf_xxxxx'

./runpod/deploy.sh auto-train-all --volume-id gol2v1emhp
```

## 🔑 Environment Variables

### Required
| Variable | Description | Where to Get |
|----------|-------------|--------------|
| `RUNPOD_API_KEY` | RunPod API key | [RunPod Settings](https://www.runpod.io/console/user/settings) |
| `RUNPOD_S3_ACCESS_KEY` | S3 access key for volume uploads | RunPod Settings → Storage → S3 Access Keys |
| `RUNPOD_S3_SECRET_KEY` | S3 secret key | Same as above |

### Optional
| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (enables model/dataset uploads) | None |
| `GROK_API_KEY` | X.AI Grok API key (enables LLM dataset enhancement) | None |
| `GITHUB_TOKEN` | GitHub PAT for Docker image authentication | None |
| `HF_REPO_NAME` | Base HuggingFace repo name | `GesturaAI/stick-gen` |
| `VERSION` | Model version tag | `1.0.0` |

## ✨ Key Features

### 1. GROK_API_KEY Integration
- **LLM-enhanced dataset generation** using X.AI's Grok API
- Automatically passed to all data preparation and training pods
- Optional - pipeline works without it using mock data
- See [ENABLE_GROK_API.md](ENABLE_GROK_API.md) for details

### 2. Retry Logic for HuggingFace Uploads
- **3 automatic retry attempts** with exponential backoff (5s → 10s → 20s)
- Handles temporary API failures gracefully
- Applies to both model and dataset uploads
- See [RUNPOD_PIPELINE_IMPROVEMENTS.md](RUNPOD_PIPELINE_IMPROVEMENTS.md) for details

### 3. Graceful Error Handling
- **Model push failures**: Preserve checkpoint, continue to next variant
- **Training failures**: Stop pipeline immediately
- **Dataset push failures**: Stop pipeline (dataset is critical)
- Comprehensive summary shows success/failure for each model

### 4. Pipeline Summary Reporting
```
╔════════════════════════════════════════════════════════════╗
║              Training Pipeline Summary                     ║
╚════════════════════════════════════════════════════════════╝

  small model:
    Training: ✅ Success
    HF Push:  ✅ Success - https://huggingface.co/GesturaAI/stick-gen-small

  base model:
    Training: ✅ Success
    HF Push:  ❌ Failed - Checkpoint preserved at /runpod-volume/checkpoints

  large model:
    Training: ✅ Success
    HF Push:  ✅ Success - https://huggingface.co/GesturaAI/stick-gen-large
```

## 📋 Available Commands

### Docker Image Management
```bash
./runpod/deploy.sh build          # Build Docker image locally
./runpod/deploy.sh push           # Push to GitHub Container Registry
./runpod/deploy.sh login          # Authenticate with ghcr.io
./runpod/deploy.sh all            # Build + push + deploy
```

### Volume Management
```bash
./runpod/deploy.sh create-volume --volume-size 200 --datacenter EU-CZ-1
./runpod/deploy.sh upload-s3 --volume-id <ID> --datacenter EU-CZ-1
```

### Training Workflows
```bash
./runpod/deploy.sh full-deploy --datacenter EU-CZ-1 --volume-size 200
./runpod/deploy.sh auto-train-all --volume-id <ID>
./runpod/deploy.sh prep-data --volume-id <ID>
./runpod/deploy.sh train-small --volume-id <ID>
./runpod/deploy.sh train-base --volume-id <ID>
./runpod/deploy.sh train-large --volume-id <ID>
```

### Serverless Endpoints
```bash
./runpod/deploy.sh create-endpoint --volume-id <ID> --workers-max 5
./runpod/deploy.sh list-endpoints
./runpod/deploy.sh submit-job --endpoint-id <ID> --input '{"prompt": "..."}'
```

## 📖 Next Steps

1. **First time?** Start with [QUICK_START.md](QUICK_START.md)
2. **Need details?** Read [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md)
3. **Want LLM enhancement?** See [ENABLE_GROK_API.md](ENABLE_GROK_API.md)
4. **Troubleshooting?** Check [RUNPOD_DEPLOYMENT.md#troubleshooting](RUNPOD_DEPLOYMENT.md#troubleshooting)

## 🔗 Resources

- [RunPod Console](https://www.runpod.io/console/pods)
- [RunPod API Documentation](https://docs.runpod.io/)
- [HuggingFace Hub](https://huggingface.co/)
- [X.AI Grok API](https://x.ai/)

