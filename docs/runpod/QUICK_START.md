# Quick Start: Complete Deployment Pipeline

## Prerequisites Setup (5 minutes)

### 1. Get RunPod API Key
1. Go to https://www.runpod.io/console/user/settings
2. Create an API key
3. Copy it

### 2. Get RunPod S3 Credentials
1. Go to https://www.runpod.io/console/user/settings
2. Click "Storage" in left menu
3. Under "S3 Access Keys", click "Generate Key"
4. Copy both the Access Key and Secret Key

### 3. Get HuggingFace Token (Optional but Recommended)
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "stick-gen-upload"
4. Type: "Write"
5. Copy the token

### 4. Set Environment Variables

```bash
# Required
export RUNPOD_API_KEY='your_runpod_api_key_here'
export RUNPOD_S3_ACCESS_KEY='user_xxxxx'
export RUNPOD_S3_SECRET_KEY='rps_xxxxx'

# Optional but recommended (for auto-push to HuggingFace)
export HF_TOKEN='hf_xxxxx'

# Optional (for Docker image push)
export GITHUB_TOKEN='ghp_xxxxx'
export GITHUB_USERNAME='your-github-username'
```

## Run Complete Pipeline (One Command!)

```bash
# This is your original command - now it works!
./runpod/deploy.sh --datacenter EU-CZ-1 --volume-size 200
```

That's it! The script will now:
1. ✅ Build and push Docker image
2. ✅ Create volume with S3 connectivity
3. ✅ Upload your ./data folder
4. ✅ Generate and process datasets
5. ✅ Train all 3 model variants
6. ✅ Push models to HuggingFace

## Monitor Progress

1. **RunPod Console**: https://www.runpod.io/console/pods
   - See active pods
   - View GPU utilization
   - Check costs

2. **SSH into Pod** (to view logs):
   ```bash
   # Get SSH command from RunPod Console
   ssh <pod_id>-<host_id>@ssh.runpod.io
   
   # View training logs
   tail -f /runpod-volume/checkpoints/training_small.log
   tail -f /runpod-volume/checkpoints/training_base.log
   tail -f /runpod-volume/checkpoints/training_large.log
   ```

## Expected Timeline

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Docker Build | 5-10 min | Building image with all dependencies |
| Volume Creation | 1 min | Creating 200GB network volume |
| Data Upload | 30-60 min | Uploading ~87GB via S3 |
| Data Prep | 30-60 min | Generating & processing datasets |
| Train Small | 1-2 hours | Training 5.6M param model |
| Train Base | 2-4 hours | Training 15.8M param model |
| Train Large | 3-6 hours | Training 28M param model |
| **Total** | **6-12 hours** | **Complete pipeline** |

## Expected Costs

| Item | Cost |
|------|------|
| Network Volume (200GB) | ~$20/month (delete after training) |
| Data Prep Pod (1 hour) | ~$0.25-0.50 |
| Training Pods (8-12 hours) | ~$40-80 |
| **Total** | **~$50-100** |

💡 **Tip**: Pods auto-terminate after training, but remember to delete the Network Volume when done!

## Troubleshooting

### "No GPUs available"
The script will automatically try multiple GPU types and cloud types. If all fail, it will provide manual instructions.

### "S3 upload failed"
Check that your S3 credentials are correct:
```bash
echo $RUNPOD_S3_ACCESS_KEY  # Should start with 'user_'
echo $RUNPOD_S3_SECRET_KEY  # Should start with 'rps_'
```

### "HuggingFace push failed"
Models are still saved on the volume. You can manually push later:
```bash
# SSH into pod and run:
cd /workspace
python scripts/push_to_huggingface.py \
  --checkpoint /runpod-volume/checkpoints/model_checkpoint_best.pth \
  --variant small \
  --token $HF_TOKEN
```

## Alternative: Train with Existing Volume

If you already have a volume with data:

```bash
# Just provide the volume ID
./runpod/deploy.sh --volume-id YOUR_VOLUME_ID

# This automatically runs: auto-train-all
# (data prep + train all 3 variants)
```

## Alternative: Individual Steps

```bash
# 1. Create volume
./runpod/deploy.sh create-volume --datacenter EU-CZ-1 --volume-size 200
# Note the VOLUME_ID returned

# 2. Upload data
./runpod/deploy.sh upload-s3 --volume-id YOUR_VOLUME_ID

# 3. Prepare data
./runpod/deploy.sh prep-data --volume-id YOUR_VOLUME_ID

# 4. Train individual models
./runpod/deploy.sh train-small --volume-id YOUR_VOLUME_ID
./runpod/deploy.sh train-base --volume-id YOUR_VOLUME_ID
./runpod/deploy.sh train-large --volume-id YOUR_VOLUME_ID
```

## What Gets Created on HuggingFace

If `HF_TOKEN` is set, these repositories are automatically created:

1. `GesturaAI/stick-gen-small` - 5.6M parameter model
2. `GesturaAI/stick-gen-base` - 15.8M parameter model
3. `GesturaAI/stick-gen-large` - 28M parameter model

To use your own account:
```bash
export HF_REPO_NAME='your-username/stick-gen'
# Creates: your-username/stick-gen-small, etc.
```

## Cleanup After Training

```bash
# Delete the Network Volume (via RunPod Console)
# This saves ~$20/month

# Or via API:
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { deleteNetworkVolume(input: { id: \"YOUR_VOLUME_ID\" }) }"}'
```

## Need Help?

- Check `DEPLOYMENT_FIX_SUMMARY.md` for detailed architecture
- Check `runpod/RUNPOD_DEPLOYMENT.md` for comprehensive documentation
- Run `./runpod/deploy.sh help` for all available commands

