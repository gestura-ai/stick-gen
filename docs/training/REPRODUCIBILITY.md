# Reproducibility Guide

This guide ensures that the Stick-Gen model training results can be exactly reproduced. We provide the specific commit hashes, environment configurations, and random seeds used for our release models.

## 1. Environment Setup

To match the training environment exactly:

```bash
# Clone the repository
git clone https://github.com/gestura-ai/stick-gen.git
cd stick-gen

# Checkout the release tag (e.g., v1.0.0)
git checkout v1.0.0

# Create a virtual environment with Python 3.9/3.10
python3 -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

### System Requirements matched
- **OS**: Linux (tested on Ubuntu 20.04/22.04) or macOS 13+
- **Python**: 3.9.x or 3.10.x
- **PyTorch**: 2.0.0+ (CUDA 11.8 for GPU training)

## 2. Configuration & Hyperparameters

We use `hydra` or `argparse` based configs. The canonical configuration for the **Medium (20.5M)** model is:

**File**: `configs/medium.yaml`

```yaml
model:
  d_model: 384
  nhead: 12
  num_layers: 8
  dim_feedforward: 1536
  dropout: 0.1

training:
  seed: 42
  batch_size: 16
  grad_accumulation_steps: 4
  learning_rate: 3e-4
  warmup_steps: 2000
  max_epochs: 50
  optimizer: adamw
  weight_decay: 0.01
```

## 3. Data Preparation

Standardized data processing is crucial for reproduction.

1. **Download AMASS**: Obtain the AMASS dataset from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) and place it in `data/amass`.
2. **Run Preparation Pipeline**:
   ```bash
   # This script converts AMASS to our .pt format deterministically
   python scripts/merge_amass_dataset.py --seed 42
   
   # Verify data hash (MD5)
   # Expected hash will be computed after first successful data preparation
   # Run: md5sum data/curated/train_data.pt > data/curated/train_data.pt.md5
   # Then verify: md5sum -c data/curated/train_data.pt.md5
   md5sum data/curated/train_data.pt
   ```

## 4. Training Command

Run the training with the fixed seed:

```bash
python -m src.train.train \
    --config configs/medium.yaml \
    --seed 42 \
    --checkpoint_dir checkpoints/repro_run_v1
```

## 5. Verification

After training, verify the model performance matches the reported metrics using our evaluation runner:

```bash
python scripts/run_comprehensive_eval.py \
    --checkpoint checkpoints/repro_run_v1/best_model.pth \
    --output verification_report.json
```

**Expected Metrics (Medium Model)**:
- **Pose MSE**: < 0.05
- **Action Accuracy**: > 85%
- **FID-like score**: < 15.0

## Troubleshooting Differences

If your results differ significantly:
1. **Check Floating Point Precision**: Ensure you are using FP32 or mixed precision (BF16) as specified in config.
2. **Deterministic Mode**: Set `torch.use_deterministic_algorithms(True)` if exact bit-wise matching is required (note: may slow down training).
3. **Hardware Differences**: Small variations between CUDA versions and GPU architectures are normal (usually < 1-2% deviation).
