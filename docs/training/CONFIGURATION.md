# Training Configuration Guide

This guide explains how to configure stick-gen training for different hardware setups.

## Configuration Files

Stick-gen uses YAML configuration files to manage all training hyperparameters. Three pre-configured files are provided in the `configs/` directory:

### 1. `configs/medium.yaml` (Default)
**Best for**: Standard CPU training with 16GB+ RAM
- **Model**: 20.5M parameters (d_model=384, 8 layers, 12 heads)
- **Architecture**: RMSNorm + SwiGLU + Pre-Norm (modern LLM standards)
- **Batch size**: 2 with 32-step gradient accumulation (effective batch: 64)
- **Training time**: ~40 hours per epoch on 8-core CPU
- **Memory usage**: ~8-12GB RAM

### 2. `configs/small.yaml` (Budget CPU)
**Best for**: Affordable CPU training with 8-16GB RAM
- **Model**: 7.2M parameters (d_model=256, 6 layers, 8 heads)
- **Architecture**: RMSNorm + SwiGLU + Pre-Norm (modern LLM standards)
- **Batch size**: 1 with 64-step gradient accumulation (effective batch: 64)
- **Training time**: ~20 hours per epoch on 8-core CPU
- **Memory usage**: ~4-6GB RAM

### 3. `configs/large.yaml` (GPU)
**Best for**: GPU training with 8GB+ VRAM
- **Model**: 44.5M parameters (d_model=512, 10 layers, 16 heads)
- **Architecture**: RMSNorm + SwiGLU + Pre-Norm (modern LLM standards)
- **Batch size**: 16 with 4-step gradient accumulation (effective batch: 64)
- **Training time**: ~1-2 hours per epoch on RTX 3080
- **Memory usage**: ~6-8GB VRAM

## Using Configuration Files

### Command Line
```bash
# Use default config (base variant)
python3.9 -m src.train.train

# Use small variant config
python3.9 -m src.train.train --config configs/small.yaml

# Use large variant config
python3.9 -m src.train.train --config configs/large.yaml
```

### In Python
```python
from src.train.config import TrainingConfig

# Load configuration
config = TrainingConfig("configs/base.yaml")

# Access parameters
d_model = config.get("model.d_model")
batch_size = config.get("training.batch_size")

# Or use properties
model_config = config.model
training_config = config.training
```

## Configuration Structure

### Model Architecture
```yaml
model:
  input_dim: 20           # 10 joints × 2 coords
  d_model: 384            # Transformer hidden dimension
  nhead: 12               # Number of attention heads
  num_layers: 8           # Number of transformer layers
  output_dim: 20          # Same as input_dim
  embedding_dim: 1024     # Text embedding dimension
  dropout: 0.1            # Dropout rate
  num_actions: 64         # Number of action classes
```

**Key parameters**:
- `d_model`: Controls model size (256=7.2M, 384=20.5M, 512=44.5M params)
- `nhead`: Must divide d_model evenly (d_model % nhead == 0)
- `num_layers`: More layers = better quality but slower training

### Training Settings
```yaml
training:
  batch_size: 2           # Batch size (reduce if OOM)
  grad_accum_steps: 32    # Gradient accumulation steps
  epochs: 50              # Number of training epochs
  learning_rate: 0.0003   # Initial learning rate
  warmup_epochs: 10       # Number of warmup epochs
  max_grad_norm: 1.0      # Gradient clipping threshold
```

**Key parameters**:
- `batch_size`: Reduce if you get Out-of-Memory errors
- `grad_accum_steps`: Effective batch = batch_size × grad_accum_steps
- `learning_rate`: Lower for larger models, higher for larger batches

### Loss Weights
```yaml
loss_weights:
  temporal: 0.1           # Temporal consistency loss weight
  action: 0.15            # Action prediction loss weight
  physics: 0.2            # Physics loss weight
  diffusion: 0.1          # Diffusion refinement loss weight
```

**Tuning tips**:
- Increase `temporal` for smoother motion
- Increase `action` for better action recognition
- Increase `physics` for more realistic dynamics

### Dataset Paths and Curated Pipeline
```yaml
data:
  train_data: "data/train_data_final.pt"                    # Legacy combined dataset (with embeddings)
  curated_pretrain_data: "data/curated/pretrain_data_embedded.pt"  # Curated pretraining dataset (with embeddings)
  curated_sft_data: "data/curated/sft_data_embedded.pt"            # Curated SFT dataset (with embeddings)
```

**Notes**:
- `train_data` remains the default when no `--data_path` override is provided.
- For curated continued pretraining, point `--data_path` (locally) or `TRAIN_DATA_PATH` (RunPod) at the curated pretraining file, e.g. `data/curated/pretrain_data_embedded.pt`.

### Device Settings
```yaml
device:
  type: "auto"            # "auto", "cpu", "cuda", or "mps"
  num_workers: 0          # DataLoader workers
  pin_memory: false       # Pin memory for faster GPU transfer
```

**Device types**:
- `auto`: Automatically detect best device
- `cpu`: Force CPU training
- `cuda`: Force NVIDIA GPU training
- `mps`: Force Apple Silicon GPU training

### Logging Settings
```yaml
logging:
  level: "INFO"           # "DEBUG", "INFO", "WARNING", "ERROR"
  log_interval: 10        # Log every N gradient steps
  save_interval: 5        # Save checkpoint every N epochs
```

**Logging levels**:
- `DEBUG`: Verbose function-level tracing (very detailed)
- `INFO`: Normal training progress (recommended)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

## Creating Custom Configurations

### Step 1: Copy a base configuration
```bash
cp configs/base.yaml configs/custom.yaml
```

### Step 2: Edit parameters
```yaml
# Example: Smaller model for faster experimentation
model:
  d_model: 192            # Smaller model (2.8M params)
  nhead: 8
  num_layers: 4

training:
  epochs: 10              # Fewer epochs for quick testing
  batch_size: 4           # Larger batch if you have memory
```

### Step 3: Use your custom config
```bash
python3.9 -m src.train.train --config configs/custom.yaml
```

## Hardware-Specific Recommendations

### 8GB RAM (Budget CPU)
```yaml
model:
  d_model: 192
  num_layers: 4
training:
  batch_size: 1
  grad_accum_steps: 64
```

### 16GB RAM (Standard CPU)
```yaml
model:
  d_model: 256
  num_layers: 6
training:
  batch_size: 2
  grad_accum_steps: 32
```

### 32GB+ RAM (High-end CPU)
```yaml
model:
  d_model: 384
  num_layers: 8
training:
  batch_size: 4
  grad_accum_steps: 16
```

### 8GB VRAM (GPU)
```yaml
model:
  d_model: 384
  num_layers: 8
training:
  batch_size: 16
  grad_accum_steps: 4
device:
  type: "cuda"
  num_workers: 4
  pin_memory: true
```

### 16GB+ VRAM (High-end GPU)
```yaml
model:
  d_model: 512
  num_layers: 10
training:
  batch_size: 32
  grad_accum_steps: 2
device:
  type: "cuda"
  num_workers: 8
  pin_memory: true
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size` (try 1)
2. Reduce `d_model` (try 256 or 192)
3. Reduce `num_layers` (try 4 or 6)
4. Disable diffusion: `diffusion.enabled: false`

### Training Too Slow
1. Increase `batch_size` if you have memory
2. Reduce `grad_accum_steps` proportionally
3. Reduce `epochs` for faster experimentation
4. Use GPU if available

### Poor Quality Results
1. Increase `d_model` (try 384 or 512)
2. Increase `num_layers` (try 8 or 10)
3. Increase `epochs` (try 100+)
4. Tune loss weights (increase `temporal` and `physics`)

## See Also

- [Training Guide](TRAINING_GUIDE.md) - Complete training pipeline
- [Installation Guide](../setup/INSTALLATION.md) - Setup instructions
- [Architecture Documentation](../architecture/AGENT.md) - Technical details

