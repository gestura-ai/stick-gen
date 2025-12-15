# Hugging Face Hub Upload Guide

This guide explains how to upload your trained stick-gen model to Hugging Face Hub for easy sharing and deployment.

## Prerequisites

### 1. Install Hugging Face Hub
```bash
pip install huggingface_hub
```

### 2. Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Verify your email

### 3. Get API Token
1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name: "stick-gen-upload"
4. Type: "Write"
5. Copy the token

### 4. Login to Hugging Face CLI
```bash
huggingface-cli login
# Paste your token when prompted
```

Or set environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Quick Upload

### Upload Medium Variant (Default)
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --version 1.0.0
```

This will:
1. ✅ Validate checkpoint matches medium variant (15.8M params)
2. ✅ Package model checkpoint
3. ✅ Copy variant-specific model card (model_cards/medium.md → README.md)
4. ✅ Copy variant-specific configuration (configs/medium.yaml)
5. ✅ Copy source code
6. ✅ Upload to `GesturaAI/stick-gen-medium`

### Upload Small Variant
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/small_model.pt \
  --variant small \
  --version 1.0.0
```

Uploads to `GesturaAI/stick-gen-small` with:
- Configuration: `configs/small.yaml`
- Model card: `model_cards/small.md`
- Expected parameters: 5.6M

### Upload Large Variant
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/large_model.pt \
  --variant large \
  --version 1.0.0
```

Uploads to `GesturaAI/stick-gen-large` with:
- Configuration: `configs/large.yaml`
- Model card: `model_cards/large.md`
- Expected parameters: 28M

## Upload Fine-Tuned Models

### Upload SFT Model
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/sft/best.pth \
  --variant medium \
  --model-card model_cards/stick-gen-medium-sft.md \
  --repo-name GesturaAI/stick-gen-medium-sft \
  --version 1.0.0
```

This uploads to `GesturaAI/stick-gen-medium-sft` with:
- Model card: `model_cards/stick-gen-medium-sft.md`
- Medium model reference in metadata
- SFT training configuration

### Upload LoRA Adapters
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/lora/lora_adapters.pth \
  --variant medium \
  --model-card model_cards/stick-gen-medium-lora.md \
  --repo-name GesturaAI/stick-gen-medium-lora \
  --version 1.0.0
```

This uploads to `GesturaAI/stick-gen-medium-lora` with:
- LoRA adapter weights only (~300K parameters)
- Instructions for merging with medium model
- LoRA configuration metadata

### Fine-Tuned Model Naming Convention

| Model Type | Repository Name | Example |
|------------|-----------------|---------|
| Pretrained | `GesturaAI/stick-gen-{variant}` | `stick-gen-medium` |
| SFT Fine-Tuned | `GesturaAI/stick-gen-{variant}-sft` | `stick-gen-medium-sft` |
| LoRA Adapters | `GesturaAI/stick-gen-{variant}-lora` | `stick-gen-medium-lora` |
| Custom Fine-Tune | `GesturaAI/stick-gen-{variant}-{method}` | `stick-gen-medium-custom` |

## Advanced Usage

### Custom Repository Name
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --repo-name your-username/my-custom-model
```

### Create Private Repository
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --private
```

### Prepare Files Without Uploading
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --no-upload
```

This creates `hf_upload/` directory with all files ready for manual upload.

### Skip Validation (Not Recommended)
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --skip-validation
```

**Warning**: Only use this if you know the checkpoint is correct. Validation ensures:
- Parameter count matches variant
- Model card has required sections
- All necessary files are present

### Custom Output Directory
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --output-dir my_upload_dir
```

### Use Specific Token
```bash
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant medium \
  --token hf_xxxxxxxxxxxxx
```

## Model Variant Selection

### Choosing the Right Variant

| Variant | Parameters | Hardware | Use Case |
|---------|-----------|----------|----------|
| **Small** | 5.6M | CPU (4+ cores, 8-16GB RAM) | Budget deployment, edge devices, testing |
| **Medium** | 15.8M | CPU (8+ cores, 16-32GB RAM) or GPU (8GB+) | Standard deployment, balanced quality/performance |
| **Large** | 28M | GPU (8GB+ VRAM) | High-quality production, maximum animation quality |

### Variant Configuration Mapping

Each variant uses specific configuration files:

| Variant | Config File | Model Card | Repository |
|---------|------------|------------|------------|
| Small | `configs/small.yaml` | `model_cards/small.md` | `GesturaAI/stick-gen-small` |
| Medium | `configs/medium.yaml` | `model_cards/medium.md` | `GesturaAI/stick-gen-medium` |
| Large | `configs/large.yaml` | `model_cards/large.md` | `GesturaAI/stick-gen-large` |

## Validation

The upload script automatically validates:

### 1. Parameter Count Validation
- Checks that checkpoint parameter count matches expected variant
- Tolerance: ±5% of expected parameters
- Example: Medium variant expects 15.8M ± 790K parameters

### 2. Model Card Validation
- Checks for required sections (Model Details, Training Details, Citation, etc.)
- Warns about "TBD" values that should be updated
- Ensures variant-specific model card is used

### 3. File Validation
- Verifies checkpoint file exists and is loadable
- Checks for required files (LICENSE, CITATIONS.md, etc.)
- Validates configuration file matches variant

If validation fails, you'll be prompted to continue or cancel.

## What Gets Uploaded

The upload script packages the following files:

```
hf_upload/
├── README.md                    # Model card (variant-specific)
├── pytorch_model.bin            # Model checkpoint
├── config.yaml                  # Training configuration (variant-specific)
├── VERSION                      # Version file (if --version specified)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── CITATIONS.md                 # Dataset citations
└── src/                         # Source code
    ├── model/                   # Model architecture
    ├── inference/               # Inference code
    └── data_gen/                # Data generation utilities
```

## Loading Model from Hub

Once uploaded, users can load your model:

### Load Medium Variant
```python
from huggingface_hub import hf_hub_download
import torch
from src.model.transformer import StickFigureTransformer

# Download model
model_path = hf_hub_download(
    repo_id="GesturaAI/stick-gen-medium",
    filename="pytorch_model.bin"
)

# Load model (medium variant: 15.8M params)
model = StickFigureTransformer(
    input_dim=20,
    d_model=384,      # Medium variant
    nhead=12,         # Medium variant
    num_layers=8,     # Medium variant
    output_dim=20,
    embedding_dim=1024,
    num_actions=64
)
model.load_state_dict(torch.load(model_path))
model.eval()
```

### Load Small Variant
```python
# Download small variant
model_path = hf_hub_download(
    repo_id="gestura-ai/stick-gen-small",
    filename="pytorch_model.bin"
)

# Load model (small variant: 5.6M params)
model = StickFigureTransformer(
    input_dim=20,
    d_model=256,      # Small variant
    nhead=8,          # Small variant
    num_layers=6,     # Small variant
    output_dim=20,
    embedding_dim=1024,
    num_actions=64
)
model.load_state_dict(torch.load(model_path))
model.eval()
```

### Load Large Variant
```python
# Download large variant
model_path = hf_hub_download(
    repo_id="gestura-ai/stick-gen-large",
    filename="pytorch_model.bin"
)

# Load model (large variant: 28M params)
model = StickFigureTransformer(
    input_dim=20,
    d_model=512,      # Large variant
    nhead=16,         # Large variant
    num_layers=10,    # Large variant
    output_dim=20,
    embedding_dim=1024,
    num_actions=64
)
model.load_state_dict(torch.load(model_path))
model.eval()
```

### Load with Configuration File
```python
import yaml
from huggingface_hub import hf_hub_download

# Download config
config_path = hf_hub_download(
    repo_id="GesturaAI/stick-gen-medium",
    filename="config.yaml"
)

# Load configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create model from config
model = StickFigureTransformer(
    input_dim=20,
    d_model=config['model']['d_model'],
    nhead=config['model']['nhead'],
    num_layers=config['model']['num_layers'],
    output_dim=20,
    embedding_dim=config['model']['embedding_dim'],
    num_actions=config['model']['num_actions']
)
```

## Updating Model Card

### Edit Model Card
1. Edit `model_card.md` in your repository
2. Update metrics, examples, or documentation
3. Re-run upload script:

```bash
python push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --repo-name your-username/stick-gen
```

### Add Training Metrics
After training completes, update `model_card.md` with actual metrics:

```markdown
### Metrics (Validation Set)
- **Pose MSE**: 0.0234
- **Temporal Consistency**: 0.0156
- **Action Accuracy**: 87.3%
- **Physics MSE**: 0.0189
```

### Add Example Outputs
Include example animations in your model card:

```markdown
## Example Outputs

### "A person walks forward and waves"
![Animation](examples/walk_wave.gif)

### "Two people playing catch"
![Animation](examples/catch.gif)
```

## Best Practices

### 1. Use Semantic Versioning
Create tags for different model versions:

```bash
# Upload v1.0.0
python push_to_huggingface.py \
  --checkpoint checkpoints/epoch_50.pt \
  --repo-name your-username/stick-gen

# Later, upload v1.1.0 with improvements
python push_to_huggingface.py \
  --checkpoint checkpoints/epoch_100.pt \
  --repo-name your-username/stick-gen
```

### 2. Include Example Code
Add usage examples to your model card showing:
- Basic inference
- Batch processing
- Custom configurations

### 3. Document Limitations
Be transparent about model limitations:
- Maximum sequence length
- Supported actions
- Known failure cases

### 4. Provide Citations
Always include proper citations for:
- AMASS dataset
- Contributing motion capture datasets
- Embedding models

## Troubleshooting

### Authentication Error
```
Error: Invalid token
```

**Solution**: Re-login with correct token
```bash
huggingface-cli login
```

### Upload Timeout
```
Error: Upload timeout
```

**Solution**: Upload files manually or increase timeout
```python
# In push_to_huggingface.py, add timeout parameter
upload_folder(..., timeout=600)
```

### Large File Error
```
Error: File too large
```

**Solution**: Use Git LFS for large files
```bash
cd hf_upload
git lfs install
git lfs track "*.bin"
git add .gitattributes
git add pytorch_model.bin
git commit -m "Add model checkpoint"
git push
```

### Repository Already Exists
```
Error: Repository already exists
```

**Solution**: The script handles this automatically with `exist_ok=True`. If you want to create a new repository, use a different name.

## See Also

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Git LFS Guide](https://huggingface.co/docs/hub/adding-a-model#using-git-lfs)
- [Model Export Guide](MODEL_EXPORT_GUIDE.md)

