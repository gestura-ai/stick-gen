---
language: en
license: mit
base_model: GesturaAI/stick-gen-large
tags:
- text-to-animation
- stick-figures
- motion-generation
- transformer
- fine-tuned
- sft
- supervised-fine-tuning
- pytorch
- high-quality
datasets:
- amass
- 100style
- interhuman
- ntu-rgbd
metrics:
- mse
- temporal_consistency
- action_accuracy
- robustness_score
library_name: pytorch
pipeline_tag: text-to-video
model-index:
- name: Stick-Gen Large SFT
  results:
  - task:
      type: text-to-animation
      name: Text-to-Animation Generation
    dataset:
      name: Curated SFT Dataset
      type: motion-capture
    metrics:
    - type: mse
      value: TBD
      name: Pose MSE
    - type: temporal_consistency
      value: TBD
      name: Temporal Consistency
    - type: accuracy
      value: TBD
      name: Action Classification Accuracy
---

# Stick-Gen Large SFT: High-Quality Supervised Fine-Tuned Model

<div align="center">
  <img src="https://img.shields.io/badge/Model-Stick--Gen--Large--SFT-blue" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Parameters-44.5M-green" alt="Parameters Badge"/>
  <img src="https://img.shields.io/badge/Variant-Large-orange" alt="Variant Badge"/>
  <img src="https://img.shields.io/badge/Fine--Tuned-SFT-purple" alt="SFT Badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License Badge"/>
</div>

## Model Description

This is a **Supervised Fine-Tuned (SFT)** version of [Stick-Gen Large](https://huggingface.co/GesturaAI/stick-gen-large). Maximum quality model optimized for GPU deployment with enhanced motion fidelity through SFT.

### Base Model
- **Base Model**: [GesturaAI/stick-gen-large](https://huggingface.co/GesturaAI/stick-gen-large)
- **Parameters**: 44.5M (same as large)
- **Architecture**: Transformer encoder with RMSNorm + SwiGLU + Pre-Norm

### Fine-Tuning Improvements
- **Highest Quality Motion**: Trained on curated samples with strictest quality thresholds
- **Best Action Accuracy**: Balanced action distribution in SFT dataset
- **Superior Physics**: Strictest physics consistency filtering
- **GPU Optimized**: Designed for high-end GPU deployment

## Training Details

### Fine-Tuning Data
- **Dataset**: `sft_data_embedded.pt` (curated SFT split)
- **Quality Filter**: High-quality samples only (score >= 0.5)
- **Action Balance**: Stratified sampling for balanced action distribution

### Fine-Tuning Procedure

| Parameter | Value |
|-----------|-------|
| Base Model | stick-gen-large |
| Learning Rate | 5e-5 (lower for large model) |
| Epochs | 20 |
| Batch Size | 2 |
| Gradient Accumulation | 64 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Training Stage | SFT |

### Training Command
```bash
python -m src.train.train \
    --config configs/sft_large.yaml \
    --init_from checkpoints/pretrain/large_best.pth \
    --data_path data/curated/sft_data_embedded.pt
```

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
from src.model.transformer import StickFigureTransformer

# Download SFT model
model_path = hf_hub_download(
    repo_id="GesturaAI/stick-gen-large-sft",
    filename="pytorch_model.bin"
)

# Load model (same architecture as large)
model = StickFigureTransformer(
    input_dim=20, d_model=512, nhead=16, num_layers=10,
    output_dim=20, embedding_dim=1024, num_actions=64
)
model.load_state_dict(torch.load(model_path))
model.eval()
```

## Model Variants

| Variant | Parameters | Use Case |
|---------|------------|----------|
| [Small](https://huggingface.co/GesturaAI/stick-gen-small) | 7.2M | CPU/Edge deployment |
| [Medium](https://huggingface.co/GesturaAI/stick-gen-medium) | 20.5M | Recommended default |
| [Large](https://huggingface.co/GesturaAI/stick-gen-large) | 44.5M | Maximum quality |

## Citation

See [Stick-Gen Large Model Card](https://huggingface.co/GesturaAI/stick-gen-large) for citation information.

## License

MIT License - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

