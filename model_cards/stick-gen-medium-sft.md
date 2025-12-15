---
language: en
license: mit
base_model: GesturaAI/stick-gen-medium
tags:
- text-to-animation
- stick-figures
- motion-generation
- transformer
- fine-tuned
- sft
- supervised-fine-tuning
- pytorch
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
- name: Stick-Gen Medium SFT
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

# Stick-Gen Medium SFT: Supervised Fine-Tuned Text-to-Animation Model

<div align="center">
  <img src="https://img.shields.io/badge/Model-Stick--Gen--Medium--SFT-blue" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Parameters-20.5M-green" alt="Parameters Badge"/>
  <img src="https://img.shields.io/badge/Variant-Medium-orange" alt="Variant Badge"/>
  <img src="https://img.shields.io/badge/Fine--Tuned-SFT-purple" alt="SFT Badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License Badge"/>
</div>

## Model Description

This is a **Supervised Fine-Tuned (SFT)** version of [Stick-Gen Medium](https://huggingface.co/GesturaAI/stick-gen-medium). It has been fine-tuned on a curated, high-quality subset of the training data to improve motion quality, temporal consistency, and action accuracy.

### Base Model
- **Base Model**: [GesturaAI/stick-gen-medium](https://huggingface.co/GesturaAI/stick-gen-medium)
- **Parameters**: 20.5M (same as medium)
- **Architecture**: Transformer encoder with RMSNorm + SwiGLU + Pre-Norm

### Fine-Tuning Improvements
- **Higher Quality Motion**: Trained on curated samples with stricter quality thresholds
- **Better Action Accuracy**: Balanced action distribution in SFT dataset
- **Improved Physics**: Stricter physics consistency filtering
- **Camera Stability**: Enhanced camera-motion alignment

## Training Details

### Fine-Tuning Data
- **Dataset**: `sft_data_embedded.pt` (curated SFT split)
- **Quality Threshold**: `min_quality_score >= 0.5`
- **Physics Filter**: Only samples passing strict physics validation
- **Action Balancing**: Uniform distribution across action categories

### Fine-Tuning Procedure

| Parameter | Value |
|-----------|-------|
| Base Model | stick-gen-medium |
| Learning Rate | 1e-4 (lower than pretraining) |
| Epochs | 20 |
| Batch Size | 2 |
| Gradient Accumulation | 32 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Training Stage | SFT |

### Training Command
```bash
python -m src.train.train \
    --config configs/sft_medium.yaml \
    --init_from checkpoints/pretrain/best.pth \
    --data_path data/curated/sft_data_embedded.pt
```

## Differences from Medium Model

| Aspect | Medium Model | SFT Model |
|--------|--------------|-----------|
| Training Data | Full pretrain dataset | Curated SFT subset |
| Quality Threshold | None | >= 0.5 |
| Action Distribution | Natural | Balanced |
| Learning Rate | 3e-4 | 1e-4 |
| Epochs | 50 | 20 |
| Focus | General motion | High-quality motion |

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
from src.model.transformer import StickFigureTransformer

# Download SFT model
model_path = hf_hub_download(
    repo_id="GesturaAI/stick-gen-medium-sft",
    filename="pytorch_model.bin"
)

# Load model (same architecture as medium)
model = StickFigureTransformer(
    input_dim=20, d_model=384, nhead=12, num_layers=8,
    output_dim=20, embedding_dim=1024, num_actions=64
)
model.load_state_dict(torch.load(model_path))
model.eval()
```

## Evaluation

See [Stick-Gen Medium](https://huggingface.co/GesturaAI/stick-gen-medium) for full evaluation methodology. SFT models are expected to show improvements in:
- Motion quality metrics
- Action classification accuracy
- Physics consistency scores

## Citation

See [Stick-Gen Medium Model Card](https://huggingface.co/GesturaAI/stick-gen-medium) for citation information.

## License

MIT License - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

