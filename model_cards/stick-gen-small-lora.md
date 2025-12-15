---
language: en
license: mit
base_model: GesturaAI/stick-gen-small
tags:
- text-to-animation
- stick-figures
- motion-generation
- transformer
- fine-tuned
- lora
- low-rank-adaptation
- parameter-efficient
- pytorch
- lightweight
datasets:
- amass
- 100style
- interhuman
- ntu-rgbd
metrics:
- mse
- temporal_consistency
- action_accuracy
library_name: pytorch
pipeline_tag: text-to-video
model-index:
- name: Stick-Gen Small LoRA
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
    - type: accuracy
      value: TBD
      name: Action Classification Accuracy
---

# Stick-Gen Small LoRA: Lightweight Parameter-Efficient Fine-Tuned Model

<div align="center">
  <img src="https://img.shields.io/badge/Model-Stick--Gen--Small--LoRA-blue" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Variant-Small-purple" alt="Variant Badge"/>
  <img src="https://img.shields.io/badge/LoRA%20Params-~200K-orange" alt="LoRA Parameters"/>
  <img src="https://img.shields.io/badge/Base%20Params-7.2M-green" alt="Base Parameters"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License Badge"/>
</div>

## Model Description

This is a **LoRA (Low-Rank Adaptation)** fine-tuned version of [Stick-Gen Small](https://huggingface.co/GesturaAI/stick-gen-small). Only ~3.5% of parameters are trainable, enabling efficient fine-tuning while preserving base model capabilities.

### Architecture
- **Base Model**: [GesturaAI/stick-gen-small](https://huggingface.co/GesturaAI/stick-gen-small) (7.2M params, frozen)
- **LoRA Adapters**: ~200K trainable parameters
- **Target Modules**: `transformer_encoder`, `pose_decoder`
- **Rank**: 8
- **Alpha**: 16

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.05 |
| Target Modules | transformer_encoder, pose_decoder |
| Trainable Params | ~200K (3.5% of total) |
| Frozen Params | 5.4M (96.5% of total) |

## Training Details

### Fine-Tuning Procedure

| Parameter | Value |
|-----------|-------|
| Base Model | stick-gen-small (frozen) |
| Learning Rate | 1e-4 |
| Epochs | 20 |
| Batch Size | 4 |
| Gradient Accumulation | 16 |
| Only LoRA Params | ✓ |

### Training Command
```bash
export USE_LORA=true
export LORA_RANK=8

python -m src.train.train \
    --config configs/sft_small.yaml \
    --init_from checkpoints/pretrain/small_best.pth \
    --data_path data/curated/sft_data_embedded.pt
```

## Usage

### Load LoRA Adapters
```python
import torch
from src.model.transformer import StickFigureTransformer
from src.model.lora import inject_lora_adapters, load_lora_state_dict

# Load base model
model = StickFigureTransformer(
    input_dim=20, d_model=256, nhead=8, num_layers=6,
    output_dim=20, embedding_dim=1024, num_actions=64
)
model.load_state_dict(torch.load("small_model.pth"))

# Inject LoRA adapters
inject_lora_adapters(model, target_modules=["transformer_encoder", "pose_decoder"], rank=8, alpha=16)

# Load LoRA weights
load_lora_state_dict(model, torch.load("lora_adapters.pth"))
model.eval()
```

## Model Variants

| Variant | Base Params | LoRA Params | Use Case |
|---------|-------------|-------------|----------|
| Small-LoRA | 7.2M | ~200K | CPU/Edge with fine-tuning |
| Medium-LoRA | 20.5M | ~300K | Recommended default |
| Large-LoRA | 44.5M | ~400K | Maximum quality |

## Citation

See [Stick-Gen Small Model Card](https://huggingface.co/GesturaAI/stick-gen-small) for citation.

## License

MIT License - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

