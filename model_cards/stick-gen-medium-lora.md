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
- lora
- low-rank-adaptation
- parameter-efficient
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
library_name: pytorch
pipeline_tag: text-to-video
model-index:
- name: Stick-Gen Medium LoRA
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

# Stick-Gen Medium LoRA: Parameter-Efficient Fine-Tuned Model

<div align="center">
  <img src="https://img.shields.io/badge/Model-Stick--Gen--Medium--LoRA-blue" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Variant-Medium-purple" alt="Variant Badge"/>
  <img src="https://img.shields.io/badge/LoRA%20Params-~300K-orange" alt="LoRA Parameters"/>
  <img src="https://img.shields.io/badge/Base%20Params-20.5M-green" alt="Base Parameters"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License Badge"/>
</div>

## Model Description

This is a **LoRA (Low-Rank Adaptation)** fine-tuned version of [Stick-Gen Medium](https://huggingface.co/GesturaAI/stick-gen-medium). Only ~2% of parameters are trainable, enabling efficient fine-tuning while preserving base model capabilities.

### Architecture
- **Base Model**: [GesturaAI/stick-gen-medium](https://huggingface.co/GesturaAI/stick-gen-medium) (20.5M params, frozen)
- **LoRA Adapters**: ~300K trainable parameters
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
| Trainable Params | ~300K (2% of total) |
| Frozen Params | 15.5M (98% of total) |

## Training Details

### Fine-Tuning Data
- **Dataset**: `sft_data_embedded.pt` (curated SFT split)
- **Quality Filter**: High-quality samples only

### Fine-Tuning Procedure

| Parameter | Value |
|-----------|-------|
| Base Model | stick-gen-medium (frozen) |
| Learning Rate | 1e-4 |
| Epochs | 20 |
| Batch Size | 2 |
| Gradient Accumulation | 32 |
| Only LoRA Params | ✓ |

### Training Command
```bash
# Enable LoRA in config or via env
export USE_LORA=true
export LORA_RANK=8

python -m src.train.train \
    --config configs/sft_medium.yaml \
    --init_from checkpoints/pretrain/best.pth \
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
    input_dim=20, d_model=384, nhead=12, num_layers=8,
    output_dim=20, embedding_dim=1024, num_actions=64
)
model.load_state_dict(torch.load("base_model.pth"))

# Inject LoRA adapters
inject_lora_adapters(
    model,
    target_modules=["transformer_encoder", "pose_decoder"],
    rank=8, alpha=16
)

# Load LoRA weights
load_lora_state_dict(model, torch.load("lora_adapters.pth"))
model.eval()
```

### Merge LoRA into Base (for deployment)
```python
from src.model.lora import merge_lora_weights

# Merge LoRA adapters into base weights
merge_lora_weights(model)

# Now model can be saved/deployed without LoRA overhead
torch.save(model.state_dict(), "merged_model.pth")
```

## Benefits of LoRA

| Benefit | Description |
|---------|-------------|
| **Efficient Training** | Only 2% of params updated |
| **Memory Savings** | ~50% less GPU memory |
| **Fast Iteration** | Quick experimentation cycles |
| **Modularity** | Swap adapters for different styles |
| **Mergeable** | Can merge back into base for deployment |

## Citation

See [Stick-Gen Medium Model Card](https://huggingface.co/GesturaAI/stick-gen-medium) for citation.

### LoRA Paper
```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## License

MIT License - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

