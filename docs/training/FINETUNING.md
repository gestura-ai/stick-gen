# Fine-Tuning Guide for Stick-Gen

This document describes the **production-oriented expert decomposition** pipeline for fine-tuning Stick-Gen's dual-phase motion generation architecture.

## Overview

Stick-Gen uses a **multi-expert LoRA (Low-Rank Adaptation)** system for domain-specific fine-tuning. Rather than training separate models for different motion types, we train lightweight LoRA adapters that can be dynamically composed at inference time.

### Expert Categories

We use **two types of experts**:

| Type | Routing | Purpose |
|------|---------|---------|
| **Style Experts** | Softmax (mutually exclusive) | Motion style and character expression |
| **Orthogonal Experts** | Sigmoid gating (always-on) | Technical aspects like camera and timing |

## Phase 1: Expert Configuration

Expert configs are stored in `configs/finetune/`:

### Style Experts (Routed via Softmax)

| Config | Focus | Target |
|--------|-------|--------|
| `dramatic_style.yaml` | Emotional pacing, slower tempo, expressive movements | Both phases |
| `action_style.yaml` | High energy, fast cuts, dynamic motion | Both phases |
| `expressive_body.yaml` | Body dynamics, natural motion quality | Both phases |
| `multi_actor.yaml` | Multi-actor coordination | Both phases |

### Orthogonal Experts (Always Active)

| Config | Focus | Target Phase |
|--------|-------|--------------|
| `camera.yaml` | Cinematic camera framing | Transformer only |
| `timing.yaml` | Dramatic pacing, holds, timing beats | Diffusion only |

### Phase-Specific LoRA Injection

The `target_phase` field controls which model components receive LoRA adapters:

- `"both"` - Inject into transformer AND diffusion modules
- `"transformer_only"` - Inject only into transformer (planning phase)
- `"diffusion_only"` - Inject only into diffusion UNet (refinement phase)

```yaml
# Example: configs/finetune/camera.yaml
lora:
  rank: 8
  alpha: 16
  dropout: 0.1
  target_modules:
    - "camera_projection"
    - "q_proj"
    - "v_proj"
  target_phase: "transformer_only"  # Camera affects planning only
```

## Phase 2: Training

### Running Fine-Tuning

```bash
# Fine-tune a single expert
python -m src.train.finetune \
  --config configs/finetune/dramatic_style.yaml \
  --data_path data/motions_processed/dramatic/ \
  --output_dir checkpoints/lora/

# Or using the CLI
python -m src.cli finetune --config configs/finetune/action_style.yaml
```

### Training Configuration

Each config includes:

```yaml
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 100
  gradient_accumulation: 4
  
domain:
  name: "dramatic_style"
  description: "Emotional pacing and expressive movements"
  action_types:
    - "emotional_gesture"
    - "dramatic_pause"
    - "expressive_movement"
```

## Phase 3: Multi-Expert Inference

At inference time, the `MultiLoRARouter` dynamically selects and blends experts based on the input prompt.

### Expert Registry

Trained experts are registered in `configs/experts/registry.yaml`:

```yaml
router:
  temperature: 0.1    # Lower = sharper routing
  min_expert_weight: 0.05

style_experts:
  dramatic_style:
    enabled: true
    checkpoint_path: "checkpoints/lora/dramatic_style.pt"
    target_phase: "both"
    prototype_prompts:
      - "A slow, emotional scene with deep sadness"
      - "Dramatic pause before revealing information"
```

### Using Multi-Expert Routing

```python
from src.inference.generator import InferenceGenerator

# Initialize generator
gen = InferenceGenerator(model_path="checkpoints/model.pt")

# Initialize multi-expert routing
gen.initialize_multi_expert_routing(
    registry_path="configs/experts/registry.yaml",
    auto_load=True
)

# Generate with dynamic expert composition
motion = gen.generate(
    prompt="A dramatic confrontation with slow, deliberate movements",
    num_frames=250
)

# Inspect routing weights
weights = gen.get_routing_weights(
    "Fast-paced chase through the city"
)
print(weights)
# {'action_style': 0.72, 'dramatic_style': 0.18, ...}
```

### Routing Mechanism

1. **Text Embedding**: Input prompt is encoded using BAAI/bge-large-en-v1.5
2. **Cosine Similarity**: Embedding compared to expert prototype embeddings
3. **Softmax Routing**: Style experts receive temperature-controlled softmax weights
4. **Sigmoid Gating**: Orthogonal experts use learned MLP gating
5. **LoRA Blending**: Active expert weights are blended and applied to model

## Files Reference

| File | Purpose |
|------|---------|
| `src/train/finetune.py` | Fine-tuning pipeline with phase-specific LoRA |
| `src/model/lora.py` | LoRA implementation + `MultiExpertLoRAManager` |
| `src/model/lora_router.py` | `MultiLoRARouter` for dynamic expert composition |
| `src/inference/generator.py` | Inference integration with router |
| `configs/finetune/*.yaml` | Expert training configurations |
| `configs/experts/registry.yaml` | Expert checkpoint registry |

## Best Practices

1. **Start with style experts** - They have the most impact on output quality
2. **Use phase-specific targeting** - Camera/timing experts should target their relevant phase
3. **Keep rank low** - rank=8 is usually sufficient; higher ranks risk overfitting
4. **Curate training data** - Quality matters more than quantity for LoRA fine-tuning
5. **Monitor routing weights** - Use `get_routing_weights()` to debug unexpected behavior

