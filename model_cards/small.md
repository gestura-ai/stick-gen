---
language: en
license: mit
tags:
- text-to-animation
- stick-figures
- motion-generation
- transformer
- amass
- pytorch
- computer-vision
- generative-ai
- 100style
- interhuman
- ntu-rgbd
- lightweight
- cpu-friendly
- checkpoint-resume
- sft-ready
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
- name: Stick-Gen Small
  results:
  - task:
      type: text-to-animation
      name: Text-to-Animation Generation
    dataset:
      name: AMASS + InterHuman + NTU-RGB+D + 100STYLE + Synthetic
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

# Stick-Gen Small: Lightweight Text-to-Animation Model

**Variant**: Small (7.2M parameters)
**Optimized for**: Budget CPU deployment, edge devices, development/testing

This is the **Small variant** of Stick-Gen, optimized for resource-constrained environments. For other variants, see:
- [Stick-Gen Medium](https://huggingface.co/GesturaAI/stick-gen-medium) (20.5M params) - Recommended for most use cases
- [Stick-Gen Large](https://huggingface.co/GesturaAI/stick-gen-large) (44.5M params) - Maximum quality

## Quick Facts

| Specification | Value |
|---------------|-------|
| **Parameters** | 7,200,000 (7.2M) |
| **Model Size** | 27.5 MB (FP32) |
| **Hardware** | CPU (4+ cores, 8-16GB RAM) |
| **Inference Speed** | ~2.0s per animation (CPU) |
| **Training Time** | ~1,000 hours (30 epochs) |
| **Use Cases** | Budget deployment, edge devices, testing |

## Model Architecture

This model uses a **transformer encoder architecture** with multi-head self-attention, following modern LLM best practices (Qwen/Llama standards).

### Core Transformer

| Component | Value | Notes |
|-----------|-------|-------|
| **Architecture** | Transformer Encoder | PyTorch `nn.TransformerEncoder` |
| **Attention Type** | Multi-Head Self-Attention | 8 parallel attention heads |
| **d_model** | 256 | Hidden dimension (reduced from 384) |
| **num_layers** | 6 | Transformer encoder layers (reduced from 8) |
| **num_heads** | 8 | Attention heads (reduced from 12) |
| **dim_feedforward** | 1024 | SwiGLU hidden dimension (4× d_model) |
| **Normalization** | RMSNorm | Root Mean Square normalization (Pre-Norm) |
| **Activation** | SwiGLU | Gated Linear Unit with Swish (SiLU) |
| **Dropout** | 0.1 | Applied to attention and FFN |
| **Total Parameters** | 7,200,000 | 7.2M parameters |

### Input/Output Specifications

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **Input (Motion)** | 20 | 10 joints × 2 coordinates (x, y) |
| **Text Embedding** | 1024 | BAAI/bge-large-en-v1.5 embeddings |
| **Action Classes** | 64 | Discrete action conditioning |
| **Output (Pose)** | 20 | Predicted joint positions |

### Multi-Task Learning Heads

The model uses multiple output heads for comprehensive motion understanding:

- **Pose Decoder**: Joint position prediction (main task)
- **Position Decoder**: Global scene position (x, y)
- **Velocity Decoder**: Movement speed prediction
- **Action Predictor**: Action classification
- **Physics Decoder**: Velocity, acceleration, momentum states

### Fine-Tuning Suitability

This model is designed as a **base model for fine-tuning**:

- ✅ **Compact size** (7.2M params) enables fast fine-tuning on consumer hardware
- ✅ **Transformer architecture** allows attention-based transfer learning
- ✅ **Multi-task heads** can be frozen or fine-tuned independently
- ✅ **Action conditioning** supports domain-specific action vocabularies
- ✅ **Modular design** allows swapping embedding models or adding new heads

## Performance vs. Medium

| Metric | Small | Medium | Difference |
|--------|-------|--------|------------|
| Parameters | 7.2M | 20.5M | -65% |
| Model Size | 27.5 MB | 78.3 MB | -65% |
| Inference (CPU) | ~2.0s | ~1.5s | +33% slower |
| Memory (RAM) | 4GB | 8GB | -50% |
| Quality (Pose MSE) | TBD | TBD | TBD |

## When to Use Small

✅ **Use Small if you:**
- Have limited computational resources (4-core CPU, 8GB RAM)
- Need to deploy on edge devices or mobile
- Are developing/testing and don't need maximum quality
- Want faster model loading and lower memory footprint
- Need to minimize deployment costs

❌ **Don't use Small if you:**
- Have access to 8+ core CPU or GPU
- Need the highest quality animations
- Are deploying in production with quality requirements
- Can afford the extra computational cost

## Training Configuration

Trained with `config_cpu.yaml`:
- Batch size: 1 (gradient accumulation: 64)
- Epochs: 30 (reduced for faster training)
- Learning rate: 3e-4
- Diffusion: Disabled (for CPU efficiency)

## Key Features

- 🎯 **Realistic Motion**: Physics-aware training with velocity, acceleration, and momentum tracking
- 😊 **Facial Expressions**: 6 expression types with smooth transitions
- 📷 **Cinematic Rendering**: 2.5D perspective projection with Z-depth ordering
- 🎥 **Camera System**: 6 movement types (Pan, Zoom, Track, Dolly, Crane, Orbit)
- 🤖 **LLM Story Generation**: Generate narratives with Grok, Ollama, or Mock backends
- 💃 **100STYLE Support**: 100+ motion styles from BVH files

## Dataset Details

This model was trained on a hybrid dataset combining synthetic generation, real motion capture, and LLM-driven narratives.

| Data Source | Type | Contribution |
|-------------|------|--------------|
| **Procedural Generation** | Synthetic | 80% of data. Provides base physics, interactions, and diverse scenarios. |
| **LLM Stories** | Synthetic (Text) | 20% of data. Complex plots (heists, battles) generated by LLM to teach narrative structure. |
| **AMASS** | Real (MoCap) | High-quality human motion data retargeted to stick figures for realistic movement. |
| **100STYLE** | Real (BVH) | 100+ motion styles with forward kinematics conversion. |
| **Camera Data** | Synthetic | Simulated camera trajectories (Pan, Zoom, Track, Dolly, Crane, Orbit) for cinematic direction. |

**Dataset Availability**: The full training dataset is available at [GesturaAI/stick-gen-dataset](https://huggingface.co/datasets/GesturaAI/stick-gen-dataset).

## Transparency & Open Source

Gestura AI is committed to full transparency in AI development.
-   **Open Weights**: All model weights are released under MIT license.
-   **Open Data**: The training dataset is fully open-sourced.
-   **Open Code**: The entire training and generation pipeline is available on GitHub.
-   **Reproducibility**: We provide the exact scripts and configs used to train this model.

## Installation & Usage

```bash
# Install dependencies
pip install torch transformers sentence-transformers

# Download model
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="GesturaAI/stick-gen-small",
    filename="pytorch_model.bin"
)

# Load and use (see main model card for complete example)
```

## Citation

```bibtex
@software{stick_gen_small_2025,
  title = {Stick-Gen Small: Lightweight Text-to-Animation Model},
  author = {Gestura AI},
  year = {2025},
  version = {1.0.0},
  url = {https://huggingface.co/GesturaAI/stick-gen-small}
}
```

## License

MIT License - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

## Links

- **GitHub**: [gestura-ai/stick-gen](https://github.com/gestura-ai/stick-gen)
- **Base Model**: [GesturaAI/stick-gen-base](https://huggingface.co/GesturaAI/stick-gen-base)
- **Large Model**: [GesturaAI/stick-gen-large](https://huggingface.co/GesturaAI/stick-gen-large)
- **Documentation**: [docs/](https://github.com/gestura-ai/stick-gen/tree/main/docs)

---

For complete model details, training procedure, and evaluation metrics, see the [Base model card](https://huggingface.co/gestura-ai/stick-gen-base).

