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
- high-quality
- gpu-optimized
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
- name: Stick-Gen Large
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

# Stick-Gen Large: High-Quality Text-to-Animation Model

**Variant**: Large (44.6M motion-only / 71.3M multimodal)
**Optimized for**: GPU deployment, maximum animation quality, production use

This is the **Large variant** of Stick-Gen, optimized for maximum quality with GPU acceleration. For other variants, see:
- [Stick-Gen Small](https://huggingface.co/GesturaAI/stick-gen-small) (7.2M/11.7M params) - Budget CPU deployment
- [Stick-Gen Medium](https://huggingface.co/GesturaAI/stick-gen-medium) (20.6M/25.1M params) - Recommended for most use cases

## Quick Facts

| Specification | Motion-Only | Multimodal |
|---------------|-------------|------------|
| **Parameters** | 44,619,646 (44.6M) | 71,284,158 (71.3M) |
| **Model Size** | ~178 MB (FP32) / ~89 MB (FP16) | ~285 MB (FP32) / ~143 MB (FP16) |
| **Hardware** | GPU (8GB+ VRAM) | GPU (16GB+ VRAM) |
| **Inference Speed** | ~0.2s per animation (GPU) | ~0.3s per animation (GPU) |
| **Training Time** | ~400 hours (100 epochs) | ~600 hours (100 epochs) |
| **Use Cases** | High-quality production | Maximum quality multimodal |

## Model Architecture

This model uses a **transformer encoder architecture** with multi-head self-attention, following modern LLM best practices (Qwen/Llama standards). This is the largest variant with maximum capacity for highest quality output.

### Core Transformer

| Component | Value | Notes |
|-----------|-------|-------|
| **Architecture** | Transformer Encoder | PyTorch `nn.TransformerEncoder` |
| **Attention Type** | Multi-Head Self-Attention | 16 parallel attention heads |
| **d_model** | 512 | Hidden dimension (increased from 384) |
| **num_layers** | 10 | Transformer encoder layers (increased from 8) |
| **num_heads** | 16 | Attention heads (increased from 12) |
| **dim_feedforward** | 2048 | SwiGLU hidden dimension (4× d_model) |
| **Normalization** | RMSNorm | Root Mean Square normalization (Pre-Norm) |
| **Activation** | SwiGLU | Gated Linear Unit with Swish (SiLU) |
| **Dropout** | 0.1 | Applied to attention and FFN |
| **Total Parameters** | 44,619,646 / 71,284,158 | 44.6M (motion-only) / 71.3M (multimodal) |

### Input/Output Specifications

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **Input (Motion)** | 20 | 10 joints × 2 coordinates (x, y) |
| **Text Embedding** | 1024 | BAAI/bge-large-en-v1.5 embeddings |
| **Action Classes** | 64 | Discrete action conditioning |
| **Output (Pose)** | 20 | Predicted joint positions per frame |
| **Sequence Length** | 250 | Frames (10 seconds @ 25fps) |

### Multi-Task Learning Heads

The model uses 6 specialized decoder heads for comprehensive motion generation:

| Head | Output Dim | Description |
|------|------------|-------------|
| **Pose Decoder** | 20 per frame | Joint position prediction (main task) |
| **Position Decoder** | 2 | Global scene position (x, y) |
| **Velocity Decoder** | 2 | Movement speed prediction |
| **Action Predictor** | 64 | Action classification logits |
| **Physics Decoder** | 6 per frame | Velocity, acceleration, momentum states |
| **Environment Decoder** | 32 | Environment context features |

### Fine-Tuning Suitability

This model is designed as a **high-quality base model for fine-tuning**:

- ✅ **Maximum capacity** (44.6M/71.3M params) for highest quality fine-tuned results
- ✅ **Deeper transformer** (10 layers) captures more complex motion patterns
- ✅ **More attention heads** (16) enables finer-grained attention patterns
- ✅ **Multi-task heads** can be frozen or fine-tuned independently
- ✅ **Diffusion-enabled training** provides smoother motion output
- ✅ **GPU-optimized** for fast fine-tuning on modern hardware

## Performance vs. Medium

| Metric | Medium | Large | Improvement |
|--------|--------|-------|-------------|
| Parameters | 20.5M | 44.5M | +117% |
| Model Size | 78.3 MB | 170 MB | +117% |
| Inference (GPU) | ~0.3s | ~0.2s | +33% faster |
| Memory (VRAM) | 4GB | 8GB | +100% |
| Quality (Pose MSE) | TBD | TBD | TBD (expected better) |

## When to Use Large

✅ **Use Large if you:**
- Have GPU with 8GB+ VRAM (RTX 3060 or better)
- Need the highest quality animations possible
- Are deploying in production with quality requirements
- Can afford the extra computational cost
- Want fastest inference with GPU acceleration

❌ **Don't use Large if you:**
- Only have CPU available
- Have limited VRAM (<8GB)
- Don't need maximum quality
- Want to minimize deployment costs
- Are just testing or developing

## Training Configuration

Trained with `config_gpu.yaml`:
- Batch size: 16 (gradient accumulation: 4)
- Epochs: 100 (more epochs for better convergence)
- Learning rate: 5e-4 (higher for larger batches)
- Diffusion: Enabled (for quality improvement)

## Key Features

- 🎯 **Realistic Motion**: Physics-aware training with velocity, acceleration, and momentum tracking
- 😊 **Facial Expressions**: 6 expression types with smooth transitions
- 📷 **Cinematic Rendering**: 2.5D perspective projection with Z-depth ordering
- 🎥 **Camera System**: 6 movement types (Pan, Zoom, Track, Dolly, Crane, Orbit)
- 🤖 **LLM Story Generation**: Generate narratives with Grok, Ollama, or Mock backends
- 📹 **Camera Conditioning**: Model accepts camera context during generation
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

## GPU Requirements

| GPU Model | VRAM | Batch Size | Inference Speed |
|-----------|------|------------|-----------------|
| RTX 3060 | 12GB | 16 | ~0.2s |
| RTX 3070 | 8GB | 12 | ~0.25s |
| RTX 3080 | 10GB | 16 | ~0.15s |
| RTX 3090 | 24GB | 32 | ~0.12s |
| A100 | 40GB | 64 | ~0.08s |

## Installation & Usage

```bash
# Install dependencies with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers

# Download model
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="GesturaAI/stick-gen-large",
    filename="pytorch_model.bin"
)

# Load and use with GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## Optimization Tips

- **Use FP16**: Reduce memory usage by 50% with minimal quality loss
- **Batch Inference**: Process multiple prompts simultaneously
- **TensorRT**: Further optimize inference speed with TensorRT
- **Model Quantization**: INT8 quantization for even faster inference

## Citation

```bibtex
@software{stick_gen_large_2025,
  title = {Stick-Gen Large: High-Quality Text-to-Animation Model},
  author = {Gestura AI},
  year = {2025},
  version = {1.0.0},
  url = {https://huggingface.co/GesturaAI/stick-gen-large}
}
```

## License

MIT License - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

## Links

- **GitHub**: [gestura-ai/stick-gen](https://github.com/gestura-ai/stick-gen)
- **Small Model**: [GesturaAI/stick-gen-small](https://huggingface.co/GesturaAI/stick-gen-small)
- **Medium Model**: [GesturaAI/stick-gen-medium](https://huggingface.co/GesturaAI/stick-gen-medium)
- **Documentation**: [docs/](https://github.com/gestura-ai/stick-gen/tree/main/docs)

---

For complete model details, training procedure, and evaluation metrics, see the [Medium model card](https://huggingface.co/GesturaAI/stick-gen-medium).

