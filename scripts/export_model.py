#!/usr/bin/env python3
"""
Model Export Script - Convert stick-gen models to modern formats

Supports:
- Hugging Face format (safetensors + config.json + model card)
- ONNX format (cross-platform deployment)
- TorchScript format (PyTorch production)

Usage:
    # Export to all formats
    python export_model.py --input checkpoint.pth --output stick-gen-v1 --formats all
    
    # Export to specific formats
    python export_model.py --input checkpoint.pth --output stick-gen-v1 --formats safetensors onnx
    
    # Push to Hugging Face Hub
    python export_model.py --input checkpoint.pth --output stick-gen-v1 --push-to-hub gestura-ai/stick-gen-v1
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIG = {
    "model_type": "stick-figure-transformer",
    "architecture": "transformer",
    "d_model": 384,
    "nhead": 12,
    "num_layers": 8,
    "dim_feedforward": 1536,
    "dropout": 0.1,
    "input_dim": 20,
    "output_dim": 20,
    "text_embedding_dim": 1024,
    "sequence_length": 250,
    "fps": 25,
    "duration_seconds": 10.0,
    "num_actions": 60,
    "action_embedding_dim": 64,
    "num_parameters": 15_500_000,
    "torch_dtype": "float32",
    "framework": "pytorch",
    "license": "apache-2.0",
    "tags": ["stick-figure", "animation", "transformer", "text-to-motion"]
}

# ============================================================================
# Export Functions
# ============================================================================

def export_to_safetensors(model_path: str, output_dir: Path) -> None:
    """Export model to Hugging Face safetensors format"""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("❌ safetensors not installed. Run: pip install safetensors")
        sys.exit(1)
    
    print("📦 Exporting to Hugging Face format (safetensors)...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Save as safetensors
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(MODEL_CONFIG, f, indent=2)
    
    # Save training metadata if available
    if 'epoch' in checkpoint:
        training_args = {
            "epoch": checkpoint.get('epoch', 0),
            "loss": checkpoint.get('loss', 0.0),
            "action_accuracy": checkpoint.get('action_accuracy', 0.0),
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "batch_size": 16,
            "gradient_accumulation_steps": 4
        }
        training_args_path = output_dir / "training_args.json"
        with open(training_args_path, 'w') as f:
            json.dump(training_args, f, indent=2)
    
    print(f"✓ Saved to {safetensors_path}")
    print(f"✓ Saved config to {config_path}")
    print(f"✓ Model size: {safetensors_path.stat().st_size / 1024 / 1024:.2f} MB")


def export_to_onnx(model_path: str, output_dir: Path) -> None:
    """Export model to ONNX format"""
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("❌ onnx/onnxruntime not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)
    
    print("📦 Exporting to ONNX format...")
    
    # Load model
    from src.model.transformer import StickFigureTransformer
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = StickFigureTransformer(
        input_dim=MODEL_CONFIG['input_dim'],
        output_dim=MODEL_CONFIG['output_dim'],
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['nhead'],
        num_layers=MODEL_CONFIG['num_layers'],
        dim_feedforward=MODEL_CONFIG['dim_feedforward'],
        dropout=MODEL_CONFIG['dropout'],
        text_embedding_dim=MODEL_CONFIG['text_embedding_dim'],
        num_actions=MODEL_CONFIG['num_actions']
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = MODEL_CONFIG['sequence_length']
    motion_sequence = torch.randn(batch_size, seq_len, MODEL_CONFIG['input_dim'])
    text_embedding = torch.randn(batch_size, MODEL_CONFIG['text_embedding_dim'])
    action_sequence = torch.randint(0, MODEL_CONFIG['num_actions'], (batch_size, seq_len))
    
    # Export to ONNX
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    
    torch.onnx.export(
        model,
        (motion_sequence, text_embedding, action_sequence),
        str(onnx_path),
        input_names=['motion_sequence', 'text_embedding', 'action_sequence'],
        output_names=['pose', 'position', 'velocity', 'action_logits'],
        dynamic_axes={
            'motion_sequence': {0: 'batch_size'},
            'text_embedding': {0: 'batch_size'},
            'action_sequence': {0: 'batch_size'},
            'pose': {0: 'batch_size'},
            'position': {0: 'batch_size'},
            'velocity': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    # Validate ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ Saved to {onnx_path}")
    print(f"✓ Model size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"✓ ONNX validation: PASSED")


def export_to_torchscript(model_path: str, output_dir: Path) -> None:
    """Export model to TorchScript format"""
    print("📦 Exporting to TorchScript format...")
    
    # Load model
    from src.model.transformer import StickFigureTransformer
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = StickFigureTransformer(
        input_dim=MODEL_CONFIG['input_dim'],
        output_dim=MODEL_CONFIG['output_dim'],
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['nhead'],
        num_layers=MODEL_CONFIG['num_layers'],
        dim_feedforward=MODEL_CONFIG['dim_feedforward'],
        dropout=MODEL_CONFIG['dropout'],
        text_embedding_dim=MODEL_CONFIG['text_embedding_dim'],
        num_actions=MODEL_CONFIG['num_actions']
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Trace model
    batch_size = 1
    seq_len = MODEL_CONFIG['sequence_length']
    motion_sequence = torch.randn(batch_size, seq_len, MODEL_CONFIG['input_dim'])
    text_embedding = torch.randn(batch_size, MODEL_CONFIG['text_embedding_dim'])
    action_sequence = torch.randint(0, MODEL_CONFIG['num_actions'], (batch_size, seq_len))
    
    traced_model = torch.jit.trace(model, (motion_sequence, text_embedding, action_sequence))
    
    # Save TorchScript
    output_dir.mkdir(parents=True, exist_ok=True)
    torchscript_path = output_dir / "model.pt"
    traced_model.save(str(torchscript_path))
    
    print(f"✓ Saved to {torchscript_path}")
    print(f"✓ Model size: {torchscript_path.stat().st_size / 1024 / 1024:.2f} MB")


def create_model_card(output_dir: Path, model_name: str = "stick-gen") -> None:
    """Create Hugging Face model card (README.md)"""
    print("📝 Creating model card...")

    model_card = f"""---
license: apache-2.0
tags:
- stick-figure
- animation
- transformer
- text-to-motion
- pytorch
library_name: pytorch
---

# {model_name}

Transformer-based model for generating stick figure animations from text descriptions.

## Model Description

**stick-gen** is a 15.5M parameter transformer model that generates realistic stick figure animations from natural language descriptions. The model uses action-conditioned generation to produce smooth, physics-aware motion sequences.

### Key Features

- **Text-to-Motion**: Generate animations from text prompts
- **Action Conditioning**: Frame-by-frame action control
- **Multi-Task Learning**: Simultaneous pose, position, and velocity prediction
- **Temporal Consistency**: Smooth, realistic motion transitions
- **10-Second Sequences**: 250 frames @ 25fps

### Model Architecture

- **Type**: Transformer Encoder-Decoder
- **Parameters**: 15.5M
- **Embedding**: BAAI/bge-large-en-v1.5 (1024-dim)
- **Hidden Size**: 384
- **Attention Heads**: 12
- **Layers**: 8
- **Sequence Length**: 250 frames (10 seconds @ 25fps)

## Intended Use

### Primary Use Cases

- Generating stick figure animations for educational content
- Prototyping character animations
- Creating motion references for animators
- Research in text-to-motion generation

### Out-of-Scope Uses

- Realistic human animation (use SMPL-based models instead)
- Real-time motion capture
- Medical or safety-critical applications

## Training Data

- **Synthetic Data**: 100,000 procedurally generated sequences
- **AMASS Dataset**: 400,000 real human motion sequences
- **Total**: 500,000 samples with 5x augmentation (2.5M training samples)
- **Actions**: 60 action types (walk, run, jump, wave, etc.)

## Training Procedure

### Hyperparameters

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 with warmup + cosine decay
- **Batch Size**: 16 (effective: 64 with gradient accumulation)
- **Epochs**: 50
- **Loss**: Multi-task (pose + temporal + action)
- **Hardware**: CPU (8 threads)

### Metrics

- **Pose Loss**: MSE on joint positions
- **Action Accuracy**: >80% on validation set
- **Temporal Consistency**: Smooth motion transitions

## Usage

### Installation

```bash
pip install torch safetensors sentence-transformers
```

### Basic Usage

```python
import torch
from safetensors.torch import load_file

# Load model
state_dict = load_file("model.safetensors")
# ... initialize model and load state_dict

# Generate animation
text_prompt = "A person walking forward then jumping"
animation = model.generate(text_prompt)
```

### Action-Conditioned Generation

```python
# Define action sequence
actions = [ActionType.WALK] * 125 + [ActionType.JUMP] * 125

# Generate with explicit action control
animation = model.generate_with_actions(text_prompt, actions)
```

## Limitations

- Limited to stick figure representation (5 lines: head, 2 arms, 2 legs)
- 2D projection only (no depth information)
- Fixed 10-second duration
- Requires text embedding model (BAAI/bge-large-en-v1.5)

## Citation

```bibtex
@software{{stick_gen_2024,
  author = {{Gestura AI}},
  title = {{stick-gen: Text-to-Stick-Figure Animation}},
  year = {{2024}},
  url = {{https://github.com/gestura-ai/stick-gen}}
}}
```

## License

Apache 2.0
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)

    print(f"✓ Created model card: {readme_path}")


def push_to_hub(output_dir: Path, repo_id: str) -> None:
    """Push model to Hugging Face Hub"""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    print(f"🚀 Pushing to Hugging Face Hub: {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")

    # Upload all files
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model"
    )

    print(f"✓ Model pushed to https://huggingface.co/{repo_id}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export stick-gen model to modern formats')
    parser.add_argument('--input', required=True, help='Input .pth checkpoint path')
    parser.add_argument('--output', required=True, help='Output directory name')
    parser.add_argument('--formats', nargs='+', default=['safetensors'],
                       choices=['safetensors', 'onnx', 'torchscript', 'all'],
                       help='Export formats (default: safetensors)')
    parser.add_argument('--push-to-hub', help='Push to Hugging Face Hub (e.g., gestura-ai/stick-gen-v1)')
    parser.add_argument('--model-name', default='stick-gen', help='Model name for card')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output)

    # Determine formats
    formats = args.formats
    if 'all' in formats:
        formats = ['safetensors', 'onnx', 'torchscript']

    print("=" * 70)
    print("Model Export Tool")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Formats: {', '.join(formats)}")
    print("=" * 70)
    print()

    # Export to each format
    if 'safetensors' in formats:
        export_to_safetensors(str(input_path), output_dir)
        create_model_card(output_dir, args.model_name)
        print()

    if 'onnx' in formats:
        export_to_onnx(str(input_path), output_dir)
        print()

    if 'torchscript' in formats:
        export_to_torchscript(str(input_path), output_dir)
        print()

    # Push to Hub if requested
    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub)
        print()

    print("=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print(f"\nExported files in: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Test the exported model")
    print("2. Update inference code to load from new format")
    print("3. Share on Hugging Face Hub (if not already done)")
    print()


if __name__ == "__main__":
    main()

