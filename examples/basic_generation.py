"""
Basic Generation Example

Generate stick figure animations from text prompts using a trained model.

Usage:
    python examples/basic_generation.py \
      --checkpoint checkpoints/best_model.pt \
      --config configs/base.yaml \
      --prompt "A person walking forward" \
      --output outputs/walking.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer

from src.model.transformer import StickFigureTransformer
from src.train.config import TrainingConfig


def load_model(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load trained model from checkpoint."""
    print(f"Loading configuration from {config_path}...")
    config = TrainingConfig(config_path)

    # Extract model parameters
    input_dim = config.get("model.input_dim", 20)
    d_model = config.get("model.d_model", 384)
    nhead = config.get("model.nhead", 12)
    num_layers = config.get("model.num_layers", 8)
    output_dim = config.get("model.output_dim", 20)
    embedding_dim = config.get("model.embedding_dim", 1024)
    num_actions = config.get("model.num_actions", 64)

    print(f"Initializing model ({d_model}d, {num_layers} layers, {nhead} heads)...")
    model = StickFigureTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        num_actions=num_actions,
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")

    return model, config


def generate_animation(model, prompt: str, config: TrainingConfig, device: str = "cpu"):
    """Generate animation from text prompt."""
    print(f"\nGenerating animation for: '{prompt}'")

    # Load embedding model
    embedding_model_name = config.get("data.embedding_model", "BAAI/bge-large-en-v1.5")
    print(f"Loading embedding model: {embedding_model_name}...")
    embedding_model = SentenceTransformer(embedding_model_name)

    # Generate text embedding
    print("Encoding prompt...")
    text_embedding = embedding_model.encode([prompt], convert_to_tensor=True)
    text_embedding = text_embedding.to(device)

    # Generate motion
    print("Generating motion...")
    with torch.no_grad():
        # Create dummy input (will be replaced by model's autoregressive generation)
        seq_len = config.get("model.seq_len", 250)
        input_dim = config.get("model.input_dim", 20)
        dummy_input = torch.zeros(1, seq_len, input_dim).to(device)

        # Forward pass
        outputs = model(dummy_input, text_embedding)

        # Extract pose predictions
        pose_output = outputs["pose"]  # Shape: (batch, seq_len, output_dim)

    print(f"✅ Generated {seq_len} frames")

    return {
        "prompt": prompt,
        "motion": pose_output.cpu().numpy().tolist(),
        "seq_len": seq_len,
        "fps": config.get("data.fps", 25),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate stick figure animation from text prompt"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Generate animation
    result = generate_animation(model, args.prompt, config, args.device)

    # Save output
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Animation saved to: {args.output}")
    print(f"   Frames: {result['seq_len']}")
    print(f"   FPS: {result['fps']}")
    print(f"   Duration: {result['seq_len'] / result['fps']:.2f}s")


if __name__ == "__main__":
    main()
