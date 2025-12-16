"""
Batch Processing Example

Process multiple prompts in batch for efficient generation.

Usage:
    # From file
    python examples/batch_processing.py \
      --checkpoint checkpoints/best_model.pt \
      --config configs/base.yaml \
      --prompts-file prompts.txt \
      --output-dir outputs/

    # From command line
    python examples/batch_processing.py \
      --checkpoint checkpoints/best_model.pt \
      --config configs/base.yaml \
      --prompts "walking" "running" "jumping" \
      --output-dir outputs/
"""

import argparse
import json
import os
import sys
import time
import csv
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer import StickFigureTransformer
from src.train.config import TrainingConfig
from sentence_transformers import SentenceTransformer


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
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")

    return model, config


def load_prompts(prompts_file: str = None, prompts_list: list = None):
    """Load prompts from file or list."""
    if prompts_file:
        print(f"Loading prompts from {prompts_file}...")
        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(prompts)} prompts")
        return prompts
    elif prompts_list:
        return prompts_list
    else:
        raise ValueError("Either prompts_file or prompts_list must be provided")


def batch_generate(
    model,
    prompts: list,
    config: TrainingConfig,
    batch_size: int = 4,
    device: str = "cpu",
):
    """Generate animations for multiple prompts in batches."""
    # Load embedding model
    embedding_model_name = config.get("data.embedding_model", "BAAI/bge-large-en-v1.5")
    print(f"Loading embedding model: {embedding_model_name}...")
    embedding_model = SentenceTransformer(embedding_model_name)

    seq_len = config.get("model.seq_len", 250)
    input_dim = config.get("model.input_dim", 20)
    fps = config.get("data.fps", 25)

    results = []

    print(f"\nGenerating {len(prompts)} animations (batch_size={batch_size})...")

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i : i + batch_size]

        # Generate embeddings for batch
        text_embeddings = embedding_model.encode(batch_prompts, convert_to_tensor=True)
        text_embeddings = text_embeddings.to(device)

        # Create dummy input
        batch_len = len(batch_prompts)
        dummy_input = torch.zeros(batch_len, seq_len, input_dim).to(device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model(dummy_input, text_embeddings)
            pose_output = outputs["pose"]
        generation_time = time.time() - start_time

        # Store results
        for j, prompt in enumerate(batch_prompts):
            results.append(
                {
                    "prompt": prompt,
                    "motion": pose_output[j].cpu().numpy().tolist(),
                    "seq_len": seq_len,
                    "fps": fps,
                    "generation_time": generation_time / batch_len,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process multiple prompts")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--prompts-file", type=str, help="File with prompts (one per line)"
    )
    parser.add_argument("--prompts", nargs="+", help="List of prompts")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for generation"
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validate arguments
    if not args.prompts_file and not args.prompts:
        parser.error("Either --prompts-file or --prompts must be provided")

    # Set random seed
    torch.manual_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Load prompts
    prompts = load_prompts(args.prompts_file, args.prompts)

    # Generate animations
    results = batch_generate(model, prompts, config, args.batch_size, args.device)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual JSON files
    metadata = []
    for i, result in enumerate(results):
        output_file = os.path.join(args.output_dir, f"animation_{i:04d}.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        metadata.append(
            {
                "index": i,
                "prompt": result["prompt"],
                "output_file": output_file,
                "frames": result["seq_len"],
                "duration": result["seq_len"] / result["fps"],
                "generation_time": result["generation_time"],
            }
        )

    # Save metadata CSV
    metadata_file = os.path.join(args.output_dir, "metadata.csv")
    with open(metadata_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\n✅ Generated {len(results)} animations")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Metadata: {metadata_file}")
    print(
        f"   Avg generation time: {sum(r['generation_time'] for r in results) / len(results):.2f}s"
    )


if __name__ == "__main__":
    main()
