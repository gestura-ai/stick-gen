"""
Fine-Tuning Example

Fine-tune a pre-trained model on custom datasets.

Usage:
    python examples/fine_tuning.py \
      --checkpoint checkpoints/best_model.pt \
      --config configs/base.yaml \
      --data custom_data.pt \
      --output checkpoints/fine_tuned.pt \
      --epochs 10 \
      --learning-rate 1e-5
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer import StickFigureTransformer
from src.train.config import TrainingConfig


def load_model(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load pre-trained model from checkpoint."""
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

    print(f"Loading pre-trained checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")

    return model, config


def load_custom_data(data_path: str, batch_size: int = 8):
    """Load custom dataset."""
    print(f"Loading custom data from {data_path}...")
    data = torch.load(data_path)

    # Assuming data is a dict with 'motion', 'embedding', 'actions' keys
    motions = data["motion"]
    embeddings = data["embedding"]
    actions = data.get("actions", None)

    print(f"  Loaded {len(motions)} samples")

    # Create dataset
    if actions is not None:
        dataset = TensorDataset(motions, embeddings, actions)
    else:
        dataset = TensorDataset(motions, embeddings)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def fine_tune(
    model, dataloader, epochs: int, learning_rate: float, device: str = "cpu"
):
    """Fine-tune the model."""
    print(f"\nFine-tuning for {epochs} epochs (lr={learning_rate})...")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            if len(batch) == 3:
                motions, embeddings, actions = batch
            else:
                motions, embeddings = batch
                actions = None

            motions = motions.to(device)
            embeddings = embeddings.to(device)
            if actions is not None:
                actions = actions.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(motions, embeddings)

            # Compute loss
            loss = criterion(outputs["pose"], motions)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    print("✅ Fine-tuning complete")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained model on custom data"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to pre-trained checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to custom dataset (.pt file)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output checkpoint path"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Load custom data
    dataloader = load_custom_data(args.data, args.batch_size)

    # Fine-tune
    model = fine_tune(model, dataloader, args.epochs, args.learning_rate, args.device)

    # Save fine-tuned model
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fine_tuned": True,
            "base_checkpoint": args.checkpoint,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        args.output,
    )

    print(f"\n✅ Fine-tuned model saved to: {args.output}")


if __name__ == "__main__":
    main()
