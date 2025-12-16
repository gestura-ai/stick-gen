#!/usr/bin/env python3
"""
Stick-Gen Training Script for RunPod
Gestura AI - https://gestura.ai

Optimized training script for cloud GPU environments with:
- Model variant selection (small/base/large)
- Checkpoint resumption
- Camera conditioning support
- Wandb logging (optional)
- Automatic checkpointing

Usage:
    python scripts/train_runpod.py --model-variant base --epochs 50
    python scripts/train_runpod.py --resume-from checkpoints/model_checkpoint.pth
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer import StickFigureTransformer
from src.data_gen.schema import NUM_ACTIONS
from src.train.config import TrainingConfig


class StickFigureDataset(Dataset):
    """Dataset for stick figure motion sequences."""

    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item["motion"]
        embedding = item["embedding"]
        actions = item.get("actions", torch.zeros(motion.shape[0], dtype=torch.long))
        physics = item.get("physics", torch.zeros(motion.shape[0], 6))
        camera = item.get("camera", torch.zeros(motion.shape[0], 3))

        # Input: frames [0:-1], Target: frames [1:]
        return (
            motion[:-1],
            embedding,
            motion[1:],
            actions[:-1],
            physics[:-1],
            camera[:-1],
        )


def temporal_consistency_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Penalize large frame-to-frame changes for smooth motion."""
    frame_diff = predictions[1:] - predictions[:-1]
    return torch.mean(frame_diff**2)


def physics_loss(physics_output: torch.Tensor, physics_targets: torch.Tensor) -> tuple:
    """Physics-aware loss with gravity, momentum, and consistency constraints."""
    pred_vx, pred_vy = physics_output[:, :, 0], physics_output[:, :, 1]
    pred_ax, pred_ay = physics_output[:, :, 2], physics_output[:, :, 3]
    pred_mx, pred_my = physics_output[:, :, 4], physics_output[:, :, 5]

    # MSE loss
    mse_loss = nn.MSELoss()(physics_output, physics_targets)

    # Gravity constraint
    gravity_loss = torch.mean((pred_ay - (-9.8)) ** 2)

    # Momentum conservation
    momentum_diff = (pred_mx[1:] - pred_mx[:-1]) ** 2 + (
        pred_my[1:] - pred_my[:-1]
    ) ** 2
    momentum_loss = torch.mean(momentum_diff)

    # Velocity-acceleration consistency
    dt = 1.0 / 25.0
    expected_vx = pred_vx[:-1] + pred_ax[:-1] * dt
    expected_vy = pred_vy[:-1] + pred_ay[:-1] * dt
    consistency_loss = torch.mean(
        (pred_vx[1:] - expected_vx) ** 2 + (pred_vy[1:] - expected_vy) ** 2
    )

    total = mse_loss + 0.1 * gravity_loss + 0.1 * momentum_loss + 0.2 * consistency_loss

    return total, {
        "physics_mse": mse_loss.item(),
        "gravity_loss": gravity_loss.item(),
        "momentum_loss": momentum_loss.item(),
        "consistency_loss": consistency_loss.item(),
    }


def train_epoch(model, loader, optimizer, device, config, epoch, grad_accum_steps=32):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (motion, embedding, targets, actions, physics, camera) in enumerate(
        pbar
    ):
        motion = motion.permute(1, 0, 2).to(device)  # [seq, batch, dim]
        embedding = embedding.to(device)
        targets = targets.permute(1, 0, 2).to(device)
        actions = actions.to(device)
        physics = physics.to(device)
        camera = camera.permute(1, 0, 2).to(device)  # [seq, batch, 3]

        # Forward pass with camera conditioning
        outputs = model(
            motion,
            embedding,
            return_all_outputs=True,
            action_sequence=actions,
            camera_data=camera,
        )

        # Compute losses
        pose_loss = nn.MSELoss()(outputs["pose"], targets)
        temporal_loss = temporal_consistency_loss(outputs["pose"])

        loss = pose_loss + 0.1 * temporal_loss

        # Action loss
        if "action_logits" in outputs:
            action_logits = outputs["action_logits"].permute(1, 2, 0)
            action_loss = nn.CrossEntropyLoss()(action_logits, actions)
            loss += 0.15 * action_loss

        # Physics loss
        if "physics" in outputs:
            physics_targets = physics.permute(1, 0, 2)
            phys_loss, _ = physics_loss(outputs["physics"], physics_targets)
            loss += 0.2 * phys_loss

        # Gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    return total_loss / len(loader)


def validate(model, loader, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for motion, embedding, targets, actions, physics, camera in loader:
            motion = motion.permute(1, 0, 2).to(device)
            embedding = embedding.to(device)
            targets = targets.permute(1, 0, 2).to(device)
            camera = camera.permute(1, 0, 2).to(device)

            outputs = model(
                motion, embedding, return_all_outputs=True, camera_data=camera
            )
            loss = nn.MSELoss()(outputs["pose"], targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, loss, path, config):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Stick-Gen on RunPod")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="Model variant to train",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train_data_final.pt",
        help="Path to training data",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="stick-gen", help="W&B project name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Stick-Gen Training - RunPod Edition")
    print("by Gestura AI")
    print("=" * 60)

    # Load config
    config_path = f"configs/{args.model_variant}.yaml"
    config = TrainingConfig(config_path)
    print(f"\nLoaded config: {config_path}")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"Device: {device} (WARNING: No GPU detected)")

    # Model parameters
    model_config = {
        "input_dim": config.get("model.input_dim", 20),
        "d_model": config.get("model.d_model", 384),
        "nhead": config.get("model.nhead", 12),
        "num_layers": config.get("model.num_layers", 8),
        "output_dim": config.get("model.output_dim", 20),
        "embedding_dim": config.get("model.embedding_dim", 1024),
        "dropout": config.get("model.dropout", 0.1),
        "num_actions": config.get("model.num_actions", NUM_ACTIONS),
    }

    # Training parameters
    batch_size = args.batch_size or config.get("training.batch_size", 2)
    learning_rate = args.lr or config.get("training.learning_rate", 0.0003)
    grad_accum = config.get("training.grad_accum_steps", 32)
    warmup_epochs = config.get("training.warmup_epochs", 10)

    print(f"\nModel: {args.model_variant}")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  layers: {model_config['num_layers']}")
    print(f"  heads: {model_config['nhead']}")
    print(f"\nTraining:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  grad_accum: {grad_accum}")
    print(f"  effective_batch: {batch_size * grad_accum}")
    print(f"  learning_rate: {learning_rate}")

    # Initialize model
    model = StickFigureTransformer(**model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # LR scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed at epoch {start_epoch}")

    # Load data
    print(f"\nLoading data: {args.data_path}")
    dataset = StickFigureDataset(args.data_path)

    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # W&B logging
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config={
                "model_variant": args.model_variant,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                **model_config,
            },
        )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_val_loss = float("inf")
    training_history = []

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, config, epoch, grad_accum
        )
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}"
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        if args.wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                f"{args.checkpoint_dir}/best_model.pth",
                model_config,
            )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pth",
                model_config,
            )

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs - 1,
        val_loss,
        f"{args.checkpoint_dir}/model_checkpoint.pth",
        model_config,
    )

    # Save training history
    with open(f"{args.checkpoint_dir}/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
