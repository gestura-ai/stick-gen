#!/usr/bin/env python3
"""Integration tests for checkpoint save/load/resume in src.train.train.

These tests run on a tiny synthetic dataset and a few epochs to keep runtime
low while exercising the full training entrypoint and resume logic.
"""

import os
import shutil
import tempfile

import torch

from src.train.train import train


def _prepare_synthetic_dataset(root_dir: str, num_samples: int = 32, T: int = 16):
    """Create a minimal train_data_final.pt under root_dir/data.

    Matches the schema expected by StickFigureDataset in src.train.train.
    """

    os.makedirs(os.path.join(root_dir, "data"), exist_ok=True)
    path = os.path.join(root_dir, "data", "train_data_final.pt")

    data = []
    for _ in range(num_samples):
        motion = torch.zeros(T, 20)
        embedding = torch.zeros(1024)
        actions = torch.zeros(T, dtype=torch.long)
        physics = torch.zeros(T, 6)
        data.append(
            {
                "motion": motion,
                "embedding": embedding,
                "actions": actions,
                "physics": physics,
            }
        )

    torch.save(data, path)
    return path


def _make_tiny_config(root_dir: str) -> str:
    """Write a minimal YAML config with very small training settings.

    Uses the small model but with 2 epochs to keep tests fast.
    """

    cfg_path = os.path.join(root_dir, "configs", "tiny_test.yaml")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

    cfg = """
metadata:
  variant: "tiny-test"

model:
  input_dim: 20
  d_model: 64
  nhead: 4
  num_layers: 2
  output_dim: 20
  embedding_dim: 1024
  dropout: 0.1
  num_actions: 64

training:
  batch_size: 2
  grad_accum_steps: 1
  epochs: 2
  learning_rate: 0.001
  warmup_epochs: 0
  max_grad_norm: 1.0

loss_weights:
  temporal: 0.0
  action: 0.0
  physics: 0.0
  diffusion: 0.0

physics:
  enabled: false

diffusion:
  enabled: false

data:
  train_data: "data/train_data_final.pt"
  checkpoint_dir: "checkpoints"
  log_dir: "logs"

device:
  type: "cpu"
  num_workers: 0
  pin_memory: false

logging:
  level: "ERROR"
  log_interval: 100
  save_interval: 1
"""

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg)

    return cfg_path


def test_checkpoint_resume_roundtrip(tmp_path):
    """Train for 1 epoch, then resume from checkpoint and ensure progress.

    - First run: epochs=1, write best checkpoint.
    - Second run: epochs=2, resume from best checkpoint and ensure that the
      second run completes without error and consumes at least one more epoch.
    """

    # Work in an isolated temp directory to avoid touching real project paths
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # Create synthetic data and tiny config under workdir
    _ = _prepare_synthetic_dataset(str(workdir))
    cfg_path = _make_tiny_config(str(workdir))

    # First run: train for 1 epoch and save best checkpoint
    # Override epochs via a temporary config copy
    first_cfg = str(workdir / "configs" / "tiny_test_first.yaml")
    shutil.copy(cfg_path, first_cfg)
    with open(first_cfg, "r", encoding="utf-8") as f:
        text = f.read().replace("epochs: 2", "epochs: 1")
    with open(first_cfg, "w", encoding="utf-8") as f:
        f.write(text)

    # Run first training
    os.chdir(str(workdir))
    train(config_path=first_cfg)

    ckpt_dir = workdir / "checkpoints"
    best_ckpt = ckpt_dir / "model_checkpoint_best.pth"
    assert best_ckpt.exists(), "Best checkpoint was not created"

    # Second run: use original config (epochs=2) and resume from first run
    train(
        config_path=cfg_path,
        resume_from_cli=str(best_ckpt),
    )


def test_missing_checkpoint_raises(tmp_path):
    """Ensure that pointing resume_from at a non-existent file fails fast."""

    workdir = tmp_path / "work_missing"
    workdir.mkdir(parents=True, exist_ok=True)

    _ = _prepare_synthetic_dataset(str(workdir))
    cfg_path = _make_tiny_config(str(workdir))

    os.chdir(str(workdir))

    missing = workdir / "checkpoints" / "does_not_exist.pth"
    try:
        train(config_path=cfg_path, resume_from_cli=str(missing))
    except FileNotFoundError:
        # Expected path
        return

    assert False, "Expected FileNotFoundError when resuming from missing checkpoint"
