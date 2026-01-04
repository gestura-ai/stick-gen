"""
Production-oriented fine-tuning pipeline for Stick-Gen.

This module provides parameter-efficient fine-tuning using LoRA for:
- The autoregressive transformer (motion planning)
- The diffusion UNet (motion refinement)

Supports production-oriented expert fine-tuning with two expert categories:

1. Style Experts (routed based on content):
   - dramatic_style: Emotional pacing, slower tempo, expressive movements
   - action_style: High energy, fast cuts, dynamic motion
   - expressive_body: Body dynamics and natural motion quality
   - multi_actor: Multi-actor coordination and scene blocking

2. Orthogonal Experts (always active, different output spaces):
   - camera: Cinematic framing, shot composition (transformer-only)
   - timing: Dramatic pacing, holds, timing beats (diffusion-only)

Usage:
    python -m src.train.finetune \
        --config configs/finetune/dramatic_style.yaml \
        --base_checkpoint checkpoints/pretrain/medium_best.pth \
        --data_path data/processed/canonical/domain_dramatic.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

import yaml

from src.model.transformer import StickFigureTransformer
from src.model.diffusion import (
    DiffusionRefinementModule,
    PoseRefinementUNet,
    DDPMScheduler,
    StyleCondition,
)
from src.model.lora import (
    inject_lora_adapters,
    freeze_base_model,
    get_lora_parameters,
    get_lora_state_dict,
    count_lora_parameters,
)
from src.train.parallax_dataset import MultimodalParallaxDataset
from src.eval.metrics import (
    compute_motion_temporal_metrics,
    compute_motion_realism_score,
    compute_synthetic_artifact_score,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LoRAConfig:
    """LoRA adapter configuration with phase-specific targeting.

    Phase targeting allows creating orthogonal experts that operate on
    specific components of the dual-phase architecture:
    - transformer_only: Only inject LoRA into transformer (e.g., camera expert)
    - diffusion_only: Only inject LoRA into diffusion UNet (e.g., timing expert)
    - both (default): Inject into both components (style experts)
    """

    enabled: bool = True
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    # Phase targeting for orthogonal experts
    # Options: "both", "transformer_only", "diffusion_only"
    target_phase: str = "both"

    # Regex patterns for target modules
    transformer_targets: list[str] = field(default_factory=lambda: [
        r"transformer_encoder\.layers\.\d+\.self_attn\.(in_proj|out_proj)",
        r"transformer_encoder\.layers\.\d+\.linear[12]",
        r"pose_decoder\.\d+",
        r"text_projection\.\d+",
    ])
    diffusion_targets: list[str] = field(default_factory=lambda: [
        r"encoder.*conv",
        r"decoder.*conv",
        r"bottleneck.*conv",
    ])


@dataclass
class DomainConfig:
    """Domain-specific fine-tuning configuration."""

    name: str = "general"
    description: str = "General motion fine-tuning"

    # Motion characteristics for this domain
    expected_velocity_range: tuple[float, float] = (0.05, 0.3)
    expected_smoothness: float = 0.7
    temporal_weight: float = 0.1
    physics_weight: float = 0.2

    # Domain-specific action categories to emphasize
    emphasized_actions: list[str] = field(default_factory=list)
    action_weight: float = 0.15

    # Multimodal conditioning weights
    image_condition_weight: float = 0.3  # Weight for image-conditioned samples
    text_only_weight: float = 0.7        # Weight for text-only samples


@dataclass
class FinetuneConfig:
    """Complete fine-tuning configuration."""

    # Training settings
    batch_size: int = 4
    grad_accum_steps: int = 8
    epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # LoRA settings
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Domain settings
    domain: DomainConfig = field(default_factory=DomainConfig)

    # Model architecture (should match base model)
    input_dim: int = 48
    d_model: int = 384
    nhead: int = 12
    num_layers: int = 8
    embedding_dim: int = 1024
    num_actions: int = 64

    # Multimodal settings
    enable_image_conditioning: bool = True
    image_encoder_arch: str = "lightweight_cnn"
    fusion_strategy: str = "gated"
    image_size: tuple[int, int] = (256, 256)

    # Diffusion settings
    enable_diffusion: bool = True
    diffusion_timesteps: int = 1000
    diffusion_weight: float = 0.1

    # Data settings
    train_split: float = 0.9
    num_workers: int = 0

    # Output settings
    checkpoint_dir: str = "checkpoints/finetune"
    save_every: int = 5
    log_every: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested configs
        lora_data = data.pop("lora", {})
        domain_data = data.pop("domain", {})

        return cls(
            lora=LoRAConfig(**lora_data) if lora_data else LoRAConfig(),
            domain=DomainConfig(**domain_data) if domain_data else DomainConfig(),
            **data
        )


# =============================================================================
# Fine-tuning Trainer
# =============================================================================


class DomainFinetuner:
    """Domain-specific fine-tuning trainer with LoRA adapters.

    This trainer:
    1. Loads a pretrained base model (transformer + optional diffusion)
    2. Injects LoRA adapters into specified layers
    3. Freezes base model weights, trains only LoRA parameters
    4. Supports multimodal conditioning (text + image)
    5. Tracks domain-specific metrics
    """

    def __init__(
        self,
        config: FinetuneConfig,
        base_checkpoint: str,
        diffusion_checkpoint: str | None = None,
        device: str = "auto",
    ):
        """Initialize the fine-tuner.

        Args:
            config: Fine-tuning configuration
            base_checkpoint: Path to pretrained transformer checkpoint
            diffusion_checkpoint: Optional path to pretrained diffusion UNet
            device: Device to use ("auto", "cpu", "cuda", "mps")
        """
        self.config = config
        self.device = self._setup_device(device)

        # Initialize models
        self.model = self._load_transformer(base_checkpoint)
        self.diffusion = self._load_diffusion(diffusion_checkpoint) if config.enable_diffusion else None

        # Inject LoRA adapters
        if config.lora.enabled:
            self._inject_lora_adapters()

        # Setup optimizer (only LoRA params if enabled)
        self.optimizer = self._setup_optimizer()
        self.scheduler = None  # Set during training

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_transformer(self, checkpoint_path: str) -> StickFigureTransformer:
        """Load pretrained transformer model."""
        logger.info(f"Loading transformer from {checkpoint_path}")

        model = StickFigureTransformer(
            input_dim=self.config.input_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            output_dim=self.config.input_dim,
            embedding_dim=self.config.embedding_dim,
            dropout=0.1,  # Keep dropout for regularization
            num_actions=self.config.num_actions,
            enable_image_conditioning=self.config.enable_image_conditioning,
            image_encoder_arch=self.config.image_encoder_arch,
            image_size=self.config.image_size,
            fusion_strategy=self.config.fusion_strategy,
        )

        # Load pretrained weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded pretrained transformer weights")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using random init")

        return model.to(self.device)

    def _load_diffusion(self, checkpoint_path: str | None) -> DiffusionRefinementModule | None:
        """Load pretrained diffusion module.

        Creates a DiffusionRefinementModule with:
        - PoseRefinementUNet: The UNet for denoising
        - DDPMScheduler: The noise scheduler
        """
        # Create UNet and scheduler
        unet = PoseRefinementUNet(
            pose_dim=self.config.input_dim,
            use_style_conditioning=True,
        )
        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion_timesteps)

        # Create the diffusion module
        diffusion = DiffusionRefinementModule(
            unet=unet,
            scheduler=scheduler,
            device=str(self.device),
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading diffusion from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("diffusion_state_dict", checkpoint)
            diffusion.unet.load_state_dict(state_dict, strict=False)
            logger.info("Loaded pretrained diffusion weights")
        else:
            logger.info("No diffusion checkpoint provided, initializing fresh")

        return diffusion

    def _inject_lora_adapters(self) -> None:
        """Inject LoRA adapters into transformer and/or diffusion models.

        Supports phase-specific injection based on lora.target_phase:
        - "both": Inject into both transformer and diffusion (default for style experts)
        - "transformer_only": Only inject into transformer (e.g., camera expert)
        - "diffusion_only": Only inject into diffusion UNet (e.g., timing expert)
        """
        lora_cfg = self.config.lora
        target_phase = lora_cfg.target_phase

        # Validate target_phase
        valid_phases = {"both", "transformer_only", "diffusion_only"}
        if target_phase not in valid_phases:
            raise ValueError(
                f"Invalid target_phase: {target_phase}. Must be one of {valid_phases}"
            )

        num_transformer = 0
        num_diffusion = 0

        # Inject into transformer (unless diffusion_only)
        if target_phase in {"both", "transformer_only"}:
            num_transformer = inject_lora_adapters(
                self.model,
                target_modules=lora_cfg.transformer_targets,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                dropout=lora_cfg.dropout,
            )
            logger.info(f"Injected LoRA into {num_transformer} transformer layers")

        # Inject into diffusion UNet (unless transformer_only)
        if target_phase in {"both", "diffusion_only"} and self.diffusion is not None:
            num_diffusion = inject_lora_adapters(
                self.diffusion.unet,
                target_modules=lora_cfg.diffusion_targets,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                dropout=lora_cfg.dropout,
            )
            logger.info(f"Injected LoRA into {num_diffusion} diffusion layers")

        # Freeze base model, keep LoRA trainable
        freeze_base_model(self.model)
        if self.diffusion is not None:
            freeze_base_model(self.diffusion.unet)

        # Log parameter counts and phase targeting
        logger.info(f"LoRA target phase: {target_phase}")

        if num_transformer > 0:
            trainable, total, lora_params = count_lora_parameters(self.model)
            logger.info(
                f"Transformer: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.2f}%)"
            )

        if num_diffusion > 0 and self.diffusion is not None:
            t2, tot2, l2 = count_lora_parameters(self.diffusion.unet)
            logger.info(
                f"Diffusion: {t2:,} trainable / {tot2:,} total "
                f"({100*t2/tot2:.2f}%)"
            )

    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer for LoRA parameters based on target phase.

        Respects the lora.target_phase setting to only include parameters
        from the targeted phase(s).
        """
        params: list[torch.nn.Parameter] = []

        if self.config.lora.enabled:
            target_phase = self.config.lora.target_phase

            # Collect transformer LoRA params (unless diffusion_only)
            if target_phase in {"both", "transformer_only"}:
                params.extend(get_lora_parameters(self.model))

            # Collect diffusion LoRA params (unless transformer_only)
            if target_phase in {"both", "diffusion_only"} and self.diffusion is not None:
                params.extend(get_lora_parameters(self.diffusion.unet))
        else:
            # Full fine-tuning (all parameters)
            params = list(self.model.parameters())
            if self.diffusion is not None:
                params.extend(self.diffusion.unet.parameters())

        if not params:
            logger.warning("No trainable parameters found! Check target_phase and model.")
            # Add a dummy parameter to avoid optimizer error
            params = [torch.nn.Parameter(torch.zeros(1))]

        return AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

    def compute_domain_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        actions: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute domain-aware multi-task loss.

        Args:
            outputs: Model outputs dict with 'pose', 'action_logits', 'physics', etc.
            targets: Ground truth motion [seq, batch, dim]
            actions: Ground truth action labels [seq, batch]
            physics: Ground truth physics [seq, batch, 6]

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        domain = self.config.domain
        loss_components = {}

        # 1. Main pose reconstruction loss
        pose_loss = F.mse_loss(outputs["pose"], targets)
        loss_components["pose"] = pose_loss.item()

        # 2. Temporal consistency loss (velocity smoothness)
        if outputs["pose"].shape[0] > 1:
            pred_vel = outputs["pose"][1:] - outputs["pose"][:-1]
            target_vel = targets[1:] - targets[:-1]
            temporal_loss = F.mse_loss(pred_vel, target_vel)
        else:
            temporal_loss = torch.tensor(0.0, device=self.device)
        loss_components["temporal"] = temporal_loss.item()

        # 3. Action prediction loss (if actions provided)
        action_loss = torch.tensor(0.0, device=self.device)
        if actions is not None and "action_logits" in outputs:
            action_logits = outputs["action_logits"]  # [seq, batch, num_actions]
            # Reshape for cross entropy
            logits_flat = action_logits.view(-1, self.config.num_actions)
            actions_flat = actions.view(-1)
            action_loss = F.cross_entropy(logits_flat, actions_flat)
        loss_components["action"] = action_loss.item()

        # 4. Physics consistency loss (if physics provided)
        physics_loss = torch.tensor(0.0, device=self.device)
        if physics is not None and "physics" in outputs:
            physics_loss = F.mse_loss(outputs["physics"], physics)
        loss_components["physics"] = physics_loss.item()

        # Combine with domain-specific weights
        total_loss = (
            pose_loss
            + domain.temporal_weight * temporal_loss
            + domain.action_weight * action_loss
            + domain.physics_weight * physics_loss
        )

        loss_components["total"] = total_loss.item()
        return total_loss, loss_components

    def compute_diffusion_loss(
        self,
        motion: torch.Tensor,
        style: StyleCondition | None = None,
    ) -> torch.Tensor:
        """Compute diffusion refinement loss.

        Args:
            motion: Ground truth motion [batch, seq, dim]
            style: Optional style conditioning

        Returns:
            Diffusion loss tensor
        """
        if self.diffusion is None:
            return torch.tensor(0.0, device=self.device)

        # Add noise and predict
        return self.diffusion.compute_loss(motion, style_condition=style)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        if self.diffusion is not None:
            self.diffusion.unet.train()

        epoch_metrics = {
            "loss": 0.0,
            "pose_loss": 0.0,
            "temporal_loss": 0.0,
            "diffusion_loss": 0.0,
        }
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch
            motion = batch["motion"].to(self.device)  # [batch, seq, dim]
            embedding = batch["embedding"].to(self.device)  # [batch, embed_dim]

            # Optional fields
            actions = batch.get("actions")
            if actions is not None:
                actions = actions.to(self.device)

            physics = batch.get("physics")
            if physics is not None:
                physics = physics.to(self.device)

            # Optional image conditioning
            image_tensor = batch.get("image")
            if image_tensor is not None:
                image_tensor = image_tensor.to(self.device)

            camera_pose = batch.get("camera_pose")
            if camera_pose is not None:
                camera_pose = camera_pose.to(self.device)

            # Transpose for transformer: [batch, seq, dim] -> [seq, batch, dim]
            motion_t = motion.transpose(0, 1)

            # Forward pass
            outputs = self.model(
                motion_t,
                embedding,
                return_all_outputs=True,
                action_sequence=actions.transpose(0, 1) if actions is not None else None,
                image_tensor=image_tensor,
                image_camera_pose=camera_pose,
            )

            # Compute transformer loss
            targets = motion_t  # Teacher forcing
            trans_loss, loss_components = self.compute_domain_loss(
                outputs, targets, actions, physics
            )

            # Compute diffusion loss
            diff_loss = self.compute_diffusion_loss(motion)

            # Combined loss
            total_loss = trans_loss + self.config.diffusion_weight * diff_loss

            # Scale for gradient accumulation
            scaled_loss = total_loss / self.config.grad_accum_steps
            scaled_loss.backward()

            # Update metrics
            epoch_metrics["loss"] += total_loss.item()
            epoch_metrics["pose_loss"] += loss_components["pose"]
            epoch_metrics["temporal_loss"] += loss_components["temporal"]
            epoch_metrics["diffusion_loss"] += diff_loss.item()
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if self.diffusion is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.diffusion.unet.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every == 0:
                    avg_loss = epoch_metrics["loss"] / num_batches
                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f}"
                    )

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation and compute metrics.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        if self.diffusion is not None:
            self.diffusion.unet.eval()

        val_metrics = {
            "val_loss": 0.0,
            "val_pose_loss": 0.0,
            "realism_score": 0.0,
            "smoothness_score": 0.0,
            "artifact_score": 0.0,
        }
        num_batches = 0

        for batch in val_loader:
            motion = batch["motion"].to(self.device)
            embedding = batch["embedding"].to(self.device)

            motion_t = motion.transpose(0, 1)

            outputs = self.model(motion_t, embedding, return_all_outputs=True)

            # Compute loss
            pose_loss = F.mse_loss(outputs["pose"], motion_t)
            val_metrics["val_loss"] += pose_loss.item()
            val_metrics["val_pose_loss"] += pose_loss.item()

            # Compute quality metrics on predictions
            pred_motion = outputs["pose"].transpose(0, 1)  # [batch, seq, dim]
            for i in range(pred_motion.shape[0]):
                sample = pred_motion[i]  # [seq, dim]

                temporal = compute_motion_temporal_metrics(sample)
                realism = compute_motion_realism_score(sample)
                artifacts = compute_synthetic_artifact_score(sample)

                val_metrics["smoothness_score"] += temporal["smoothness_score"]
                val_metrics["realism_score"] += realism["realism_score"]
                val_metrics["artifact_score"] += artifacts["artifact_score"]

            num_batches += 1

        # Average
        batch_size = val_loader.batch_size or 1
        total_samples = num_batches * batch_size
        for key in val_metrics:
            if key in ["smoothness_score", "realism_score", "artifact_score"]:
                val_metrics[key] /= max(total_samples, 1)
            else:
                val_metrics[key] /= max(num_batches, 1)

        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save training checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training/validation metrics
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config,
            "metrics": metrics,
        }

        # Save LoRA weights only if enabled
        if self.config.lora.enabled:
            checkpoint["lora_state_dict"] = get_lora_state_dict(self.model)
            if self.diffusion is not None:
                checkpoint["diffusion_lora_state_dict"] = get_lora_state_dict(
                    self.diffusion.unet
                )
        else:
            checkpoint["model_state_dict"] = self.model.state_dict()
            if self.diffusion is not None:
                checkpoint["diffusion_state_dict"] = self.diffusion.unet.state_dict()

        # Save optimizer state
        checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        # Save regular checkpoint
        domain_name = self.config.domain.name
        checkpoint_path = checkpoint_dir / f"{domain_name}_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / f"{domain_name}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

        return str(checkpoint_path)

    def train(
        self,
        train_data_path: str,
        val_split: float = 0.1,
    ) -> dict[str, Any]:
        """Run full training loop.

        Args:
            train_data_path: Path to training data (.pt file)
            val_split: Fraction of data to use for validation

        Returns:
            Dictionary of final training metrics
        """
        logger.info(f"Loading training data from {train_data_path}")

        # Load dataset
        dataset = MultimodalParallaxDataset(
            data_path=train_data_path,
            parallax_root=None,  # No parallax augmentation during fine-tuning
            image_size=self.config.image_size,
        )

        # Split into train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        logger.info(f"Train: {train_size}, Val: {val_size}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        # Setup scheduler
        total_steps = len(train_loader) // self.config.grad_accum_steps * self.config.epochs
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 4,
            T_mult=2,
        )

        # Training loop
        logger.info(f"Starting training for {self.config.epochs} epochs")
        history = {"train": [], "val": []}

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train"].append(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader)
            history["val"].append(val_metrics)

            epoch_time = time.time() - epoch_start

            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Realism: {val_metrics['realism_score']:.3f}"
            )

            # Save checkpoint
            is_best = val_metrics["val_loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["val_loss"]

            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    is_best=is_best,
                )

        logger.info("Training complete!")
        return history



# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Stick-Gen model with LoRA adapters"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to fine-tuning config YAML",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        required=True,
        help="Path to pretrained transformer checkpoint",
    )
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained diffusion checkpoint (optional)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (.pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = FinetuneConfig.from_yaml(args.config)
    logger.info(f"Loaded config for domain: {config.domain.name}")

    # Initialize trainer
    trainer = DomainFinetuner(
        config=config,
        base_checkpoint=args.base_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        device=args.device,
    )

    # Run training
    history = trainer.train(
        train_data_path=args.data_path,
        val_split=args.val_split,
    )

    # Log final metrics
    final_train = history["train"][-1]
    final_val = history["val"][-1]
    logger.info(
        f"Final metrics - Train Loss: {final_train['loss']:.4f}, "
        f"Val Loss: {final_val['val_loss']:.4f}, "
        f"Realism: {final_val['realism_score']:.3f}"
    )


if __name__ == "__main__":
    main()

