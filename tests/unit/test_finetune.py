"""Unit tests for the fine-tuning pipeline.

Tests cover:
- Style experts: dramatic_style, action_style, expressive_body, multi_actor
- Orthogonal experts: camera (transformer-only), timing (diffusion-only)
- Phase-specific LoRA injection
"""

import pytest
import torch

from src.train.finetune import (
    FinetuneConfig,
    LoRAConfig,
    DomainConfig,
    DomainFinetuner,
)


class TestFinetuneConfig:
    """Tests for FinetuneConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FinetuneConfig()

        assert config.batch_size == 4
        assert config.epochs == 20
        assert config.learning_rate == 1e-4
        assert config.input_dim == 48
        assert config.d_model == 384
        assert config.lora.enabled is True
        assert config.lora.rank == 8
        assert config.lora.target_phase == "both"  # Default phase targeting
        assert config.domain.name == "general"

    def test_load_dramatic_style_config(self):
        """Test loading dramatic style expert config from YAML."""
        config = FinetuneConfig.from_yaml("configs/finetune/dramatic_style.yaml")

        assert config.domain.name == "dramatic_style"
        assert config.lora.rank == 16  # Higher rank for expressive nuance
        assert config.lora.target_phase == "both"  # Style expert
        assert config.epochs == 25
        assert "stand" in config.domain.emphasized_actions
        assert config.diffusion_weight == 0.18

    def test_load_action_style_config(self):
        """Test loading action style expert config from YAML."""
        config = FinetuneConfig.from_yaml("configs/finetune/action_style.yaml")

        assert config.domain.name == "action_style"
        assert config.domain.physics_weight == 0.35  # Higher for action
        assert config.lora.target_phase == "both"  # Style expert
        assert "run" in config.domain.emphasized_actions

    def test_load_expressive_body_config(self):
        """Test loading expressive body expert config from YAML."""
        config = FinetuneConfig.from_yaml("configs/finetune/expressive_body.yaml")

        assert config.domain.name == "expressive_body"
        assert config.lora.rank == 14
        assert config.lora.target_phase == "both"  # Style expert
        assert config.fusion_strategy == "cross_attention"
        assert "wave" in config.domain.emphasized_actions

    def test_load_multi_actor_config(self):
        """Test loading multi-actor expert config from YAML."""
        config = FinetuneConfig.from_yaml("configs/finetune/multi_actor.yaml")

        assert config.domain.name == "multi_actor"
        assert config.batch_size == 2  # Smaller for multi-actor complexity
        assert config.lora.target_phase == "both"  # Style expert
        assert "handshake" in config.domain.emphasized_actions

    def test_load_camera_config(self):
        """Test loading camera orthogonal expert config from YAML."""
        config = FinetuneConfig.from_yaml("configs/finetune/camera.yaml")

        assert config.domain.name == "camera"
        assert config.lora.target_phase == "transformer_only"  # Orthogonal expert
        assert config.enable_diffusion is False  # Camera doesn't use diffusion
        assert len(config.lora.diffusion_targets) == 0  # Empty diffusion targets
        assert "camera_projection" in config.lora.transformer_targets[0]

    def test_load_timing_config(self):
        """Test loading timing orthogonal expert config from YAML."""
        config = FinetuneConfig.from_yaml("configs/finetune/timing.yaml")

        assert config.domain.name == "timing"
        assert config.lora.target_phase == "diffusion_only"  # Orthogonal expert
        assert config.enable_diffusion is True  # Timing requires diffusion
        assert len(config.lora.transformer_targets) == 0  # Empty transformer targets
        assert config.diffusion_weight == 1.0  # Full weight on diffusion


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_lora_config(self):
        """Test default LoRA configuration."""
        lora = LoRAConfig()

        assert lora.enabled is True
        assert lora.rank == 8
        assert lora.alpha == 16.0
        assert lora.dropout == 0.05
        assert lora.target_phase == "both"  # Default phase targeting
        assert len(lora.transformer_targets) > 0
        assert len(lora.diffusion_targets) > 0

    def test_phase_targeting_options(self):
        """Test that all phase targeting options are valid."""
        # Style experts use "both"
        style_lora = LoRAConfig(target_phase="both")
        assert style_lora.target_phase == "both"

        # Camera expert uses transformer_only
        camera_lora = LoRAConfig(target_phase="transformer_only")
        assert camera_lora.target_phase == "transformer_only"

        # Timing expert uses diffusion_only
        timing_lora = LoRAConfig(target_phase="diffusion_only")
        assert timing_lora.target_phase == "diffusion_only"

    def test_orthogonal_expert_config(self):
        """Test configuration for orthogonal experts."""
        # Camera expert: transformer-only with camera-specific targets
        camera_lora = LoRAConfig(
            target_phase="transformer_only",
            transformer_targets=[r"camera_projection\.\d+"],
            diffusion_targets=[],  # Empty for orthogonal
        )
        assert len(camera_lora.diffusion_targets) == 0

        # Timing expert: diffusion-only
        timing_lora = LoRAConfig(
            target_phase="diffusion_only",
            transformer_targets=[],  # Empty for orthogonal
            diffusion_targets=[r"encoder.*conv", r"decoder.*conv"],
        )
        assert len(timing_lora.transformer_targets) == 0


class TestDomainConfig:
    """Tests for DomainConfig dataclass."""
    
    def test_default_domain_config(self):
        """Test default domain configuration."""
        domain = DomainConfig()
        
        assert domain.name == "general"
        assert domain.expected_smoothness == 0.7
        assert domain.temporal_weight == 0.1
        assert domain.physics_weight == 0.2
    
    def test_custom_domain_config(self):
        """Test custom domain configuration."""
        domain = DomainConfig(
            name="custom",
            expected_velocity_range=(0.1, 0.5),
            emphasized_actions=["jump", "spin"],
        )
        
        assert domain.name == "custom"
        assert domain.expected_velocity_range == (0.1, 0.5)
        assert "jump" in domain.emphasized_actions


class TestDomainFinetuner:
    """Tests for DomainFinetuner class."""
    
    def test_device_setup_cpu(self):
        """Test CPU device setup."""
        config = FinetuneConfig()
        
        # Create finetuner with non-existent checkpoint (will use random init)
        finetuner = DomainFinetuner(
            config=config,
            base_checkpoint="nonexistent.pt",
            device="cpu",
        )
        
        assert finetuner.device == torch.device("cpu")
        assert finetuner.model is not None
    
    def test_lora_injection(self):
        """Test LoRA adapter injection with default 'both' phase."""
        config = FinetuneConfig()
        config.enable_diffusion = False  # Disable diffusion for faster test

        finetuner = DomainFinetuner(
            config=config,
            base_checkpoint="nonexistent.pt",
            device="cpu",
        )

        # Check that LoRA parameters exist
        lora_params = list(finetuner.optimizer.param_groups[0]["params"])
        assert len(lora_params) > 0

    def test_transformer_only_lora_injection(self):
        """Test LoRA injection for transformer-only expert (e.g., camera)."""
        config = FinetuneConfig()
        config.enable_diffusion = False
        config.lora.target_phase = "transformer_only"

        finetuner = DomainFinetuner(
            config=config,
            base_checkpoint="nonexistent.pt",
            device="cpu",
        )

        # Check that only transformer LoRA parameters exist
        lora_params = list(finetuner.optimizer.param_groups[0]["params"])
        assert len(lora_params) > 0
        # Diffusion is disabled, so all params should be from transformer

    def test_diffusion_only_lora_injection(self):
        """Test LoRA injection for diffusion-only expert (e.g., timing)."""
        config = FinetuneConfig()
        config.enable_diffusion = True  # Required for diffusion-only
        config.lora.target_phase = "diffusion_only"

        finetuner = DomainFinetuner(
            config=config,
            base_checkpoint="nonexistent.pt",
            device="cpu",
        )

        # Check that diffusion LoRA parameters exist
        lora_params = list(finetuner.optimizer.param_groups[0]["params"])
        assert len(lora_params) > 0
        # All trainable params should be from diffusion UNet
    
    def test_compute_domain_loss(self):
        """Test domain-aware loss computation."""
        config = FinetuneConfig()
        config.enable_diffusion = False
        
        finetuner = DomainFinetuner(
            config=config,
            base_checkpoint="nonexistent.pt",
            device="cpu",
        )
        
        # Create dummy outputs and targets
        seq_len, batch_size, dim = 50, 2, 48
        outputs = {
            "pose": torch.randn(seq_len, batch_size, dim),
            "action_logits": torch.randn(seq_len, batch_size, 64),
            "physics": torch.randn(seq_len, batch_size, 6),
        }
        targets = torch.randn(seq_len, batch_size, dim)
        actions = torch.randint(0, 64, (seq_len, batch_size))
        physics = torch.randn(seq_len, batch_size, 6)
        
        loss, components = finetuner.compute_domain_loss(
            outputs, targets, actions, physics
        )
        
        assert loss.ndim == 0  # Scalar
        assert "pose" in components
        assert "temporal" in components
        assert "action" in components
        assert "physics" in components
        assert "total" in components

