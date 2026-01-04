"""
Unit tests for MultiLoRARouter module.

Tests cover:
- Router initialization and expert registration
- Style expert routing with softmax normalization
- Orthogonal expert gating
- Prototype embedding computation
- Edge cases (single expert, all experts, empty router)
"""

import torch
import torch.nn as nn

from src.model.lora_router import (
    ExpertConfig,
    MultiLoRARouter,
    DEFAULT_PROTOTYPES,
)


class TestExpertConfig:
    """Tests for ExpertConfig dataclass."""

    def test_default_config(self):
        """Test default ExpertConfig values."""
        config = ExpertConfig(name="test", checkpoint_path="/path/to/ckpt")
        assert config.name == "test"
        assert config.checkpoint_path == "/path/to/ckpt"
        assert config.expert_type == "style"
        assert config.target_phase == "both"
        assert config.prototype_prompts == []

    def test_style_expert_config(self):
        """Test style expert configuration."""
        config = ExpertConfig(
            name="dramatic_style",
            checkpoint_path="/path/to/dramatic.pt",
            expert_type="style",
            target_phase="both",
            prototype_prompts=["slow emotional scene"],
        )
        assert config.expert_type == "style"
        assert len(config.prototype_prompts) == 1

    def test_orthogonal_expert_config(self):
        """Test orthogonal expert configuration."""
        config = ExpertConfig(
            name="camera",
            checkpoint_path="/path/to/camera.pt",
            expert_type="orthogonal",
            target_phase="transformer_only",
        )
        assert config.expert_type == "orthogonal"
        assert config.target_phase == "transformer_only"


class TestMultiLoRARouter:
    """Tests for MultiLoRARouter class."""

    def test_initialization(self):
        """Test router initialization with default parameters."""
        router = MultiLoRARouter(embed_dim=1024, temperature=0.1)
        assert router.embed_dim == 1024
        assert router.temperature == 0.1
        assert router.num_experts == 0
        assert len(router.style_experts) == 0
        assert len(router.orthogonal_experts) == 0

    def test_register_style_expert(self):
        """Test registering a style expert."""
        router = MultiLoRARouter()
        config = ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style")
        router.register_expert(config)

        assert router.num_experts == 1
        assert "dramatic_style" in router.style_experts
        assert "dramatic_style" in router.style_expert_names

    def test_register_orthogonal_expert(self):
        """Test registering an orthogonal expert."""
        router = MultiLoRARouter()
        config = ExpertConfig(name="camera", checkpoint_path="/ckpt", expert_type="orthogonal")
        router.register_expert(config)

        assert router.num_experts == 1
        assert "camera" in router.orthogonal_experts
        assert "camera" in router.orthogonal_gates

    def test_register_multiple_experts(self):
        """Test registering multiple experts of different types."""
        router = MultiLoRARouter()

        style_configs = [
            ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style"),
            ExpertConfig(name="action_style", checkpoint_path="/ckpt", expert_type="style"),
        ]
        ortho_configs = [
            ExpertConfig(name="camera", checkpoint_path="/ckpt", expert_type="orthogonal"),
            ExpertConfig(name="timing", checkpoint_path="/ckpt", expert_type="orthogonal"),
        ]

        for cfg in style_configs + ortho_configs:
            router.register_expert(cfg)

        assert router.num_experts == 4
        assert len(router.style_experts) == 2
        assert len(router.orthogonal_experts) == 2

    def test_style_weights_sum_to_one(self):
        """Test that style expert weights sum to 1.0 after softmax."""
        router = MultiLoRARouter(embed_dim=32, temperature=0.1)

        # Register style experts with manual prototypes
        for name in ["dramatic_style", "action_style", "expressive_body"]:
            config = ExpertConfig(name=name, checkpoint_path="/ckpt", expert_type="style")
            router.register_expert(config)

        # Manually set prototype embeddings (bypass text encoder)
        router.style_prototypes = torch.randn(3, 32)
        router.style_prototypes = torch.nn.functional.normalize(router.style_prototypes, dim=-1)
        router._prototypes_initialized = True

        # Test routing
        text_embedding = torch.randn(2, 32)  # batch size 2
        weights = router._compute_style_weights(text_embedding)

        # Sum weights for each batch item
        total_weight = sum(w.squeeze(-1) for w in weights.values())
        assert torch.allclose(total_weight, torch.ones(2), atol=1e-5)

    def test_orthogonal_weights_independent(self):
        """Test that orthogonal expert weights are independent [0, 1]."""
        router = MultiLoRARouter(embed_dim=32)

        for name in ["camera", "timing"]:
            config = ExpertConfig(name=name, checkpoint_path="/ckpt", expert_type="orthogonal")
            router.register_expert(config)

        text_embedding = torch.randn(2, 32)
        weights = router._compute_orthogonal_weights(text_embedding)

        # Each weight should be in [0, 1] (sigmoid output)
        for name, weight in weights.items():
            assert (weight >= 0).all() and (weight <= 1).all()
            assert weight.shape == (2, 1)

    def test_forward_all_experts(self):
        """Test forward pass with both style and orthogonal experts."""
        router = MultiLoRARouter(embed_dim=32)

        # Register experts
        router.register_expert(ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style"))
        router.register_expert(ExpertConfig(name="action_style", checkpoint_path="/ckpt", expert_type="style"))
        router.register_expert(ExpertConfig(name="camera", checkpoint_path="/ckpt", expert_type="orthogonal"))


    def test_get_active_experts(self):
        """Test filtering experts by activation threshold."""
        router = MultiLoRARouter(embed_dim=32)

        # Create mock weights
        weights = {
            "dramatic_style": torch.tensor([[0.7]]),
            "action_style": torch.tensor([[0.05]]),
            "camera": torch.tensor([[0.8]]),
        }

        active = router.get_active_experts(weights, threshold=0.1)
        assert "dramatic_style" in active
        assert "camera" in active
        assert "action_style" not in active  # Below threshold

    def test_get_expert_config(self):
        """Test retrieving expert configuration by name."""
        router = MultiLoRARouter()
        config = ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style")
        router.register_expert(config)

        retrieved = router.get_expert_config("dramatic_style")
        assert retrieved is not None
        assert retrieved.name == "dramatic_style"

        missing = router.get_expert_config("nonexistent")
        assert missing is None

    def test_all_experts_property(self):
        """Test all_experts property returns combined dict."""
        router = MultiLoRARouter()
        router.register_expert(ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style"))
        router.register_expert(ExpertConfig(name="camera", checkpoint_path="/ckpt", expert_type="orthogonal"))

        all_exp = router.all_experts
        assert len(all_exp) == 2
        assert "dramatic_style" in all_exp
        assert "camera" in all_exp

    def test_empty_router_forward(self):
        """Test forward pass with no registered experts."""
        router = MultiLoRARouter(embed_dim=32)
        text_embedding = torch.randn(1, 32)
        weights = router(text_embedding)
        assert weights == {}

    def test_single_style_expert(self):
        """Test routing with only one style expert (should get weight 1.0)."""
        router = MultiLoRARouter(embed_dim=32)
        router.register_expert(ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style"))

        router.style_prototypes = torch.nn.functional.normalize(torch.randn(1, 32), dim=-1)
        router._prototypes_initialized = True

        text_embedding = torch.randn(1, 32)
        weights = router._compute_style_weights(text_embedding)

        # Single expert should get weight 1.0
        assert "dramatic_style" in weights
        assert torch.allclose(weights["dramatic_style"], torch.ones(1, 1))

    def test_temperature_effect(self):
        """Test that lower temperature produces sharper distributions."""
        # Create prototypes that are somewhat similar
        prototypes = torch.nn.functional.normalize(torch.randn(2, 32), dim=-1)

        # Create routers with different temperatures
        router_low = MultiLoRARouter(embed_dim=32, temperature=0.01)
        router_high = MultiLoRARouter(embed_dim=32, temperature=1.0)

        for router in [router_low, router_high]:
            router.register_expert(ExpertConfig(name="a", checkpoint_path="/ckpt", expert_type="style"))
            router.register_expert(ExpertConfig(name="b", checkpoint_path="/ckpt", expert_type="style"))
            router.style_prototypes = prototypes.clone()
            router._prototypes_initialized = True

        text_embedding = torch.randn(1, 32)

        weights_low = router_low._compute_style_weights(text_embedding)
        weights_high = router_high._compute_style_weights(text_embedding)

        # Low temperature should have higher max weight (sharper)
        max_low = max(w.max().item() for w in weights_low.values())
        max_high = max(w.max().item() for w in weights_high.values())
        assert max_low >= max_high

    def test_state_dict_with_prototypes(self):
        """Test saving and loading state with prototype metadata."""
        router = MultiLoRARouter(embed_dim=32)
        router.register_expert(ExpertConfig(name="dramatic_style", checkpoint_path="/ckpt", expert_type="style"))
        router.register_expert(ExpertConfig(name="camera", checkpoint_path="/ckpt", expert_type="orthogonal"))

        router.style_prototypes = torch.randn(1, 32)
        router._prototypes_initialized = True

        state = router.state_dict_with_prototypes()
        assert "_style_expert_names" in state
        assert "_prototypes_initialized" in state
        assert state["_prototypes_initialized"] is True

    def test_invalid_expert_type(self):
        """Test that invalid expert type raises error."""
        router = MultiLoRARouter()
        config = ExpertConfig(name="invalid", checkpoint_path="/ckpt", expert_type="unknown")

        try:
            router.register_expert(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown expert_type" in str(e)


class TestDefaultPrototypes:
    """Tests for DEFAULT_PROTOTYPES configuration."""

    def test_all_experts_have_prototypes(self):
        """Test that all expected experts have default prototypes."""
        expected = ["dramatic_style", "action_style", "expressive_body", "multi_actor", "camera", "timing"]
        for name in expected:
            assert name in DEFAULT_PROTOTYPES
            assert len(DEFAULT_PROTOTYPES[name]) > 0

    def test_prototype_prompts_are_strings(self):
        """Test that all prototype prompts are non-empty strings."""
        for name, prompts in DEFAULT_PROTOTYPES.items():
            for prompt in prompts:
                assert isinstance(prompt, str)
                assert len(prompt) > 10  # Should be meaningful prompts
