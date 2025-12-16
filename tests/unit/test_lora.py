"""
Unit tests for LoRA (Low-Rank Adaptation) module.

Tests cover:
- LoRALinear layer forward pass and initialization
- LoRA adapter injection into models
- Parameter freezing and counting
- LoRA weight merging
- LoRA state dict extraction and loading
"""

import torch
import torch.nn as nn

from src.model.lora import (
    LoRALinear,
    count_lora_parameters,
    freeze_base_model,
    get_lora_parameters,
    get_lora_state_dict,
    inject_lora_adapters,
)


class TestLoRALinear:
    """Tests for the LoRALinear layer."""

    def test_lora_linear_initialization(self):
        """Test LoRALinear initializes correctly."""
        base_layer = nn.Linear(64, 32)
        lora_layer = LoRALinear(base_layer, rank=8, alpha=16.0, dropout=0.0)

        assert lora_layer.rank == 8
        assert lora_layer.alpha == 16.0
        assert lora_layer.scaling == 2.0  # alpha / rank
        assert lora_layer.lora_A.shape == (8, 64)
        assert lora_layer.lora_B.shape == (32, 8)

    def test_lora_linear_forward_pass(self):
        """Test LoRALinear forward pass produces correct output shape."""
        base_layer = nn.Linear(64, 32)
        lora_layer = LoRALinear(base_layer, rank=8, alpha=16.0)

        x = torch.randn(4, 64)  # batch of 4
        output = lora_layer(x)

        assert output.shape == (4, 32)

    def test_lora_starts_as_identity(self):
        """Test that LoRA starts as identity (B initialized to zeros)."""
        base_layer = nn.Linear(64, 32, bias=False)
        lora_layer = LoRALinear(base_layer, rank=8, alpha=16.0, dropout=0.0)

        x = torch.randn(4, 64)
        base_output = base_layer(x)
        lora_output = lora_layer(x)

        # Since B is initialized to zeros, LoRA contribution should be zero
        # So lora_output should equal base_output
        torch.testing.assert_close(lora_output, base_output)

    def test_lora_base_layer_frozen(self):
        """Test that base layer parameters are frozen after LoRA wrapping."""
        base_layer = nn.Linear(64, 32)
        lora_layer = LoRALinear(base_layer, rank=8, alpha=16.0)

        # Base layer should be frozen
        assert not lora_layer.base_layer.weight.requires_grad
        if lora_layer.base_layer.bias is not None:
            assert not lora_layer.base_layer.bias.requires_grad

        # LoRA parameters should be trainable
        assert lora_layer.lora_A.requires_grad
        assert lora_layer.lora_B.requires_grad

    def test_lora_merge_weights(self):
        """Test merging LoRA weights into base layer."""
        base_layer = nn.Linear(64, 32, bias=False)
        original_weight = base_layer.weight.clone()

        lora_layer = LoRALinear(base_layer, rank=8, alpha=16.0, dropout=0.0)

        # Modify LoRA weights
        lora_layer.lora_B.data.fill_(1.0)

        # Merge weights
        lora_layer.merge_weights()

        # Base weight should have changed
        assert not torch.allclose(lora_layer.base_layer.weight, original_weight)


class TestLoRAInjection:
    """Tests for LoRA adapter injection into models."""

    def test_inject_lora_into_simple_model(self):
        """Test injecting LoRA into a simple model."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Inject LoRA into all Linear layers
        num_modified = inject_lora_adapters(
            model,
            target_modules=["0", "2"],  # Match layer indices
            rank=4,
            alpha=8.0,
        )

        assert num_modified == 2
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)

    def test_inject_lora_with_regex_pattern(self):
        """Test injecting LoRA using regex patterns."""
        model = nn.ModuleDict(
            {
                "encoder": nn.Linear(64, 32),
                "decoder": nn.Linear(32, 16),
                "other": nn.Linear(16, 8),
            }
        )

        # Only inject into encoder and decoder
        num_modified = inject_lora_adapters(
            model,
            target_modules=["encoder", "decoder"],
            rank=4,
            alpha=8.0,
        )

        assert num_modified == 2
        assert isinstance(model["encoder"], LoRALinear)
        assert isinstance(model["decoder"], LoRALinear)
        assert isinstance(model["other"], nn.Linear)  # Not modified


class TestLoRAParameterManagement:
    """Tests for LoRA parameter freezing and counting."""

    def test_freeze_base_model(self):
        """Test freezing base model parameters."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Inject LoRA
        inject_lora_adapters(model, target_modules=["0", "2"], rank=4)

        # Freeze base model
        freeze_base_model(model)

        # Check that base parameters are frozen
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_count_lora_parameters(self):
        """Test counting LoRA parameters."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 16),
        )

        inject_lora_adapters(model, target_modules=["0", "1"], rank=4)
        freeze_base_model(model)

        trainable, total, lora_params = count_lora_parameters(model)

        # LoRA params: layer 0: (4*64 + 32*4) = 384, layer 1: (4*32 + 16*4) = 192
        expected_lora = 384 + 192
        assert lora_params == expected_lora
        assert trainable == lora_params

    def test_get_lora_parameters(self):
        """Test getting only LoRA parameters."""
        model = nn.Sequential(nn.Linear(64, 32))
        inject_lora_adapters(model, target_modules=["0"], rank=4)

        lora_params = list(get_lora_parameters(model))
        assert len(lora_params) == 2  # lora_A and lora_B


class TestLoRAStateDictManagement:
    """Tests for LoRA state dict extraction and loading."""

    def test_get_lora_state_dict(self):
        """Test extracting LoRA state dict."""
        model = nn.Sequential(nn.Linear(64, 32))
        inject_lora_adapters(model, target_modules=["0"], rank=4)

        lora_state = get_lora_state_dict(model)

        assert len(lora_state) == 2
        assert any("lora_A" in k for k in lora_state.keys())
        assert any("lora_B" in k for k in lora_state.keys())
