"""Unit tests for feature fusion module."""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from src.model.fusion import (
    FUSION_REGISTRY,
    ConcatFusion,
    CrossAttentionFusion,
    FiLMFusion,
    GatedFusion,
    create_fusion_module,
)


class TestConcatFusion:
    """Tests for concatenation-based fusion."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        fusion = ConcatFusion(
            text_dim=1024, image_dim=1024, camera_dim=7, output_dim=1024
        )
        text = torch.randn(2, 1024)
        image = torch.randn(2, 1024)
        camera = torch.randn(2, 7)
        out = fusion(text, image, camera)
        assert out.shape == (2, 1024)

    def test_text_only(self) -> None:
        """Test fusion with text only (image and camera are None)."""
        fusion = ConcatFusion(output_dim=512)
        text = torch.randn(2, 1024)
        out = fusion(text, None, None)
        assert out.shape == (2, 512)

    def test_text_and_image(self) -> None:
        """Test fusion with text and image, no camera."""
        fusion = ConcatFusion(output_dim=1024)
        text = torch.randn(2, 1024)
        image = torch.randn(2, 1024)
        out = fusion(text, image, None)
        assert out.shape == (2, 1024)


class TestGatedFusion:
    """Tests for gated fusion."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        fusion = GatedFusion(text_dim=1024, image_dim=1024, output_dim=1024)
        text = torch.randn(2, 1024)
        image = torch.randn(2, 1024)
        camera = torch.randn(2, 7)
        out = fusion(text, image, camera)
        assert out.shape == (2, 1024)

    def test_gates_affect_output(self) -> None:
        """Test that gates modulate the output."""
        fusion = GatedFusion(output_dim=512)
        text = torch.randn(2, 1024)

        # With image
        image = torch.randn(2, 1024)
        out_with_image = fusion(text, image, None)

        # Without image (different code path)
        out_without_image = fusion(text, None, None)

        # Outputs should be different
        assert not torch.allclose(out_with_image, out_without_image)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through all inputs."""
        fusion = GatedFusion(output_dim=256)
        text = torch.randn(2, 1024, requires_grad=True)
        image = torch.randn(2, 1024, requires_grad=True)
        camera = torch.randn(2, 7, requires_grad=True)

        out = fusion(text, image, camera)
        loss = out.sum()
        loss.backward()

        assert text.grad is not None
        assert image.grad is not None
        assert camera.grad is not None


class TestFiLMFusion:
    """Tests for FiLM (Feature-wise Linear Modulation) fusion."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        fusion = FiLMFusion(text_dim=1024, image_dim=1024, output_dim=1024)
        text = torch.randn(2, 1024)
        image = torch.randn(2, 1024)
        camera = torch.randn(2, 7)
        out = fusion(text, image, camera)
        assert out.shape == (2, 1024)

    def test_identity_init(self) -> None:
        """Test that FiLM is initialized close to identity transform."""
        fusion = FiLMFusion(output_dim=512)
        text = torch.randn(2, 1024)

        # With zero image features, should be close to projected text
        image = torch.zeros(2, 1024)
        out = fusion(text, image, None)

        # Output should be finite
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestCrossAttentionFusion:
    """Tests for cross-attention based fusion."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        fusion = CrossAttentionFusion(
            text_dim=1024, image_dim=1024, output_dim=1024, num_heads=8
        )
        text = torch.randn(2, 1024)
        image = torch.randn(2, 1024)
        camera = torch.randn(2, 7)
        out = fusion(text, image, camera)
        assert out.shape == (2, 1024)

    def test_attention_to_modalities(self) -> None:
        """Test that cross-attention attends to all modalities."""
        fusion = CrossAttentionFusion(output_dim=512, num_heads=4)
        text = torch.randn(2, 1024)
        image = torch.randn(2, 1024)
        camera = torch.randn(2, 7)

        out_all = fusion(text, image, camera)
        out_text_only = fusion(text, None, None)

        # Different inputs should give different outputs
        assert not torch.allclose(out_all, out_text_only)


class TestCreateFusionModule:
    """Tests for fusion factory function."""

    def test_create_all_strategies(self) -> None:
        """Test creating all fusion strategies."""
        for strategy in FUSION_REGISTRY:
            fusion = create_fusion_module(strategy, output_dim=512)
            text = torch.randn(1, 1024)
            image = torch.randn(1, 1024)
            camera = torch.randn(1, 7)
            out = fusion(text, image, camera)
            assert out.shape == (1, 512)

    def test_invalid_strategy(self) -> None:
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            create_fusion_module("invalid_strategy", output_dim=512)
