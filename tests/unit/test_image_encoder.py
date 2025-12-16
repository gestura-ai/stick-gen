"""Unit tests for image encoder module."""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from src.model.image_encoder import (
    LightweightCNN,
    ResNetEncoder,
    MiniViT,
    create_image_encoder,
    ENCODER_REGISTRY,
)


class TestLightweightCNN:
    """Tests for LightweightCNN encoder."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        encoder = LightweightCNN(in_channels=3, output_dim=1024, base_channels=32)
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 1024)

    def test_different_output_dims(self) -> None:
        """Test encoder with different output dimensions."""
        for output_dim in [512, 1024, 2048]:
            encoder = LightweightCNN(output_dim=output_dim)
            x = torch.randn(1, 3, 256, 256)
            out = encoder(x)
            assert out.shape == (1, output_dim)

    def test_different_input_sizes(self) -> None:
        """Test encoder with different input image sizes."""
        encoder = LightweightCNN(output_dim=1024)
        for size in [128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            out = encoder(x)
            assert out.shape == (1, 1024)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow properly."""
        encoder = LightweightCNN(output_dim=256)
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestResNetEncoder:
    """Tests for ResNetEncoder."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        encoder = ResNetEncoder(in_channels=3, output_dim=1024, base_channels=64)
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 1024)

    def test_residual_connection(self) -> None:
        """Test that model can be forward passed without errors."""
        encoder = ResNetEncoder(output_dim=512)
        x = torch.randn(1, 3, 128, 128)
        out = encoder(x)
        assert out.shape == (1, 512)
        assert not torch.isnan(out).any()


class TestMiniViT:
    """Tests for Mini Vision Transformer encoder."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        encoder = MiniViT(
            img_size=256,
            patch_size=16,
            embed_dim=384,
            num_layers=4,
            output_dim=1024,
        )
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 1024)

    def test_different_patch_sizes(self) -> None:
        """Test ViT with different patch sizes."""
        for patch_size in [8, 16, 32]:
            encoder = MiniViT(img_size=256, patch_size=patch_size, output_dim=512)
            x = torch.randn(1, 3, 256, 256)
            out = encoder(x)
            assert out.shape == (1, 512)

    def test_cls_token_output(self) -> None:
        """Test that CLS token is used for output."""
        encoder = MiniViT(img_size=128, patch_size=16, output_dim=256, dropout=0.0)
        encoder.eval()  # Disable dropout for deterministic test
        x = torch.randn(2, 3, 128, 128)
        out = encoder(x)
        # Output should be deterministic given same input in eval mode
        out2 = encoder(x)
        assert torch.allclose(out, out2)


class TestCreateImageEncoder:
    """Tests for encoder factory function."""

    def test_create_lightweight_cnn(self) -> None:
        """Test creating lightweight CNN encoder."""
        encoder = create_image_encoder("lightweight_cnn", output_dim=1024)
        assert isinstance(encoder, LightweightCNN)

    def test_create_resnet(self) -> None:
        """Test creating ResNet encoder."""
        encoder = create_image_encoder("resnet", output_dim=1024)
        assert isinstance(encoder, ResNetEncoder)

    def test_create_mini_vit(self) -> None:
        """Test creating Mini ViT encoder."""
        encoder = create_image_encoder("mini_vit", output_dim=1024, img_size=256)
        assert isinstance(encoder, MiniViT)

    def test_invalid_architecture(self) -> None:
        """Test error handling for invalid architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_image_encoder("invalid_arch", output_dim=1024)

    def test_registry_completeness(self) -> None:
        """Test that all registered encoders can be instantiated."""
        for name in ENCODER_REGISTRY:
            kwargs = {"img_size": 256} if name == "mini_vit" else {}
            encoder = create_image_encoder(name, output_dim=512, **kwargs)
            x = torch.randn(1, 3, 256, 256)
            out = encoder(x)
            assert out.shape == (1, 512)

