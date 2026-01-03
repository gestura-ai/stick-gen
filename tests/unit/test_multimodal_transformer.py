"""Integration tests for multimodal transformer with image conditioning."""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from src.model.transformer import StickFigureTransformer


class TestMultimodalTransformer:
    """Tests for transformer with image conditioning enabled."""

    @pytest.fixture
    def multimodal_model(self):
        """Create a small multimodal transformer for testing."""
        return StickFigureTransformer(
            input_dim=48,
            d_model=64,  # Small for fast testing
            nhead=4,
            num_layers=2,
            output_dim=48,
            embedding_dim=128,  # Small embedding dim
            dropout=0.0,  # Disable dropout for deterministic tests
            num_actions=8,
            enable_image_conditioning=True,
            image_encoder_arch="lightweight_cnn",
            image_size=(64, 64),  # Small images for fast testing
            fusion_strategy="gated",
        )

    @pytest.fixture
    def motion_only_model(self):
        """Create a motion-only transformer (no image conditioning)."""
        return StickFigureTransformer(
            input_dim=48,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_dim=48,
            embedding_dim=128,
            dropout=0.0,
            num_actions=8,
            enable_image_conditioning=False,
        )

    def test_multimodal_forward_with_image(self, multimodal_model) -> None:
        """Test forward pass with image tensor."""
        batch_size = 2
        seq_len = 10

        motion = torch.randn(seq_len, batch_size, 48)
        text_emb = torch.randn(batch_size, 128)
        image = torch.randn(batch_size, 3, 64, 64)
        camera_pose = torch.randn(batch_size, 7)

        outputs = multimodal_model(
            motion,
            text_emb,
            return_all_outputs=True,
            image_tensor=image,
            image_camera_pose=camera_pose,
        )

        assert "pose" in outputs
        assert outputs["pose"].shape == (seq_len, batch_size, 48)

    def test_multimodal_forward_without_image(self, multimodal_model) -> None:
        """Test forward pass without image (should still work)."""
        batch_size = 2
        seq_len = 10

        motion = torch.randn(seq_len, batch_size, 48)
        text_emb = torch.randn(batch_size, 128)

        # Should work without image
        outputs = multimodal_model(
            motion,
            text_emb,
            return_all_outputs=True,
            image_tensor=None,
            image_camera_pose=None,
        )

        assert "pose" in outputs
        assert outputs["pose"].shape == (seq_len, batch_size, 48)

    def test_motion_only_backward_compatible(self, motion_only_model) -> None:
        """Test that motion-only model still works (backward compatibility)."""
        batch_size = 2
        seq_len = 10

        motion = torch.randn(seq_len, batch_size, 48)
        text_emb = torch.randn(batch_size, 128)

        outputs = motion_only_model(
            motion,
            text_emb,
            return_all_outputs=True,
        )

        assert "pose" in outputs
        assert outputs["pose"].shape == (seq_len, batch_size, 48)

    def test_image_affects_output(self, multimodal_model) -> None:
        """Test that image input affects the model output."""
        batch_size = 2
        seq_len = 5

        motion = torch.randn(seq_len, batch_size, 48)
        text_emb = torch.randn(batch_size, 128)
        image1 = torch.randn(batch_size, 3, 64, 64)
        image2 = torch.randn(batch_size, 3, 64, 64)  # Different image

        out1 = multimodal_model(motion, text_emb, image_tensor=image1)
        out2 = multimodal_model(motion, text_emb, image_tensor=image2)

        # Different images should produce different outputs
        assert not torch.allclose(out1, out2)

    def test_gradient_flow_multimodal(self, multimodal_model) -> None:
        """Test that gradients flow through image encoder."""
        batch_size = 2
        seq_len = 5

        motion = torch.randn(seq_len, batch_size, 48)
        text_emb = torch.randn(batch_size, 128, requires_grad=True)
        image = torch.randn(batch_size, 3, 64, 64, requires_grad=True)
        camera_pose = torch.randn(batch_size, 7, requires_grad=True)

        outputs = multimodal_model(
            motion,
            text_emb,
            return_all_outputs=True,
            image_tensor=image,
            image_camera_pose=camera_pose,
        )

        loss = outputs["pose"].sum()
        loss.backward()

        # Gradients should flow to all inputs
        assert text_emb.grad is not None
        assert image.grad is not None
        assert camera_pose.grad is not None

    def test_all_fusion_strategies(self) -> None:
        """Test transformer with different fusion strategies."""
        for strategy in ["concat", "gated", "film", "cross_attention"]:
            model = StickFigureTransformer(
                input_dim=48,
                d_model=32,
                nhead=2,
                num_layers=1,
                output_dim=48,
                embedding_dim=64,
                num_actions=4,
                enable_image_conditioning=True,
                image_encoder_arch="lightweight_cnn",
                image_size=(64, 64),  # Larger image to avoid batchnorm issue
                fusion_strategy=strategy,
            )
            model.eval()  # Disable batchnorm running stats requirement

            motion = torch.randn(5, 1, 48)
            text_emb = torch.randn(1, 64)
            image = torch.randn(1, 3, 64, 64)

            out = model(motion, text_emb, image_tensor=image)
            assert out.shape == (5, 1, 48), f"Failed for strategy: {strategy}"

    def test_all_encoder_architectures(self) -> None:
        """Test transformer with different image encoder architectures."""
        for arch in ["lightweight_cnn", "resnet"]:
            model = StickFigureTransformer(
                input_dim=48,
                d_model=32,
                nhead=2,
                num_layers=1,
                output_dim=48,
                embedding_dim=64,
                num_actions=4,
                enable_image_conditioning=True,
                image_encoder_arch=arch,
                image_size=(64, 64),
                fusion_strategy="gated",
            )

            motion = torch.randn(5, 1, 48)
            text_emb = torch.randn(1, 64)
            image = torch.randn(1, 3, 64, 64)

            out = model(motion, text_emb, image_tensor=image)
            assert out.shape == (5, 1, 48), f"Failed for architecture: {arch}"
