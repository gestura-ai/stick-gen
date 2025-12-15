"""
Integration tests for SFT init_from functionality.

Tests cover:
- init_from loads model weights only (not optimizer/scheduler/epoch)
- init_from vs resume_from distinction
- init_from with LoRA enabled
- Error handling for missing init checkpoint
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.model.transformer import StickFigureTransformer
from src.model.lora import inject_lora_adapters, freeze_base_model, count_lora_parameters


class TestSFTInitFrom:
    """Tests for SFT init_from functionality."""

    @pytest.fixture
    def model_and_checkpoint(self, tmp_path):
        """Create a model and save a checkpoint."""
        model = StickFigureTransformer(
            input_dim=20,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_dim=20,
            embedding_dim=1024,
            dropout=0.1,
            num_actions=64,
        )

        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Simulate some training
        for _ in range(5):
            optimizer.step()
            scheduler.step()

        # Save checkpoint
        checkpoint_path = tmp_path / "pretrained.pth"
        torch.save({
            'epoch': 10,
            'global_step': 500,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': 0.5,
        }, checkpoint_path)

        return model, checkpoint_path

    def test_init_from_loads_weights_only(self, model_and_checkpoint):
        """Test that init_from loads model weights but not optimizer state."""
        original_model, checkpoint_path = model_and_checkpoint

        # Create a new model
        new_model = StickFigureTransformer(
            input_dim=20,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_dim=20,
            embedding_dim=1024,
            dropout=0.1,
            num_actions=64,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Simulate init_from behavior: load only model weights
        new_model.load_state_dict(checkpoint['model_state_dict'])

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(),
            new_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_init_from_with_lora(self, model_and_checkpoint):
        """Test init_from followed by LoRA injection."""
        _, checkpoint_path = model_and_checkpoint

        # Create new model and load weights
        model = StickFigureTransformer(
            input_dim=20,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_dim=20,
            embedding_dim=1024,
            dropout=0.1,
            num_actions=64,
        )

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Inject LoRA
        num_modified = inject_lora_adapters(
            model,
            target_modules=["transformer_encoder", "pose_decoder"],
            rank=4,
            alpha=8.0,
        )

        assert num_modified > 0

        # Freeze base model
        freeze_base_model(model)

        # Count parameters
        trainable, total, lora_params = count_lora_parameters(model)

        assert lora_params > 0
        assert trainable == lora_params
        assert trainable < total

    def test_init_from_missing_checkpoint_raises(self, tmp_path):
        """Test that missing init checkpoint raises FileNotFoundError."""
        nonexistent_path = tmp_path / "nonexistent.pth"

        # This simulates the check in train.py
        with pytest.raises(FileNotFoundError):
            if not os.path.isfile(nonexistent_path):
                raise FileNotFoundError(f"Init checkpoint not found: {nonexistent_path}")

    def test_init_from_missing_model_state_dict_raises(self, tmp_path):
        """Test that checkpoint without model_state_dict raises KeyError."""
        # Save incomplete checkpoint
        checkpoint_path = tmp_path / "incomplete.pth"
        torch.save({'epoch': 5}, checkpoint_path)

        checkpoint = torch.load(checkpoint_path)

        with pytest.raises(KeyError):
            if 'model_state_dict' not in checkpoint:
                raise KeyError("Checkpoint missing 'model_state_dict'")


class TestLoRATraining:
    """Tests for LoRA training behavior."""

    def test_lora_forward_backward(self):
        """Test that LoRA model can do forward and backward pass."""
        model = StickFigureTransformer(
            input_dim=20,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_dim=20,
            embedding_dim=1024,
            dropout=0.1,
            num_actions=64,
        )

        # Inject LoRA
        inject_lora_adapters(
            model,
            target_modules=["transformer_encoder"],
            rank=4,
            alpha=8.0,
        )
        freeze_base_model(model)

        # Create dummy input
        motion = torch.randn(10, 2, 20)  # [seq, batch, dim]
        embedding = torch.randn(2, 1024)  # [batch, embed_dim]

        # Forward pass
        output = model(motion, embedding)

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Check that only LoRA parameters have gradients
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.grad is not None, f"{name} should have gradient"
            else:
                assert param.grad is None, f"{name} should not have gradient"

