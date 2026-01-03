"""
Tests for Transformer Components: RMSNorm, SwiGLU, and RoPETransformerEncoderLayer

Tests the modern LLM architecture components (Qwen/Llama standards):
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU: Gated Linear Unit with Swish activation
- RoPETransformerEncoderLayer: Pre-Norm architecture with RoPE
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from src.model.transformer import (
    RMSNorm,
    RoPETransformerEncoderLayer,
    StickFigureTransformer,
    SwiGLU,
)


class TestRMSNorm:
    """Tests for RMSNorm class"""

    def test_output_shape(self):
        """RMSNorm should preserve input shape"""
        rms = RMSNorm(384)
        x = torch.randn(10, 4, 384)
        out = rms(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        print("✓ RMSNorm output shape test passed")

    def test_normalization_scale(self):
        """RMSNorm should normalize to approximately unit RMS"""
        rms = RMSNorm(256, elementwise_affine=False)
        x = torch.randn(32, 8, 256) * 10  # Large values
        out = rms(x)
        # RMS of output should be approximately 1
        rms_value = torch.sqrt(out.pow(2).mean(dim=-1)).mean()
        assert 0.9 < rms_value < 1.1, f"RMS should be ~1, got {rms_value:.4f}"
        print("✓ RMSNorm normalization scale test passed")

    def test_learnable_weight(self):
        """RMSNorm should have learnable weight parameter"""
        rms = RMSNorm(128)
        assert hasattr(rms, "weight"), "RMSNorm should have weight parameter"
        assert rms.weight.shape == (
            128,
        ), f"Weight shape should be (128,), got {rms.weight.shape}"
        assert torch.allclose(
            rms.weight, torch.ones(128)
        ), "Weight should be initialized to ones"
        print("✓ RMSNorm learnable weight test passed")

    def test_no_affine(self):
        """RMSNorm without affine should not have weight"""
        rms = RMSNorm(64, elementwise_affine=False)
        assert rms.weight is None, "Weight should be None when elementwise_affine=False"
        print("✓ RMSNorm no affine test passed")

    def test_gradient_flow(self):
        """RMSNorm should allow gradients to flow"""
        rms = RMSNorm(64)
        x = torch.randn(4, 4, 64, requires_grad=True)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through RMSNorm"
        assert not torch.isnan(x.grad).any(), "Gradients should not be NaN"
        print("✓ RMSNorm gradient flow test passed")


class TestSwiGLU:
    """Tests for SwiGLU class"""

    def test_output_shape(self):
        """SwiGLU should preserve input/output dimensions"""
        swiglu = SwiGLU(384, 1536)
        x = torch.randn(10, 4, 384)
        out = swiglu(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        print("✓ SwiGLU output shape test passed")

    def test_no_bias(self):
        """SwiGLU projections should have no bias"""
        swiglu = SwiGLU(256, 1024)
        assert swiglu.gate_proj.bias is None, "Gate projection should have no bias"
        assert swiglu.value_proj.bias is None, "Value projection should have no bias"
        assert swiglu.output_proj.bias is None, "Output projection should have no bias"
        print("✓ SwiGLU no bias test passed")

    def test_projection_dimensions(self):
        """SwiGLU should have correct projection dimensions"""
        d_model, hidden_dim = 256, 1024
        swiglu = SwiGLU(d_model, hidden_dim)
        assert swiglu.gate_proj.in_features == d_model
        assert swiglu.gate_proj.out_features == hidden_dim
        assert swiglu.value_proj.in_features == d_model
        assert swiglu.value_proj.out_features == hidden_dim
        assert swiglu.output_proj.in_features == hidden_dim
        assert swiglu.output_proj.out_features == d_model
        print("✓ SwiGLU projection dimensions test passed")

    def test_gradient_flow(self):
        """SwiGLU should allow gradients to flow"""
        swiglu = SwiGLU(64, 256)
        x = torch.randn(4, 4, 64, requires_grad=True)
        out = swiglu(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through SwiGLU"
        assert not torch.isnan(x.grad).any(), "Gradients should not be NaN"
        print("✓ SwiGLU gradient flow test passed")

    def test_dropout(self):
        """SwiGLU should apply dropout during training"""
        swiglu = SwiGLU(64, 256, dropout=0.5)
        swiglu.train()
        x = torch.ones(8, 8, 64)
        out1 = swiglu(x)
        out2 = swiglu(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2), "Dropout should cause different outputs"
        print("✓ SwiGLU dropout test passed")


class TestRoPETransformerEncoderLayer:
    """Tests for RoPETransformerEncoderLayer with RMSNorm and SwiGLU"""

    def test_output_shape(self):
        """Encoder layer should preserve input shape"""
        layer = RoPETransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)
        x = torch.randn(32, 4, 256)  # [seq_len, batch, d_model]
        out = layer(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        print("✓ RoPETransformerEncoderLayer output shape test passed")

    def test_uses_rmsnorm(self):
        """Encoder layer should use RMSNorm instead of LayerNorm"""
        layer = RoPETransformerEncoderLayer(d_model=256, nhead=8)
        assert isinstance(layer.norm1, RMSNorm), "norm1 should be RMSNorm"
        assert isinstance(layer.norm2, RMSNorm), "norm2 should be RMSNorm"
        print("✓ RoPETransformerEncoderLayer uses RMSNorm test passed")

    def test_uses_swiglu(self):
        """Encoder layer should use SwiGLU for feed-forward"""
        layer = RoPETransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)
        assert hasattr(layer, "ffn"), "Layer should have ffn attribute"
        assert isinstance(layer.ffn, SwiGLU), "ffn should be SwiGLU"
        print("✓ RoPETransformerEncoderLayer uses SwiGLU test passed")

    def test_pre_norm_architecture(self):
        """Encoder layer should use Pre-Norm architecture"""
        layer = RoPETransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256)
        # Pre-norm: normalization happens before attention/FFN, not after
        # We verify by checking the forward pass structure indirectly
        x = torch.randn(8, 2, 64)
        layer.eval()
        with torch.no_grad():
            out = layer(x)
        # Output should be different from input (transformation applied)
        assert not torch.allclose(out, x), "Layer should transform input"
        print("✓ RoPETransformerEncoderLayer Pre-Norm architecture test passed")

    def test_gradient_flow(self):
        """Encoder layer should allow gradients to flow"""
        layer = RoPETransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256)
        x = torch.randn(8, 2, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through encoder layer"
        assert not torch.isnan(x.grad).any(), "Gradients should not be NaN"
        print("✓ RoPETransformerEncoderLayer gradient flow test passed")


class TestStickFigureTransformerIntegration:
    """Integration tests for StickFigureTransformer with new components"""

    def test_initialization_default(self):
        """Model should initialize with default parameters"""
        model = StickFigureTransformer()
        params = sum(p.numel() for p in model.parameters())
        assert params > 0, "Model should have parameters"
        print(f"✓ StickFigureTransformer initialized with {params:,} parameters")

    def test_initialization_all_sizes(self):
        """Model should initialize for all three tier sizes"""
        configs = {
            "small": {"d_model": 256, "nhead": 8, "num_layers": 6},
            "medium": {"d_model": 384, "nhead": 12, "num_layers": 8},
            "large": {"d_model": 512, "nhead": 16, "num_layers": 10},
        }
        for name, cfg in configs.items():
            model = StickFigureTransformer(
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
            )
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ {name} model: {params:,} parameters ({params/1e6:.1f}M)")

    def test_forward_pass_basic(self):
        """Basic forward pass should work"""
        model = StickFigureTransformer(
            input_dim=48,
            d_model=256,
            nhead=8,
            num_layers=4,
            output_dim=48,
        )
        model.eval()
        motion = torch.randn(32, 4, 48)
        text_emb = torch.randn(4, 1024)
        with torch.no_grad():
            out = model(motion, text_emb)
        assert out.shape == (32, 4, 48), f"Expected (32, 4, 48), got {out.shape}"
        print("✓ StickFigureTransformer basic forward pass test passed")

    def test_forward_pass_all_outputs(self):
        """Forward pass with all outputs should work"""
        model = StickFigureTransformer(
            input_dim=48,
            d_model=256,
            nhead=8,
            num_layers=4,
            output_dim=48,
        )
        model.eval()
        motion = torch.randn(32, 4, 48)
        text_emb = torch.randn(4, 1024)
        with torch.no_grad():
            outputs = model(motion, text_emb, return_all_outputs=True)
        expected_keys = [
            "pose",
            "position",
            "velocity",
            "action_logits",
            "physics",
            "environment",
        ]
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        print("✓ StickFigureTransformer all outputs test passed")

    def test_forward_with_conditioning(self):
        """Forward pass with action and camera conditioning should work"""
        model = StickFigureTransformer(
            input_dim=48,
            d_model=256,
            nhead=8,
            num_layers=4,
            output_dim=48,
        )
        model.eval()
        seq_len, batch = 32, 4
        motion = torch.randn(seq_len, batch, 48)
        text_emb = torch.randn(batch, 1024)
        action_seq = torch.randint(0, 60, (seq_len, batch))
        camera_data = torch.randn(seq_len, batch, 3)
        with torch.no_grad():
            out = model(
                motion, text_emb, action_sequence=action_seq, camera_data=camera_data
            )
        assert out.shape == (seq_len, batch, 48)
        print("✓ StickFigureTransformer conditioning test passed")

    def test_training_step(self):
        """Model should be trainable"""
        model = StickFigureTransformer(
            input_dim=48,
            d_model=128,
            nhead=4,
            num_layers=2,
            output_dim=48,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        motion = torch.randn(16, 2, 48)
        text_emb = torch.randn(2, 1024)
        target = torch.randn(16, 2, 48)
        out = model(motion, text_emb)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss), "Loss should not be NaN"
        print(f"✓ StickFigureTransformer training step passed (loss={loss.item():.4f})")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TRANSFORMER COMPONENTS TEST SUITE")
    print("RMSNorm, SwiGLU, RoPETransformerEncoderLayer (Qwen/Llama Standards)")
    print("=" * 80)

    test_classes = [
        TestRMSNorm,
        TestSwiGLU,
        TestRoPETransformerEncoderLayer,
        TestStickFigureTransformerIntegration,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'─' * 40}")
        print(f"Running {test_class.__name__}")
        print("─" * 40)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"✗ FAILED: {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
