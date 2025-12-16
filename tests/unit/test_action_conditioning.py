"""
Phase 1: Action-Conditioned Generation - Test Suite

Tests for action conditioning functionality.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_gen.schema import (
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NUM_ACTIONS,
    ActionType,
    generate_per_frame_actions,
)
from src.model.transformer import StickFigureTransformer


def test_action_mappings():
    """Test 1: Verify action-to-index mappings"""
    print("\n" + "=" * 80)
    print("TEST 1: Action Mappings")
    print("=" * 80)

    assert len(ACTION_TO_IDX) == NUM_ACTIONS
    assert len(IDX_TO_ACTION) == NUM_ACTIONS

    for action, idx in ACTION_TO_IDX.items():
        assert IDX_TO_ACTION[idx] == action

    print(f"✓ All {NUM_ACTIONS} actions have unique indices")


def test_per_frame_actions():
    """Test 2: Per-frame action generation"""
    print("\n" + "=" * 80)
    print("TEST 2: Per-Frame Actions")
    print("=" * 80)

    actions = [(0.0, ActionType.WALK), (5.0, ActionType.RUN)]
    per_frame = generate_per_frame_actions(actions, fps=25, total_duration=10.0)

    assert len(per_frame) == 250
    assert per_frame[0] == ActionType.WALK
    assert per_frame[125] == ActionType.RUN

    print("✓ Generated 250 frames correctly")


def test_model_architecture():
    """Test 3: Model architecture"""
    print("\n" + "=" * 80)
    print("TEST 3: Model Architecture")
    print("=" * 80)

    model = StickFigureTransformer(
        input_dim=10,
        d_model=384,
        nhead=12,
        num_layers=8,
        output_dim=10,
        embedding_dim=1024,
        dropout=0.1,
        num_actions=NUM_ACTIONS,
    )

    assert hasattr(model, "action_embedding")
    assert hasattr(model, "action_projection")
    assert hasattr(model, "action_predictor")

    print("✓ All action components present")


def test_forward_pass():
    """Test 4: Forward pass"""
    print("\n" + "=" * 80)
    print("TEST 4: Forward Pass")
    print("=" * 80)

    model = StickFigureTransformer(
        input_dim=10,
        d_model=384,
        nhead=12,
        num_layers=8,
        output_dim=10,
        embedding_dim=1024,
        dropout=0.1,
        num_actions=NUM_ACTIONS,
    )
    model.eval()

    motion = torch.randn(50, 2, 10)
    text_embedding = torch.randn(2, 1024)
    action_sequence = torch.randint(0, NUM_ACTIONS, (50, 2))

    with torch.no_grad():
        outputs = model(
            motion,
            text_embedding,
            return_all_outputs=True,
            action_sequence=action_sequence,
        )

    assert "pose" in outputs
    assert "action_logits" in outputs
    assert outputs["action_logits"].shape == (50, 2, NUM_ACTIONS)

    print("✓ Forward pass successful")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PHASE 1: ACTION-CONDITIONED GENERATION - TEST SUITE")
    print("=" * 80)

    tests = [
        test_action_mappings,
        test_per_frame_actions,
        test_model_architecture,
        test_forward_pass,
    ]
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
