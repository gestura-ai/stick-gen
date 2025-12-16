"""
Tests for Physics-Aware World Model (Phase 2)

Tests:
- Physics decoder heads
- Gravity constraints
- Momentum conservation
- Collision detection
- Ground contact
"""

import sys

sys.path.insert(0, "/Users/bc/gestura/stick-gen")

import torch

from src.data_gen.renderer import Renderer
from src.model.transformer import StickFigureTransformer


def test_physics_decoder_heads():
    """Test that physics decoder heads exist and output correct shapes"""
    model = StickFigureTransformer()

    # Check physics decoder exists
    assert hasattr(model, "physics_decoder"), "Model should have physics_decoder"
    assert hasattr(
        model, "environment_decoder"
    ), "Model should have environment_decoder"

    # Test forward pass
    batch_size = 4
    seq_len = 250
    motion = torch.randn(seq_len, batch_size, 20)
    text_embedding = torch.randn(batch_size, 1024)

    outputs = model(motion, text_embedding, return_all_outputs=True)

    assert "physics" in outputs, "Outputs should include physics predictions"
    assert "environment" in outputs, "Outputs should include environment predictions"

    # Physics: [seq, batch, 6] - vx, vy, ax, ay, momentum_x, momentum_y
    assert outputs["physics"].shape == (
        seq_len,
        batch_size,
        6,
    ), f"Physics shape should be (250, 4, 6), got {outputs['physics'].shape}"

    # Environment: [batch, 32] - ground_level, obstacles, context
    assert outputs["environment"].shape == (
        batch_size,
        32,
    ), f"Environment shape should be (4, 32), got {outputs['environment'].shape}"

    print("✓ Physics decoder heads test passed")


def test_gravity_constraint():
    """Test that jumping applies realistic gravity"""
    # TODO: Implement after training
    # Generate jumping animation
    # Extract vertical positions over time
    # Verify parabolic trajectory
    # Verify peak height is realistic (1-2 units)
    print("⏳ Gravity constraint test - pending training")


def test_momentum_conservation():
    """Test momentum conservation during running"""
    # TODO: Implement after training
    # Generate running animation
    # Extract velocities over time
    # Verify smooth velocity transitions
    # Verify no sudden jumps in momentum
    print("⏳ Momentum conservation test - pending training")


def test_collision_detection():
    """Test ground collision detection"""
    renderer = Renderer()

    # Test collision detection function
    if hasattr(renderer, "detect_collisions"):
        # Test ground collision
        ground_level = 0.0

        # Position above ground
        collision_info = renderer.detect_collisions((0, 1.0), ground_level)
        assert not collision_info[
            "ground_collision"
        ], "Should not collide when above ground"

        # Position on ground
        collision_info = renderer.detect_collisions((0, 0.0), ground_level)
        assert collision_info["ground_collision"], "Should collide when on ground"

        # Position below ground
        collision_info = renderer.detect_collisions((0, -0.5), ground_level)
        assert collision_info["ground_collision"], "Should collide when below ground"

        print("✓ Collision detection test passed")
    else:
        print("⏳ Collision detection test - pending implementation")


def test_physics_constraints():
    """Test physics constraints in renderer"""
    renderer = Renderer()

    if hasattr(renderer, "apply_physics_constraints"):
        # Test gravity application
        position = (0, 5.0)  # Start 5 units above ground
        velocity = (0, 0)  # No initial velocity
        dt = 0.04

        new_position, new_velocity = renderer.apply_physics_constraints(
            position, velocity, dt
        )

        # Velocity should be negative (falling)
        assert new_velocity[1] < 0, "Vertical velocity should be negative (falling)"

        # Position should be lower
        assert new_position[1] < position[1], "Position should be lower after gravity"

        print("✓ Physics constraints test passed")
    else:
        print("⏳ Physics constraints test - pending implementation")


def test_no_ground_penetration():
    """Test that figures don't go below ground"""
    # TODO: Implement after training
    # Generate various animations
    # Extract all y-coordinates
    # Verify all y >= ground_level
    print("⏳ Ground penetration test - pending training")


def test_realistic_jump_height():
    """Test that jump heights are realistic"""
    # TODO: Implement after training
    # Generate jumping animation
    # Measure peak height
    # Verify 1-2 units (realistic for stick figure)
    print("⏳ Realistic jump height test - pending training")


def test_velocity_smoothness():
    """Test that velocities are smooth (no sudden jumps)"""
    # TODO: Implement after training
    # Generate animation
    # Compute frame-to-frame velocity changes
    # Verify changes are gradual (no spikes)
    print("⏳ Velocity smoothness test - pending training")


if __name__ == "__main__":
    print("Running Physics-Aware World Model Tests...\n")

    # Run tests that don't require trained model
    try:
        test_physics_decoder_heads()
    except AssertionError as e:
        print(f"✗ Physics decoder heads test failed: {e}")

    try:
        test_collision_detection()
    except Exception as e:
        print(f"⚠ Collision detection test error: {e}")

    try:
        test_physics_constraints()
    except Exception as e:
        print(f"⚠ Physics constraints test error: {e}")

    # Tests requiring training
    test_gravity_constraint()
    test_momentum_conservation()
    test_no_ground_penetration()
    test_realistic_jump_height()
    test_velocity_smoothness()

    print("\n✅ Physics-aware world model test suite complete")
