"""
Phase 3: Diffusion Refinement Module Tests

Tests for:
- DDPM Scheduler
- UNet architecture
- Diffusion refinement
- Training step
- Style conditioning (NEW)
- Classifier-free guidance (NEW)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from src.model.diffusion import (
    DDPMScheduler,
    DiffusionRefinementModule,
    PoseRefinementUNet,
    StyleCondition,
    StyleConditioningModule,
    count_parameters,
    extract_style_conditions_from_batch,
)


def test_scheduler():
    print("\n" + "=" * 60)
    print("TEST 1: DDPM Scheduler")
    print("=" * 60)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    clean_poses = torch.randn(4, 250, 20)
    noise = torch.randn_like(clean_poses)
    timesteps = torch.randint(0, 1000, (4,))
    noisy_poses = scheduler.add_noise(clean_poses, noise, timesteps)
    assert noisy_poses.shape == clean_poses.shape
    print(f"✓ Noise addition: {clean_poses.shape} -> {noisy_poses.shape}")
    scheduler.set_timesteps(50)
    assert len(scheduler.timesteps) == 50
    print(f"✓ Timestep setting: {len(scheduler.timesteps)} steps")
    print("\n✅ DDPM Scheduler: PASSED")


def test_unet():
    print("\n" + "=" * 60)
    print("TEST 2: UNet Architecture")
    print("=" * 60)
    unet = PoseRefinementUNet(pose_dim=20, hidden_dims=[64, 128, 256], time_emb_dim=128)
    num_params = count_parameters(unet)
    print(f"✓ Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    assert num_params < 5e6
    noisy_poses = torch.randn(4, 250, 20)
    timesteps = torch.randint(0, 1000, (4,))
    noise_pred = unet(noisy_poses, timesteps)
    assert noise_pred.shape == noisy_poses.shape
    print(f"✓ Forward pass: {noisy_poses.shape} -> {noise_pred.shape}")
    print("\n✅ UNet Architecture: PASSED")


def test_refinement():
    print("\n" + "=" * 60)
    print("TEST 3: Diffusion Refinement")
    print("=" * 60)
    unet = PoseRefinementUNet(pose_dim=20, hidden_dims=[32, 64], time_emb_dim=64)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    refinement = DiffusionRefinementModule(unet, scheduler, device="cpu")
    transformer_output = torch.randn(2, 250, 20)
    refined_poses = refinement.refine_poses(transformer_output, num_inference_steps=10)
    assert refined_poses.shape == transformer_output.shape
    print(f"✓ Refinement: {transformer_output.shape} -> {refined_poses.shape}")
    print("\n✅ Diffusion Refinement: PASSED")


def test_training():
    print("\n" + "=" * 60)
    print("TEST 4: Training Step")
    print("=" * 60)
    unet = PoseRefinementUNet(pose_dim=20, hidden_dims=[32, 64], time_emb_dim=64)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    refinement = DiffusionRefinementModule(unet, scheduler, device="cpu")
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    clean_poses = torch.randn(2, 250, 20)
    result = refinement.train_step(clean_poses, optimizer)
    assert "loss" in result
    print(f"✓ Training step: loss = {result['loss']:.6f}")
    print("\n✅ Training Step: PASSED")


# =============================================================================
# Style Conditioning Tests (NEW)
# =============================================================================


def test_style_condition_creation():
    """Test StyleCondition dataclass creation and conversion."""
    print("\n" + "=" * 60)
    print("TEST 5: StyleCondition Creation")
    print("=" * 60)

    # Direct creation
    sc = StyleCondition(tempo=0.8, energy_level=0.6, valence=0.2)
    assert sc.tempo == 0.8
    assert sc.energy_level == 0.6
    assert sc.smoothness is None
    print("✓ Direct creation with partial fields")

    # to_tensor with missing values
    tensor = sc.to_tensor()
    assert tensor.shape == (5,)
    assert tensor[0] == 0.8  # tempo
    assert tensor[2] == -1.0  # smoothness (missing)
    print("✓ to_tensor handles missing values")

    # From enhanced_meta
    meta = {
        "motion_style": {"tempo": 0.5, "energy_level": 0.3, "smoothness": 0.9},
        "emotion": {"valence": 0.1, "arousal": 0.7},
    }
    sc2 = StyleCondition.from_enhanced_meta(meta)
    assert sc2.tempo == 0.5
    assert sc2.smoothness == 0.9
    assert sc2.arousal == 0.7
    print("✓ from_enhanced_meta extracts all fields")

    # From None
    sc3 = StyleCondition.from_enhanced_meta(None)
    assert sc3.tempo is None
    print("✓ from_enhanced_meta handles None")

    print("\n✅ StyleCondition Creation: PASSED")


def test_style_conditioning_module():
    """Test StyleConditioningModule forward pass."""
    print("\n" + "=" * 60)
    print("TEST 6: StyleConditioningModule")
    print("=" * 60)

    scm = StyleConditioningModule(output_dim=128)
    assert scm.output_dim == 128
    print(f"✓ Module created with output_dim=128")

    # With conditions
    sc1 = StyleCondition(tempo=0.8, energy_level=0.6)
    sc2 = StyleCondition(smoothness=0.9, valence=-0.5, arousal=0.3)
    style_emb = scm([sc1, sc2], batch_size=2, device="cpu")
    assert style_emb.shape == (2, 128)
    print(f"✓ Forward with conditions: {style_emb.shape}")

    # Without conditions (null embedding)
    null_emb = scm(None, batch_size=3, device="cpu")
    assert null_emb.shape == (3, 128)
    print(f"✓ Forward without conditions: {null_emb.shape}")

    print("\n✅ StyleConditioningModule: PASSED")


def test_unet_with_style_conditioning():
    """Test PoseRefinementUNet with style conditioning."""
    print("\n" + "=" * 60)
    print("TEST 7: UNet with Style Conditioning")
    print("=" * 60)

    # With conditioning
    unet = PoseRefinementUNet(
        pose_dim=20,
        hidden_dims=[32, 64],
        time_emb_dim=64,
        style_emb_dim=64,
        use_style_conditioning=True,
    )
    num_params = count_parameters(unet)
    print(f"✓ Conditioned UNet: {num_params:,} params")

    x = torch.randn(2, 100, 20)
    t = torch.tensor([500, 300])

    # Forward with explicit style embedding
    style_emb = torch.randn(2, 64)
    out = unet(x, t, style_emb)
    assert out.shape == x.shape
    print(f"✓ Forward with style_embedding: {out.shape}")

    # Forward with None (uses null embedding)
    out2 = unet(x, t, None)
    assert out2.shape == x.shape
    print(f"✓ Forward with None style: {out2.shape}")

    # Without conditioning
    unet_uncond = PoseRefinementUNet(
        pose_dim=20, hidden_dims=[32, 64], use_style_conditioning=False
    )
    out3 = unet_uncond(x, t)
    assert out3.shape == x.shape
    print(f"✓ Forward unconditioned: {out3.shape}")

    print("\n✅ UNet with Style Conditioning: PASSED")


def test_diffusion_with_cfg():
    """Test DiffusionRefinementModule with classifier-free guidance."""
    print("\n" + "=" * 60)
    print("TEST 8: Diffusion with Classifier-Free Guidance")
    print("=" * 60)

    unet = PoseRefinementUNet(
        pose_dim=20,
        hidden_dims=[32, 64],
        use_style_conditioning=True,
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    module = DiffusionRefinementModule(unet, scheduler, device="cpu")

    x = torch.randn(2, 100, 20)
    sc1 = StyleCondition(tempo=0.8, energy_level=0.6)
    sc2 = StyleCondition(tempo=0.3, energy_level=0.2)

    # Unconditional (guidance_scale=1.0)
    refined_uncond = module.refine_poses(x, num_inference_steps=5, guidance_scale=1.0)
    assert refined_uncond.shape == x.shape
    print(f"✓ Unconditional refinement: {refined_uncond.shape}")

    # Conditional without guidance
    refined_cond = module.refine_poses(
        x, style_conditions=[sc1, sc2], num_inference_steps=5, guidance_scale=1.0
    )
    assert refined_cond.shape == x.shape
    print(f"✓ Conditional refinement (no CFG): {refined_cond.shape}")

    # Conditional with CFG
    refined_cfg = module.refine_poses(
        x, style_conditions=[sc1, sc2], num_inference_steps=5, guidance_scale=2.5
    )
    assert refined_cfg.shape == x.shape
    print(f"✓ Conditional refinement (CFG=2.5): {refined_cfg.shape}")

    print("\n✅ Diffusion with CFG: PASSED")


def test_training_with_style():
    """Test training step with style conditioning."""
    print("\n" + "=" * 60)
    print("TEST 9: Training with Style Conditioning")
    print("=" * 60)

    unet = PoseRefinementUNet(
        pose_dim=20,
        hidden_dims=[32, 64],
        use_style_conditioning=True,
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    module = DiffusionRefinementModule(unet, scheduler, device="cpu", cfg_dropout_prob=0.1)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    clean_poses = torch.randn(2, 100, 20)
    sc1 = StyleCondition(tempo=0.8, energy_level=0.6)
    sc2 = StyleCondition(tempo=0.3, energy_level=0.2)

    # Train with style
    result = module.train_step(clean_poses, optimizer, style_conditions=[sc1, sc2])
    assert "loss" in result
    print(f"✓ Training with style: loss = {result['loss']:.6f}")

    # Train without style
    result2 = module.train_step(clean_poses, optimizer, style_conditions=None)
    assert "loss" in result2
    print(f"✓ Training without style: loss = {result2['loss']:.6f}")

    print("\n✅ Training with Style Conditioning: PASSED")


def test_extract_style_from_batch():
    """Test extracting StyleConditions from batch dict."""
    print("\n" + "=" * 60)
    print("TEST 10: Extract Style from Batch")
    print("=" * 60)

    # Batch with enhanced_meta
    batch = {
        "motion": torch.randn(2, 100, 20),
        "enhanced_meta": [
            {"motion_style": {"tempo": 0.5, "energy_level": 0.3}, "emotion": {"valence": 0.1}},
            {"motion_style": {"tempo": 0.8}, "emotion": None},
        ],
    }
    conditions = extract_style_conditions_from_batch(batch)
    assert conditions is not None
    assert len(conditions) == 2
    assert conditions[0].tempo == 0.5
    assert conditions[1].tempo == 0.8
    print(f"✓ Extracted {len(conditions)} conditions from batch")

    # Batch without enhanced_meta
    batch2 = {"motion": torch.randn(2, 100, 20)}
    conditions2 = extract_style_conditions_from_batch(batch2)
    assert conditions2 is None
    print("✓ Returns None when no enhanced_meta")

    print("\n✅ Extract Style from Batch: PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 3: DIFFUSION REFINEMENT MODULE - TEST SUITE")
    print("=" * 70)
    tests = [
        test_scheduler,
        test_unet,
        test_refinement,
        test_training,
        test_style_condition_creation,
        test_style_conditioning_module,
        test_unet_with_style_conditioning,
        test_diffusion_with_cfg,
        test_training_with_style,
        test_extract_style_from_batch,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 70)
    sys.exit(0 if passed == len(tests) else 1)
