"""
Phase 3: Diffusion Refinement Module Tests
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.model.diffusion import DDPMScheduler, PoseRefinementUNet, DiffusionRefinementModule, count_parameters

def test_scheduler():
    print("\n" + "="*60)
    print("TEST 1: DDPM Scheduler")
    print("="*60)
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
    print("\n" + "="*60)
    print("TEST 2: UNet Architecture")
    print("="*60)
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
    print("\n" + "="*60)
    print("TEST 3: Diffusion Refinement")
    print("="*60)
    unet = PoseRefinementUNet(pose_dim=20, hidden_dims=[32, 64], time_emb_dim=64)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    refinement = DiffusionRefinementModule(unet, scheduler, device='cpu')
    transformer_output = torch.randn(2, 250, 20)
    refined_poses = refinement.refine_poses(transformer_output, num_inference_steps=10)
    assert refined_poses.shape == transformer_output.shape
    print(f"✓ Refinement: {transformer_output.shape} -> {refined_poses.shape}")
    print("\n✅ Diffusion Refinement: PASSED")


def test_training():
    print("\n" + "="*60)
    print("TEST 4: Training Step")
    print("="*60)
    unet = PoseRefinementUNet(pose_dim=20, hidden_dims=[32, 64], time_emb_dim=64)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    refinement = DiffusionRefinementModule(unet, scheduler, device='cpu')
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    clean_poses = torch.randn(2, 250, 20)
    result = refinement.train_step(clean_poses, optimizer)
    assert 'loss' in result
    print(f"✓ Training step: loss = {result['loss']:.6f}")
    print("\n✅ Training Step: PASSED")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 3: DIFFUSION REFINEMENT MODULE - TEST SUITE")
    print("="*70)
    tests = [test_scheduler, test_unet, test_refinement, test_training]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    sys.exit(0 if passed == len(tests) else 1)
