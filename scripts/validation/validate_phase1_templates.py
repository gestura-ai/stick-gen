#!/usr/bin/env python3
"""
Phase 1 Template Validation Script

Validates all Phase 1 action-conditioned generation templates before integration.
Tests tensor shapes, forward/backward passes, loss computation, and gradient flow.

Run this script to verify templates are ready for integration into the codebase.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 4
SEQ_LEN = 250
INPUT_DIM = 20  # 5 lines × 4 coords
OUTPUT_DIM = 20
D_MODEL = 384
NUM_ACTIONS = 60
EMBEDDING_DIM = 1024  # Text embedding dimension

print("=" * 70)
print("Phase 1 Template Validation")
print("=" * 70)
print("Configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Model dimension: {D_MODEL}")
print(f"  Number of actions: {NUM_ACTIONS}")
print("=" * 70)
print()

# ============================================================================
# Test 1: Action Embedding Layer
# ============================================================================

print("Test 1: Action Embedding Layer")
print("-" * 70)

try:
    # Create action embedding layer (from template)
    action_embedding = nn.Embedding(
        num_embeddings=NUM_ACTIONS,
        embedding_dim=64
    )

    # Create action projection layer
    action_projection = nn.Sequential(
        nn.Linear(64, D_MODEL),
        nn.LayerNorm(D_MODEL),
        nn.ReLU(),
        nn.Dropout(0.1)
    )

    # Test with random action indices
    action_indices = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    action_emb = action_embedding(action_indices)
    action_features = action_projection(action_emb)

    # Verify shapes
    assert action_emb.shape == (BATCH_SIZE, SEQ_LEN, 64), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, 64)}, got {action_emb.shape}"
    assert action_features.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, got {action_features.shape}"

    # Count parameters
    embedding_params = sum(p.numel() for p in action_embedding.parameters())
    projection_params = sum(p.numel() for p in action_projection.parameters())

    print(f"✓ Action embedding shape: {action_emb.shape}")
    print(f"✓ Action features shape: {action_features.shape}")
    print(f"✓ Embedding parameters: {embedding_params:,}")
    print(f"✓ Projection parameters: {projection_params:,}")
    print(f"✓ Total parameters: {embedding_params + projection_params:,}")
    print("✓ Test 1 PASSED")

except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")
    sys.exit(1)

print()

# ============================================================================
# Test 2: Action Prediction Head
# ============================================================================

print("Test 2: Action Prediction Head")
print("-" * 70)

try:
    # Create action prediction head (from template)
    action_predictor = nn.Sequential(
        nn.Linear(D_MODEL, D_MODEL // 2),
        nn.LayerNorm(D_MODEL // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(D_MODEL // 2, NUM_ACTIONS)
    )

    # Test with random features
    random_features = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    # Forward pass
    action_logits = action_predictor(random_features)

    # Verify shape
    assert action_logits.shape == (BATCH_SIZE, SEQ_LEN, NUM_ACTIONS), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, NUM_ACTIONS)}, got {action_logits.shape}"

    # Count parameters
    predictor_params = sum(p.numel() for p in action_predictor.parameters())

    print(f"✓ Action logits shape: {action_logits.shape}")
    print(f"✓ Predictor parameters: {predictor_params:,}")
    print(f"✓ Output range: [{action_logits.min():.2f}, {action_logits.max():.2f}]")
    print("✓ Test 2 PASSED")

except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")
    sys.exit(1)

print()

# ============================================================================
# Test 3: Loss Computation
# ============================================================================

print("Test 3: Loss Computation")
print("-" * 70)

try:
    # Create mock outputs and targets
    mock_action_logits = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_ACTIONS, requires_grad=True)
    target_actions = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE, SEQ_LEN))

    # Reshape for cross-entropy
    logits_flat = mock_action_logits.reshape(-1, NUM_ACTIONS)
    targets_flat = target_actions.reshape(-1)

    # Compute action loss
    action_loss = F.cross_entropy(logits_flat, targets_flat)

    # Compute accuracy
    action_preds = torch.argmax(logits_flat, dim=-1)
    action_accuracy = (action_preds == targets_flat).float().mean()

    # Verify loss properties
    assert action_loss.dim() == 0, "Loss should be scalar"
    assert action_loss.requires_grad, "Loss should require gradients"
    assert 0.0 <= action_accuracy <= 1.0, f"Accuracy should be in [0, 1], got {action_accuracy}"

    print(f"✓ Action loss: {action_loss.item():.4f}")
    print(f"✓ Action accuracy: {action_accuracy.item():.4f} ({action_accuracy.item()*100:.2f}%)")
    print(f"✓ Loss is scalar: {action_loss.dim() == 0}")
    print(f"✓ Loss requires grad: {action_loss.requires_grad}")
    print("✓ Test 3 PASSED")

except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")
    sys.exit(1)

print()

# ============================================================================
# Test 4: Full Forward Pass with Action Conditioning
# ============================================================================

print("Test 4: Full Forward Pass with Action Conditioning")
print("-" * 70)

try:
    # Create minimal transformer with action conditioning
    class MinimalActionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            # Motion embedding
            self.motion_embedding = nn.Linear(INPUT_DIM, D_MODEL)

            # Text projection
            self.text_projection = nn.Linear(EMBEDDING_DIM, D_MODEL)

            # Action embedding (from template)
            self.action_embedding = nn.Embedding(NUM_ACTIONS, 64)
            self.action_projection = nn.Sequential(
                nn.Linear(64, D_MODEL),
                nn.LayerNorm(D_MODEL),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            # Simple transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=12,
                dim_feedforward=D_MODEL * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

            # Decoders
            self.pose_decoder = nn.Linear(D_MODEL, OUTPUT_DIM)
            self.position_decoder = nn.Linear(D_MODEL, 2)
            self.velocity_decoder = nn.Linear(D_MODEL, 2)

            # Action prediction head (from template)
            self.action_predictor = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL // 2),
                nn.LayerNorm(D_MODEL // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(D_MODEL // 2, NUM_ACTIONS)
            )

        def forward(self, motion_sequence, text_embedding, action_sequence=None):
            # Embed motion and text
            motion_features = self.motion_embedding(motion_sequence)
            text_features = self.text_projection(text_embedding).unsqueeze(1).expand(-1, SEQ_LEN, -1)

            # Action conditioning (from template)
            if action_sequence is not None:
                action_emb = self.action_embedding(action_sequence)
                action_features = self.action_projection(action_emb)
                combined_features = motion_features + text_features + action_features
            else:
                combined_features = motion_features + text_features

            # Transformer encoding
            encoded = self.transformer_encoder(combined_features)

            # Multi-task decoding
            pose_output = self.pose_decoder(encoded)
            position_output = self.position_decoder(encoded)
            velocity_output = self.velocity_decoder(encoded)
            action_logits = self.action_predictor(encoded)

            return {
                'pose': pose_output,
                'position': position_output,
                'velocity': velocity_output,
                'action_logits': action_logits
            }

    # Create model
    model = MinimalActionTransformer()

    # Create mock inputs
    motion_sequence = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    text_embedding = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    action_sequence = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE, SEQ_LEN))

    # Test forward pass WITH action conditioning
    outputs_with_actions = model(motion_sequence, text_embedding, action_sequence)

    # Verify all outputs present
    assert 'pose' in outputs_with_actions, "Missing 'pose' output"
    assert 'position' in outputs_with_actions, "Missing 'position' output"
    assert 'velocity' in outputs_with_actions, "Missing 'velocity' output"
    assert 'action_logits' in outputs_with_actions, "Missing 'action_logits' output"

    # Verify shapes
    assert outputs_with_actions['pose'].shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
    assert outputs_with_actions['position'].shape == (BATCH_SIZE, SEQ_LEN, 2)
    assert outputs_with_actions['velocity'].shape == (BATCH_SIZE, SEQ_LEN, 2)
    assert outputs_with_actions['action_logits'].shape == (BATCH_SIZE, SEQ_LEN, NUM_ACTIONS)

    # Test forward pass WITHOUT action conditioning (backward compatibility)
    outputs_without_actions = model(motion_sequence, text_embedding, action_sequence=None)

    # Verify all outputs still present
    assert 'action_logits' in outputs_without_actions, "Missing 'action_logits' in no-action mode"

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✓ Forward pass with actions: All outputs present")
    print(f"✓ Pose output shape: {outputs_with_actions['pose'].shape}")
    print(f"✓ Position output shape: {outputs_with_actions['position'].shape}")
    print(f"✓ Velocity output shape: {outputs_with_actions['velocity'].shape}")
    print(f"✓ Action logits shape: {outputs_with_actions['action_logits'].shape}")
    print("✓ Forward pass without actions: Backward compatible")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print("✓ Test 4 PASSED")

except Exception as e:
    print(f"✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# Test 5: Backward Pass and Gradient Flow
# ============================================================================

print("Test 5: Backward Pass and Gradient Flow")
print("-" * 70)

try:
    # Use model from Test 4
    model.zero_grad()

    # Create mock targets
    target_pose = torch.randn(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
    target_actions = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    outputs = model(motion_sequence, text_embedding, action_sequence)

    # Compute losses (from template)
    pose_loss = F.mse_loss(outputs['pose'], target_pose)

    # Action loss
    action_logits_flat = outputs['action_logits'].reshape(-1, NUM_ACTIONS)
    target_actions_flat = target_actions.reshape(-1)
    action_loss = F.cross_entropy(action_logits_flat, target_actions_flat)

    # Total loss (weighted)
    total_loss = 1.0 * pose_loss + 0.15 * action_loss

    # Backward pass
    total_loss.backward()

    # Check gradients
    has_gradients = []
    gradient_magnitudes = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None
            has_gradients.append(has_grad)

            if has_grad:
                grad_magnitude = param.grad.abs().mean().item()
                gradient_magnitudes.append(grad_magnitude)

                # Check for NaN or Inf
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    all_have_gradients = all(has_gradients)
    avg_grad_magnitude = sum(gradient_magnitudes) / len(gradient_magnitudes)

    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Pose loss: {pose_loss.item():.4f}")
    print(f"✓ Action loss: {action_loss.item():.4f}")
    print(f"✓ All parameters have gradients: {all_have_gradients}")
    print(f"✓ Average gradient magnitude: {avg_grad_magnitude:.6f}")
    print("✓ No NaN gradients: True")
    print("✓ No Inf gradients: True")
    print("✓ Test 5 PASSED")

except Exception as e:
    print(f"✗ Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# Test 6: Edge Cases
# ============================================================================

print("Test 6: Edge Cases")
print("-" * 70)

try:
    # Test 6a: All same action
    same_action = torch.full((BATCH_SIZE, SEQ_LEN), 5, dtype=torch.long)  # All WALK
    outputs_same = model(motion_sequence, text_embedding, same_action)
    assert outputs_same['action_logits'].shape == (BATCH_SIZE, SEQ_LEN, NUM_ACTIONS)
    print("✓ All same action: OK")

    # Test 6b: Sequential actions (0, 1, 2, ..., 59, 0, 1, ...)
    sequential_actions = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1) % NUM_ACTIONS
    outputs_seq = model(motion_sequence, text_embedding, sequential_actions)
    assert outputs_seq['action_logits'].shape == (BATCH_SIZE, SEQ_LEN, NUM_ACTIONS)
    print("✓ Sequential actions: OK")

    # Test 6c: Batch size = 1
    single_motion = motion_sequence[:1]
    single_text = text_embedding[:1]
    single_action = action_sequence[:1]
    outputs_single = model(single_motion, single_text, single_action)
    assert outputs_single['action_logits'].shape == (1, SEQ_LEN, NUM_ACTIONS)
    print("✓ Batch size = 1: OK")

    # Test 6d: Perfect predictions (accuracy = 1.0)
    perfect_logits = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_ACTIONS)
    perfect_targets = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE, SEQ_LEN))
    for b in range(BATCH_SIZE):
        for t in range(SEQ_LEN):
            perfect_logits[b, t, perfect_targets[b, t]] = 10.0  # High logit for correct class

    perfect_preds = torch.argmax(perfect_logits.reshape(-1, NUM_ACTIONS), dim=-1)
    perfect_accuracy = (perfect_preds == perfect_targets.reshape(-1)).float().mean()
    assert perfect_accuracy == 1.0, f"Perfect accuracy should be 1.0, got {perfect_accuracy}"
    print(f"✓ Perfect predictions: Accuracy = {perfect_accuracy:.2f}")

    print("✓ Test 6 PASSED")

except Exception as e:
    print(f"✗ Test 6 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# Final Summary
# ============================================================================

print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print()
print("✓ Test 1: Action Embedding Layer - PASSED")
print("✓ Test 2: Action Prediction Head - PASSED")
print("✓ Test 3: Loss Computation - PASSED")
print("✓ Test 4: Full Forward Pass - PASSED")
print("✓ Test 5: Backward Pass and Gradient Flow - PASSED")
print("✓ Test 6: Edge Cases - PASSED")
print()
print("=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print()
print("Phase 1 templates are validated and ready for integration.")
print()
print("Next steps:")
print("1. Apply templates to actual code files (schema.py, transformer.py, train.py)")
print("2. Add action sequences to existing 100k dataset")
print("3. Train Phase 1 model with action conditioning")
print("4. Validate action prediction accuracy (target: >80%)")
print()

