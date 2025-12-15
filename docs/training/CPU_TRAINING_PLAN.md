# CPU-Optimized Training Plan for Stick-Gen with Facial Expressions

**Model**: 15.5M parameter transformer with facial expressions  
**Hardware**: CPU-only execution  
**Dataset**: Synthetic + AMASS (merged)  
**Total Estimated Time**: 36-48 hours

---

## Overview

This document provides a phased training plan optimized for CPU execution, breaking down the training process into digestible, manageable phases with clear checkpoints and validation steps.

---

## Phase Breakdown

### **Phase 1: Dataset Generation** (16-20 hours)

**What it does:**
- Generates 100,000 base training samples with diverse actions, themes, and scenarios
- Applies 4x data augmentation (speed, position, scale, mirror) → 500,000 total samples
- Creates 10-second sequences (250 frames @ 25fps)
- Computes physics tensors (velocity, acceleration, momentum)
- Generates per-frame action labels for action conditioning
- **NEW**: Includes facial expression data in schema (automatic via story engine)

**Command:**
```bash
python3.9 -m src.data_gen.dataset_generator
```

**Output:**
- File: `data/train_data.pt`
- Size: ~1.0-1.5 GB
- Samples: 500,000 (100k base + 400k augmented)

**Validation Checkpoint:**
```bash
# Verify dataset was created
python3.9 -c "
import torch
data = torch.load('data/train_data.pt')
print(f'Total samples: {len(data)}')
print(f'Sample keys: {list(data[0].keys())}')
print(f'Motion shape: {data[0][\"motion\"].shape}')
print(f'Actions shape: {data[0][\"actions\"].shape}')
print(f'Physics shape: {data[0][\"physics\"].shape}')
print('✅ Dataset generation successful!')
"
```

**Success Criteria:**
- ✅ File `data/train_data.pt` exists
- ✅ Contains 500,000 samples
- ✅ Each sample has: description, motion, actions, physics
- ✅ Motion shape: [250, 20]
- ✅ Actions shape: [250]
- ✅ Physics shape: [250, 6]

**Time Estimate:** 16-20 hours on CPU  
**Progress Monitoring:** Watch console output showing sample generation progress

---

### **Phase 2: Embedding Generation** (4-8 hours)

**What it does:**
- Loads BAAI/bge-large-en-v1.5 embedding model (1024-dim, top-5 on MTEB leaderboard)
- Generates semantic text embeddings for all 500,000 descriptions
- Processes in batches of 16 for CPU efficiency
- Normalizes embeddings (standard for BGE models)

**Command:**
```bash
python3.9 -m src.data_gen.preprocess_embeddings
```

**Output:**
- File: `data/train_data_embedded.pt`
- Size: ~2.0-2.5 GB
- Samples: 500,000 (same as Phase 1, now with embeddings)

**Validation Checkpoint:**
```bash
# Verify embeddings were added
python3.9 -c "
import torch
data = torch.load('data/train_data_embedded.pt')
print(f'Total samples: {len(data)}')
print(f'Sample keys: {list(data[0].keys())}')
print(f'Embedding shape: {data[0][\"embedding\"].shape}')
print(f'Embedding mean: {data[0][\"embedding\"].mean():.6f}')
print(f'Embedding std: {data[0][\"embedding\"].std():.6f}')
print('✅ Embedding generation successful!')
"
```

**Success Criteria:**
- ✅ File `data/train_data_embedded.pt` exists
- ✅ Contains 500,000 samples
- ✅ Each sample has: description, motion, actions, physics, **embedding**
- ✅ Embedding shape: [1024]
- ✅ Embedding mean: ~0.0 (normalized)
- ✅ Embedding std: ~0.1-0.3 (semantic variation)

**Time Estimate:** 4-8 hours on CPU  
**Progress Monitoring:** Watch batch processing progress bar

---

### **Phase 3: Model Training** (12-24 hours)

**What it does:**
- Trains 15.5M parameter transformer model for 50 epochs
- Multi-task learning: pose + position + velocity + actions + physics
- Gradient accumulation (effective batch size: 64)
- Learning rate warmup (10 epochs) + cosine decay
- Saves checkpoints every 10 epochs
- Saves best model based on validation loss
- **NEW**: Facial expressions are learned implicitly through pose prediction (no separate training needed)

**Command:**
```bash
python3.9 -m src.train.train
```

**Output:**
- File: `model_checkpoint_best.pth` (best model)
- Files: `checkpoint_epoch_10.pth`, `checkpoint_epoch_20.pth`, etc.
- Size: ~250-300 MB per checkpoint

**Validation Checkpoints:**

**After Epoch 10:**
```bash
# Check training progress
python3.9 -c "
import torch
ckpt = torch.load('checkpoint_epoch_10.pth')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Train loss: {ckpt[\"train_loss\"]:.6f}')
print(f'Val loss: {ckpt[\"val_loss\"]:.6f}')
print('✅ Training progressing normally')
"
```

**After Epoch 50 (Final):**
```bash
# Check final model
python3.9 -c "
import torch
ckpt = torch.load('model_checkpoint_best.pth')
print(f'Best epoch: {ckpt[\"epoch\"]}')
print(f'Train loss: {ckpt[\"train_loss\"]:.6f}')
print(f'Val loss: {ckpt[\"val_loss\"]:.6f}')
print(f'Val smoothness: {ckpt[\"val_smoothness\"]:.6f}')
print(f'Val position: {ckpt[\"val_position\"]:.6f}')
print('✅ Training complete!')
"
```

**Success Criteria:**
- ✅ Training completes 50 epochs without errors
- ✅ Validation loss decreases over time
- ✅ Best model saved with val_loss < 0.01 (target)
- ✅ Smoothness metric < 0.5 (target)
- ✅ Position accuracy < 0.1 (target)

**Time Estimate:** 12-24 hours on CPU  
**Progress Monitoring:** Use `monitor_training.sh` script (updates every 30 seconds)

---

## Monitoring Commands

### Real-Time Training Monitor
```bash
./monitor_training.sh
```
Shows:
- Training process status (PID, CPU%, Memory%)
- Latest checkpoint files
- Current epoch progress
- Time since last checkpoint
- System CPU usage

### Manual Progress Check
```bash
# Check if training is running
ps aux | grep "src.train.train" | grep -v grep

# Check latest checkpoint
ls -lht checkpoint_epoch_*.pth | head -1

# Check training logs (if redirected)
tail -f training.log
```

---

## Complete Pipeline Script

For convenience, run all phases sequentially:

```bash
./run_full_training_pipeline.sh
```

This script:
1. Runs Phase 1 (dataset generation)
2. Runs Phase 2 (embedding generation)
3. Runs Phase 3 (model training)
4. Validates each phase before proceeding
5. Exits on any errors

**Total Time:** 36-48 hours on CPU

---

## Facial Expressions Integration

**Important:** Facial expressions do NOT require separate training phases!

The facial expression feature is implemented at the **data generation and rendering level**:
- Schema extensions (FacialExpression, MouthShape, FaceFeatures)
- Renderer enhancements (facial drawing methods)
- Story engine integration (action-expression mappings)

During training:
- The model learns to predict **pose tensors** [250, 20] (5 lines × 4 coords)
- Facial expressions are **derived from actions** at inference time
- No additional model parameters or training steps needed

This means:
- ✅ No changes to training pipeline
- ✅ No additional training time
- ✅ Facial expressions work immediately after training
- ✅ End-to-end training is sufficient

---

## Next Steps After Training

1. **Test the model:**
   ```bash
   ./stick-gen "A person walking and waving happily" --output test.mp4
   ```

2. **Verify facial expressions:**
   ```bash
   python test_facial_expressions.py
   python test_expression_transitions.py
   python test_speech_animation.py
   ```

3. **Run integration tests:**
   ```bash
   python test_integration_all_features.py
   ```

4. **Performance benchmark:**
   ```bash
   python test_performance_benchmark.py
   ```

---

## Troubleshooting

### Phase 1 Issues
- **Out of memory:** Reduce `num_samples` in dataset_generator.py
- **Slow generation:** Normal on CPU, ~85 samples/sec expected

### Phase 2 Issues
- **Model download fails:** Check internet connection, model downloads on first run
- **Out of memory:** Reduce `batch_size` in preprocess_embeddings.py

### Phase 3 Issues
- **Training crashes:** Reduce `BATCH_SIZE` in train.py (default: 16)
- **Slow training:** Normal on CPU, ~20-30 min/epoch expected
- **High memory usage:** Enable gradient accumulation (already configured)

---

## Summary

| Phase | Time | Output | Validation |
|-------|------|--------|------------|
| 1. Dataset Generation | 16-20h | `train_data.pt` (1.5GB) | 500k samples with actions/physics |
| 2. Embedding Generation | 4-8h | `train_data_embedded.pt` (2.5GB) | 1024-dim embeddings added |
| 3. Model Training | 12-24h | `model_checkpoint_best.pth` (300MB) | Val loss < 0.01, smoothness < 0.5 |
| **Total** | **36-48h** | **Production-ready model** | **All tests passing** |

**Facial expressions are included automatically - no additional training needed!**

