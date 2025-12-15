# Production Readiness Checklist

This document provides a comprehensive checklist to verify production readiness
of the Stick-Gen training and deployment infrastructure.

---

## Resource Requirements

### Data Storage Requirements

| Component | Size | Description |
|-----------|------|-------------|
| **Raw Datasets** | | |
| AMASS (all subsets) | ~45 GB | SMPL motion capture data |
| InterHuman | ~8 GB | Multi-human interaction data |
| NTU-RGB+D 60/120 | ~25 GB | RGB+D skeleton data |
| 100STYLE | ~2 GB | Stylized BVH motions |
| HumanML3D | ~5 GB | Motion + text pairs |
| KIT-ML | ~2 GB | Motion + text pairs |
| AIST++ | ~3 GB | Dance + camera data |
| LSMB19 | ~1 GB | Long-horizon sequences |
| SMPL Models | ~1 GB | Required for AMASS conversion |
| **Raw Total** | **~92 GB** | |
| **Canonical/Processed** | | |
| Canonical .pt files (all converters) | ~15 GB | One per dataset source |
| Synthetic generated data | ~3 GB | 50k-100k samples |
| pretrain_data.pt (curated) | ~5 GB | Filtered pretraining set |
| sft_data.pt (curated) | ~1 GB | High-quality SFT set |
| **Canonical Total** | **~24 GB** | |
| **Embedded/Training-Ready** | | |
| pretrain_data_embedded.pt | ~8 GB | With BGE embeddings |
| sft_data_embedded.pt | ~2 GB | With BGE embeddings |
| train_data_final.pt (legacy) | ~5 GB | Combined dataset |
| BGE embeddings cache | ~2 GB | Sentence transformer cache |
| **Embedded Total** | **~17 GB** | |
| **Checkpoints (All 9 Models)** | | |
| Small Pretrain (30 epochs, 6 ckpts) | ~150 MB | 25 MB each |
| Medium Pretrain (50 epochs, 10 ckpts) | ~650 MB | 65 MB each |
| Large Pretrain (100 epochs, 20 ckpts) | ~2.2 GB | 110 MB each |
| Small/Medium/Large SFT | ~800 MB | 4 checkpoints each |
| Small/Medium/Large LoRA | ~140 MB | Adapter weights only |
| **Checkpoint Total** | **~4 GB** | |
| **Other Artifacts** | | |
| Training logs (W&B, stdout) | ~500 MB | Per-training-run logs |
| Intermediate/temp files | ~1 GB | During data conversion |
| Docker/code workspace | ~2 GB | Container disk |
| **Other Total** | **~3.5 GB** | |
| | | |
| **GRAND TOTAL** | **~140.5 GB** | All data + all 9 models |
| **With 30% Buffer** | **~183 GB** | Recommended headroom |
| **RunPod Network Volume** | **≥200 GB** | **Recommended minimum** |

### Training Compute Requirements

| Variant | Parameters | VRAM Required | Training Time (50 epochs) | Checkpoint Size | Est. RunPod Cost |
|---------|------------|---------------|---------------------------|-----------------|------------------|
| **Small** | 5.6M | 4 GB | ~20 GPU-hours | 25 MB | ~$8-15 |
| **Medium** | 15.8M | 8 GB | ~50 GPU-hours | 65 MB | ~$20-40 |
| **Large** | 28M | 16 GB | ~100 GPU-hours | 110 MB | ~$40-80 |

### GPU Recommendations by Variant

| Variant | Minimum GPU | Recommended GPU | RunPod GPU Type |
|---------|-------------|-----------------|-----------------|
| **Small** | RTX 3060 (12GB) | RTX 3070 | RTX A4000 |
| **Medium** | RTX 3080 (10GB) | RTX 3090 | RTX A5000 |
| **Large** | RTX 3090 (24GB) | A100 40GB | A100 PCIe |

### SFT/LoRA Additional Requirements

| Training Type | Base Variant | Additional VRAM | Additional Time | Notes |
|---------------|--------------|-----------------|-----------------|-------|
| **SFT (Full)** | Medium | +2 GB | ~15 GPU-hours | 20 epochs on curated data |
| **LoRA** | Medium | +0.5 GB | ~8 GPU-hours | Only ~2% params trainable |
| **SFT (Full)** | Large | +4 GB | ~30 GPU-hours | 20 epochs on curated data |
| **LoRA** | Large | +1 GB | ~15 GPU-hours | Only ~2% params trainable |

### RunPod Configuration Defaults

From `runpod/config.yaml`:

```yaml
training:
  network_volume:
    size_gb: 100    # Minimum for all datasets
  pod:
    container_disk_gb: 50  # Sufficient for code + temp files
gpu:
  min_vram_gb: 16   # Supports all variants
```

---

## 1. Infrastructure Validation

### 1.1 Training Pipeline

- [ ] **Training entrypoints run without syntax errors**
  - `bash -n runpod/train_entrypoint.sh` → passes
  - `bash -n runpod/sft_entrypoint.sh` → passes

- [ ] **All model variants train successfully**
  - `configs/small.yaml` → produces checkpoint
  - `configs/base.yaml` → produces checkpoint
  - `configs/large.yaml` → produces checkpoint

- [ ] **Checkpoint resume works**
  - `--resume_from` CLI flag → loads and continues
  - `RESUME_FROM_CHECKPOINT` env var → loads and continues
  - `training.resume_from` config → loads and continues

- [ ] **SFT/LoRA workflow works**
  - `--init_from` loads pretrained weights only
  - LoRA injection freezes base parameters
  - Only LoRA parameters are optimized

### 1.2 Data Pipeline

- [ ] **Canonical data conversion works**
  - AMASS converter runs without errors
  - InterHuman converter runs without errors
  - 100STYLE converter runs without errors

- [ ] **Curation pipeline works**
  - `scripts/prepare_curated_datasets.py` produces pretrain + SFT splits
  - Quality thresholds applied correctly
  - Action balancing applied for SFT

- [ ] **Embedding pipeline works**
  - `scripts/build_dataset_for_training.py` adds embeddings
  - Output compatible with `StickFigureDataset`

### 1.3 Evaluation Pipeline

- [ ] **Evaluation scripts run**
  - `scripts/evaluate.py --checkpoint <path> --data <path>` → produces report
  - `scripts/run_adversarial_eval.py --checkpoint <path>` → produces JSON report

- [ ] **Safety critic integration works**
  - `InferenceGenerator(enable_safety_check=True)` initializes
  - `generator.check_motion_safety(motion)` returns results

## 2. Test Suite

### 2.1 Test Coverage

- [ ] **All tests pass**: `PYTHONPATH=. pytest tests/`
  - Expected: 136+ tests passing
  - Warnings: acceptable (deprecation, PytestReturnNotNoneWarning)

- [ ] **Key test categories covered**:
  - Unit tests: transformer, diffusion, LoRA, safety critic, metrics
  - Integration tests: checkpoint resume, SFT init_from
  - Performance tests: baseline, expressions, speech, multi-actor

### 2.2 Critical Test Files

| Test File | Description | Status |
|-----------|-------------|--------|
| `tests/unit/test_lora.py` | LoRA injection and freezing | ✓ |
| `tests/unit/test_safety_critic.py` | Safety critic detection | ✓ |
| `tests/integration/test_checkpoint_resume.py` | Resume from checkpoint | ✓ |
| `tests/integration/test_sft_init_from.py` | SFT weight initialization | ✓ |

## 3. Model Selection Criteria

### 3.1 Evaluation Metrics

| Metric | Threshold | Weight |
|--------|-----------|--------|
| Validation Loss | < 0.5 | 30% |
| Motion MSE | < 0.1 | 20% |
| Temporal Consistency | > 0.8 | 15% |
| Action Accuracy | > 70% | 15% |
| Physics Consistency | > 0.7 | 10% |
| Robustness (adversarial safe ratio) | > 50% | 10% |

### 3.2 Model Comparison Template

```bash
# Run evaluation on all candidate checkpoints
python scripts/evaluate.py --checkpoint checkpoints/pretrain/best.pth --output eval_pretrain.json
python scripts/evaluate.py --checkpoint checkpoints/sft/best.pth --output eval_sft.json
python scripts/evaluate.py --checkpoint checkpoints/lora/best.pth --output eval_lora.json

# Run adversarial evaluation
python scripts/run_adversarial_eval.py --checkpoint checkpoints/pretrain/best.pth
python scripts/run_adversarial_eval.py --checkpoint checkpoints/sft/best.pth
```

### 3.3 Selection Decision

Document the selection rationale:

1. **Baseline model**: Metrics from initial training
2. **Pretrained model**: Metrics after curated pretraining
3. **SFT model**: Metrics after supervised fine-tuning
4. **LoRA model**: Metrics after LoRA fine-tuning

**Selected model**: [TBD after evaluation]
**Rationale**: [Document why this model was selected]

## 4. Deployment Configuration

### 4.1 Model Checkpoint

- [ ] **Best checkpoint identified and accessible**
  - Path: `checkpoints/production/best_model.pth`
  - Size: ~62 MB (15.8M params)
  - Contains: model_state_dict, optimizer_state_dict, epoch, config

- [ ] **HuggingFace upload complete**
  - Repository: `GesturaAI/stick-gen-{variant}`
  - Model card updated with training details
  - License: MIT

### 4.2 Inference Configuration

- [ ] **Safety thresholds configured**
  - `rejection_severity_threshold`: 0.7 (production default)
  - `min_quality_score`: 0.3

- [ ] **Diffusion refinement settings**
  - `diffusion_steps`: 20 (balance of quality vs speed)
  - `use_diffusion`: true/false based on hardware

## 5. Monitoring and Observability

### 5.1 Training Logs

- [ ] **Log files produced**
  - `training_verbose.log` in checkpoint directory
  - Training metrics logged per epoch
  - Validation metrics logged per epoch

- [ ] **Checkpoint naming convention**
  - `model_checkpoint_best.pth` - best validation loss
  - `checkpoint_epoch_{N}.pth` - periodic checkpoints (every 10 epochs)
  - `model_checkpoint.pth` - final checkpoint

### 5.2 Evaluation Reports

- [ ] **Reports generated and stored**
  - `eval_results/` directory contains JSON reports
  - `eval_results/adversarial/` contains robustness reports
  - Reports include timestamps for tracking

## 6. Final Verification

### 6.1 End-to-End Smoke Test

```bash
# 1. Generate motion from prompt
python -c "
from src.inference.generator import InferenceGenerator
gen = InferenceGenerator(
    model_path='checkpoints/best_model.pth',
    enable_safety_check=True
)
# Minimal test - just verify initialization works
print('✓ InferenceGenerator initialized successfully')
"

# 2. Run safety check
python -c "
from src.eval.safety_critic import SafetyCritic, evaluate_motion_safety
import torch
motion = torch.randn(100, 20) * 0.01
motion = torch.cumsum(motion, dim=0)
result = evaluate_motion_safety(motion)
print(f'✓ Safety check: is_safe={result.is_safe}, score={result.overall_score:.2f}')
"

# 3. Verify all imports work
python -c "
from src.train.train import train
from src.model.transformer import StickFigureTransformer
from src.model.lora import inject_lora_adapters
from src.eval.safety_critic import SafetyCritic
from src.data_gen.curation import curate_samples
print('✓ All critical modules import successfully')
"
```

### 6.2 Sign-off

| Reviewer | Role | Date | Status |
|----------|------|------|--------|
| [Name] | Engineer | [Date] | [ ] Approved |
| [Name] | Lead | [Date] | [ ] Approved |

---

## Appendix: Quick Commands

```bash
# Run full test suite
PYTHONPATH=. pytest tests/ -v

# Train with curated data
python -m src.train.train --config configs/base.yaml --data_path data/curated/pretrain_data_embedded.pt

# Resume training
python -m src.train.train --config configs/base.yaml --resume_from checkpoints/checkpoint_epoch_20.pth

# Run SFT with LoRA
python -m src.train.train --config configs/sft_base.yaml --init_from checkpoints/pretrain/best.pth

# Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/test.pt

# Run adversarial evaluation
python scripts/run_adversarial_eval.py --checkpoint checkpoints/best_model.pth
```

