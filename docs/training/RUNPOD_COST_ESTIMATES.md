# RunPod Training Cost Estimates for Stick-Gen

**Pricing Date**: December 2024  
**Source**: RunPod public pricing (runpod.io/pricing)

> ⚠️ **Note**: Prices are subject to change. Always verify current pricing at [runpod.io/pricing](https://www.runpod.io/pricing) before provisioning resources.

---

## GPU Pricing Reference

### Community Cloud (On-Demand)

| GPU Type | VRAM | Price/Hour | Best For |
|----------|------|------------|----------|
| RTX A4000 | 16 GB | $0.20-0.30 | Small variant, data prep |
| RTX A5000 | 24 GB | $0.30-0.40 | Medium variant (recommended) |
| RTX 4090 | 24 GB | $0.34-0.44 | Medium variant, fast training |
| RTX A6000 | 48 GB | $0.50-0.70 | Large variant |
| A100 PCIe 40GB | 40 GB | $0.95-1.20 | Large variant |
| A100 PCIe 80GB | 80 GB | $1.50-2.00 | Large variant, multi-GPU |
| H100 PCIe | 80 GB | $2.50-3.50 | Maximum performance |

### Secure Cloud (Enterprise)

Secure Cloud pricing is typically 20-40% higher than Community Cloud but offers:
- Certified data centers
- Enhanced security compliance
- Guaranteed availability

### Storage Pricing

| Storage Type | Price |
|--------------|-------|
| Network Volume (running) | $0.10/GB/month |
| Network Volume (idle) | $0.20/GB/month |
| Container Disk | $0.10/GB/month |

**Recommended**: 200GB Network Volume = **$20-40/month**

> **Note**: Full pipeline requires ~140GB (raw datasets ~92GB + processed ~24GB + embedded ~17GB + checkpoints ~4GB + other ~3.5GB). 200GB provides 30% buffer for growth.

---

## Training Cost Breakdown by Stage

### 1. Data Preparation (One-Time)

| Task | GPU | Hours | Cost/Hour | Total |
|------|-----|-------|-----------|-------|
| Text embedding (CLIP) | RTX A4000 | 4 | $0.25 | $1.00 |
| Data curation pipeline | RTX A4000 | 2 | $0.25 | $0.50 |
| **Data Prep Total** | | **6** | | **$1.50** |

### 2. Pretraining Costs

| Variant | Epochs | GPU | Hours | Cost/Hour | Total |
|---------|--------|-----|-------|-----------|-------|
| Small | 30 | RTX A4000 | 12 | $0.25 | **$3.00** |
| Medium | 50 | RTX A5000 | 50 | $0.35 | **$17.50** |
| Large | 100 | A100 PCIe | 100 | $1.00 | **$100.00** |

### 3. SFT Fine-Tuning Costs (20 epochs each)

| Variant | GPU | Hours | Cost/Hour | Total |
|---------|-----|-------|-----------|-------|
| Small-SFT | RTX A4000 | 8 | $0.25 | **$2.00** |
| Medium-SFT | RTX A5000 | 15 | $0.35 | **$5.25** |
| Large-SFT | A100 PCIe | 30 | $1.00 | **$30.00** |

### 4. LoRA Fine-Tuning Costs (20 epochs each)

| Variant | GPU | Hours | Cost/Hour | Total |
|---------|-----|-------|-----------|-------|
| Small-LoRA | RTX A4000 | 5 | $0.25 | **$1.25** |
| Medium-LoRA | RTX A5000 | 8 | $0.35 | **$2.80** |
| Large-LoRA | A100 PCIe | 15 | $1.00 | **$15.00** |

---

## Total Cost Scenarios

### Scenario A: Medium Variant Only (Recommended)

| Stage | Cost |
|-------|------|
| Data Preparation | $1.50 |
| Medium Pretrain (50 epochs) | $17.50 |
| Medium SFT (20 epochs) | $5.25 |
| Medium LoRA (20 epochs) | $2.80 |
| Storage (200GB, 1 month) | $20.00 |
| **Total** | **$47.05** |

### Scenario B: All 3 Pretrained Models

| Stage | Cost |
|-------|------|
| Data Preparation | $1.50 |
| Small Pretrain | $3.00 |
| Medium Pretrain | $17.50 |
| Large Pretrain | $100.00 |
| Storage (200GB, 1 month) | $20.00 |
| **Total** | **$142.00** |

### Scenario C: Full Pipeline (All 9 Models)

| Stage | Cost |
|-------|------|
| Data Preparation | $1.50 |
| All Pretraining (3 variants) | $120.50 |
| All SFT (3 variants) | $37.25 |
| All LoRA (3 variants) | $19.05 |
| Storage (200GB, 2 months) | $40.00 |
| **Total** | **$218.30** |

---

## Cost Optimization Strategies

### 1. Use Spot/Community Instances
- Community Cloud is 20-40% cheaper than Secure Cloud
- Spot instances (when available) can save additional 30-50%
- Enable `spot_instances: true` in `runpod/config.yaml`

### 2. Checkpoint Resume
- Use `--resume` flag to continue from interruptions
- Avoid re-training from scratch if spot instance is preempted
- Checkpoints saved every epoch by default

### 3. Start with Small Variant
- Validate pipeline with Small variant first (~$3)
- Only scale to Medium/Large after confirming setup works

### 4. Use LoRA Instead of Full SFT
- LoRA training is ~50% faster than full SFT
- Similar quality for most use cases
- Medium-LoRA: $2.80 vs Medium-SFT: $5.25

### 5. Auto-Shutdown
- Enable `auto_shutdown: true` in config
- Set `max_cost_per_hour: 2.0` as safety limit
- Prevents runaway costs from forgotten instances

### 6. Off-Peak Hours
- GPU availability is often better during off-peak hours
- Prices may be slightly lower on weekends

---

## Quick Reference: Recommended Setup

For most users, we recommend:

```yaml
# runpod/config.yaml defaults
gpu:
  type: "NVIDIA RTX A5000"  # 24GB VRAM, good price/performance
training:
  network_volume:
    size_gb: 200  # Full pipeline requires ~140GB + buffer
cost:
  spot_instances: true
  auto_shutdown: true
  max_cost_per_hour: 2.0
```

**Expected Total Cost**: ~$45-50 for Medium variant with SFT and LoRA (including 200GB storage)

---

## Notes

1. **Prices are estimates** based on December 2024 Community Cloud rates
2. **Actual costs may vary** based on GPU availability and region
3. **Storage costs accumulate** even when pods are stopped
4. **Per-second billing** means you only pay for actual usage
5. **No API keys or credentials** are stored in this document

