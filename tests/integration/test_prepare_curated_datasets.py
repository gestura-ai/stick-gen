from pathlib import Path

import torch

from scripts.prepare_curated_datasets import run_curation
from src.data_gen.curation import CurationConfig


def make_canonical_sample(quality: float):
    motion = torch.zeros(10, 1, 20)
    physics = torch.zeros(10, 1, 6)
    camera = torch.zeros(10, 3)
    annotations = {"quality": {"score": quality}, "actions": {"dominant": ["walk"]}}
    return {
        "description": "test",
        "motion": motion,
        "physics": physics,
        "camera": camera,
        "annotations": annotations,
        "quality_score": quality,
    }


def test_run_curation_creates_outputs(tmp_path: Path):
    data = [make_canonical_sample(0.9) for _ in range(3)] + [make_canonical_sample(0.4)]
    input_path = tmp_path / "canonical.pt"
    torch.save(data, input_path)

    out_dir = tmp_path / "out"

    cfg = CurationConfig(min_quality_pretrain=0.5, min_quality_sft=0.8)
    run_curation([str(input_path)], str(out_dir), cfg)

    pretrain_path = out_dir / "pretrain_data.pt"
    sft_path = out_dir / "sft_data.pt"
    stats_path = out_dir / "curation_stats.json"

    assert pretrain_path.exists()
    assert sft_path.exists()
    assert stats_path.exists()

    pretrain = torch.load(pretrain_path)
    sft = torch.load(sft_path)

    assert len(pretrain) == 3
    # With default balance_max_fraction=0.3 and a single dominant action,
    # SFT should keep a strict subset of the pretraining samples.
    assert 0 < len(sft) < len(pretrain)

    # Check that samples keep canonical fields
    sample = pretrain[0]
    assert "motion" in sample and "physics" in sample and "camera" in sample
