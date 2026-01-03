from pathlib import Path

import torch

from scripts.prepare_curated_datasets import run_curation
from src.data_gen.curation import CurationConfig


def make_canonical_sample(quality: float):
    """Create a minimal canonical-like sample for integration testing.

    We use sequences longer than the minimum curation length (25 frames) so that
    enhanced filtering keeps high-quality samples.
    """

    motion = torch.zeros(50, 1, 20)
    physics = torch.zeros(50, 1, 6)
    camera = torch.zeros(50, 3)
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

    # Enhanced curation and source balancing may downsample small test sets,
    # but we should still keep at least one high-quality sample in pretrain,
    # and SFT should never exceed the size of pretrain.
    assert len(pretrain) >= 1
    assert len(sft) <= len(pretrain)

    # Check that samples keep canonical fields
    sample = pretrain[0]
    assert "motion" in sample and "physics" in sample and "camera" in sample
