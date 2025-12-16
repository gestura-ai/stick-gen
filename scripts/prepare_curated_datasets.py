#!/usr/bin/env python3
"""Prepare curated pretraining and SFT datasets from canonical .pt files.

This script loads one or more canonical datasets, applies quality and physics
filters, optionally balances action types for SFT, and writes:

- ``pretrain_data.pt``
- ``sft_data.pt``
- ``curation_stats.json``
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.data_gen.curation import (
    CurationConfig,
    curate_samples,
    load_canonical_datasets,
)


def run_curation(input_paths: list[str], output_dir: str, cfg: CurationConfig) -> None:
    samples = load_canonical_datasets(input_paths)
    pretrain, sft, stats = curate_samples(samples, cfg)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pretrain, out_dir / "pretrain_data.pt")
    torch.save(sft, out_dir / "sft_data.pt")

    stats_path = out_dir / "curation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(
        f"Curation complete. Pretrain: {len(pretrain)} samples, SFT: {len(sft)} samples"
    )
    print(f"Stats written to {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare curated pretraining and SFT datasets"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to canonical .pt files (e.g. AMASS, InterHuman, NTU, 100STYLE, synthetic)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/curated",
        help="Directory to write curated splits and stats",
    )
    parser.add_argument(
        "--min-quality-pretrain",
        type=float,
        default=0.5,
        help="Minimum quality_score for inclusion in pretraining split",
    )
    parser.add_argument(
        "--min-quality-sft",
        type=float,
        default=0.8,
        help="Minimum quality_score for inclusion in SFT split",
    )
    parser.add_argument(
        "--min-camera-stability-sft",
        type=float,
        default=0.6,
        help="Minimum camera stability score for SFT when camera is available",
    )
    parser.add_argument(
        "--balance-max-fraction",
        type=float,
        default=0.3,
        help="Maximum fraction of SFT samples allowed for any single dominant action",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CurationConfig(
        min_quality_pretrain=args.min_quality_pretrain,
        min_quality_sft=args.min_quality_sft,
        min_camera_stability_sft=args.min_camera_stability_sft,
        balance_max_fraction=args.balance_max_fraction,
    )

    # Expand any directories passed in as inputs to *.pt files
    input_paths: list[str] = []
    for p in args.inputs:
        if os.path.isdir(p):
            for entry in sorted(Path(p).glob("*.pt")):
                input_paths.append(str(entry))
        else:
            input_paths.append(p)

    if not input_paths:
        raise SystemExit("No input .pt files found")

    run_curation(input_paths, args.output_dir, cfg)


if __name__ == "__main__":
    main()
