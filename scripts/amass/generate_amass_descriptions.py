#!/usr/bin/env python3
"""
Generate Text Descriptions for Converted AMASS Data

This script:
1. Scans all converted AMASS sequences (.pt files)
2. Infers action types from filenames
3. Generates natural language descriptions
4. Creates a metadata file with descriptions for each sequence
5. Validates conversion quality

Usage:
    python generate_amass_descriptions.py
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from src.data_gen.convert_amass import (
    infer_action_from_filename,
    generate_description_from_action,
    AMASS_ACTION_MAPPING,
)
from src.data_gen.schema import ActionType, ACTION_TO_IDX
import numpy as np


def analyze_motion_quality(motion_tensor):
    """Analyze motion quality metrics"""
    # motion_tensor: [250, 20]

    # Compute smoothness (frame-to-frame differences)
    diffs = motion_tensor[1:] - motion_tensor[:-1]
    smoothness = torch.norm(diffs, dim=-1).mean().item()

    # Compute temporal consistency (second-order differences)
    second_diffs = diffs[1:] - diffs[:-1]
    consistency = torch.norm(second_diffs, dim=-1).mean().item()

    # Check for NaN or Inf values
    has_nan = torch.isnan(motion_tensor).any().item()
    has_inf = torch.isinf(motion_tensor).any().item()

    # Compute motion range (how much movement)
    motion_range = (motion_tensor.max() - motion_tensor.min()).item()

    return {
        "smoothness": smoothness,
        "consistency": consistency,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "motion_range": motion_range,
        "is_valid": not (has_nan or has_inf) and motion_range > 0.01,
    }


def generate_enhanced_description(action, dataset_name, filename):
    """Generate enhanced description with dataset context"""
    base_description = generate_description_from_action(action)

    # Add dataset-specific context
    dataset_contexts = {
        "ACCAD": "from motion capture data",
        "BMLmovi": "from the BioMotionLab dataset",
        "HDM05": "from the HDM05 motion database",
        "TotalCapture": "from full-body capture",
        "BMLhandball": "performing handball movements",
        "BMLrub": "from the BioMotionLab dataset",
        "DanceDB": "performing dance movements",
        "GRAB": "interacting with objects",
        "SFU": "from the SFU motion database",
        "Transitions": "transitioning between poses",
        "HumanEva": "from the HumanEva dataset",
        "MoSh": "from motion synthesis",
        "CMU": "from the CMU motion capture database",
        "CNRS": "from the CNRS dataset",
    }

    context = dataset_contexts.get(dataset_name, "from motion capture")
    enhanced_description = f"{base_description} {context}"

    return enhanced_description


def main():
    print("=" * 70)
    print("AMASS TEXT DESCRIPTION GENERATOR")
    print("=" * 70)

    converted_dir = Path("data/amass_converted")
    output_file = Path("data/amass_descriptions.json")

    if not converted_dir.exists():
        print(f"❌ Error: {converted_dir} does not exist")
        return

    # Find all converted .pt files
    pt_files = list(converted_dir.rglob("*.pt"))
    print(f"\nFound {len(pt_files)} converted sequences")

    descriptions = {}
    quality_stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "has_nan": 0,
        "has_inf": 0,
        "low_motion": 0,
    }

    action_counts = {}
    dataset_counts = {}

    print("\nGenerating descriptions and validating quality...")
    for pt_file in tqdm(pt_files, desc="Processing"):
        try:
            # Load motion tensor
            motion = torch.load(pt_file)

            # Get dataset name and filename
            rel_path = pt_file.relative_to(converted_dir)
            dataset_name = rel_path.parts[0]
            filename = pt_file.stem

            # Infer action from original filename
            # Reconstruct original .npz path
            original_path = str(Path("data/amass") / rel_path.with_suffix(".npz"))
            action = infer_action_from_filename(original_path)

            # Generate description
            description = generate_enhanced_description(action, dataset_name, filename)

            # Analyze quality
            quality = analyze_motion_quality(motion)

            # Store metadata
            descriptions[str(rel_path)] = {
                "description": description,
                "action": action.value,
                "action_idx": ACTION_TO_IDX[action],
                "dataset": dataset_name,
                "filename": filename,
                "frames": motion.shape[0],
                "quality": quality,
            }

            # Update statistics
            quality_stats["total"] += 1
            if quality["is_valid"]:
                quality_stats["valid"] += 1
            else:
                quality_stats["invalid"] += 1
            if quality["has_nan"]:
                quality_stats["has_nan"] += 1
            if quality["has_inf"]:
                quality_stats["has_inf"] += 1
            if quality["motion_range"] < 0.01:
                quality_stats["low_motion"] += 1

            # Count actions and datasets
            action_counts[action.value] = action_counts.get(action.value, 0) + 1
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

        except Exception as e:
            print(f"\n❌ Error processing {pt_file}: {e}")
            continue

    # Save descriptions to JSON
    print(f"\nSaving descriptions to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(descriptions, f, indent=2)

    # Print statistics
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(f"\n📊 Quality Statistics:")
    print(f"  Total sequences: {quality_stats['total']}")
    print(
        f"  Valid: {quality_stats['valid']} ({quality_stats['valid']/quality_stats['total']*100:.1f}%)"
    )
    print(
        f"  Invalid: {quality_stats['invalid']} ({quality_stats['invalid']/quality_stats['total']*100:.1f}%)"
    )
    print(f"  Has NaN: {quality_stats['has_nan']}")
    print(f"  Has Inf: {quality_stats['has_inf']}")
    print(f"  Low motion: {quality_stats['low_motion']}")

    print(f"\n📁 Dataset Distribution:")
    for dataset, count in sorted(
        dataset_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {dataset}: {count} sequences")

    print(f"\n🎬 Action Distribution (Top 15):")
    for action, count in sorted(
        action_counts.items(), key=lambda x: x[1], reverse=True
    )[:15]:
        print(f"  {action}: {count} sequences")

    print(f"\n✅ Descriptions saved to: {output_file}")
    print(f"✅ Total descriptions generated: {len(descriptions)}")


if __name__ == "__main__":
    main()
