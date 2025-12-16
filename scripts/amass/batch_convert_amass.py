#!/usr/bin/env python3
"""
Batch convert all verified AMASS datasets to stick figure format

This script converts all SMPL+H compatible datasets to stick figure format
and saves the output to data/amass_converted/
"""

import time
from pathlib import Path

import torch

from src.data_gen.convert_amass import AMASSConverter

# Datasets that are SMPL+H compatible (from verification)
COMPATIBLE_DATASETS = [
    "ACCAD",  # 252 files (100% SMPL+H)
    "BMLmovi",  # 1,801 files (100% SMPL+H) - excluding shape.npz files
    "HDM05",  # 215 files (100% SMPL+H)
    "TotalCapture",  # 37 files (100% SMPL+H)
    "BMLhandball",  # 649 files (98.5% SMPL+H)
    "BMLrub",  # 3,061 files (100% SMPL+H)
    "DanceDB",  # 153 files (88.4% SMPL+H) - excluding shape.npz files
    "SFU",  # 44 files (100% SMPL+H)
    "Transitions",  # 110 files (100% SMPL+H)
    "HumanEva",  # 28 files (100% SMPL+H)
    "MoSh",  # 77 files (100% SMPL+H)
    "GRAB",  # 1,350 files (100% SMPL+H) - grasping and object interaction
]

# Datasets that need SMPL-X support (not yet implemented)
SMPLX_DATASETS = [
    "CMU",  # 2,079 files (165 params - SMPL-X with facial features)
    "CNRS",  # 81 files (165 params - SMPL-X with facial features)
]


def find_npz_files(dataset_path: Path) -> list:
    """Find all .npz files in dataset, excluding shape.npz files"""
    npz_files = []
    for npz_file in dataset_path.rglob("*.npz"):
        # Skip shape.npz files (they contain body shape params, not motion)
        if npz_file.name == "shape.npz":
            continue
        npz_files.append(npz_file)
    return sorted(npz_files)


def convert_dataset(dataset_name: str, converter: AMASSConverter, output_dir: Path):
    """Convert all sequences in a dataset"""
    dataset_path = Path("data/amass") / dataset_name
    output_dataset_dir = output_dir / dataset_name
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Converting dataset: {dataset_name}")
    print(f"{'='*70}")

    npz_files = find_npz_files(dataset_path)
    print(f"Found {len(npz_files)} .npz files")

    stats = {
        "total": len(npz_files),
        "converted": 0,
        "skipped": 0,
        "errors": 0,
        "total_frames": 0,
        "total_size_mb": 0,
    }

    start_time = time.time()

    for i, npz_file in enumerate(npz_files, 1):
        try:
            # Convert sequence
            motion = converter.convert_sequence(str(npz_file))

            # Create output path (preserve directory structure)
            rel_path = npz_file.relative_to(dataset_path)
            output_file = output_dataset_dir / rel_path.with_suffix(".pt")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save converted motion
            torch.save(motion, output_file)

            # Update stats
            stats["converted"] += 1
            stats["total_frames"] += motion.shape[0]
            stats["total_size_mb"] += output_file.stat().st_size / (1024 * 1024)

            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(npz_files) - i) / rate if rate > 0 else 0
                print(
                    f"  Progress: {i}/{len(npz_files)} ({i/len(npz_files)*100:.1f}%) - "
                    f"{rate:.1f} files/sec - ETA: {remaining/60:.1f} min"
                )

        except Exception as e:
            print(f"  ❌ Error converting {npz_file.name}: {e}")
            stats["errors"] += 1

    elapsed = time.time() - start_time

    print(f"\n{dataset_name} Conversion Complete:")
    print(f"  ✅ Converted: {stats['converted']} files")
    print(f"  ⏭️  Skipped: {stats['skipped']} files")
    print(f"  ❌ Errors: {stats['errors']} files")
    print(f"  📊 Total frames: {stats['total_frames']:,}")
    print(f"  💾 Total size: {stats['total_size_mb']:.1f} MB")
    print(
        f"  ⏱️  Time: {elapsed/60:.1f} minutes ({stats['converted']/elapsed:.1f} files/sec)"
    )

    return stats


def main():
    print("=" * 70)
    print("AMASS Batch Conversion to Stick Figure Format")
    print("=" * 70)

    # Initialize converter
    print("\nInitializing AMASS converter...")
    converter = AMASSConverter(smpl_model_path="data/smpl_models")

    # Create output directory
    output_dir = Path("data/amass_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert all compatible datasets
    all_stats = []
    total_start = time.time()

    for dataset_name in COMPATIBLE_DATASETS:
        dataset_path = Path("data/amass") / dataset_name
        if not dataset_path.exists():
            print(f"\n⚠️  Skipping {dataset_name} - not found")
            continue

        stats = convert_dataset(dataset_name, converter, output_dir)
        all_stats.append((dataset_name, stats))

    total_elapsed = time.time() - total_start

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_converted = sum(s["converted"] for _, s in all_stats)
    total_errors = sum(s["errors"] for _, s in all_stats)
    total_frames = sum(s["total_frames"] for _, s in all_stats)
    total_size = sum(s["total_size_mb"] for _, s in all_stats)

    print(f"\nDatasets processed: {len(all_stats)}")
    print(f"Total files converted: {total_converted:,}")
    print(f"Total errors: {total_errors}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total output size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(
        f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)"
    )
    print(f"Average rate: {total_converted/total_elapsed:.1f} files/sec")

    print(f"\n✅ Conversion complete! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
