#!/usr/bin/env python3
"""
Convert GRAB dataset to stick figure format
Auto-generated conversion script
"""

import os
import sys
import time
from pathlib import Path
from src.data_gen.convert_amass import AMASSConverter
import torch


def find_npz_files(dataset_path):
    """Find all .npz files excluding shape.npz"""
    npz_files = []
    for npz_file in Path(dataset_path).rglob("*.npz"):
        if npz_file.name != "shape.npz":
            npz_files.append(npz_file)
    return sorted(npz_files)


def main():
    dataset_name = "GRAB"
    dataset_path = Path("data/amass") / dataset_name
    output_dir = Path("data/amass_converted") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {dataset_name} dataset...")
    print(f"Output: {output_dir}")

    converter = AMASSConverter(smpl_model_path="data/smpl_models")
    npz_files = find_npz_files(dataset_path)

    print(f"Found {len(npz_files)} files to convert")

    converted = 0
    errors = 0
    start_time = time.time()

    for i, npz_file in enumerate(npz_files, 1):
        try:
            # Check if already converted
            rel_path = npz_file.relative_to(dataset_path)
            output_file = output_dir / rel_path.with_suffix(".pt")

            if output_file.exists():
                print(f"[{i}/{len(npz_files)}] Skipping (exists): {npz_file.name}")
                continue

            # Convert
            motion = converter.convert_sequence(str(npz_file))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(motion, output_file)

            converted += 1

            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(npz_files) - i) / rate / 60
                print(
                    f"[{i}/{len(npz_files)}] {converted} converted, {errors} errors, {rate:.1f} files/sec, ETA: {eta:.1f} min"
                )

        except Exception as e:
            errors += 1
            print(f"[{i}/{len(npz_files)}] ERROR: {npz_file.name}: {e}")

    elapsed = time.time() - start_time
    print(
        f"\nCompleted: {converted} converted, {errors} errors in {elapsed/60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
