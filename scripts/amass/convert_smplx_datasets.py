#!/usr/bin/env python3
"""
Convert SMPL-X datasets (CMU and CNRS) to stick figure format

This script converts CMU and CNRS datasets which use SMPL-X format (165 params)
Requires SMPL-X models to be installed in data/smpl_models/smplx/
"""

import os
import sys
import time
from pathlib import Path
from src.data_gen.convert_amass import AMASSConverter
import torch
import numpy as np

# SMPL-X datasets
SMPLX_DATASETS = [
    'CMU',        # 2,079 files (165 params - SMPL-X with facial features)
    'CNRS',       # 81 files (165 params - SMPL-X with facial features)
]

def find_npz_files(dataset_path: Path) -> list:
    """Find all .npz files in dataset, excluding shape.npz files"""
    npz_files = []
    for npz_file in dataset_path.rglob('*.npz'):
        # Skip shape.npz files (they contain body shape params, not motion)
        if npz_file.name == 'shape.npz':
            continue
        npz_files.append(npz_file)
    return sorted(npz_files)

def convert_dataset(dataset_name: str, converter: AMASSConverter, output_dir: Path):
    """Convert all sequences in a dataset"""
    dataset_path = Path('data/amass') / dataset_name
    output_dataset_dir = output_dir / dataset_name
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Converting SMPL-X dataset: {dataset_name}")
    print(f"{'='*70}")
    
    npz_files = find_npz_files(dataset_path)
    print(f"Found {len(npz_files)} .npz files")
    
    # Check for existing conversions
    existing_files = set()
    for pt_file in output_dataset_dir.rglob('*.pt'):
        existing_files.add(pt_file.stem)
    
    if existing_files:
        print(f"Found {len(existing_files)} already converted files (will skip)")
    
    stats = {
        'total': len(npz_files),
        'converted': 0,
        'skipped': 0,
        'errors': 0,
        'total_frames': 0,
        'total_size_mb': 0,
    }
    
    start_time = time.time()
    
    for i, npz_file in enumerate(npz_files, 1):
        try:
            # Check if already converted
            rel_path = npz_file.relative_to(dataset_path)
            output_file = output_dataset_dir / rel_path.with_suffix('.pt')
            
            if output_file.exists():
                stats['skipped'] += 1
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(npz_files)} ({i/len(npz_files)*100:.1f}%) - "
                          f"{stats['converted']} converted, {stats['skipped']} skipped, {stats['errors']} errors")
                continue
            
            # Convert sequence
            motion = converter.convert_sequence(str(npz_file))
            
            # Create output path (preserve directory structure)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save converted motion
            torch.save(motion, output_file)
            
            # Update stats
            stats['converted'] += 1
            stats['total_frames'] += motion.shape[0]
            stats['total_size_mb'] += output_file.stat().st_size / (1024 * 1024)
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = stats['converted'] / elapsed if elapsed > 0 else 0
                remaining_files = len(npz_files) - i
                remaining_time = remaining_files / rate if rate > 0 else 0
                print(f"  Progress: {i}/{len(npz_files)} ({i/len(npz_files)*100:.1f}%) - "
                      f"{rate:.1f} files/sec - ETA: {remaining_time/60:.1f} min - "
                      f"{stats['converted']} converted, {stats['skipped']} skipped, {stats['errors']} errors")
        
        except Exception as e:
            print(f"  ❌ Error converting {npz_file.name}: {e}")
            stats['errors'] += 1
            if stats['errors'] <= 5:  # Show first 5 errors in detail
                import traceback
                traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    print(f"\n{dataset_name} Conversion Complete:")
    print(f"  ✅ Converted: {stats['converted']} files")
    print(f"  ⏭️  Skipped: {stats['skipped']} files (already converted)")
    print(f"  ❌ Errors: {stats['errors']} files")
    print(f"  📊 Total frames: {stats['total_frames']:,}")
    print(f"  💾 Total size: {stats['total_size_mb']:.1f} MB")
    if elapsed > 0:
        print(f"  ⏱️  Time: {elapsed/60:.1f} minutes ({stats['converted']/elapsed:.1f} files/sec)")
    
    return stats

def main():
    print("="*70)
    print("SMPL-X Dataset Conversion (CMU and CNRS)")
    print("="*70)
    
    # Check if SMPL-X models exist
    smplx_model_dir = Path('data/smpl_models/smplx')
    required_models = ['SMPLX_NEUTRAL.npz', 'SMPLX_MALE.npz', 'SMPLX_NEUTRAL.pkl', 'SMPLX_MALE.pkl']
    
    has_model = False
    for model_file in required_models:
        if (smplx_model_dir / model_file).exists():
            has_model = True
            print(f"✅ Found SMPL-X model: {model_file}")
            break
    
    if not has_model:
        print("\n❌ ERROR: SMPL-X models not found!")
        print(f"   Expected location: {smplx_model_dir}")
        print(f"   Required files: SMPLX_NEUTRAL.npz or SMPLX_MALE.npz (or .pkl versions)")
        print("\n   Please download SMPL-X models from: https://smpl-x.is.tue.mpg.de/")
        print("   See SMPLX_DOWNLOAD_GUIDE.md for instructions")
        sys.exit(1)
    
    # Initialize converter
    print("\nInitializing AMASS converter with SMPL-X support...")
    try:
        converter = AMASSConverter(smpl_model_path='data/smpl_models')
        print("✅ Converter initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize converter: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create output directory
    output_dir = Path('data/amass_converted')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert all SMPL-X datasets
    all_stats = []
    total_start = time.time()
    
    for dataset_name in SMPLX_DATASETS:
        dataset_path = Path('data/amass') / dataset_name
        if not dataset_path.exists():
            print(f"\n⚠️  Skipping {dataset_name} - not found at {dataset_path}")
            continue
        
        stats = convert_dataset(dataset_name, converter, output_dir)
        all_stats.append((dataset_name, stats))
    
    total_elapsed = time.time() - total_start
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    total_converted = sum(s['converted'] for _, s in all_stats)
    total_skipped = sum(s['skipped'] for _, s in all_stats)
    total_errors = sum(s['errors'] for _, s in all_stats)
    total_frames = sum(s['total_frames'] for _, s in all_stats)
    total_size = sum(s['total_size_mb'] for _, s in all_stats)
    
    print(f"\nDatasets processed: {len(all_stats)}")
    print(f"Total files converted: {total_converted:,}")
    print(f"Total files skipped: {total_skipped:,}")
    print(f"Total errors: {total_errors}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total output size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    if total_elapsed > 0:
        print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
        if total_converted > 0:
            print(f"Average rate: {total_converted/total_elapsed:.1f} files/sec")
    
    print(f"\n✅ Conversion complete! Output saved to: {output_dir}")
    
    # Show final dataset counts
    print("\n" + "="*70)
    print("FINAL DATASET COUNTS")
    print("="*70)
    for dataset_name in SMPLX_DATASETS:
        output_dataset_dir = output_dir / dataset_name
        if output_dataset_dir.exists():
            pt_count = len(list(output_dataset_dir.rglob('*.pt')))
            print(f"  {dataset_name}: {pt_count} files")

if __name__ == "__main__":
    main()

