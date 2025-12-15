#!/usr/bin/env python3
"""
Generate Comprehensive AMASS Conversion Statistics Report

This script generates a detailed report of the AMASS dataset conversion process,
including file counts, sizes, errors, and processing times.

Usage:
    python generate_conversion_stats.py
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import torch

def get_directory_size(path):
    """Get total size of directory in bytes"""
    result = subprocess.run(['du', '-sb', str(path)], capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.split()[0])
    return 0

def count_files_in_dir(path, extension='.pt'):
    """Count files with specific extension in directory"""
    return len(list(Path(path).rglob(f'*{extension}')))

def get_dataset_stats(converted_dir):
    """Get statistics for each dataset"""
    stats = {}
    
    for dataset_dir in sorted(Path(converted_dir).iterdir()):
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            pt_files = list(dataset_dir.rglob('*.pt'))
            
            total_frames = 0
            total_size = 0
            
            for pt_file in pt_files:
                try:
                    motion = torch.load(pt_file)
                    total_frames += motion.shape[0]
                    total_size += pt_file.stat().st_size
                except Exception as e:
                    print(f"Warning: Could not load {pt_file}: {e}")
            
            stats[dataset_name] = {
                'files': len(pt_files),
                'frames': total_frames,
                'size_bytes': total_size,
                'size_mb': total_size / (1024 * 1024),
                'avg_file_size_kb': (total_size / len(pt_files) / 1024) if len(pt_files) > 0 else 0
            }
    
    return stats

def main():
    print("="*70)
    print("AMASS CONVERSION STATISTICS GENERATOR")
    print("="*70)
    
    converted_dir = Path('data/amass_converted')
    amass_dir = Path('data/amass')
    descriptions_file = Path('data/amass_descriptions.json')
    
    if not converted_dir.exists():
        print(f"❌ Error: {converted_dir} does not exist")
        return
    
    # Get overall statistics
    print("\n📊 Collecting statistics...")
    
    total_converted_files = count_files_in_dir(converted_dir, '.pt')
    total_size_bytes = get_directory_size(converted_dir)
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_mb / 1024
    
    # Get per-dataset statistics
    dataset_stats = get_dataset_stats(converted_dir)
    
    # Load descriptions if available
    action_distribution = {}
    total_frames = 0
    
    if descriptions_file.exists():
        with open(descriptions_file, 'r') as f:
            descriptions = json.load(f)
        
        for metadata in descriptions.values():
            action = metadata['action']
            action_distribution[action] = action_distribution.get(action, 0) + 1
            total_frames += metadata.get('frames', 250)
    else:
        # Estimate total frames
        total_frames = total_converted_files * 250
    
    # Count available source files
    available_datasets = {}
    for dataset_dir in sorted(amass_dir.iterdir()):
        if dataset_dir.is_dir():
            npz_count = len([f for f in dataset_dir.rglob('*.npz') if f.name != 'shape.npz'])
            available_datasets[dataset_dir.name] = npz_count
    
    # Generate report
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_converted_files': total_converted_files,
            'total_frames': total_frames,
            'total_size_mb': round(total_size_mb, 2),
            'total_size_gb': round(total_size_gb, 2),
            'avg_file_size_kb': round((total_size_bytes / total_converted_files / 1024), 2) if total_converted_files > 0 else 0,
            'datasets_converted': len(dataset_stats),
            'datasets_available': len(available_datasets)
        },
        'dataset_statistics': dataset_stats,
        'available_datasets': available_datasets,
        'action_distribution': action_distribution
    }
    
    # Save report
    output_file = Path('AMASS_CONVERSION_STATISTICS.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("CONVERSION STATISTICS SUMMARY")
    print("="*70)
    
    print(f"\n📁 Overall Statistics:")
    print(f"  Total Converted Files: {total_converted_files:,}")
    print(f"  Total Frames: {total_frames:,}")
    print(f"  Total Size: {total_size_gb:.2f} GB ({total_size_mb:.2f} MB)")
    print(f"  Average File Size: {report['summary']['avg_file_size_kb']:.2f} KB")
    print(f"  Datasets Converted: {len(dataset_stats)}")
    print(f"  Datasets Available: {len(available_datasets)}")
    
    print(f"\n📊 Per-Dataset Statistics:")
    for dataset, stats in sorted(dataset_stats.items(), key=lambda x: x[1]['files'], reverse=True):
        available = available_datasets.get(dataset, 0)
        conversion_rate = (stats['files'] / available * 100) if available > 0 else 0
        print(f"  {dataset}:")
        print(f"    Files: {stats['files']:,} / {available:,} ({conversion_rate:.1f}%)")
        print(f"    Frames: {stats['frames']:,}")
        print(f"    Size: {stats['size_mb']:.2f} MB")
        print(f"    Avg File Size: {stats['avg_file_size_kb']:.2f} KB")
    
    if action_distribution:
        print(f"\n🎬 Action Distribution (Top 10):")
        for action, count in sorted(action_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {action}: {count:,} sequences ({count/total_converted_files*100:.1f}%)")
    
    print(f"\n✅ Statistics saved to: {output_file}")

if __name__ == '__main__':
    main()

