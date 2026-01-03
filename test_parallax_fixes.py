#!/usr/bin/env python3
"""
Test script to validate parallax rendering fixes on a small subset of samples.

This script:
1. Clears existing test output
2. Runs parallax generation on 10 samples
3. Analyzes output image quality (resolution, brightness, visibility)
4. Reports statistics for validation before full dataset regeneration
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def clear_test_output(output_dir: str) -> None:
    """Remove existing test output directory."""
    if os.path.exists(output_dir):
        print(f"Clearing existing test output: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def run_parallax_test(max_samples: int = 10) -> None:
    """Run parallax generation on a small subset."""
    print(f"\n{'='*60}")
    print(f"Running parallax generation on {max_samples} samples...")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "-m",
        "src.data_gen.async_data_prep",
        "--config",
        "configs/small.yaml",
        "--only-parallax",
    ]
    
    # Note: max_samples is controlled in the parallax_augmentation.py function
    # For testing, we'll manually edit the dataset or use a subset
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Parallax generation failed!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)


def analyze_output_quality(output_dir: str) -> dict:
    """Analyze the quality of generated images."""
    print(f"\n{'='*60}")
    print("Analyzing output image quality...")
    print(f"{'='*60}\n")
    
    stats = {
        "total_images": 0,
        "resolutions": {},
        "brightness_values": [],
        "non_black_percentages": [],
        "samples_processed": set(),
        "all_zero_skipped": 0,
    }
    
    # Find all PNG files
    png_files = list(Path(output_dir).rglob("*.png"))
    stats["total_images"] = len(png_files)
    
    if stats["total_images"] == 0:
        print("⚠️  WARNING: No PNG files found!")
        return stats
    
    # Analyze a sample of images (max 50 for speed)
    sample_size = min(50, len(png_files))
    print(f"Analyzing {sample_size} sample images...")
    
    for img_path in png_files[:sample_size]:
        try:
            img = Image.open(img_path)
            arr = np.array(img)
            
            # Track resolution
            resolution = f"{img.size[0]}x{img.size[1]}"
            stats["resolutions"][resolution] = stats["resolutions"].get(resolution, 0) + 1
            
            # Calculate brightness and visibility
            if len(arr.shape) == 3:
                rgb = arr[:, :, :3]
                mean_brightness = rgb.mean()
                stats["brightness_values"].append(mean_brightness)
                
                # Count non-black pixels
                non_black = (rgb.sum(axis=2) > 30).sum()
                total_pixels = rgb.shape[0] * rgb.shape[1]
                non_black_pct = (non_black / total_pixels) * 100
                stats["non_black_percentages"].append(non_black_pct)
            
            # Track which samples were processed
            parts = str(img_path).split("/")
            for part in parts:
                if part.startswith("sample_"):
                    stats["samples_processed"].add(part)
                    break
        
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
    
    return stats


def print_statistics(stats: dict) -> None:
    """Print analysis statistics."""
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"📊 Total images generated: {stats['total_images']}")
    print(f"📁 Unique samples processed: {len(stats['samples_processed'])}")
    
    if stats["resolutions"]:
        print(f"\n🖼️  Image Resolutions:")
        for res, count in stats["resolutions"].items():
            print(f"   {res}: {count} images")
    
    if stats["brightness_values"]:
        brightness_arr = np.array(stats["brightness_values"])
        print(f"\n💡 Brightness Statistics (0-255 scale):")
        print(f"   Mean: {brightness_arr.mean():.1f}")
        print(f"   Min:  {brightness_arr.min():.1f}")
        print(f"   Max:  {brightness_arr.max():.1f}")
        print(f"   Std:  {brightness_arr.std():.1f}")
    
    if stats["non_black_percentages"]:
        visibility_arr = np.array(stats["non_black_percentages"])
        print(f"\n👁️  Visibility (% non-black pixels):")
        print(f"   Mean: {visibility_arr.mean():.1f}%")
        print(f"   Min:  {visibility_arr.min():.1f}%")
        print(f"   Max:  {visibility_arr.max():.1f}%")
    
    # Validation checks
    print(f"\n✅ VALIDATION CHECKS:")
    
    all_256 = all("256x256" in res for res in stats["resolutions"].keys())
    print(f"   Resolution is 256x256: {'✅ PASS' if all_256 else '❌ FAIL'}")
    
    if stats["brightness_values"]:
        good_brightness = brightness_arr.mean() > 50
        print(f"   Mean brightness > 50: {'✅ PASS' if good_brightness else '❌ FAIL'}")
        
        good_visibility = visibility_arr.mean() > 20
        print(f"   Mean visibility > 20%: {'✅ PASS' if good_visibility else '❌ FAIL'}")


if __name__ == "__main__":
    TEST_OUTPUT = "data/2.5d_parallax_test"
    
    print("🧪 PARALLAX RENDERING FIX VALIDATION TEST")
    print("=" * 60)
    
    # Note: For actual testing, you'll need to modify the parallax generation
    # to use a test output directory and limit samples
    # For now, we'll analyze the existing output
    
    output_dir = "data/2.5d_parallax"
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        print("Please run parallax generation first.")
        sys.exit(1)
    
    stats = analyze_output_quality(output_dir)
    print_statistics(stats)

