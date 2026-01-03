#!/bin/bash
# Test script to validate parallax rendering fixes on 10 samples

set -e

echo "=========================================="
echo "PARALLAX RENDERING FIX VALIDATION TEST"
echo "=========================================="
echo ""

# Create test output directory
TEST_DIR="data/2.5d_parallax_test"
echo "Creating test output directory: $TEST_DIR"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Create a temporary test dataset with only 10 samples
echo "Creating test dataset (10 samples)..."
python3 << 'EOF'
import torch
import os

# Load full dataset
data = torch.load('data/processed/curated/pretrain_data.pt')

# Take first 10 samples (including some that might have issues)
test_data = data[:10]

# Save test dataset
test_path = 'data/processed/curated/pretrain_data_test.pt'
torch.save(test_data, test_path)
print(f"Created test dataset with {len(test_data)} samples at {test_path}")
EOF

# Run parallax generation on test dataset
echo ""
echo "Running parallax generation on test dataset..."
echo "This will generate ~2,500 images (10 samples × 250 views × 1 actor)"
echo ""

python3 << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())

from src.data_gen.parallax_augmentation import generate_parallax_for_dataset

generate_parallax_for_dataset(
    dataset_path='data/processed/curated/pretrain_data_test.pt',
    output_dir='data/2.5d_parallax_test',
    views_per_motion=250,
    node_script='src/data_gen/renderers/threejs_parallax_renderer.js',
    max_samples=10,
    fps=25,
    frames_per_view=4,
)
EOF

# Analyze results
echo ""
echo "=========================================="
echo "ANALYZING TEST RESULTS"
echo "=========================================="
echo ""

python3 << 'EOF'
import os
from pathlib import Path
from PIL import Image
import numpy as np

output_dir = 'data/2.5d_parallax_test'

# Find all PNG files
png_files = list(Path(output_dir).rglob("*.png"))
print(f"📊 Total images generated: {len(png_files)}")

# Count samples
samples = set()
for f in png_files:
    parts = str(f).split('/')
    for part in parts:
        if part.startswith('sample_'):
            samples.add(part)
            break
print(f"📁 Unique samples processed: {len(samples)}")
print(f"   Samples: {sorted(samples)}")

# Analyze image quality (sample 20 images)
if png_files:
    sample_size = min(20, len(png_files))
    print(f"\n🔍 Analyzing {sample_size} sample images...")
    
    resolutions = {}
    brightness_vals = []
    visibility_vals = []
    
    for img_path in png_files[:sample_size]:
        img = Image.open(img_path)
        arr = np.array(img)
        
        # Resolution
        res = f"{img.size[0]}x{img.size[1]}"
        resolutions[res] = resolutions.get(res, 0) + 1
        
        # Brightness and visibility
        if len(arr.shape) == 3:
            rgb = arr[:, :, :3]
            brightness = rgb.mean()
            brightness_vals.append(brightness)
            
            non_black = (rgb.sum(axis=2) > 30).sum()
            total = rgb.shape[0] * rgb.shape[1]
            visibility = (non_black / total) * 100
            visibility_vals.append(visibility)
    
    print(f"\n🖼️  Image Resolutions:")
    for res, count in resolutions.items():
        print(f"   {res}: {count} images")
    
    if brightness_vals:
        brightness_arr = np.array(brightness_vals)
        print(f"\n💡 Brightness (0-255 scale):")
        print(f"   Mean: {brightness_arr.mean():.1f}")
        print(f"   Range: [{brightness_arr.min():.1f}, {brightness_arr.max():.1f}]")
        
        visibility_arr = np.array(visibility_vals)
        print(f"\n👁️  Visibility (% non-black pixels):")
        print(f"   Mean: {visibility_arr.mean():.1f}%")
        print(f"   Range: [{visibility_arr.min():.1f}%, {visibility_arr.max():.1f}%]")
    
    # Validation
    print(f"\n✅ VALIDATION CHECKS:")
    all_256 = all("256x256" in res for res in resolutions.keys())
    print(f"   ✓ Resolution is 256x256: {'PASS ✅' if all_256 else 'FAIL ❌'}")
    
    if brightness_vals:
        good_brightness = brightness_arr.mean() > 50
        print(f"   ✓ Mean brightness > 50: {'PASS ✅' if good_brightness else 'FAIL ❌'} ({brightness_arr.mean():.1f})")
        
        good_visibility = visibility_arr.mean() > 20
        print(f"   ✓ Mean visibility > 20%: {'PASS ✅' if good_visibility else 'FAIL ❌'} ({visibility_arr.mean():.1f}%)")
    
    # Show sample file paths
    print(f"\n📁 Sample output files:")
    for f in sorted(png_files)[:5]:
        print(f"   {f}")

EOF

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo ""
echo "Review the results above. If all validation checks pass:"
echo "  - Resolution should be 256x256 (not 512x512)"
echo "  - Mean brightness should be > 50 (not < 5)"
echo "  - Mean visibility should be > 20% (not < 5%)"
echo ""
echo "If tests pass, you can proceed with full dataset regeneration."
echo "If tests fail, review the fixes and try again."

