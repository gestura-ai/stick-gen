#!/usr/bin/env bash
#
# Launch Asynchronous AMASS Dataset Conversions
#
# This script launches remaining AMASS dataset conversions as background processes
# in separate terminal sessions so they can run asynchronously without blocking.
#
# Usage:
#   ./launch_async_amass_conversions.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "ASYNCHRONOUS AMASS DATASET CONVERSION LAUNCHER"
echo "========================================================================"
echo ""

# Check which datasets need conversion
echo "Checking conversion status..."
echo ""

# Function to count files
count_npz() {
    find "data/amass/$1" -name "*.npz" 2>/dev/null | grep -v "shape.npz" | wc -l | tr -d ' '
}

count_pt() {
    find "data/amass_converted/$1" -name "*.pt" 2>/dev/null | wc -l | tr -d ' '
}

# Check CMU
CMU_NPZ=$(count_npz "CMU")
CMU_PT=$(count_pt "CMU")
echo "CMU: $CMU_NPZ NPZ files, $CMU_PT PT files converted"

# Check CNRS
CNRS_NPZ=$(count_npz "CNRS")
CNRS_PT=$(count_pt "CNRS")
echo "CNRS: $CNRS_NPZ NPZ files, $CNRS_PT PT files converted"

# Check GRAB
GRAB_NPZ=$(count_npz "GRAB")
GRAB_PT=$(count_pt "GRAB")
echo "GRAB: $GRAB_NPZ NPZ files, $GRAB_PT PT files converted"

echo ""
echo "========================================================================"
echo "LAUNCHING CONVERSIONS"
echo "========================================================================"
echo ""

# Create conversion script for individual datasets
create_conversion_script() {
    local dataset=$1
    local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local script_name="convert_${dataset_lower}.py"
    
    cat > "$script_name" << EOF
#!/usr/bin/env python3
"""
Convert $dataset dataset to stick figure format
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
    for npz_file in Path(dataset_path).rglob('*.npz'):
        if npz_file.name != 'shape.npz':
            npz_files.append(npz_file)
    return sorted(npz_files)

def main():
    dataset_name = "$dataset"
    dataset_path = Path('data/amass') / dataset_name
    output_dir = Path('data/amass_converted') / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {dataset_name} dataset...")
    print(f"Output: {output_dir}")
    
    converter = AMASSConverter(smpl_model_path='data/smpl_models')
    npz_files = find_npz_files(dataset_path)
    
    print(f"Found {len(npz_files)} files to convert")
    
    converted = 0
    errors = 0
    start_time = time.time()
    
    for i, npz_file in enumerate(npz_files, 1):
        try:
            # Check if already converted
            rel_path = npz_file.relative_to(dataset_path)
            output_file = output_dir / rel_path.with_suffix('.pt')
            
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
                print(f"[{i}/{len(npz_files)}] {converted} converted, {errors} errors, {rate:.1f} files/sec, ETA: {eta:.1f} min")
        
        except Exception as e:
            errors += 1
            print(f"[{i}/{len(npz_files)}] ERROR: {npz_file.name}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\\nCompleted: {converted} converted, {errors} errors in {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$script_name"
    echo "$script_name"
}

# Launch CMU conversion (largest dataset - 2,079 files)
if [ "$CMU_PT" -lt "$CMU_NPZ" ]; then
    echo "Launching CMU conversion (2,079 files)..."
    CMU_SCRIPT=$(create_conversion_script "CMU")
    nohup python3 "$CMU_SCRIPT" > amass_conversion_cmu.log 2>&1 &
    CMU_PID=$!
    echo "  ✓ CMU conversion started (PID: $CMU_PID)"
    echo "  Log: amass_conversion_cmu.log"
    echo "  Monitor: tail -f amass_conversion_cmu.log"
    echo ""
else
    echo "✓ CMU already converted"
    echo ""
fi

# Launch CNRS conversion (81 files)
if [ "$CNRS_PT" -lt "$CNRS_NPZ" ]; then
    echo "Launching CNRS conversion (81 files)..."
    CNRS_SCRIPT=$(create_conversion_script "CNRS")
    nohup python3 "$CNRS_SCRIPT" > amass_conversion_cnrs.log 2>&1 &
    CNRS_PID=$!
    echo "  ✓ CNRS conversion started (PID: $CNRS_PID)"
    echo "  Log: amass_conversion_cnrs.log"
    echo "  Monitor: tail -f amass_conversion_cnrs.log"
    echo ""
else
    echo "✓ CNRS already converted"
    echo ""
fi

# Launch GRAB completion (10 remaining files)
if [ "$GRAB_PT" -lt "$GRAB_NPZ" ]; then
    echo "Launching GRAB completion ($(($GRAB_NPZ - $GRAB_PT)) remaining files)..."
    GRAB_SCRIPT=$(create_conversion_script "GRAB")
    nohup python3 "$GRAB_SCRIPT" > amass_conversion_grab_complete.log 2>&1 &
    GRAB_PID=$!
    echo "  ✓ GRAB completion started (PID: $GRAB_PID)"
    echo "  Log: amass_conversion_grab_complete.log"
    echo "  Monitor: tail -f amass_conversion_grab_complete.log"
    echo ""
else
    echo "✓ GRAB already converted"
    echo ""
fi

echo "========================================================================"
echo "CONVERSION STATUS"
echo "========================================================================"
echo ""
echo "All conversions launched as background processes!"
echo ""
echo "Monitor progress:"
echo "  CMU:  tail -f amass_conversion_cmu.log"
echo "  CNRS: tail -f amass_conversion_cnrs.log"
echo "  GRAB: tail -f amass_conversion_grab_complete.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep 'convert_' | grep -v grep"
echo ""
echo "Estimated completion times (CPU):"
echo "  CMU:  ~15-20 hours (2,079 files @ 2 files/sec)"
echo "  CNRS: ~1 hour (81 files @ 2 files/sec)"
echo "  GRAB: ~5 minutes (10 files @ 2 files/sec)"
echo ""
echo "You can continue with other work while conversions run in background!"
echo "========================================================================"

