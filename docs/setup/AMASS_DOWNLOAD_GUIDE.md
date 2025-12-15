# AMASS Dataset Download Guide

Complete step-by-step instructions for downloading and setting up the AMASS dataset for stick-gen training.

**🚀 NEW**: Automated download script available! See "Quick Start" below.

---

## ⚡ **Quick Start (Automated)**

We've created an automated download script to simplify the process:

```bash
# Install dependencies
pip install requests tqdm

# Download recommended subset (5 datasets, ~10GB)
python download_amass.py

# Or download specific datasets
python download_amass.py --datasets CMU BMLmovi ACCAD

# Or download small test subset (~2GB)
python download_amass.py --subset

# Or download everything (~50GB)
python download_amass.py --all
```

**Note**: You still need to register at https://amass.is.tue.mpg.de/ first (see Step 1 below).

The script will:
1. Prompt for your AMASS credentials
2. Guide you through manual downloads (AMASS requires browser-based download)
3. Automatically extract and verify datasets
4. Prepare data for processing

**For manual download instructions, continue reading below.**

---

## 📋 **Overview**

**AMASS** (Archive of Motion Capture as Surface Shapes) is a large-scale motion capture dataset containing:
- **40+ hours** of motion data
- **11,000+ subjects**
- **300+ motion categories**
- **15 different source datasets** unified into SMPL format

**For stick-gen**, we'll use AMASS to expand our training data from 100k to 500k samples.

---

## 🎯 **What You Need**

- **Storage**: ~50GB for full dataset (or ~10GB for subset)
- **Time**: 2-4 hours download time (depends on connection speed)
- **Account**: Free registration at amass.is.tue.mpg.de

---

## 📝 **Step 1: Register for AMASS**

### 1.1 Create Account

1. Go to: **https://amass.is.tue.mpg.de/**
2. Click **"Register"** in the top right
3. Fill out registration form:
   - Name
   - Email
   - Institution (can be "Independent Researcher")
   - Purpose: "Machine learning research for stick figure animation"
4. Accept the license agreement
5. Click **"Submit"**

### 1.2 Verify Email

1. Check your email for verification link
2. Click the link to activate your account
3. Log in to the AMASS website

---

## 📥 **Step 2: Download AMASS Data**

### 2.1 Choose Format ⚠️ CRITICAL

AMASS provides multiple formats. **Select SMPL+H (preferred) or SMPL-X (compatible).**

**✅ SMPL+H Format (PREFERRED)**
- **52 joints** (22 body + 30 hand joints)
- **156 parameters** (52 joints × 3)
- **Most common format** in AMASS
- **Compatible with `convert_amass.py`** ← Automatically detected
- **Includes detailed hand movements**

**✅ SMPL-X Format (COMPATIBLE)**
- **54 joints** (22 body + 30 hand + 2 jaw joints)
- **162 parameters** (54 joints × 3 + 10 facial expression params)
- **Use when SMPL+H is not available**
- **Compatible with `convert_amass.py`** ← Automatically detected and handled
- **Facial expressions ignored** (not used for stick figures)

**⚠️ SMPL Format (FALLBACK ONLY)**
- **22 joints** (body only)
- **72 parameters** (24 joints × 3)
- Use ONLY if SMPL+H/SMPL-X is unavailable for a specific dataset
- Requires minor code adjustments in `convert_amass.py`

**❌ DO NOT DOWNLOAD**
- **DMPL** (Dynamic SMPL) - Not compatible
- **Other specialized formats** - Not compatible

**Why SMPL+H?**
Your `convert_amass.py` script is already configured for SMPL+H:
```python
model_type='smplh'  # ← Expects SMPL+H format
poses = data['poses']  # [num_frames, 156] ← SMPL+H dimensions
```

**Format Selection on AMASS Website**:
When downloading each dataset, you'll see format options like:
- SMPL+H ← **SELECT THIS**
- SMPL
- DMPL
- Others

**Always select SMPL+H for consistency across all datasets.**

### 2.2 Recommended Datasets (Priority Order)

Download these datasets first (total ~10GB):

1. **CMU** (~2GB)
   - Carnegie Mellon University Motion Capture Database
   - 2,500+ sequences
   - Actions: walking, running, jumping, sports, dancing

2. **BMLmovi** (~3GB)
   - Berlin Motion Library
   - High-quality motion capture
   - Actions: everyday movements, gestures

3. **ACCAD** (~1.5GB)
   - Advanced Computing Center for the Arts and Design
   - Actions: sports, martial arts, dance

4. **HDM05** (~2GB)
   - HDM05 Motion Capture Database
   - Actions: walking, running, grabbing, throwing

5. **TotalCapture** (~1.5GB)
   - Full-body motion capture
   - Actions: walking, running, acting

### 2.3 Download Instructions

1. Log in to https://amass.is.tue.mpg.de/
2. Click **"Download"** in the navigation
3. Select **"SMPL+H G"** format
4. Check the datasets you want (start with CMU, BMLmovi, ACCAD)
5. Click **"Download Selected"**
6. Save the .tar.bz2 files to your Downloads folder

**Alternative**: Use `wget` or `curl` if download links are provided:
```bash
# Example (replace with actual download links from AMASS)
wget https://amass.is.tue.mpg.de/download/CMU_SMPLH_G.tar.bz2
wget https://amass.is.tue.mpg.de/download/BMLmovi_SMPLH_G.tar.bz2
wget https://amass.is.tue.mpg.de/download/ACCAD_SMPLH_G.tar.bz2
```

---

## 📂 **Step 3: Extract and Organize**

### 3.1 Create Directory Structure

```bash
cd /Users/bc/gestura/stick-gen
mkdir -p data/amass
```

### 3.2 Extract Downloaded Files

```bash
# Extract each dataset
cd data/amass

# CMU
tar -xjf ~/Downloads/CMU_SMPLH_G.tar.bz2

# BMLmovi
tar -xjf ~/Downloads/BMLmovi_SMPLH_G.tar.bz2

# ACCAD
tar -xjf ~/Downloads/ACCAD_SMPLH_G.tar.bz2

# HDM05
tar -xjf ~/Downloads/HDM05_SMPLH_G.tar.bz2

# TotalCapture
tar -xjf ~/Downloads/TotalCapture_SMPLH_G.tar.bz2
```

### 3.3 Verify Directory Structure

After extraction, you should have:

```
/Users/bc/gestura/stick-gen/data/amass/
├── CMU/
│   ├── 01/
│   │   ├── 01_01_poses.npz
│   │   ├── 01_02_poses.npz
│   │   └── ...
│   ├── 02/
│   └── ...
├── BMLmovi/
│   ├── Subject_1/
│   ├── Subject_2/
│   └── ...
├── ACCAD/
├── HDM05/
└── TotalCapture/
```

---

## ✅ **Step 4: Verify Download**

Run the verification script:

```bash
cd /Users/bc/gestura/stick-gen
python3.9 -c "
import os
from pathlib import Path

amass_root = Path('data/amass')
datasets = ['CMU', 'BMLmovi', 'ACCAD', 'HDM05', 'TotalCapture']

print('AMASS Dataset Verification')
print('=' * 50)

for dataset in datasets:
    dataset_path = amass_root / dataset
    if dataset_path.exists():
        npz_files = list(dataset_path.rglob('*.npz'))
        print(f'✓ {dataset:20s} - {len(npz_files):,} sequences')
    else:
        print(f'✗ {dataset:20s} - NOT FOUND')

print('=' * 50)
total_files = len(list(amass_root.rglob('*.npz')))
print(f'Total sequences: {total_files:,}')
"
```

Expected output:
```
AMASS Dataset Verification
==================================================
✓ CMU                  - 2,235 sequences
✓ BMLmovi              - 3,911 sequences
✓ ACCAD                - 1,534 sequences
✓ HDM05                - 2,337 sequences
✓ TotalCapture         - 480 sequences
==================================================
Total sequences: 10,497
```

---

## 🚀 **Step 5: Process AMASS Data**

Once downloaded and verified, process the data:

```bash
cd /Users/bc/gestura/stick-gen

# Process AMASS dataset (this will take 8-12 hours)
python3.9 src/data_gen/process_amass.py \
    --amass_root data/amass \
    --output data/amass_stick_data.pt \
    --max_samples 400000 \
    --batch_size 100
```

This will:
1. Convert SMPL format to stick figure format
2. Infer actions from filenames
3. Generate text descriptions
4. Save processed data to `data/amass_stick_data.pt`

---

## 📊 **Expected Results**

After processing, you should have:

- **Input**: 10,000+ AMASS sequences (.npz files)
- **Output**: 400,000 stick figure samples (data/amass_stick_data.pt)
- **Format**: Each sample contains:
  - `motion`: [250, 20] tensor (10s @ 25fps, 5 stick lines)
  - `action`: ActionType enum
  - `actions`: [250] tensor of action indices
  - `description`: Text description
  - `source`: 'amass'

---

## ⚠️ **Troubleshooting**

### Wrong Format Downloaded

**Problem**: Downloaded SMPL instead of SMPL+H, or downloaded DMPL

**Solution**:
1. **If you downloaded SMPL** (72 params instead of 156):
   - You can still use it, but need to adjust `convert_amass.py`
   - Change line 149: `body_pose = torch.tensor(poses[:, 3:66], ...)`
   - To: `body_pose = torch.tensor(poses[:, 3:69], ...)`  # SMPL has different indexing
   - **Better**: Re-download in SMPL+H format for consistency

2. **If you downloaded DMPL or other format**:
   - ❌ Not compatible with `convert_amass.py`
   - Must re-download in SMPL+H format

**How to check format**:
```python
import numpy as np
data = np.load('path/to/sequence.npz')
print(f"Pose shape: {data['poses'].shape}")
# SMPL+H: (num_frames, 156) ✅
# SMPL:   (num_frames, 72)  ⚠️
# DMPL:   Different structure ❌
```

### Format Inconsistency Across Datasets

**Problem**: Some datasets in SMPL+H, others in SMPL

**Solution**:
- **Recommended**: Re-download all datasets in SMPL+H for consistency
- **Alternative**: Create separate processing pipelines for each format
- **Not recommended**: Mixing formats will cause dimension mismatches

### Download is slow
- Try downloading during off-peak hours
- Use a download manager for resume capability
- Download one dataset at a time

### Extraction fails
- Ensure you have enough disk space (~50GB)
- Use `tar -xjvf` for verbose output to see progress
- Check file integrity with `md5sum` if provided

### Missing .npz files
- Some datasets have subdirectories - use `rglob('*.npz')` to find all files
- Verify extraction completed successfully

### Convert script fails with dimension error

**Problem**: `RuntimeError: Expected tensor of size [X, 156], got [X, 72]`

**Cause**: Downloaded SMPL format instead of SMPL+H

**Solution**: See "Wrong Format Downloaded" above

---

## 📚 **Additional Resources**

- **AMASS Website**: https://amass.is.tue.mpg.de/
- **AMASS Paper**: https://files.is.tue.mpg.de/black/papers/amass.pdf
- **SMPL-X Documentation**: https://github.com/vchoutas/smplx

---

**Next Steps**: After downloading and processing AMASS data, proceed to Phase 1 implementation (Action-Conditioned Generation).

