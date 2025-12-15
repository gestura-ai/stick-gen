# SMPL-X Model Download Guide for CMU and CNRS Dataset Conversion

**Purpose**: Download SMPL-X model files to enable conversion of CMU (2,079 files) and CNRS (81 files) AMASS datasets  
**Current Issue**: `Failed to load SMPL model: Path data/smpl_models/smplx/SMPLX_MALE.npz does not exist!`  
**Total Files to Unlock**: 2,160 additional motion capture sequences

---

## 1. Official Download Link

**SMPL-X Official Website**: https://smpl-x.is.tue.mpg.de/

**Direct Download Page**: https://smpl-x.is.tue.mpg.de/download.php

---

## 2. Registration and Licensing Requirements

### ⚠️ Important: Registration Required

SMPL-X models are **free for research purposes** but require registration:

1. **Create an Account**:
   - Go to: https://smpl-x.is.tue.mpg.de/register.php
   - Fill out the registration form
   - Provide institutional email (academic/research)
   - Agree to the license terms

2. **License Terms**:
   - ✅ **Free for research and non-commercial use**
   - ✅ **Free for academic projects**
   - ❌ **Commercial use requires separate license**
   - ❌ **Cannot redistribute the models**

3. **Approval Time**:
   - Usually **instant** for academic emails
   - May take 1-2 business days for other emails

---

## 3. Required Files to Download

After registration and login, download the following:

### **Option 1: SMPL-X Models (NPZ format) - RECOMMENDED**

Download: **"SMPL-X v1.1 (NPZ format)"**

This package includes:
- `SMPLX_NEUTRAL.npz` - Neutral gender model (recommended)
- `SMPLX_MALE.npz` - Male model
- `SMPLX_FEMALE.npz` - Female model

**File Size**: ~50-100 MB (compressed)

### **Option 2: SMPL-X Models (PKL format) - ALTERNATIVE**

Download: **"SMPL-X v1.1 (PKL format)"**

This package includes:
- `SMPLX_NEUTRAL.pkl` - Neutral gender model
- `SMPLX_MALE.pkl` - Male model
- `SMPLX_FEMALE.pkl` - Female model

**Note**: The converter supports both NPZ and PKL formats, but NPZ is preferred.

### **Minimum Required Files**

You only need **ONE** of the following:
- `SMPLX_NEUTRAL.npz` (or `.pkl`) - **RECOMMENDED** (works for all genders)
- `SMPLX_MALE.npz` (or `.pkl`) - Fallback if neutral not available

**Recommendation**: Download all three (NEUTRAL, MALE, FEMALE) for maximum compatibility.

---

## 4. Installation Instructions

### Step 1: Download the Files

1. Log in to https://smpl-x.is.tue.mpg.de/
2. Navigate to the download page
3. Download **"SMPL-X v1.1 (NPZ format)"**
4. Extract the downloaded archive (usually a `.zip` or `.tar.gz` file)

### Step 2: Place Files in Correct Directory

**Target Directory**: `data/smpl_models/smplx/`

**Expected Structure**:
```
stick-gen/
└── data/
    └── smpl_models/
        ├── smplh/                    # Existing SMPL+H models
        │   ├── male/
        │   │   └── model.npz
        │   ├── female/
        │   │   └── model.npz
        │   └── neutral/
        │       └── model.npz
        └── smplx/                    # NEW: SMPL-X models
            ├── SMPLX_NEUTRAL.npz     # Required
            ├── SMPLX_MALE.npz        # Optional (fallback)
            └── SMPLX_FEMALE.npz      # Optional
```

### Step 3: Copy Files

**From your terminal** (in the stick-gen project root):

```bash
# Create the smplx directory if it doesn't exist
mkdir -p data/smpl_models/smplx

# Copy the downloaded files
# Replace /path/to/downloaded/smplx with your actual download location
cp /path/to/downloaded/smplx/SMPLX_NEUTRAL.npz data/smpl_models/smplx/
cp /path/to/downloaded/smplx/SMPLX_MALE.npz data/smpl_models/smplx/
cp /path/to/downloaded/smplx/SMPLX_FEMALE.npz data/smpl_models/smplx/

# Verify files are in place
ls -lh data/smpl_models/smplx/
```

**Expected Output**:
```
-rw-r--r--  1 user  staff   30M Dec  9 10:00 SMPLX_FEMALE.npz
-rw-r--r--  1 user  staff   30M Dec  9 10:00 SMPLX_MALE.npz
-rw-r--r--  1 user  staff   30M Dec  9 10:00 SMPLX_NEUTRAL.npz
```

### Step 4: Verify Installation

```bash
# Check that files exist
python3.9 -c "
import os
smplx_dir = 'data/smpl_models/smplx'
required_files = ['SMPLX_NEUTRAL.npz', 'SMPLX_MALE.npz', 'SMPLX_FEMALE.npz']
for f in required_files:
    path = os.path.join(smplx_dir, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f'✅ {f}: {size_mb:.1f} MB')
    else:
        print(f'❌ {f}: NOT FOUND')
"
```

---

## 5. Re-run CMU and CNRS Conversions

Once the SMPL-X models are in place, re-run the conversions:

### Option 1: Automated Script

```bash
# Re-run the async conversion script
./launch_async_amass_conversions.sh
```

This will detect that CMU and CNRS need conversion and launch them automatically.

### Option 2: Manual Conversion

**Convert CMU** (2,079 files, ~2-3 hours):
```bash
nohup python3.9 -m src.data_gen.convert_amass \
    --dataset CMU \
    --input_dir data/amass/CMU \
    --output_dir data/amass_converted/CMU \
    > amass_conversion_cmu.log 2>&1 &
```

**Convert CNRS** (81 files, ~5-10 minutes):
```bash
nohup python3.9 -m src.data_gen.convert_amass \
    --dataset CNRS \
    --input_dir data/amass/CNRS \
    --output_dir data/amass_converted/CNRS \
    > amass_conversion_cnrs.log 2>&1 &
```

### Monitor Progress

```bash
# Check running processes
ps aux | grep 'convert_amass' | grep -v grep

# Monitor CMU conversion
tail -f amass_conversion_cmu.log

# Monitor CNRS conversion
tail -f amass_conversion_cnrs.log

# Check conversion statistics
echo "CMU: $(find data/amass_converted/CMU -name '*.pt' 2>/dev/null | wc -l)/2079 converted"
echo "CNRS: $(find data/amass_converted/CNRS -name '*.pt' 2>/dev/null | wc -l)/81 converted"
```

---

## 6. Expected Results After Conversion

### Before SMPL-X Installation
- ✅ 12/14 datasets converted (7,767 files)
- ❌ CMU: 0/2,079 converted
- ❌ CNRS: 0/81 converted
- **Total**: 7,767 files (182 MB)

### After SMPL-X Installation
- ✅ 14/14 datasets converted (9,927 files)
- ✅ CMU: 2,079/2,079 converted
- ✅ CNRS: 81/81 converted
- **Total**: 9,927 files (~250-300 MB)

**Additional Motion Capture Data**: +2,160 sequences (+28% more data!)

---

## 7. Troubleshooting

### Issue 1: "SMPLX_NEUTRAL.npz does not exist"

**Solution**: Make sure you downloaded the NPZ format, not PKL format.

If you only have PKL files:
- The converter will automatically use `.pkl` files if `.npz` files are not found
- Or download the NPZ version from the SMPL-X website

### Issue 2: "Permission denied" when copying files

**Solution**:
```bash
# Make sure you have write permissions
chmod 755 data/smpl_models/smplx
```

### Issue 3: "smplx library not installed"

**Solution**:
```bash
pip install smplx
```

### Issue 4: Conversion still fails after installing SMPL-X models

**Check**:
1. Verify files are in the correct location:
   ```bash
   ls -lh data/smpl_models/smplx/SMPLX_*.npz
   ```

2. Verify file integrity (should be ~30 MB each):
   ```bash
   du -h data/smpl_models/smplx/SMPLX_*.npz
   ```

3. Test loading the model:
   ```bash
   python3.9 -c "
   import smplx
   model = smplx.create('data/smpl_models', model_type='smplx', gender='neutral')
   print('✅ SMPL-X model loaded successfully!')
   "
   ```

---

## 8. Alternative: Skip CMU and CNRS

If you cannot obtain SMPL-X models, you can proceed with training using only the 12 successfully converted datasets:

**Available Data**:
- 7,767 AMASS sequences (12 datasets)
- 500,000 synthetic sequences
- **Total**: 507,767 training samples

**Impact**:
- ✅ Still sufficient for high-quality training
- ✅ No changes to training pipeline needed
- ⚠️  Slightly less motion diversity (missing 2,160 sequences)

**Recommendation**: The 12 converted datasets are sufficient for production training. CMU and CNRS are **optional enhancements**, not requirements.

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Register at smpl-x.is.tue.mpg.de | 5 min |
| 2 | Download SMPL-X v1.1 (NPZ format) | 5-10 min |
| 3 | Extract and copy files to `data/smpl_models/smplx/` | 2 min |
| 4 | Verify installation | 1 min |
| 5 | Re-run CMU conversion | 2-3 hours |
| 6 | Re-run CNRS conversion | 5-10 min |
| **Total** | **~3-4 hours** | **+2,160 sequences** |

**License**: Free for research/academic use (registration required)  
**Files Needed**: SMPLX_NEUTRAL.npz (minimum) or all three (NEUTRAL, MALE, FEMALE)  
**Directory**: `data/smpl_models/smplx/`  
**Benefit**: +28% more motion capture data (2,160 additional sequences)

---

## Quick Start Commands

```bash
# 1. Create directory
mkdir -p data/smpl_models/smplx

# 2. Copy downloaded files (replace /path/to/downloaded)
cp /path/to/downloaded/smplx/*.npz data/smpl_models/smplx/

# 3. Verify installation
ls -lh data/smpl_models/smplx/

# 4. Re-run conversions
./launch_async_amass_conversions.sh

# 5. Monitor progress
tail -f amass_conversion_cmu.log
```

**That's it! Once the SMPL-X models are in place, the CMU and CNRS conversions will work automatically.**

