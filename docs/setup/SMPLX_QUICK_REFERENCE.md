# SMPL-X Quick Reference Card

**Goal**: Enable CMU (2,079 files) and CNRS (81 files) AMASS dataset conversions

---

## 📥 Download

**Website**: https://smpl-x.is.tue.mpg.de/

**Steps**:
1. Register (free for research): https://smpl-x.is.tue.mpg.de/register.php
2. Login and download: **"SMPL-X v1.1 (NPZ format)"**
3. Extract the archive

---

## 📂 Required Files

**Minimum** (choose one):
- `SMPLX_NEUTRAL.npz` ← **RECOMMENDED**
- `SMPLX_MALE.npz`

**Recommended** (all three):
- `SMPLX_NEUTRAL.npz`
- `SMPLX_MALE.npz`
- `SMPLX_FEMALE.npz`

**Size**: ~30 MB each (~90 MB total)

---

## 📍 Installation Location

**Target Directory**: `data/smpl_models/smplx/`

**Expected Structure**:
```
stick-gen/
└── data/
    └── smpl_models/
        ├── smplh/              # Existing (SMPL+H)
        │   ├── male/model.npz
        │   ├── female/model.npz
        │   └── neutral/model.npz
        └── smplx/              # NEW (SMPL-X)
            ├── SMPLX_NEUTRAL.npz
            ├── SMPLX_MALE.npz
            └── SMPLX_FEMALE.npz
```

---

## ⚡ Quick Install

```bash
# 1. Create directory
mkdir -p data/smpl_models/smplx

# 2. Copy files (replace /path/to/downloaded)
cp /path/to/downloaded/smplx/SMPLX_*.npz data/smpl_models/smplx/

# 3. Verify
ls -lh data/smpl_models/smplx/

# 4. Re-run conversions
./launch_async_amass_conversions.sh
```

---

## ✅ Verification

```bash
# Check files exist
python3.9 -c "
import os
for f in ['SMPLX_NEUTRAL.npz', 'SMPLX_MALE.npz', 'SMPLX_FEMALE.npz']:
    path = f'data/smpl_models/smplx/{f}'
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f'✅ {f}: {size:.1f} MB')
    else:
        print(f'❌ {f}: NOT FOUND')
"
```

---

## 🔄 Re-run Conversions

**Automated**:
```bash
./launch_async_amass_conversions.sh
```

**Manual**:
```bash
# CMU (2,079 files, ~2-3 hours)
nohup python3.9 -m src.data_gen.convert_amass \
    --dataset CMU \
    --input_dir data/amass/CMU \
    --output_dir data/amass_converted/CMU \
    > amass_conversion_cmu.log 2>&1 &

# CNRS (81 files, ~5-10 minutes)
nohup python3.9 -m src.data_gen.convert_amass \
    --dataset CNRS \
    --input_dir data/amass/CNRS \
    --output_dir data/amass_converted/CNRS \
    > amass_conversion_cnrs.log 2>&1 &
```

---

## 📊 Expected Results

**Before**:
- 12/14 datasets (7,767 files, 182 MB)

**After**:
- 14/14 datasets (9,927 files, ~300 MB)
- +2,160 sequences (+28% more data!)

---

## 🔍 Monitor Progress

```bash
# Check running processes
ps aux | grep 'convert_amass' | grep -v grep

# Monitor logs
tail -f amass_conversion_cmu.log
tail -f amass_conversion_cnrs.log

# Check conversion count
echo "CMU: $(find data/amass_converted/CMU -name '*.pt' 2>/dev/null | wc -l)/2079"
echo "CNRS: $(find data/amass_converted/CNRS -name '*.pt' 2>/dev/null | wc -l)/81"
```

---

## ⚠️ Important Notes

- **License**: Free for research/academic use (registration required)
- **No Redistribution**: Cannot share the model files
- **Commercial Use**: Requires separate license
- **Approval Time**: Usually instant for academic emails
- **Optional**: Training works fine with 12 datasets if you can't get SMPL-X

---

## 🆘 Troubleshooting

**"SMPLX_NEUTRAL.npz does not exist"**
→ Make sure files are in `data/smpl_models/smplx/` (not a subdirectory)

**"Permission denied"**
→ `chmod 755 data/smpl_models/smplx`

**"smplx library not installed"**
→ `pip install smplx`

**Files are PKL instead of NPZ**
→ Both formats work, but NPZ is preferred

---

## 📚 Full Documentation

See `SMPLX_DOWNLOAD_GUIDE.md` for complete instructions.

---

**Summary**: Register → Download → Copy to `data/smpl_models/smplx/` → Re-run conversions → +2,160 sequences!

