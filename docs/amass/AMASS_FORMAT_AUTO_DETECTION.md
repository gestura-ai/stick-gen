# AMASS Format Auto-Detection

**✅ UPDATED:** The stick-gen pipeline now automatically detects and handles SMPL+H and all SMPL-X variants, including Stage II files with explicit pose component fields!

---

## 🎯 **What This Means**

You can now download **any** SMPL+H or SMPL-X format from AMASS, and the converter will automatically:
1. **Detect** which format you downloaded (156 params = SMPL+H, 162/165 params = SMPL-X)
2. **Handle** different pose array layouts (including Stage II files with explicit fields)
3. **Load** the appropriate SMPL model (SMPL+H or SMPL-X)
4. **Extract** the body joints (same 22 joints for both formats)
5. **Convert** to stick figure format (5 lines, 20 coordinates)
6. **Filter** metadata-only files automatically (shape.npz, *_stagei.npz)

**No manual configuration needed!**

---

## 📋 **Supported Formats**

| Format | Params | Joints | Status | Notes |
|--------|--------|--------|--------|-------|
| **SMPL+H** | 156 | 52 (22 body + 30 hands) | ✅ **Preferred** | Most common in AMASS |
| **SMPL-X** | 162 | 54 (22 body + 30 hands + 2 jaw) | ✅ **Compatible** | Use when SMPL+H unavailable |
| **SMPL-X Stage II** | 165 | 54+ (with explicit fields) | ✅ **Compatible** | CNRS, MoSh, etc. subsets |
| SMPL | 72 | 22 (body only) | ⚠️ Fallback | Requires code changes |
| DMPL | 300 | Various | ❌ Not supported | Incompatible |

---

## 🚀 **Download Strategy**

### Recommended Approach

When downloading from AMASS:

1. **First choice:** Select **SMPL+H** format
   - Most widely available
   - Includes hand detail
   - 156 parameters

2. **Second choice:** Select **SMPL-X** format (if SMPL+H not available)
   - Includes facial expressions (not used for stick figures)
   - 162 parameters
   - Automatically handled

3. **Verify format:**
   ```bash
   python verify_amass_format.py data/amass/CMU
   ```

### Example Output

**SMPL+H dataset:**
```
✅ SMPL+H (PREFERRED): 2,235 files (100.0%)
🎉 All files are in compatible format - ready for processing!
```

**SMPL-X dataset:**
```
✅ SMPL-X (COMPATIBLE): 1,500 files (100.0%)
   → Automatically handled by converter
🎉 All files are in compatible format - ready for processing!
```

**Mixed dataset:**
```
✅ SMPL+H (PREFERRED): 1,500 files (60.0%)
✅ SMPL-X (COMPATIBLE):   1,000 files (40.0%)
🎉 All files are in compatible format - ready for processing!
```

---

## 🔧 **How It Works**

### 1. Format Detection

The converter automatically detects format by checking pose parameter count:

```python
def _detect_format(self, npz_path: str) -> str:
    data = np.load(npz_path)
    pose_params = data['poses'].shape[1]

    if pose_params == 156:
        return 'smplh'   # SMPL+H
    elif pose_params == 162:
        return 'smplx'   # SMPL-X (standard)
    elif pose_params == 165:
        return 'smplx'   # SMPL-X Stage II (with jaw + eyes)
    else:
        raise ValueError(f"Unknown format: {pose_params} params")
```

### 2. Pose Component Extraction

Different AMASS subsets store pose data differently. The converter handles both:

**Explicit Fields (Stage II files - CNRS, MoSh, etc.):**
```python
# These files have explicit component fields - more reliable
if 'root_orient' in raw_data and 'pose_body' in raw_data:
    data['root_orient'] = raw_data['root_orient']      # [N, 3]
    data['pose_body'] = raw_data['pose_body']          # [N, 63]
    data['left_hand_pose'] = raw_data['pose_hand'][:, :45]
    data['right_hand_pose'] = raw_data['pose_hand'][:, 45:]
```

**Poses Array Slicing (Standard files):**
```python
# Fall back to slicing the poses array
poses = raw_data['poses']
data['root_orient'] = poses[:, :3]
data['pose_body'] = poses[:, 3:66]
data['left_hand_pose'] = poses[:, 66:111]
data['right_hand_pose'] = poses[:, 111:156]
```

**Why this matters:** Stage II files (CNRS, etc.) have a different `poses` array layout:
- Standard: `root(3) + body(63) + hands(90) + jaw(3) + eye(6)`
- Stage II: `root(3) + body(63) + jaw(3) + eye(6) + hands(90)`

Using explicit fields avoids layout ambiguity.

### 3. Model Loading

The converter loads the appropriate SMPL model:

```python
def _load_smpl_model(self, model_type: str):
    if model_type == 'smplh':
        self.smpl_model = smplx.create(model_type='smplh', ...)
    elif model_type == 'smplx':
        self.smplx_model = smplx.create(model_type='smplx', ...)
```

### 4. File Filtering

The converter automatically skips metadata-only files that don't contain motion data:

```python
# Files filtered out (no motion data):
skip_patterns = {'shape.npz'}  # Body shape parameters only

# Also skipped via suffix matching:
# - *_stagei.npz: Stage 1 fitting data (marker/beta calibration)
npz_files = [p for p in amass_root.rglob('*.npz')
             if p.name not in skip_patterns
             and not p.name.endswith('_stagei.npz')]
```

### 5. Stick Figure Conversion

The same conversion logic works for all formats:

```python
# Map 22 body joints → 9 key positions → 5 lines
stick_lines = self.smpl_to_stick_figure(smpl_joints)
```

---

## 📊 **Body Motion Quality**

**Key Finding:** SMPL+H and SMPL-X provide **identical body motion quality**

| Aspect | SMPL+H | SMPL-X | Difference |
|--------|--------|--------|------------|
| Body joints | 22 | 22 | ✅ **Identical** |
| Body pose params | 63 | 63 | ✅ **Identical** |
| Hand joints | 30 | 30 | ✅ **Identical** |
| Jaw joints | 0 | 2 | ⚠️ Not used for stick figures |
| Facial expressions | 0 | 10 params | ⚠️ Not used for stick figures |

**Conclusion:** For stick figure body motion, SMPL+H and SMPL-X are functionally equivalent.

---

## ⚙️ **Technical Details**

### SMPL+H Format (156 params)
```
Global orientation:  3 params
Body pose:          63 params (21 joints × 3)
Left hand pose:     45 params (15 joints × 3)
Right hand pose:    45 params (15 joints × 3)
Total:             156 params
```

### SMPL-X Format (162 params)
```
Global orientation:  3 params
Body pose:          63 params (21 joints × 3)
Left hand pose:     45 params (15 joints × 3)
Right hand pose:    45 params (15 joints × 3)
Jaw pose:            3 params  ← Extra (ignored for stick figures)
Facial expression:  10 params  ← Extra (ignored for stick figures)
Total:             162 params
```

### SMPL-X Stage II Format (165 params)
```
Global orientation:  3 params  (or explicit 'root_orient' field)
Body pose:          63 params  (or explicit 'pose_body' field)
Left hand pose:     45 params  (or explicit 'pose_hand[:, :45]')
Right hand pose:    45 params  (or explicit 'pose_hand[:, 45:]')
Jaw pose:            3 params  (or explicit 'pose_jaw')
Eye pose:            6 params  (or explicit 'pose_eye')
Total:             165 params

Note: Stage II files have explicit pose component fields which
are preferred over slicing the 'poses' array.
```

### File Types in AMASS

| File Pattern | Contents | Processed? |
|--------------|----------|------------|
| `*_poses.npz` | Motion sequences with pose data | ✅ Yes |
| `*_stageii.npz` | Stage II fitted motion data | ✅ Yes |
| `shape.npz` | Body shape (betas) only, no poses | ❌ Skipped |
| `*_stagei.npz` | Stage I calibration data | ❌ Skipped |

### Stick Figure Extraction

Both formats extract the same 9 key positions:

```python
SMPL_TO_STICK_MAPPING = {
    'head': 15,           # Body joint (same in both)
    'left_shoulder': 16,  # Body joint (same in both)
    'right_shoulder': 17, # Body joint (same in both)
    'left_hip': 1,        # Body joint (same in both)
    'right_hip': 2,       # Body joint (same in both)
    'left_hand': 20,      # Body joint (same in both)
    'right_hand': 21,     # Body joint (same in both)
    'left_foot': 7,       # Body joint (same in both)
    'right_foot': 8       # Body joint (same in both)
}
```

---

## ✅ **Benefits**

1. **Flexibility:** Download whichever format is available
2. **No manual config:** Automatic detection and handling
3. **Consistent quality:** Same body motion quality from all formats
4. **Stage II support:** Handles CNRS, MoSh, and other Stage II subsets
5. **Smart filtering:** Automatically skips metadata-only files (shape.npz, *_stagei.npz)
6. **Future-proof:** Ready for facial expressions if needed later
7. **Mixed datasets:** Can use SMPL+H and SMPL-X datasets together

---

## 🎯 **Quick Reference**

### Download
```bash
python download_amass.py
# Select SMPL+H (preferred) or SMPL-X (compatible)
```

### Verify
```bash
python verify_amass_format.py data/amass/CMU
# Should show ✅ for SMPL+H or SMPL-X
```

### Convert
```bash
python src/data_gen/convert_amass.py
# Automatically detects and handles all formats
# Silently skips shape.npz and *_stagei.npz files
```

---

**Document Status:** ✅ COMPLETE
**Last Updated:** 2025-12-21
**Pipeline Status:** ✅ Automatic format detection with Stage II support enabled

