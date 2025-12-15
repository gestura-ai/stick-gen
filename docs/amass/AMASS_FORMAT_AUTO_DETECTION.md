# AMASS Format Auto-Detection

**✅ UPDATED:** The stick-gen pipeline now automatically detects and handles both SMPL+H and SMPL-X formats!

---

## 🎯 **What This Means**

You can now download **either** SMPL+H or SMPL-X format from AMASS, and the converter will automatically:
1. **Detect** which format you downloaded (156 params = SMPL+H, 162 params = SMPL-X)
2. **Load** the appropriate SMPL model (SMPL+H or SMPL-X)
3. **Extract** the body joints (same 22 joints for both formats)
4. **Convert** to stick figure format (5 lines, 20 coordinates)

**No manual configuration needed!**

---

## 📋 **Supported Formats**

| Format | Params | Joints | Status | Notes |
|--------|--------|--------|--------|-------|
| **SMPL+H** | 156 | 52 (22 body + 30 hands) | ✅ **Preferred** | Most common in AMASS |
| **SMPL-X** | 162 | 54 (22 body + 30 hands + 2 jaw) | ✅ **Compatible** | Use when SMPL+H unavailable |
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
        return 'smplh'  # SMPL+H
    elif pose_params == 162:
        return 'smplx'  # SMPL-X
    else:
        raise ValueError(f"Unknown format: {pose_params} params")
```

### 2. Model Loading

The converter loads the appropriate SMPL model:

```python
def _load_smpl_model(self, model_type: str):
    if model_type == 'smplh':
        self.smpl_model = smplx.create(
            model_type='smplh',  # SMPL+H model
            ...
        )
    elif model_type == 'smplx':
        self.smplx_model = smplx.create(
            model_type='smplx',  # SMPL-X model
            ...
        )
```

### 3. Joint Extraction

Both formats provide the same 22 body joints:

```python
# Extract body joints (same for both formats)
smpl_joints = output.joints.numpy()  # [frames, 22 or 54, 3]
smpl_joints = smpl_joints[:, :22, :]  # Use first 22 (body joints)
```

### 4. Stick Figure Conversion

The same conversion logic works for both formats:

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
Jaw pose:            3 params  ← Extra (ignored)
Facial expression:  10 params  ← Extra (ignored)
Total:             162 params
```

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
3. **Consistent quality:** Same body motion quality from both formats
4. **Future-proof:** Ready for facial expressions if needed later
5. **Mixed datasets:** Can use SMPL+H and SMPL-X datasets together

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
# Automatically detects and handles both formats
```

---

**Document Status:** ✅ COMPLETE  
**Last Updated:** 2025-12-08  
**Pipeline Status:** ✅ Automatic format detection enabled

