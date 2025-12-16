# RunPod Upload S3 Pagination Error Fix

## Problem
```
fatal error: Error during pagination: The same next token was received twice:
{'ContinuationToken': 'ZGF0YS9IdW1hbk1MM0QvSHVtYW5NTDNEL3RleHRzLzAwMDM1MS50eHQ='}
```

This is a known AWS S3 CLI bug that occurs when listing large directories with many files.

## Root Cause
- AWS S3 API pagination bug when the bucket contains thousands of files
- The `aws s3 sync` command lists the ENTIRE bucket to compare files, not just the target directory
- Even when syncing `smpl_models`, it tries to list `HumanML3D` files, causing pagination errors
- The continuation token gets stuck in an infinite loop
- **Critical insight**: The bug is triggered by the total number of files in the bucket, not just the directory being synced

## Solutions

### ⭐ **RECOMMENDED: Use rclone (Most Reliable)**

The AWS CLI has fundamental pagination bugs with large buckets. **rclone** is much more reliable:

```bash
# Install rclone (if not already installed)
brew install rclone  # macOS
# or
curl https://rclone.org/install.sh | sudo bash  # Linux

# Use the rclone upload script
./runpod/resume_upload_rclone.sh --volume-id 3pcn16qk2s
```

**Why rclone is better:**
- ✅ No pagination bugs
- ✅ Faster with `--fast-list` option
- ✅ Better progress reporting
- ✅ More reliable retries
- ✅ Handles large directories (100k+ files) without issues

### Alternative: Clear Problematic Directory First

If you must use AWS CLI, clear the problematic directory from S3 first:

```bash
# Clear HumanML3D (or other problematic directory)
./runpod/clear_problematic_dir.sh --volume-id 3pcn16qk2s --dir HumanML3D

# Then re-upload
make resume-upload VOLUME_ID=3pcn16qk2s
```

### AWS CLI Improvements (Partial Fix)

The `resume_upload.sh` script has been updated with:
1. Very small `--page-size 10` (reduced from 100)
2. `--size-only` flag to skip timestamp checks
3. Retry logic (3 attempts)
4. Skip functionality

**Note:** These help but don't fully solve the pagination bug when the bucket has many files.

## Usage

### ⭐ **WORKING SOLUTION: Upload File-by-File**

The AWS S3 `sync` command has a pagination bug. Upload files individually instead:

```bash
# Upload a specific directory (e.g., smpl_models)
./runpod/upload_by_file.sh --volume-id 3pcn16qk2s --dir smpl_models

# Upload another directory
./runpod/upload_by_file.sh --volume-id 3pcn16qk2s --dir amass
```

**Benefits:**
- ✅ Avoids S3 pagination bug entirely (no bucket-wide listing)
- ✅ Skips already-uploaded files (checks file size)
- ✅ Shows progress every 100 files
- ✅ Continues on errors (doesn't abort)
- ✅ Works with any number of files

### Alternative: Clear Problematic Directory First

If you want to use the original `sync` approach:

```bash
# Step 1: Clear ALL data from the bucket (fresh start)
./runpod/clear_problematic_dir.sh --volume-id 3pcn16qk2s --dir data

# Step 2: Upload with sync (will work on empty bucket)
make resume-upload VOLUME_ID=3pcn16qk2s
```

## Files Modified

1. **`runpod/resume_upload.sh`**
   - Added `--page-size 100` to all sync commands
   - Added retry logic with 3 attempts
   - Added `--skip` parameter to skip directories
   - Continues with next directory on failure instead of exiting

2. **`runpod/check_upload_status.sh`** (NEW)
   - Checks upload status for each directory
   - Compares local file count vs S3 file count
   - Provides skip command suggestions

## Quick Reference

```bash
# Check what's uploaded
./runpod/check_upload_status.sh --volume-id 3pcn16qk2s

# Resume upload (all directories)
make resume-upload VOLUME_ID=3pcn16qk2s

# Resume upload (skip completed directories)
./runpod/resume_upload.sh --volume-id 3pcn16qk2s --skip 'amass,100Style'

# Upload specific directory manually
aws s3 sync ./data/smpl_models s3://3pcn16qk2s/data/smpl_models/ \
    --endpoint-url https://s3api-eu-cz-1.runpod.io \
    --region eu-cz-1 \
    --page-size 100
```

## Prevention
- The `--page-size` parameter should prevent most pagination errors
- Retry logic handles transient network issues
- Skip functionality allows working around problematic directories
- Smaller page sizes (50-100) are more reliable than default (1000)

