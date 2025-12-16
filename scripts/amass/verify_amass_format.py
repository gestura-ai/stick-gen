#!/usr/bin/env python3
"""
AMASS Format Verification Script

Verifies that downloaded AMASS datasets are in the correct format (SMPL+H).

Usage:
    # Check single file
    python verify_amass_format.py path/to/sequence.npz

    # Check entire dataset directory
    python verify_amass_format.py data/amass/CMU

    # Check all datasets
    python verify_amass_format.py data/amass
"""

import sys
from pathlib import Path

import numpy as np


def check_npz_format(npz_path: Path) -> tuple[str, dict]:
    """
    Check format of a single .npz file

    Returns:
        format_type: 'SMPL+H', 'SMPL-X', 'SMPL', 'DMPL', or 'UNKNOWN'
        info: Dictionary with file information
    """
    try:
        data = np.load(npz_path)

        # Check for required keys
        if "poses" not in data:
            return "UNKNOWN", {"error": "Missing poses key"}

        poses_shape = data["poses"].shape
        num_frames = poses_shape[0]
        pose_params = poses_shape[1] if len(poses_shape) > 1 else 0

        # Determine format based on pose parameters
        if pose_params == 156:
            format_type = "SMPL+H"
            status = "✅"
        elif pose_params == 162:
            format_type = "SMPL-X"
            status = "✅"
        elif pose_params == 72:
            format_type = "SMPL"
            status = "⚠️"
        elif pose_params == 300:
            format_type = "DMPL"
            status = "❌"
        else:
            format_type = "UNKNOWN"
            status = "❓"

        info = {
            "status": status,
            "format": format_type,
            "num_frames": num_frames,
            "pose_params": pose_params,
            "has_trans": "trans" in data,
            "has_betas": "betas" in data,
        }

        return format_type, info

    except Exception as e:
        return "ERROR", {"error": str(e)}


def verify_directory(directory: Path) -> dict[str, list[Path]]:
    """
    Verify all .npz files in a directory

    Returns:
        Dictionary mapping format types to file lists
    """
    results = {
        "SMPL+H": [],
        "SMPL-X": [],
        "SMPL": [],
        "DMPL": [],
        "UNKNOWN": [],
        "ERROR": [],
    }

    npz_files = list(directory.rglob("*.npz"))

    if not npz_files:
        print(f"⚠️  No .npz files found in {directory}")
        return results

    print(f"Checking {len(npz_files)} files in {directory}...")
    print()

    for npz_file in npz_files:
        format_type, info = check_npz_format(npz_file)
        results[format_type].append(npz_file)

        # Print progress for first few files
        if (
            len(results["SMPL+H"])
            + len(results["SMPL-X"])
            + len(results["SMPL"])
            + len(results["DMPL"])
            <= 5
        ):
            print(
                f"{info.get('status', '❓')} {npz_file.name}: {format_type} ({info.get('pose_params', 0)} params)"
            )

    return results


def print_summary(results: dict[str, list[Path]], directory: Path):
    """Print verification summary"""
    total = sum(len(files) for files in results.values())

    print()
    print("=" * 70)
    print(f"AMASS Format Verification - {directory.name}")
    print("=" * 70)

    # SMPL+H (preferred format)
    if results["SMPL+H"]:
        print(
            f"✅ SMPL+H (PREFERRED): {len(results['SMPL+H']):5d} files ({len(results['SMPL+H'])/total*100:.1f}%)"
        )

    # SMPL-X (compatible format)
    if results["SMPL-X"]:
        print(
            f"✅ SMPL-X (COMPATIBLE): {len(results['SMPL-X']):5d} files ({len(results['SMPL-X'])/total*100:.1f}%)"
        )
        print("   → Automatically handled by converter")

    # SMPL (fallback)
    if results["SMPL"]:
        print(
            f"⚠️  SMPL (FALLBACK):  {len(results['SMPL']):5d} files ({len(results['SMPL'])/total*100:.1f}%)"
        )
        print("   → Consider re-downloading in SMPL+H or SMPL-X format")

    # DMPL (incompatible)
    if results["DMPL"]:
        print(
            f"❌ DMPL (INCOMPATIBLE): {len(results['DMPL']):5d} files ({len(results['DMPL'])/total*100:.1f}%)"
        )
        print("   → MUST re-download in SMPL+H or SMPL-X format")

    # Unknown/Error
    if results["UNKNOWN"]:
        print(f"❓ UNKNOWN:          {len(results['UNKNOWN']):5d} files")
    if results["ERROR"]:
        print(f"❌ ERROR:            {len(results['ERROR']):5d} files")

    print("=" * 70)
    print(f"Total files: {total}")
    print()

    # Recommendation
    compatible_count = len(results["SMPL+H"]) + len(results["SMPL-X"])
    if compatible_count == total:
        print(
            "🎉 All files are in compatible format (SMPL+H or SMPL-X) - ready for processing!"
        )
    elif len(results["SMPL"]) > 0 and len(results["DMPL"]) == 0:
        print(
            "⚠️  Some files are in SMPL format. Recommended: re-download in SMPL+H or SMPL-X."
        )
    elif len(results["DMPL"]) > 0:
        print(
            "❌ DMPL format detected - NOT compatible. Must re-download in SMPL+H or SMPL-X."
        )
    else:
        print("⚠️  Mixed or unknown formats detected. Check individual files.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_amass_format.py <path_to_npz_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"❌ Path not found: {path}")
        sys.exit(1)

    if path.is_file() and path.suffix == ".npz":
        # Check single file
        format_type, info = check_npz_format(path)
        print(f"File: {path.name}")
        print(f"Format: {info.get('status', '❓')} {format_type}")
        print(f"Frames: {info.get('num_frames', 0)}")
        print(f"Pose params: {info.get('pose_params', 0)}")
        print(f"Has trans: {info.get('has_trans', False)}")
        print(f"Has betas: {info.get('has_betas', False)}")

        if format_type == "SMPL+H":
            print("\n✅ SMPL+H format - preferred, compatible with convert_amass.py")
        elif format_type == "SMPL-X":
            print(
                "\n✅ SMPL-X format - compatible, automatically handled by convert_amass.py"
            )
        elif format_type == "SMPL":
            print("\n⚠️  SMPL format - consider re-downloading in SMPL+H or SMPL-X")
        elif format_type == "DMPL":
            print(
                "\n❌ DMPL format - NOT compatible, must re-download in SMPL+H or SMPL-X"
            )

    elif path.is_dir():
        # Check directory
        results = verify_directory(path)
        print_summary(results, path)
    else:
        print(f"❌ Invalid path: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
