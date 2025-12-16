#!/usr/bin/env python3
"""
AMASS Dataset Download Script

Automated download and setup for AMASS dataset.

Prerequisites:
1. Register at https://amass.is.tue.mpg.de/
2. Accept license terms
3. Get download credentials

IMPORTANT - Format Selection:
    When downloading from AMASS website, prefer SMPL+H but SMPL-X is also supported.

    ✅ SMPL+H (52 joints, 156 params) - PREFERRED
       - Most common format in AMASS
       - Includes detailed hand joints
       - Fully compatible with convert_amass.py

    ✅ SMPL-X (54 joints, 162 params) - COMPATIBLE
       - Use when SMPL+H is not available
       - Includes facial expressions (not used for stick figures)
       - Automatically handled by convert_amass.py

    ⚠️  SMPL (22 joints, 72 params) - FALLBACK ONLY
       - Use only if SMPL+H/SMPL-X unavailable
       - Requires minor code adjustments

    ❌ DMPL, other formats - NOT COMPATIBLE
       - Do not download these formats

Usage:
    # Interactive mode (prompts for credentials)
    python download_amass.py

    # Specify datasets
    python download_amass.py --datasets CMU BMLmovi ACCAD

    # Download to custom directory
    python download_amass.py --output-dir /path/to/amass

    # Download subset only (for testing)
    python download_amass.py --subset
"""

import argparse
import getpass
import hashlib
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("❌ Required packages not installed.")
    print("Run: pip install requests tqdm")
    sys.exit(1)

# ============================================================================
# AMASS Dataset Configuration
# ============================================================================

AMASS_BASE_URL = "https://amass.is.tue.mpg.de/download.php"

# Recommended datasets with sizes
DATASETS = {
    "CMU": {
        "size_gb": 2.0,
        "sequences": 2500,
        "description": "Carnegie Mellon University Motion Capture Database",
        "priority": 1,
    },
    "BMLmovi": {
        "size_gb": 3.0,
        "sequences": 3000,
        "description": "BioMotionLab Motion Viewer Database",
        "priority": 2,
    },
    "ACCAD": {
        "size_gb": 1.5,
        "sequences": 1200,
        "description": "Advanced Computing Center for the Arts and Design",
        "priority": 3,
    },
    "HDM05": {
        "size_gb": 2.0,
        "sequences": 1500,
        "description": "HDM05 Motion Capture Database",
        "priority": 4,
    },
    "TotalCapture": {
        "size_gb": 1.5,
        "sequences": 1000,
        "description": "Total Capture Dataset",
        "priority": 5,
    },
    "HumanEva": {
        "size_gb": 0.5,
        "sequences": 400,
        "description": "HumanEva Dataset",
        "priority": 6,
    },
    "MPI_HDM05": {
        "size_gb": 2.5,
        "sequences": 2000,
        "description": "MPI HDM05 Dataset",
        "priority": 7,
    },
    "SFU": {
        "size_gb": 1.0,
        "sequences": 800,
        "description": "Simon Fraser University Motion Capture Database",
        "priority": 8,
    },
    "MPI_mosh": {
        "size_gb": 4.0,
        "sequences": 4000,
        "description": "MPI MoSh Dataset",
        "priority": 9,
    },
    "Transitions_mocap": {
        "size_gb": 0.8,
        "sequences": 600,
        "description": "Transitions Motion Capture",
        "priority": 10,
    },
}

# Subset for testing (smaller, faster download)
SUBSET_DATASETS = ["HumanEva", "Transitions_mocap"]

# ============================================================================
# Download Functions
# ============================================================================


def download_file(url: str, output_path: Path, session: requests.Session) -> bool:
    """Download a file with progress bar"""
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract tar.gz or zip archive"""
    print(f"📦 Extracting {archive_path.name}...")

    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(
            ".tgz"
        ):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"⚠️  Unknown archive format: {archive_path}")
            return False

        print(f"✓ Extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_dataset(dataset_dir: Path, expected_sequences: int) -> bool:
    """Verify dataset was extracted correctly"""
    if not dataset_dir.exists():
        return False

    # Count .npz files (AMASS format)
    npz_files = list(dataset_dir.rglob("*.npz"))

    print(f"✓ Found {len(npz_files)} sequences (expected ~{expected_sequences})")

    # Allow some variance (±20%)
    if len(npz_files) < expected_sequences * 0.8:
        print(f"⚠️  Warning: Fewer sequences than expected")
        return False

    return True


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    session: requests.Session,
    username: str,
    password: str,
) -> bool:
    """Download and extract a single AMASS dataset"""

    dataset_info = DATASETS[dataset_name]

    print("=" * 70)
    print(f"Downloading: {dataset_name}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_gb']} GB")
    print(f"Sequences: ~{dataset_info['sequences']}")
    print("=" * 70)

    # Create output directory
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download archive
    archive_name = f"{dataset_name}.tar.bz2"
    archive_path = output_dir / archive_name

    # Note: Actual AMASS download requires authentication
    # This is a placeholder - users need to download manually from the website
    print(f"\n⚠️  MANUAL DOWNLOAD REQUIRED")
    print(f"\nPlease download {dataset_name} manually:")
    print(f"1. Go to https://amass.is.tue.mpg.de/download.php")
    print(f"2. Log in with your credentials")
    print(f"3. Select dataset: {dataset_name}")
    print(f"4. ✅ IMPORTANT: Choose 'SMPL+H' format (preferred)")
    print(f"   If SMPL+H not available, choose 'SMPL-X' (also compatible)")
    print(f"5. Download: {archive_name}")
    print(f"6. Save to: {archive_path}")
    print(f"\n📋 Format Selection:")
    print(f"   ✅ SMPL+H - PREFERRED (156 params, most common)")
    print(f"   ✅ SMPL-X - COMPATIBLE (162 params, auto-detected)")
    print(f"   ⚠️  SMPL - Fallback only (72 params, requires adjustments)")
    print(f"   ❌ DMPL - NOT compatible")
    print(f"\nPress Enter when download is complete...")
    input()

    # Check if file exists
    if not archive_path.exists():
        print(f"❌ Archive not found: {archive_path}")
        return False

    # Extract
    if not extract_archive(archive_path, dataset_dir):
        return False

    # Verify
    if not verify_dataset(dataset_dir, dataset_info["sequences"]):
        print(f"⚠️  Dataset verification failed")
        return False

    print(f"✅ {dataset_name} downloaded and verified!")
    return True


# ============================================================================
# Main CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Download AMASS dataset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to download (default: recommended subset)",
    )
    parser.add_argument(
        "--subset", action="store_true", help="Download small subset for testing"
    )
    parser.add_argument(
        "--output-dir",
        default="data/amass",
        help="Output directory (default: data/amass)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all datasets (~50GB)"
    )

    args = parser.parse_args()

    # Determine which datasets to download
    if args.all:
        datasets_to_download = list(DATASETS.keys())
    elif args.subset:
        datasets_to_download = SUBSET_DATASETS
    elif args.datasets:
        datasets_to_download = args.datasets
    else:
        # Default: top 5 recommended
        datasets_to_download = sorted(
            DATASETS.keys(), key=lambda x: DATASETS[x]["priority"]
        )[:5]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total size
    total_size = sum(DATASETS[d]["size_gb"] for d in datasets_to_download)
    total_sequences = sum(DATASETS[d]["sequences"] for d in datasets_to_download)

    print("=" * 70)
    print("AMASS Dataset Download")
    print("=" * 70)
    print(f"Datasets: {len(datasets_to_download)}")
    print(f"Total size: ~{total_size:.1f} GB")
    print(f"Total sequences: ~{total_sequences}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 70)
    print()

    # Show datasets
    print("Datasets to download:")
    for i, dataset in enumerate(datasets_to_download, 1):
        info = DATASETS[dataset]
        print(
            f"{i}. {dataset:20s} - {info['size_gb']:4.1f} GB - {info['sequences']:5d} sequences"
        )
    print()

    # Confirm
    response = input("Continue? [y/N]: ")
    if response.lower() != "y":
        print("Cancelled.")
        return

    # Get credentials
    print("\nAMASS Download Credentials")
    print("(Register at https://amass.is.tue.mpg.de/)")
    username = input("Username/Email: ")
    password = getpass.getpass("Password: ")

    # Create session
    session = requests.Session()

    # Download each dataset
    successful = []
    failed = []

    for dataset in datasets_to_download:
        print()
        if download_dataset(dataset, output_dir, session, username, password):
            successful.append(dataset)
        else:
            failed.append(dataset)

    # Summary
    print()
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)}/{len(datasets_to_download)}")
    if successful:
        for dataset in successful:
            print(f"   - {dataset}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for dataset in failed:
            print(f"   - {dataset}")

    print()
    print("Next steps:")
    print(f"1. Verify data in: {output_dir.absolute()}")
    print("2. Run: python src/data_gen/process_amass.py")
    print("3. Convert to stick figures: python src/data_gen/convert_amass.py")
    print()


if __name__ == "__main__":
    main()
