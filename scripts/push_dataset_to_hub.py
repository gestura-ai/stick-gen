#!/usr/bin/env python3
"""
Stick-Gen Dataset Uploader.

Uploads the processed dataset (.pt files or streaming samples) to the Hugging Face Hub.
Automatically includes the dataset card (README.md) for proper documentation.

Usage:
    # Upload merged .pt file
    python scripts/push_dataset_to_hub.py --dataset-path data/train_data.pt --repo-id GesturaAI/stick-gen-dataset

    # Upload streaming samples directory
    python scripts/push_dataset_to_hub.py --dataset-path data/train_samples --repo-id GesturaAI/stick-gen-dataset

    # Include custom dataset card
    python scripts/push_dataset_to_hub.py --dataset-path data/train_data.pt --repo-id GesturaAI/stick-gen-dataset --dataset-card dataset_cards/stick-gen-dataset.md
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

# Default dataset card path
DEFAULT_DATASET_CARD = "dataset_cards/stick-gen-dataset.md"


def get_dataset_stats(dataset_path: Path) -> dict:
    """Extract statistics from dataset for README generation."""
    stats = {
        "total_samples": 0,
        "format": "unknown",
        "generated_at": datetime.now().isoformat(),
    }

    if dataset_path.is_file() and dataset_path.suffix == ".pt":
        try:
            import torch

            data = torch.load(dataset_path, weights_only=False)
            stats["total_samples"] = len(data)
            stats["format"] = "merged (.pt)"
            if data:
                sample = data[0]
                if "motion" in sample:
                    stats["motion_shape"] = list(sample["motion"].shape)
                if "description" in sample:
                    stats["sample_description"] = sample["description"][:100] + "..."
        except Exception as e:
            print(f"   ⚠️ Could not read dataset stats: {e}")

    elif dataset_path.is_dir():
        meta_path = dataset_path / "generation_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                stats["total_samples"] = meta.get("total_samples_written", 0)
                stats["format"] = "streaming (JSON)"
                stats["generated_count"] = meta.get("generated_count", 0)
                stats["completed"] = meta.get("completed", False)
            except Exception as e:
                print(f"   ⚠️ Could not read metadata: {e}")
        else:
            # Count sample files
            sample_files = list(dataset_path.glob("sample_*.json")) + list(
                dataset_path.glob("sample_*.json.gz")
            )
            stats["total_samples"] = len(sample_files)
            stats["format"] = "streaming (JSON)"

    return stats


def prepare_upload_folder(
    dataset_path: Path,
    dataset_card_path: Path | None,
    stats: dict,
) -> Path:
    """Prepare a temporary folder with all files to upload."""
    upload_dir = Path(tempfile.mkdtemp(prefix="stick-gen-dataset-"))

    # Copy dataset card as README.md
    readme_path = upload_dir / "README.md"
    if dataset_card_path and dataset_card_path.exists():
        shutil.copy(dataset_card_path, readme_path)
        print(f"   ✓ Using dataset card: {dataset_card_path}")

        # Append stats to README
        with open(readme_path, "a") as f:
            f.write("\n\n## Upload Information\n\n")
            f.write(f"- **Upload Date**: {stats['generated_at']}\n")
            f.write(f"- **Total Samples**: {stats['total_samples']:,}\n")
            f.write(f"- **Format**: {stats['format']}\n")
            if stats.get("motion_shape"):
                f.write(f"- **Motion Shape**: {stats['motion_shape']}\n")
    else:
        print("   ⚠️ No dataset card found, creating minimal README")
        with open(readme_path, "w") as f:
            f.write("# Stick-Gen Dataset\n\n")
            f.write(f"Total samples: {stats['total_samples']:,}\n")
            f.write(f"Format: {stats['format']}\n")

    # Copy dataset files
    if dataset_path.is_file():
        shutil.copy(dataset_path, upload_dir / dataset_path.name)
    elif dataset_path.is_dir():
        # Copy entire directory contents
        dest_dir = upload_dir / dataset_path.name
        shutil.copytree(dataset_path, dest_dir)

    return upload_dir


def main():
    parser = argparse.ArgumentParser(
        description="Upload Stick-Gen dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the dataset file (.pt) or samples directory",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Repo ID (e.g., GesturaAI/stick-gen-dataset)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HF Token (default: $HF_TOKEN env var)",
    )
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument(
        "--dataset-card",
        default=DEFAULT_DATASET_CARD,
        help=f"Path to dataset card markdown (default: {DEFAULT_DATASET_CARD})",
    )
    parser.add_argument(
        "--no-card",
        action="store_true",
        help="Skip dataset card upload",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message",
    )
    args = parser.parse_args()

    print(f"\n🚀 Uploading dataset to {args.repo_id}...")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Path not found: {dataset_path}")
        sys.exit(1)

    # Get dataset statistics
    print("\n📊 Analyzing dataset...")
    stats = get_dataset_stats(dataset_path)
    print(f"   Total samples: {stats['total_samples']:,}")
    print(f"   Format: {stats['format']}")

    api = HfApi(token=args.token)

    # Create repo
    print(f"\n📦 Creating/verifying repository...")
    try:
        url = create_repo(
            args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=args.token,
        )
        print(f"   ✓ Repo ready: {url}")
    except Exception as e:
        print(f"   ❌ Error creating repo: {e}")
        return 1

    # Prepare upload folder
    print("\n📁 Preparing files...")
    dataset_card_path = None if args.no_card else Path(args.dataset_card)
    upload_dir = prepare_upload_folder(dataset_path, dataset_card_path, stats)

    # Upload
    print(f"\n⬆️  Uploading to {args.repo_id}...")
    commit_message = args.commit_message or f"Upload dataset ({stats['total_samples']:,} samples)"

    try:
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token,
            commit_message=commit_message,
        )
        print(f"\n✅ Dataset upload complete!")
        print(f"   View at: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return 1
    finally:
        # Cleanup temp directory
        shutil.rmtree(upload_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
