#!/usr/bin/env python3
"""
Stick-Gen Dataset Uploader.

Uploads the processed dataset (.pt files) to the Hugging Face Hub.

Usage:
    python scripts/push_dataset_to_hub.py --dataset-path data/curated/train_data.pt --repo-id GesturaAI/stick-gen-dataset
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--dataset-path", required=True, help="Path to the dataset file (or directory)")
    parser.add_argument("--repo-id", required=True, help="HF Repo ID (e.g., GesturaAI/stick-gen-dataset)")
    parser.add_argument("--token", help="HF Token (optional if env var set)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    print(f"🚀 Uploading dataset to {args.repo_id}...")

    api = HfApi(token=args.token)

    # Create repo
    try:
        url = create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=args.token)
        print(f"   ✓ Repo exists at {url}")
    except Exception as e:
        print(f"   ❌ Error creating repo: {e}")
        return

    path = Path(args.dataset_path)
    if not path.exists():
        print(f"❌ Path not found: {path}")
        sys.exit(1)

    if path.is_file():
        print(f"   ⬆️  Uploading file {path.name}...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token
        )
    elif path.is_dir():
        print(f"   ⬆️  Uploading folder {path.name}...")
        api.upload_folder(
            folder_path=str(path),
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token
        )

    print("✅ Dataset upload complete!")

if __name__ == "__main__":
    main()
