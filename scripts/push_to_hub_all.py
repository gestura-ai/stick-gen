#!/usr/bin/env python3
"""
Batch Model Uploader for Stick-Gen.

This script scans the checkpoints directory for trained models (Small, Medium, Large)
and facilitates pushing them to the Hugging Face Hub with their corresponding model cards.

Usage:
    export HF_TOKEN=your_token
    python scripts/push_to_hub_all.py --org gestura-ai
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)


DEFINED_MODELS = {
    "small": "stick-gen-small",
    "medium": "stick-gen-medium",
    "large": "stick-gen-large",
}


def push_model(variant: str, checkpoint_path: Path, org: str):
    """Push a single model variant to HF."""
    model_name = DEFINED_MODELS.get(variant)
    if not model_name:
        print(f"⚠️  Unknown variant: {variant}")
        return

    repo_id = f"{org}/{model_name}"
    print(f"\n🚀 Processing {variant.upper()} -> {repo_id}...")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    api = HfApi()

    # 1. Create Repo
    try:
        url = create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"   ✓ Repo exists at {url}")
    except Exception as e:
        print(f"   ❌ Error checking repo: {e}")
        return

    # 2. Upload Checkpoint (as pytorch_model.bin or similar)
    print("   ⬆️  Uploading weights...")
    try:
        # In a real scenario, we might want to convert to safetensors first
        # For this script we assume the user has the file ready to go
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="pytorch_model.bin",
            repo_id=repo_id,
            repo_type="model",
        )
        print("   ✓ Weights uploaded")
    except Exception as e:
        print(f"   ❌ Error uploading weights: {e}")

    # 3. Upload Model Card
    card_path = Path(f"model_cards/{variant}.md")
    if card_path.exists():
        print("   ⬆️  Uploading model card...")
        try:
            api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            print("   ✓ Model card uploaded")
        except Exception as e:
            print(f"   ❌ Error uploading card: {e}")
    else:
        print(f"   ⚠️  Model card not found at {card_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch push models to Hugging Face")
    parser.add_argument("--org", default="gestura-ai", help="Hugging Face Organization")
    parser.add_argument(
        "--checkpoints_dir",
        default="checkpoints",
        help="Directory containing .pth files",
    )
    args = parser.parse_args()

    print("Gestura AI - Batch Model Uploader")
    print(f"Target Org: {args.org}")
    print("=" * 40)

    base_dir = Path(args.checkpoints_dir)

    # Check for expected filenames like 'small.pth', 'medium.pth', 'best_model.pth' etc.
    # This is slightly heuristic.

    for variant in DEFINED_MODELS.keys():
        # Heuristic: look for {variant}.pth or stick-gen-{variant}.pth
        candidates = [
            base_dir / f"{variant}.pth",
            base_dir / f"stick-gen-{variant}.pth",
            # Fallback if user manually specifies one file, but this script is for batch
        ]

        found = False
        for p in candidates:
            if p.exists():
                push_model(variant, p, args.org)
                found = True
                break

        if not found:
            print(
                f"\n⚪️ Skipping {variant.upper()} (Checkpoint not found in {base_dir})"
            )

    print("\n✅ Batch process complete.")


if __name__ == "__main__":
    main()
