#!/usr/bin/env python3
"""
Stick-Gen Unified Backup to HuggingFace Hub
Gestura AI - https://gestura.ai

Unified script for uploading training artifacts to HuggingFace Hub.
Handles both model checkpoints (to Model repos) and datasets (to Dataset repos).

Usage:
    # Upload model only
    python scripts/backup_to_hub.py \
        --checkpoint checkpoints/best_model.pth \
        --variant medium \
        --stage pretrain

    # Upload model with metrics
    python scripts/backup_to_hub.py \
        --checkpoint checkpoints/best_model.pth \
        --metrics evaluation_results.json \
        --variant medium

    # Upload dataset only
    python scripts/backup_to_hub.py \
        --dataset generation/curated/sft_data_embedded.pt \
        --dataset-repo GesturaAI/stick-gen-sft-data

    # Full backup (model + dataset)
    python scripts/backup_to_hub.py \
        --checkpoint checkpoints/best_model.pth \
        --dataset generation/curated/pretrain_data_embedded.pt \
        --metrics evaluation_results.json \
        --variant medium \
        --version 1.0.0
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)


# Default repository configuration
DEFAULT_MODEL_ORG = "GesturaAI"
DEFAULT_DATASET_REPO = "GesturaAI/stick-gen-dataset"

VARIANT_REPO_MAP = {
    "small": "stick-gen-small",
    "medium": "stick-gen-medium",
    "base": "stick-gen-medium",  # Alias
    "large": "stick-gen-large",
}

STAGE_SUFFIX_MAP = {
    "pretrain": "",
    "sft": "-sft",
    "lora": "-lora",
}


def upload_model(
    checkpoint_path: str,
    variant: str,
    stage: str = "pretrain",
    version: str = None,
    metrics_path: str = None,
    token: str = None,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Upload model checkpoint to HuggingFace Model Hub."""
    from scripts.push_to_hub import main as push_to_hub_main
    
    # Build repo name
    repo_base = VARIANT_REPO_MAP.get(variant, f"stick-gen-{variant}")
    stage_suffix = STAGE_SUFFIX_MAP.get(stage, "")
    repo_name = f"{DEFAULT_MODEL_ORG}/{repo_base}{stage_suffix}"
    
    print(f"\n📦 Uploading model to: {repo_name}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Variant: {variant}")
    print(f"   Stage: {stage}")
    if version:
        print(f"   Version: {version}")
    if metrics_path:
        print(f"   Metrics: {metrics_path}")
    
    if dry_run:
        print("   [DRY RUN] Would upload model")
        return repo_name
    
    # Build args for push_to_hub
    args = [
        "--checkpoint", checkpoint_path,
        "--variant", variant if variant != "base" else "medium",
        "--repo-name", repo_name,
    ]
    
    if version:
        args.extend(["--version", version])
    if metrics_path and os.path.exists(metrics_path):
        args.extend(["--metrics", metrics_path])
    if token:
        args.extend(["--token", token])
    if private:
        args.append("--private")
    
    # Import and run push_to_hub with args
    import subprocess
    cmd = ["python", "scripts/push_to_hub.py"] + args
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ Model uploaded: https://huggingface.co/{repo_name}")
        return repo_name
    else:
        print(f"❌ Model upload failed")
        return None


def upload_dataset(
    dataset_path: str,
    repo_id: str = None,
    token: str = None,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Upload dataset to HuggingFace Dataset Hub."""
    repo_id = repo_id or DEFAULT_DATASET_REPO
    
    print(f"\n📊 Uploading dataset to: {repo_id}")
    print(f"   Path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    if dry_run:
        print("   [DRY RUN] Would upload dataset")
        return repo_id
    
    api = HfApi(token=token)
    
    # Create repo
    try:
        url = create_repo(
            repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"   ✓ Repo ready: {url}")
    except Exception as e:
        print(f"   ❌ Error creating repo: {e}")
        return None
    
    path = Path(dataset_path)
    
    try:
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
        elif path.is_dir():
            api.upload_folder(
                folder_path=str(path),
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
        
        print(f"✅ Dataset uploaded: https://huggingface.co/datasets/{repo_id}")
        return repo_id
        
    except Exception as e:
        print(f"❌ Dataset upload failed: {e}")
        return None


def create_backup_report(
    model_repo: str = None,
    dataset_repo: str = None,
    version: str = None,
    output_path: str = None,
) -> dict:
    """Create a backup report JSON."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "artifacts": {
            "model": {
                "repo": model_repo,
                "url": f"https://huggingface.co/{model_repo}" if model_repo else None,
            },
            "dataset": {
                "repo": dataset_repo,
                "url": f"https://huggingface.co/datasets/{dataset_repo}" if dataset_repo else None,
            },
        },
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📝 Backup report saved: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Unified backup script for Stick-Gen artifacts to HuggingFace Hub"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint to upload",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="medium",
        choices=["small", "medium", "base", "large"],
        help="Model variant (default: medium)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft", "lora"],
        help="Training stage (default: pretrain)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to evaluation_results.json",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file/directory to upload",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=DEFAULT_DATASET_REPO,
        help=f"Dataset repo ID (default: {DEFAULT_DATASET_REPO})",
    )
    
    # Common arguments
    parser.add_argument(
        "--version",
        type=str,
        help="Version tag (e.g., 1.0.0)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repositories private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save backup report JSON",
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token and not args.dry_run:
        print("⚠️  Warning: No HF_TOKEN set. Upload may fail.")
    
    if not args.checkpoint and not args.dataset:
        parser.error("At least one of --checkpoint or --dataset is required")
    
    print("=" * 60)
    print("Stick-Gen Backup to HuggingFace Hub")
    print("by Gestura AI")
    print("=" * 60)
    
    model_repo = None
    dataset_repo = None
    
    # Upload model if specified
    if args.checkpoint:
        model_repo = upload_model(
            checkpoint_path=args.checkpoint,
            variant=args.variant,
            stage=args.stage,
            version=args.version,
            metrics_path=args.metrics,
            token=token,
            private=args.private,
            dry_run=args.dry_run,
        )
    
    # Upload dataset if specified
    if args.dataset:
        dataset_repo = upload_dataset(
            dataset_path=args.dataset,
            repo_id=args.dataset_repo,
            token=token,
            private=args.private,
            dry_run=args.dry_run,
        )
    
    # Create backup report
    if args.report or (model_repo or dataset_repo):
        report = create_backup_report(
            model_repo=model_repo,
            dataset_repo=dataset_repo,
            version=args.version,
            output_path=args.report,
        )
    
    print("\n" + "=" * 60)
    print("Backup Complete!")
    print("=" * 60)
    
    if model_repo:
        print(f"  Model:   https://huggingface.co/{model_repo}")
    if dataset_repo:
        print(f"  Dataset: https://huggingface.co/datasets/{dataset_repo}")


if __name__ == "__main__":
    main()
