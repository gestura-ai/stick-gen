"""
Upload stick-gen model to Hugging Face Hub.

This script packages the trained model checkpoint and uploads it to Hugging Face Hub
along with the model card, configuration, and necessary files.

Supports multiple model variants (small, base, large) with automatic configuration
selection and validation.

Usage:
    # Upload base variant (default) - run from repository root
    python scripts/push_to_huggingface.py --checkpoint checkpoints/best_model.pt --variant base

    # Upload small variant
    python scripts/push_to_huggingface.py --checkpoint checkpoints/small_model.pt --variant small

    # Upload with specific version
    python scripts/push_to_huggingface.py --checkpoint checkpoints/best_model.pt --variant base --version 1.0.0

Requirements:
    pip install huggingface_hub
"""

import argparse
import os
import shutil
import time
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
from huggingface_hub import HfApi, create_repo, upload_folder

# Variant configuration mapping
VARIANT_CONFIG = {
    "small": {
        "config_file": "configs/small.yaml",
        "model_card": "model_cards/small.md",
        "repo_suffix": "stick-gen-small",
        "expected_params": 5_600_000,
        "tolerance": 0.05  # 5% tolerance
    },
    "base": {
        "config_file": "configs/base.yaml",
        "model_card": "model_cards/base.md",
        "repo_suffix": "stick-gen-base",
        "expected_params": 15_800_000,
        "tolerance": 0.05
    },
    "large": {
        "config_file": "configs/large.yaml",
        "model_card": "model_cards/large.md",
        "repo_suffix": "stick-gen-large",
        "expected_params": 28_000_000,
        "tolerance": 0.05
    }
}


def count_parameters(checkpoint_path: str) -> int:
    """Count total parameters in model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        total_params = sum(p.numel() for p in state_dict.values())
        return total_params
    except Exception as e:
        print(f"  ⚠️  Warning: Could not count parameters: {e}")
        return 0


def validate_checkpoint(checkpoint_path: str, variant: str) -> bool:
    """
    Validate checkpoint matches expected variant specifications.

    Args:
        checkpoint_path: Path to model checkpoint
        variant: Model variant (small, base, large)

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n🔍 Validating checkpoint for variant '{variant}'...")

    variant_info = VARIANT_CONFIG[variant]
    expected_params = variant_info["expected_params"]
    tolerance = variant_info["tolerance"]

    # Count parameters
    actual_params = count_parameters(checkpoint_path)
    if actual_params == 0:
        print("  ⚠️  Warning: Could not validate parameter count")
        return True  # Continue anyway

    # Check parameter count
    min_params = expected_params * (1 - tolerance)
    max_params = expected_params * (1 + tolerance)

    print(f"  Expected parameters: {expected_params:,}")
    print(f"  Actual parameters: {actual_params:,}")

    if min_params <= actual_params <= max_params:
        print(f"  ✅ Parameter count matches variant '{variant}'")
        return True
    else:
        print(f"  ❌ Parameter count mismatch!")
        print(f"     Expected: {expected_params:,} (±{tolerance*100:.0f}%)")
        print(f"     Got: {actual_params:,}")
        print(f"     Difference: {abs(actual_params - expected_params):,}")
        return False


def validate_model_card(model_card_path: str) -> bool:
    """
    Validate model card has no TBD values and required sections.

    Args:
        model_card_path: Path to model card file

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n🔍 Validating model card: {model_card_path}...")

    if not os.path.exists(model_card_path):
        print(f"  ❌ Model card not found: {model_card_path}")
        return False

    with open(model_card_path, 'r') as f:
        content = f.read()

    # Check for TBD values
    tbd_count = content.count("TBD")
    if tbd_count > 0:
        print(f"  ⚠️  Warning: Model card contains {tbd_count} 'TBD' values")
        print("     Consider updating metrics before release")

    # Check for required sections
    required_sections = [
        "## Model Details",
        "## Intended Uses",
        "## Training Details",
        "## Citation"
    ]

    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)

    if missing_sections:
        print(f"  ⚠️  Warning: Missing sections: {', '.join(missing_sections)}")
    else:
        print("  ✅ All required sections present")

    return True


def prepare_model_files(
    checkpoint_path: str,
    variant: str,
    output_dir: str = "hf_upload",
    version: Optional[str] = None
):
    """
    Prepare model files for Hugging Face upload.

    Args:
        checkpoint_path: Path to trained model checkpoint
        variant: Model variant (small, base, large)
        output_dir: Directory to prepare files for upload
        version: Model version (e.g., "1.0.0")
    """
    print(f"\n📦 Preparing model files for variant '{variant}' in {output_dir}/...")

    variant_info = VARIANT_CONFIG[variant]

    # Get repository root (parent of scripts/ directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Copy checkpoint
    print(f"  Copying checkpoint: {checkpoint_path}")
    shutil.copy(checkpoint_path, os.path.join(output_dir, "pytorch_model.bin"))

    # Copy variant-specific model card
    model_card_file = os.path.join(repo_root, variant_info["model_card"])
    print(f"  Copying model card: {variant_info['model_card']}")
    if os.path.exists(model_card_file):
        shutil.copy(model_card_file, os.path.join(output_dir, "README.md"))
    else:
        print(f"    ⚠️  Warning: {variant_info['model_card']} not found!")
        # Fall back to base model card
        fallback_card = os.path.join(repo_root, "model_cards/base.md")
        if os.path.exists(fallback_card):
            print("    Using model_cards/base.md as fallback")
            shutil.copy(fallback_card, os.path.join(output_dir, "README.md"))

    # Copy variant-specific configuration
    config_file = os.path.join(repo_root, variant_info["config_file"])
    print(f"  Copying configuration: {variant_info['config_file']}")
    if os.path.exists(config_file):
        shutil.copy(config_file, os.path.join(output_dir, "config.yaml"))
    else:
        print(f"    ⚠️  Warning: {variant_info['config_file']} not found!")

    # Copy source code
    print("  Copying source code...")
    src_dirs = ["src/model", "src/inference", "src/data_gen"]
    for src_dir in src_dirs:
        src_path = os.path.join(repo_root, src_dir)
        if os.path.exists(src_path):
            dest_dir = os.path.join(output_dir, src_dir)
            os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
            shutil.copytree(src_path, dest_dir, dirs_exist_ok=True)

    # Copy requirements
    print("  Copying requirements.txt...")
    requirements_file = os.path.join(repo_root, "requirements.txt")
    if os.path.exists(requirements_file):
        shutil.copy(requirements_file, os.path.join(output_dir, "requirements.txt"))

    # Copy LICENSE
    print("  Copying LICENSE...")
    license_file = os.path.join(repo_root, "LICENSE")
    if os.path.exists(license_file):
        shutil.copy(license_file, os.path.join(output_dir, "LICENSE"))
    else:
        print("    ⚠️  Warning: LICENSE not found!")

    # Copy CITATIONS
    print("  Copying CITATIONS.md...")
    citations_file = os.path.join(repo_root, "CITATIONS.md")
    if os.path.exists(citations_file):
        shutil.copy(citations_file, os.path.join(output_dir, "CITATIONS.md"))

    # Create version file if specified
    if version:
        version_file = os.path.join(output_dir, "VERSION")
        with open(version_file, 'w') as f:
            f.write(f"{version}\n")
        print(f"  Created VERSION file: {version}")

    print(f"✅ Model files prepared in {output_dir}/")


def upload_to_hub_with_retry(
    repo_name: str,
    output_dir: str = "hf_upload",
    token: Optional[str] = None,
    private: bool = False,
    version: Optional[str] = None,
    max_retries: int = 3,
    initial_delay: float = 5.0
):
    """
    Upload model to Hugging Face Hub with retry logic and exponential backoff.

    Args:
        repo_name: Repository name (e.g., "GesturaAI/stick-gen-base")
        output_dir: Directory containing files to upload
        token: Hugging Face API token (or set HF_TOKEN environment variable)
        private: Whether to create a private repository
        version: Model version for commit message (e.g., "1.0.0")
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 5.0)

    Raises:
        Exception: If all retry attempts fail
    """
    print(f"\n🚀 Uploading to Hugging Face Hub: {repo_name}")
    print(f"   Retry policy: {max_retries} attempts with exponential backoff")

    # Create repository (with retry)
    print(f"  Creating repository: {repo_name} (private={private})")
    repo_created = False
    for attempt in range(1, max_retries + 1):
        try:
            create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
            print("  ✅ Repository created/verified")
            repo_created = True
            break
        except Exception as e:
            if attempt < max_retries:
                delay = initial_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(f"  ⚠️  Repository creation failed (attempt {attempt}/{max_retries}): {e}")
                print(f"  ⏳ Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                print(f"  ⚠️  Repository creation failed after {max_retries} attempts: {e}")
                print("  Continuing with upload (repository may already exist)...")

    # Prepare commit message
    if version:
        commit_message = f"Upload stick-gen model v{version}"
    else:
        commit_message = "Upload stick-gen model"

    # Upload files with retry and exponential backoff
    print(f"  Uploading files from {output_dir}/...")
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                token=token,
                commit_message=commit_message
            )
            print("  ✅ Files uploaded successfully!")
            print(f"\n🎉 Model uploaded successfully!")
            print(f"   View at: https://huggingface.co/{repo_name}")
            if version:
                print(f"   Version: {version}")
            return  # Success!

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = initial_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(f"  ⚠️  Upload failed (attempt {attempt}/{max_retries}): {e}")
                print(f"  ⏳ Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                print(f"  ❌ Upload failed after {max_retries} attempts: {e}")
                raise Exception(f"Failed to upload model after {max_retries} attempts: {last_error}")


def upload_to_hub(
    repo_name: str,
    output_dir: str = "hf_upload",
    token: Optional[str] = None,
    private: bool = False,
    version: Optional[str] = None
):
    """
    Upload model to Hugging Face Hub (legacy function, calls upload_to_hub_with_retry).

    Args:
        repo_name: Repository name (e.g., "GesturaAI/stick-gen-base")
        output_dir: Directory containing files to upload
        token: Hugging Face API token (or set HF_TOKEN environment variable)
        private: Whether to create a private repository
        version: Model version for commit message (e.g., "1.0.0")
    """
    # Call the retry version with default retry settings
    upload_to_hub_with_retry(
        repo_name=repo_name,
        output_dir=output_dir,
        token=token,
        private=private,
        version=version,
        max_retries=3,
        initial_delay=5.0
    )


def upload_dataset(
    dataset_path: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False
):
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset_path: Path to dataset file or directory
        repo_name: Repository name (e.g., "GesturaAI/stick-gen-dataset")
        token: HF token
        private: Whether to make repo private
    """
    print(f"\n🚀 Uploading dataset to {repo_name}...")
    
    # Create repo
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="dataset",
            exist_ok=True
        )
        print("  ✅ Dataset repository created/verified")
    except Exception as e:
        print(f"  ⚠️  Dataset repo creation failed: {e}")
    
    # Upload
    try:
        if os.path.isfile(dataset_path):
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=dataset_path,
                path_in_repo=os.path.basename(dataset_path),
                repo_id=repo_name,
                repo_type="dataset",
                commit_message="Upload dataset"
            )
        else:
            upload_folder(
                folder_path=dataset_path,
                repo_id=repo_name,
                repo_type="dataset",
                token=token,
                commit_message="Upload dataset"
            )
        print("  ✅ Dataset uploaded successfully!")
        print(f"   View at: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"  ❌ Dataset upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload stick-gen model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from repository root):
  # Upload base variant (default)
  python scripts/push_to_huggingface.py --checkpoint checkpoints/best_model.pt --variant base

  # Upload small variant with version
  python scripts/push_to_huggingface.py --checkpoint checkpoints/small_model.pt --variant small --version 1.0.0

  # Upload to custom repository
  python scripts/push_to_huggingface.py --checkpoint checkpoints/best_model.pt --variant base --repo-name myorg/my-model

  # Prepare files without uploading
  python scripts/push_to_huggingface.py --checkpoint checkpoints/best_model.pt --variant base --no-upload
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g., checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Model variant: small (5.6M), base (15.8M), or large (28M) (default: base)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version (e.g., 1.0.0) for semantic versioning"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Hugging Face repository name (default: GesturaAI/stick-gen-{variant})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hf_upload",
        help="Directory to prepare files for upload (default: hf_upload)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only prepare files, don't upload"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks (not recommended)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file/folder to upload (optional)"
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="GesturaAI/stick-gen-dataset",
        help="Dataset repository name (default: GesturaAI/stick-gen-dataset)"
    )

    args = parser.parse_args()

    # Determine repository name
    if args.repo_name is None:
        variant_info = VARIANT_CONFIG[args.variant]
        args.repo_name = f"GesturaAI/{variant_info['repo_suffix']}"
        print(f"Using default repository: {args.repo_name}")

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    # Validate checkpoint
    if not args.skip_validation:
        if not validate_checkpoint(args.checkpoint, args.variant):
            response = input("\n⚠️  Validation failed. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Upload cancelled")
                return 1

        # Validate model card
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        variant_info = VARIANT_CONFIG[args.variant]
        model_card_file = os.path.join(repo_root, variant_info["model_card"])
        if os.path.exists(model_card_file):
            validate_model_card(model_card_file)

    # Prepare files
    prepare_model_files(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        output_dir=args.output_dir,
        version=args.version
    )

    # Upload to Hub
    if not args.no_upload:
        upload_to_hub(
            repo_name=args.repo_name,
            output_dir=args.output_dir,
            token=args.token,
            private=args.private,
            version=args.version
        )
        
        # Upload dataset if requested
        if args.dataset:
            upload_dataset(
                dataset_path=args.dataset,
                repo_name=args.dataset_repo,
                token=args.token,
                private=args.private
            )
    else:
        print(f"\n✅ Files prepared in {args.output_dir}/")
        print("   Run without --no-upload to upload to Hugging Face Hub")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

