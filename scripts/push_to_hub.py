#!/usr/bin/env python3
"""
Stick-Gen HuggingFace Hub Push Script
Gestura AI - https://gestura.ai

Enhanced wrapper around push_to_huggingface.py that:
- Integrates evaluation metrics into model card
- Supports versioned releases
- Updates model card with training results

Usage:
    python scripts/push_to_hub.py --checkpoint checkpoints/best_model.pth \
        --variant base --version 1.0.0 --metrics evaluation_results.json
"""

import os
import sys
import shutil
import argparse
import json
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.push_to_huggingface import (
    validate_checkpoint,
    validate_model_card,
    prepare_model_files,
    upload_to_hub,
    VARIANT_CONFIG
)


def update_model_card_with_metrics(model_card_path: str, metrics: dict, version: str = None):
    """Update model card with evaluation metrics."""
    if not os.path.exists(model_card_path):
        print(f"Model card not found: {model_card_path}")
        return
    
    with open(model_card_path, 'r') as f:
        content = f.read()
    
    # Update TBD values with actual metrics
    if 'mse' in metrics:
        mse_value = metrics['mse'].get('mean', 'TBD')
        content = re.sub(r'MSE:\s*TBD', f'MSE: {mse_value:.6f}', content)
        content = re.sub(r'mse:\s*TBD', f'mse: {mse_value:.6f}', content)
    
    if 'temporal_consistency' in metrics:
        smoothness = metrics['temporal_consistency'].get('smoothness_score', 'TBD')
        content = re.sub(r'Smoothness:\s*TBD', f'Smoothness: {smoothness:.4f}', content)
    
    if 'action_accuracy' in metrics:
        accuracy = metrics['action_accuracy'].get('mean', 'TBD')
        content = re.sub(r'Action Accuracy:\s*TBD', f'Action Accuracy: {accuracy:.4f}', content)
    
    if 'physics' in metrics:
        physics_score = metrics['physics'].get('physics_score', 'TBD')
        content = re.sub(r'Physics Score:\s*TBD', f'Physics Score: {physics_score:.4f}', content)
    
    # Add version if specified
    if version:
        # Update version in YAML frontmatter if present
        content = re.sub(r'version:\s*[\d.]+', f'version: {version}', content)
    
    # Add last updated date
    today = datetime.now().strftime('%Y-%m-%d')
    content = re.sub(r'last_updated:\s*\d{4}-\d{2}-\d{2}', f'last_updated: {today}', content)
    
    with open(model_card_path, 'w') as f:
        f.write(content)
    
    print(f"Updated model card with metrics: {model_card_path}")


def create_release_notes(version: str, metrics: dict, training_history: list = None) -> str:
    """Create release notes for the version."""
    notes = f"""# Stick-Gen v{version} Release Notes

**Release Date:** {datetime.now().strftime('%Y-%m-%d')}

## Evaluation Metrics

"""
    
    if 'mse' in metrics:
        notes += f"- **MSE:** {metrics['mse']['mean']:.6f} ± {metrics['mse']['std']:.6f}\n"
    
    if 'temporal_consistency' in metrics:
        notes += f"- **Smoothness Score:** {metrics['temporal_consistency']['smoothness_score']:.4f}\n"
    
    if 'action_accuracy' in metrics:
        notes += f"- **Action Accuracy:** {metrics['action_accuracy']['mean']:.4f}\n"
    
    if 'physics' in metrics:
        notes += f"- **Physics Score:** {metrics['physics']['physics_score']:.4f}\n"
    
    if training_history:
        notes += f"\n## Training Summary\n\n"
        notes += f"- **Total Epochs:** {len(training_history)}\n"
        notes += f"- **Final Train Loss:** {training_history[-1]['train_loss']:.4f}\n"
        notes += f"- **Final Val Loss:** {training_history[-1]['val_loss']:.4f}\n"
    
    notes += """
## Features

- Text-to-animation generation
- Camera conditioning (Pan, Zoom, Track, Dolly, Crane, Orbit)
- Action-conditioned generation
- Physics-aware motion synthesis
- 100STYLE dataset integration

## Usage

```python
from stick_gen import StickGenPipeline

pipeline = StickGenPipeline.from_pretrained("GesturaAI/stick-gen-base")
animation = pipeline("A person walking confidently")
```

---
*by Gestura AI*
"""
    
    return notes


def main():
    parser = argparse.ArgumentParser(description="Push Stick-Gen to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Model variant")
    parser.add_argument("--version", type=str, default=None,
                        help="Version number (e.g., 1.0.0)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to evaluation_results.json")
    parser.add_argument("--training-history", type=str, default=None,
                        help="Path to training_history.json")
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="Directory containing TensorBoard logs (e.g., runs/)")
    parser.add_argument("--output-dir", type=str, default="hf_upload",
                        help="Output directory for prepared files")
    parser.add_argument("--repo-name", type=str, default=None,
                        help="HuggingFace repo name")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token")
    parser.add_argument("--private", action="store_true",
                        help="Create private repository")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare files without uploading")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stick-Gen HuggingFace Hub Push")
    print("by Gestura AI")
    print("=" * 60)
    
    # Load metrics if provided
    metrics = {}
    if args.metrics and os.path.exists(args.metrics):
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
        print(f"\nLoaded metrics from: {args.metrics}")
    
    # Load training history if provided
    training_history = None
    if args.training_history and os.path.exists(args.training_history):
        with open(args.training_history, 'r') as f:
            training_history = json.load(f)
        print(f"Loaded training history from: {args.training_history}")
    
    # Validate checkpoint
    print(f"\nValidating checkpoint: {args.checkpoint}")
    if not validate_checkpoint(args.checkpoint, args.variant):
        print("WARNING: Checkpoint validation failed")
    
    # Prepare files
    print(f"\nPreparing files in: {args.output_dir}")
    prepare_model_files(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        output_dir=args.output_dir,
        version=args.version
    )

    # Copy TensorBoard logs if requested
    if args.logs_dir and os.path.exists(args.logs_dir):
        print(f"\n📈 Copying TensorBoard logs from {args.logs_dir}...")
        logs_dest = os.path.join(args.output_dir, "runs")
        if os.path.exists(logs_dest):
            shutil.rmtree(logs_dest)
        shutil.copytree(args.logs_dir, logs_dest)
        print(f"✅ Logs copied to {logs_dest}")
    model_card_path = os.path.join(args.output_dir, "README.md")
    if metrics:
        update_model_card_with_metrics(model_card_path, metrics, args.version)

    # Create release notes
    if args.version:
        release_notes = create_release_notes(args.version, metrics, training_history)
        release_notes_path = os.path.join(args.output_dir, "RELEASE_NOTES.md")
        with open(release_notes_path, 'w') as f:
            f.write(release_notes)
        print(f"Created release notes: {release_notes_path}")

    # Upload to Hub
    if not args.dry_run:
        repo_name = args.repo_name or f"GesturaAI/{VARIANT_CONFIG[args.variant]['repo_suffix']}"
        print(f"\nUploading to: {repo_name}")

        upload_to_hub(
            repo_name=repo_name,
            output_dir=args.output_dir,
            token=args.token,
            private=args.private,
            version=args.version
        )

        print("\n" + "=" * 60)
        print("Upload Complete!")
        print("=" * 60)
        print(f"Model: https://huggingface.co/{repo_name}")
        if args.version:
            print(f"Version: {args.version}")
    else:
        print("\n[DRY RUN] Files prepared but not uploaded")
        print(f"Files ready in: {args.output_dir}/")


if __name__ == "__main__":
    main()

