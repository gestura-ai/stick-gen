#!/usr/bin/env python3
"""
Stick-Gen Training Report Generator
Gestura AI - https://gestura.ai

Generates comprehensive Markdown training reports with:
- Training configuration summary
- Loss curves (ASCII art)
- Evaluation metrics
- Model comparison tables
- Recommendations

Usage:
    python scripts/generate_report.py --training-history checkpoints/training_history.json \
        --metrics evaluation_results.json --output reports/training_report.md
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_ascii_chart(values: list, title: str, width: int = 50, height: int = 10) -> str:
    """Generate ASCII art chart for loss curves."""
    if not values:
        return "No data available"
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    chart = [f"\n{title}"]
    chart.append(f"Max: {max_val:.4f}")
    
    # Create chart rows
    for row in range(height, 0, -1):
        threshold = min_val + (row / height) * range_val
        line = "│"
        step = max(1, len(values) // width)
        for i in range(0, min(len(values), width * step), step):
            if values[i] >= threshold:
                line += "█"
            else:
                line += " "
        chart.append(line)
    
    chart.append("└" + "─" * min(len(values) // max(1, len(values) // width), width))
    chart.append(f"Min: {min_val:.4f}  Epochs: {len(values)}")
    
    return "\n".join(chart)


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as Markdown table."""
    table = ["| Metric | Value |", "|--------|-------|"]
    
    if 'mse' in metrics:
        table.append(f"| MSE (mean) | {metrics['mse']['mean']:.6f} |")
        table.append(f"| MSE (std) | {metrics['mse']['std']:.6f} |")
    
    if 'temporal_consistency' in metrics:
        tc = metrics['temporal_consistency']
        table.append(f"| Smoothness Score | {tc['smoothness_score']:.4f} |")
        table.append(f"| Mean Velocity | {tc['mean_velocity']:.4f} |")
        table.append(f"| Mean Jerk | {tc['mean_jerk']:.4f} |")
    
    if 'action_accuracy' in metrics:
        table.append(f"| Action Accuracy | {metrics['action_accuracy']['mean']:.4f} |")
    
    if 'physics' in metrics:
        phys = metrics['physics']
        table.append(f"| Physics Score | {phys['physics_score']:.4f} |")
        table.append(f"| Velocity MSE | {phys['velocity_mse']:.6f} |")
        table.append(f"| Gravity Error | {phys['gravity_error']:.4f} |")
    
    if 'diversity' in metrics:
        table.append(f"| Diversity Score | {metrics['diversity']['diversity_score']:.4f} |")
    
    return "\n".join(table)


def generate_report(
    training_history: list,
    metrics: dict,
    config: dict = None,
    variant: str = "base"
) -> str:
    """Generate comprehensive training report."""
    
    report = f"""# Stick-Gen Training Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Variant:** {variant}  
**by Gestura AI**

---

## Training Summary

"""
    
    if training_history:
        epochs = len(training_history)
        final_train = training_history[-1]['train_loss']
        final_val = training_history[-1]['val_loss']
        best_val = min(h['val_loss'] for h in training_history)
        best_epoch = next(i for i, h in enumerate(training_history) if h['val_loss'] == best_val)
        
        report += f"""| Parameter | Value |
|-----------|-------|
| Total Epochs | {epochs} |
| Final Train Loss | {final_train:.4f} |
| Final Val Loss | {final_val:.4f} |
| Best Val Loss | {best_val:.4f} |
| Best Epoch | {best_epoch} |

"""
        
        # Loss curves
        train_losses = [h['train_loss'] for h in training_history]
        val_losses = [h['val_loss'] for h in training_history]
        
        report += "### Training Loss Curve\n```\n"
        report += generate_ascii_chart(train_losses, "Train Loss")
        report += "\n```\n\n"
        
        report += "### Validation Loss Curve\n```\n"
        report += generate_ascii_chart(val_losses, "Val Loss")
        report += "\n```\n\n"
    
    # Evaluation metrics
    report += "## Evaluation Metrics\n\n"
    if metrics:
        report += format_metrics_table(metrics)
    else:
        report += "*No evaluation metrics available*"
    
    report += "\n\n"
    
    # Configuration
    if config:
        report += "## Model Configuration\n\n```yaml\n"
        for key, value in config.items():
            report += f"{key}: {value}\n"
        report += "```\n\n"
    
    # Recommendations
    report += """## Recommendations

"""
    
    if metrics and 'mse' in metrics:
        mse = metrics['mse']['mean']
        if mse > 0.1:
            report += "- ⚠️ MSE is high. Consider training for more epochs or increasing model capacity.\n"
        elif mse < 0.01:
            report += "- ✅ MSE is excellent. Model is well-trained.\n"
        else:
            report += "- ℹ️ MSE is acceptable. Fine-tuning may improve results.\n"
    
    if metrics and 'temporal_consistency' in metrics:
        smoothness = metrics['temporal_consistency']['smoothness_score']
        if smoothness < 0.5:
            report += "- ⚠️ Smoothness is low. Consider adding temporal consistency loss weight.\n"
        else:
            report += "- ✅ Motion smoothness is good.\n"

    if metrics and 'action_accuracy' in metrics:
        acc = metrics['action_accuracy']['mean']
        if acc < 0.7:
            report += "- ⚠️ Action accuracy is low. Consider more action-diverse training data.\n"
        else:
            report += "- ✅ Action prediction is accurate.\n"

    report += """
---

## Next Steps

1. **Deploy to RunPod** - Use `runpod/deploy.sh` to deploy the model
2. **Push to HuggingFace** - Use `scripts/push_to_hub.py` to publish
3. **Run inference tests** - Validate with example prompts
4. **Monitor production** - Track inference latency and quality

---

*Report generated by Stick-Gen MLOps Pipeline*
*Gestura AI - https://gestura.ai*
"""

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate Stick-Gen training report")
    parser.add_argument("--training-history", type=str, default=None,
                        help="Path to training_history.json")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to evaluation_results.json")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Model variant")
    parser.add_argument("--output", type=str, default="reports/training_report.md",
                        help="Output path for report")

    args = parser.parse_args()

    print("=" * 60)
    print("Stick-Gen Report Generator")
    print("by Gestura AI")
    print("=" * 60)

    # Load training history
    training_history = []
    if args.training_history and os.path.exists(args.training_history):
        with open(args.training_history, 'r') as f:
            training_history = json.load(f)
        print(f"\nLoaded training history: {len(training_history)} epochs")

    # Load metrics
    metrics = {}
    if args.metrics and os.path.exists(args.metrics):
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
        print(f"Loaded evaluation metrics")

    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {args.config}")

    # Generate report
    print(f"\nGenerating report...")
    report = generate_report(training_history, metrics, config, args.variant)

    # Save report
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\n✅ Report saved to: {args.output}")

    # Print summary
    if training_history:
        print(f"\nTraining Summary:")
        print(f"  Epochs: {len(training_history)}")
        print(f"  Final Val Loss: {training_history[-1]['val_loss']:.4f}")

    if metrics and 'mse' in metrics:
        print(f"\nEvaluation Summary:")
        print(f"  MSE: {metrics['mse']['mean']:.6f}")


if __name__ == "__main__":
    main()

