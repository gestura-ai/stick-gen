#!/usr/bin/env python3
"""
Validate AMASS Conversion Quality

This script:
1. Loads converted AMASS sequences
2. Renders sample sequences as stick figure animations
3. Validates motion quality (smoothness, consistency, range)
4. Generates comparison videos

Usage:
    python validate_amass_quality.py --samples 10
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


def render_sequence(motion_tensor, output_path, title="AMASS Sequence"):
    """Render a motion sequence as a video"""
    # motion_tensor: [250, 20]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Initialize lines
    lines = []
    for _i in range(5):
        (line,) = ax.plot([], [], "o-", linewidth=2, markersize=4)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        # Get frame data: [20] -> reshape to [5, 4]
        frame_data = motion_tensor[frame].reshape(5, 4)

        for i, line in enumerate(lines):
            x1, y1, x2, y2 = frame_data[i]
            line.set_data([x1, x2], [y1, y2])

        return lines

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=motion_tensor.shape[0],
        interval=40,  # 25 FPS
        blit=True,
    )

    # Save video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="ffmpeg", fps=25, dpi=100)
    plt.close()

    print(f"✅ Saved: {output_path}")


def analyze_sequence_quality(motion_tensor):
    """Analyze motion quality metrics"""
    # Compute frame-to-frame differences
    diffs = motion_tensor[1:] - motion_tensor[:-1]
    smoothness = torch.norm(diffs, dim=-1).mean().item()

    # Compute second-order differences (acceleration)
    second_diffs = diffs[1:] - diffs[:-1]
    consistency = torch.norm(second_diffs, dim=-1).mean().item()

    # Compute motion range
    motion_range = (motion_tensor.max() - motion_tensor.min()).item()

    # Compute velocity statistics
    velocities = torch.norm(diffs, dim=-1)
    avg_velocity = velocities.mean().item()
    max_velocity = velocities.max().item()

    return {
        "smoothness": smoothness,
        "consistency": consistency,
        "motion_range": motion_range,
        "avg_velocity": avg_velocity,
        "max_velocity": max_velocity,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate AMASS conversion quality")
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples to validate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="validation_videos", help="Output directory"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AMASS CONVERSION QUALITY VALIDATION")
    print("=" * 70)

    # Load descriptions
    descriptions_file = Path("data/amass_descriptions.json")
    if not descriptions_file.exists():
        print(f"❌ Error: {descriptions_file} does not exist")
        print("Run generate_amass_descriptions.py first")
        return

    with open(descriptions_file) as f:
        descriptions = json.load(f)

    print(f"\nLoaded {len(descriptions)} sequence descriptions")

    # Sample sequences from different datasets and actions
    sequences_by_dataset = {}
    sequences_by_action = {}

    for seq_path, metadata in descriptions.items():
        dataset = metadata["dataset"]
        action = metadata["action"]

        if dataset not in sequences_by_dataset:
            sequences_by_dataset[dataset] = []
        sequences_by_dataset[dataset].append(seq_path)

        if action not in sequences_by_action:
            sequences_by_action[action] = []
        sequences_by_action[action].append(seq_path)

    # Sample diverse sequences
    sampled_sequences = []

    # Sample from each dataset
    for dataset, seqs in sequences_by_dataset.items():
        if len(seqs) > 0:
            sampled_sequences.append(random.choice(seqs))

    # Sample from different actions
    for action, seqs in sequences_by_action.items():
        if len(seqs) > 0 and len(sampled_sequences) < args.samples:
            seq = random.choice(seqs)
            if seq not in sampled_sequences:
                sampled_sequences.append(seq)

    # Limit to requested number of samples
    sampled_sequences = sampled_sequences[: args.samples]

    print(f"\nValidating {len(sampled_sequences)} sequences...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_metrics = []

    for i, seq_path in enumerate(sampled_sequences, 1):
        metadata = descriptions[seq_path]

        print(f"\n[{i}/{len(sampled_sequences)}] Processing: {seq_path}")
        print(f"  Dataset: {metadata['dataset']}")
        print(f"  Action: {metadata['action']}")
        print(f"  Description: {metadata['description']}")

        # Load motion tensor
        full_path = Path("data/amass_converted") / seq_path
        motion = torch.load(full_path)

        # Analyze quality
        quality = analyze_sequence_quality(motion)
        quality["sequence"] = seq_path
        quality["dataset"] = metadata["dataset"]
        quality["action"] = metadata["action"]
        quality_metrics.append(quality)

        print("  Quality Metrics:")
        print(f"    Smoothness: {quality['smoothness']:.4f}")
        print(f"    Consistency: {quality['consistency']:.4f}")
        print(f"    Motion Range: {quality['motion_range']:.4f}")
        print(f"    Avg Velocity: {quality['avg_velocity']:.4f}")
        print(f"    Max Velocity: {quality['max_velocity']:.4f}")

        # Render video
        output_path = (
            output_dir
            / f"sample_{i:02d}_{metadata['dataset']}_{metadata['action']}.mp4"
        )
        title = (
            f"{metadata['dataset']} - {metadata['action']}\n{metadata['description']}"
        )

        try:
            render_sequence(motion, output_path, title)
        except Exception as e:
            print(f"  ❌ Error rendering: {e}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    avg_smoothness = np.mean([m["smoothness"] for m in quality_metrics])
    avg_consistency = np.mean([m["consistency"] for m in quality_metrics])
    avg_range = np.mean([m["motion_range"] for m in quality_metrics])
    avg_velocity = np.mean([m["avg_velocity"] for m in quality_metrics])

    print("\n📊 Average Quality Metrics:")
    print(f"  Smoothness: {avg_smoothness:.4f}")
    print(f"  Consistency: {avg_consistency:.4f}")
    print(f"  Motion Range: {avg_range:.4f}")
    print(f"  Avg Velocity: {avg_velocity:.4f}")

    print("\n✅ Validation complete!")
    print(f"✅ Videos saved to: {output_dir}")

    # Save quality metrics
    metrics_file = output_dir / "quality_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(quality_metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
