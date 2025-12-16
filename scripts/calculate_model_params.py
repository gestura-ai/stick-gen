#!/usr/bin/env python3
"""
Calculate and display parameter counts for all stick-gen model variants.

This script serves as the single source of truth for model parameter counts.
Run this after any architectural changes to update documentation.

Usage:
    python scripts/calculate_model_params.py
    python scripts/calculate_model_params.py --json  # Machine-readable output
    python scripts/calculate_model_params.py --markdown  # For docs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


@dataclass
class ModelConfig:
    """Configuration for a model variant."""

    name: str
    d_model: int
    nhead: int
    num_layers: int
    input_dim: int = 20
    output_dim: int = 20
    embedding_dim: int = 1024
    dropout: float = 0.1
    num_actions: int = 64
    # Multimodal settings
    image_encoder_arch: str = "lightweight_cnn"
    fusion_strategy: str = "gated"
    image_size: tuple[int, int] = (256, 256)


# Official model configurations
MODEL_CONFIGS = {
    "small": ModelConfig(
        name="small",
        d_model=256,
        nhead=8,
        num_layers=6,
        image_encoder_arch="lightweight_cnn",
        fusion_strategy="gated",
    ),
    "medium": ModelConfig(
        name="medium",
        d_model=384,
        nhead=12,
        num_layers=8,
        image_encoder_arch="lightweight_cnn",
        fusion_strategy="gated",
    ),
    "large": ModelConfig(
        name="large",
        d_model=512,
        nhead=16,
        num_layers=10,
        image_encoder_arch="resnet",
        fusion_strategy="cross_attention",
    ),
}


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_component(model: torch.nn.Module) -> dict[str, int]:
    """Count parameters grouped by major component (all params, not just trainable)."""
    components = {
        "text_projection": 0,
        "motion_embedding": 0,
        "action_embedding": 0,
        "action_projection": 0,
        "camera_projection": 0,
        "partner_projection": 0,
        "transformer_encoder": 0,
        "pose_decoder": 0,
        "position_decoder": 0,
        "velocity_decoder": 0,
        "action_predictor": 0,
        "physics_decoder": 0,
        "environment_decoder": 0,
        "image_encoder": 0,
        "fusion_module": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        matched = False
        for component in components:
            if component in name:
                components[component] += param.numel()
                matched = True
                break

        if not matched:
            components["other"] += param.numel()

    return {k: v for k, v in components.items() if v > 0}


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def calculate_all_variants() -> dict[str, Any]:
    """Calculate parameter counts for all model variants."""
    from src.model.transformer import StickFigureTransformer

    results = {}

    for variant_name, config in MODEL_CONFIGS.items():
        variant_results = {
            "name": variant_name,
            "config": {
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "embedding_dim": config.embedding_dim,
                "num_actions": config.num_actions,
            },
            "motion_only": {},
            "multimodal": {},
        }

        # Motion-only model
        model_motion_only = StickFigureTransformer(
            input_dim=config.input_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_actions=config.num_actions,
            enable_image_conditioning=False,
        )

        motion_only_total = count_parameters(model_motion_only)
        motion_only_breakdown = count_parameters_by_component(model_motion_only)

        variant_results["motion_only"] = {
            "total": motion_only_total,
            "formatted": format_params(motion_only_total),
            "breakdown": motion_only_breakdown,
        }

        # Multimodal model
        model_multimodal = StickFigureTransformer(
            input_dim=config.input_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_actions=config.num_actions,
            enable_image_conditioning=True,
            image_encoder_arch=config.image_encoder_arch,
            image_size=config.image_size,
            fusion_strategy=config.fusion_strategy,
        )

        multimodal_total = count_parameters(model_multimodal)
        multimodal_breakdown = count_parameters_by_component(model_multimodal)

        # Calculate overhead
        overhead = multimodal_total - motion_only_total

        variant_results["multimodal"] = {
            "total": multimodal_total,
            "formatted": format_params(multimodal_total),
            "breakdown": multimodal_breakdown,
            "image_encoder_arch": config.image_encoder_arch,
            "fusion_strategy": config.fusion_strategy,
            "overhead": overhead,
            "overhead_formatted": format_params(overhead),
        }

        results[variant_name] = variant_results

        # Clean up
        del model_motion_only
        del model_multimodal

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("=" * 80)
    print("STICK-GEN MODEL PARAMETER COUNTS")
    print("=" * 80)
    print()

    # Summary table
    print("SUMMARY TABLE")
    print("-" * 80)
    print(
        f"{'Variant':<10} {'Motion-Only':>15} {'Multimodal':>15} {'Overhead':>15} {'Encoder':>15}"
    )
    print("-" * 80)

    for name, data in results.items():
        mo = data["motion_only"]["formatted"]
        mm = data["multimodal"]["formatted"]
        oh = data["multimodal"]["overhead_formatted"]
        enc = data["multimodal"]["image_encoder_arch"]
        print(f"{name:<10} {mo:>15} {mm:>15} {oh:>15} {enc:>15}")

    print("-" * 80)
    print()

    # Detailed breakdown for each variant
    for name, data in results.items():
        print(f"\n{'=' * 40}")
        print(f"  {name.upper()} VARIANT")
        print(f"{'=' * 40}")
        print(f"  d_model: {data['config']['d_model']}")
        print(f"  nhead: {data['config']['nhead']}")
        print(f"  num_layers: {data['config']['num_layers']}")
        print()

        print(
            f"  MOTION-ONLY: {data['motion_only']['formatted']} ({data['motion_only']['total']:,} params)"
        )
        print("  Breakdown:")
        for component, count in sorted(
            data["motion_only"]["breakdown"].items(), key=lambda x: -x[1]
        ):
            print(f"    {component}: {format_params(count)} ({count:,})")

        print()
        print(
            f"  MULTIMODAL: {data['multimodal']['formatted']} ({data['multimodal']['total']:,} params)"
        )
        print(f"    Image encoder: {data['multimodal']['image_encoder_arch']}")
        print(f"    Fusion: {data['multimodal']['fusion_strategy']}")
        print("  Breakdown:")
        for component, count in sorted(
            data["multimodal"]["breakdown"].items(), key=lambda x: -x[1]
        ):
            print(f"    {component}: {format_params(count)} ({count:,})")


def print_markdown(results: dict[str, Any]) -> None:
    """Print markdown table for documentation."""
    print("## Model Parameter Counts\n")
    print("| Variant | Motion-Only | Multimodal | Overhead | Image Encoder | Fusion |")
    print("|---------|-------------|------------|----------|---------------|--------|")

    for name, data in results.items():
        mo = data["motion_only"]["formatted"]
        mm = data["multimodal"]["formatted"]
        oh = data["multimodal"]["overhead_formatted"]
        enc = data["multimodal"]["image_encoder_arch"]
        fus = data["multimodal"]["fusion_strategy"]
        print(f"| {name.capitalize()} | {mo} | {mm} | +{oh} | {enc} | {fus} |")

    print("\n### Detailed Breakdown\n")

    for name, data in results.items():
        print(
            f"#### {name.capitalize()} ({data['config']['d_model']}d, {data['config']['num_layers']}L)\n"
        )
        print("| Component | Motion-Only | Multimodal |")
        print("|-----------|-------------|------------|")

        all_components = set(data["motion_only"]["breakdown"].keys()) | set(
            data["multimodal"]["breakdown"].keys()
        )
        for comp in sorted(all_components):
            mo_val = data["motion_only"]["breakdown"].get(comp, 0)
            mm_val = data["multimodal"]["breakdown"].get(comp, 0)
            print(f"| {comp} | {format_params(mo_val)} | {format_params(mm_val)} |")

        print()


def print_json(results: dict[str, Any]) -> None:
    """Print JSON output for programmatic use."""
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Calculate model parameter counts")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--markdown", action="store_true", help="Output Markdown format"
    )
    args = parser.parse_args()

    # Suppress model initialization messages
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        results = calculate_all_variants()

    if args.json:
        print_json(results)
    elif args.markdown:
        print_markdown(results)
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
