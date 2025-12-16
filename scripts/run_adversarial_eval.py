#!/usr/bin/env python3
"""
Adversarial and Robustness Evaluation for Stick-Gen.
Gestura AI - https://gestura.ai

This script evaluates model robustness against adversarial and edge-case prompts.
It uses the safety critic to detect degenerate outputs and produces a detailed report.

Usage:
    python scripts/run_adversarial_eval.py --checkpoint checkpoints/best_model.pth
    python scripts/run_adversarial_eval.py --checkpoint checkpoints/best_model.pth --suites adversarial_contradictory adversarial_extreme_actions
    python scripts/run_adversarial_eval.py --checkpoint checkpoints/best_model.pth --output_dir eval_results/adversarial
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoTokenizer

from src.data_gen.schema import NUM_ACTIONS
from src.eval.safety_critic import (
    SafetyCritic,
)
from src.model.transformer import StickFigureTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_prompt_suites(
    config_path: str = "configs/eval/prompt_suites.yaml",
) -> dict[str, Any]:
    """Load prompt suites from YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_adversarial_suites(suites: dict[str, Any]) -> list[str]:
    """Get list of adversarial suite names (those starting with 'adversarial_')."""
    return [
        name
        for name in suites.get("suites", {}).keys()
        if name.startswith("adversarial_")
    ]


def load_model(checkpoint_path: str, device: str = "cpu") -> StickFigureTransformer:
    """Load the transformer model from checkpoint."""
    # Default architecture (can be overridden by checkpoint metadata)
    model = StickFigureTransformer(
        input_dim=20,
        d_model=384,
        nhead=12,
        num_layers=8,
        output_dim=20,
        embedding_dim=1024,
        dropout=0.1,
        num_actions=NUM_ACTIONS,
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.warning(
            f"Checkpoint not found: {checkpoint_path}. Using random weights."
        )

    model.to(device)
    model.eval()
    return model


def get_text_embedding(
    text: str, tokenizer, embed_model, device: str = "cpu"
) -> torch.Tensor:
    """Get text embedding using BAAI/bge-large-en-v1.5."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = embed_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding


def generate_motion_for_prompt(
    prompt: str,
    model: StickFigureTransformer,
    tokenizer,
    embed_model,
    num_frames: int = 250,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate motion sequence for a given prompt."""
    text_embedding = get_text_embedding(prompt, tokenizer, embed_model, device)

    # Use IDLE action for all frames (baseline)
    action_indices = torch.zeros(1, num_frames, dtype=torch.long, device=device)

    # Initialize motion sequence
    motion_sequence = torch.zeros(1, num_frames, 20, device=device)

    with torch.no_grad():
        for t in range(1, num_frames):
            motion_input = motion_sequence[:, :t, :].permute(1, 0, 2)
            action_input = action_indices[:, :t].permute(1, 0)

            output = model(
                motion_input,
                text_embedding,
                return_all_outputs=False,
                action_sequence=action_input,
            )

            next_frame = output[-1, 0, :]
            motion_sequence[0, t, :] = next_frame

    return motion_sequence[0]  # [T, D]


def evaluate_suite(
    suite_name: str,
    suite_config: dict[str, Any],
    model: StickFigureTransformer,
    tokenizer,
    embed_model,
    safety_critic: SafetyCritic,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a single prompt suite."""
    prompts = suite_config.get("prompts", [])
    expect_degradation = suite_config.get("expect_degradation", False)

    results = []
    for prompt_config in prompts:
        prompt_id = prompt_config.get("id", "unknown")
        prompt_text = prompt_config.get("text", "")
        seconds = prompt_config.get("seconds", 10)
        num_frames = min(max(int(seconds * 25), 25), 250)  # 25 FPS, clamp to [25, 250]

        logger.info(f"  Evaluating prompt: {prompt_id}")

        try:
            # Generate motion
            motion = generate_motion_for_prompt(
                prompt_text, model, tokenizer, embed_model, num_frames, device
            )

            # Run safety critic
            safety_result = safety_critic.evaluate(motion)

            results.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": (
                        prompt_text[:100] + "..."
                        if len(prompt_text) > 100
                        else prompt_text
                    ),
                    "seconds": seconds,
                    "num_frames": num_frames,
                    "is_safe": safety_result.is_safe,
                    "overall_score": safety_result.overall_score,
                    "issue_count": len(safety_result.issues),
                    "issue_types": [i.issue_type.value for i in safety_result.issues],
                    "rejection_reasons": safety_result.get_rejection_reasons(),
                    "check_results": {
                        k: {
                            kk: float(vv) if isinstance(vv, (int, float)) else vv
                            for kk, vv in v.items()
                        }
                        for k, v in safety_result.check_results.items()
                    },
                }
            )
        except Exception as e:
            logger.error(f"  Error evaluating {prompt_id}: {e}")
            results.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": (
                        prompt_text[:100] + "..."
                        if len(prompt_text) > 100
                        else prompt_text
                    ),
                    "error": str(e),
                    "is_safe": False,
                    "overall_score": 0.0,
                }
            )

    # Aggregate suite statistics
    safe_count = sum(1 for r in results if r.get("is_safe", False))
    total = len(results)
    mean_score = (
        sum(r.get("overall_score", 0) for r in results) / total if total > 0 else 0
    )

    # Issue distribution
    issue_counts: dict[str, int] = {}
    for r in results:
        for itype in r.get("issue_types", []):
            issue_counts[itype] = issue_counts.get(itype, 0) + 1

    return {
        "suite_name": suite_name,
        "description": suite_config.get("description", ""),
        "expect_degradation": expect_degradation,
        "total_prompts": total,
        "safe_count": safe_count,
        "safe_ratio": safe_count / total if total > 0 else 0,
        "mean_score": mean_score,
        "issue_distribution": issue_counts,
        "prompts": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation on Stick-Gen model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt_suites",
        type=str,
        default="configs/eval/prompt_suites.yaml",
        help="Path to prompt suites YAML",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help="Specific suites to evaluate (default: all adversarial_* suites)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/adversarial",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu/cuda)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompt suites
    logger.info(f"Loading prompt suites from {args.prompt_suites}")
    suites_config = load_prompt_suites(args.prompt_suites)
    all_suites = suites_config.get("suites", {})

    # Determine which suites to evaluate
    if args.suites:
        suite_names = [s for s in args.suites if s in all_suites]
    else:
        suite_names = get_adversarial_suites(suites_config) + ["robustness_baseline"]

    logger.info(f"Evaluating suites: {suite_names}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # Load embedding model
    logger.info("Loading embedding model (BAAI/bge-large-en-v1.5)...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    embed_model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(args.device)
    embed_model.eval()

    # Initialize safety critic
    safety_critic = SafetyCritic()

    # Evaluate each suite
    all_results = []
    for suite_name in suite_names:
        logger.info(f"Evaluating suite: {suite_name}")
        suite_config = all_suites[suite_name]
        result = evaluate_suite(
            suite_name,
            suite_config,
            model,
            tokenizer,
            embed_model,
            safety_critic,
            args.device,
        )
        all_results.append(result)

    # Compute overall statistics
    total_prompts = sum(r["total_prompts"] for r in all_results)
    total_safe = sum(r["safe_count"] for r in all_results)
    overall_safe_ratio = total_safe / total_prompts if total_prompts > 0 else 0

    # Separate baseline vs adversarial
    baseline_results = [
        r for r in all_results if not r.get("expect_degradation", False)
    ]
    adversarial_results = [r for r in all_results if r.get("expect_degradation", False)]

    baseline_safe_ratio = (
        (
            sum(r["safe_count"] for r in baseline_results)
            / sum(r["total_prompts"] for r in baseline_results)
        )
        if baseline_results and sum(r["total_prompts"] for r in baseline_results) > 0
        else 0
    )

    adversarial_safe_ratio = (
        (
            sum(r["safe_count"] for r in adversarial_results)
            / sum(r["total_prompts"] for r in adversarial_results)
        )
        if adversarial_results
        and sum(r["total_prompts"] for r in adversarial_results) > 0
        else 0
    )

    # Build final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "summary": {
            "total_prompts": total_prompts,
            "total_safe": total_safe,
            "overall_safe_ratio": overall_safe_ratio,
            "baseline_safe_ratio": baseline_safe_ratio,
            "adversarial_safe_ratio": adversarial_safe_ratio,
            "robustness_gap": baseline_safe_ratio - adversarial_safe_ratio,
        },
        "suites": all_results,
    }

    # Save report
    report_path = os.path.join(
        args.output_dir,
        f"adversarial_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ADVERSARIAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total prompts evaluated: {total_prompts}")
    print(f"Overall safe ratio: {overall_safe_ratio:.2%}")
    print(f"Baseline safe ratio: {baseline_safe_ratio:.2%}")
    print(f"Adversarial safe ratio: {adversarial_safe_ratio:.2%}")
    print(f"Robustness gap: {baseline_safe_ratio - adversarial_safe_ratio:.2%}")
    print("=" * 60)

    for result in all_results:
        status = (
            "✓"
            if result["safe_ratio"] > 0.8
            else "⚠️" if result["safe_ratio"] > 0.5 else "✗"
        )
        print(
            f"{status} {result['suite_name']}: {result['safe_ratio']:.2%} safe ({result['safe_count']}/{result['total_prompts']})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
