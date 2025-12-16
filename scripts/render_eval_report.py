#!/usr/bin/env python3
"""Render an HTML evaluation report from JSON metric files.

Typical usage:

    python scripts/render_eval_report.py \
        --model-results evaluation_results.json \
        --dataset-results dataset_quality.json \
        --output report.html
"""

import argparse
from pathlib import Path

# Ensure project root is importable
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent.parent))

from src.eval.reporting import write_html_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HTML evaluation report")
    parser.add_argument(
        "--model-results", type=str, required=True, help="Path to model evaluation JSON"
    )
    parser.add_argument(
        "--dataset-results",
        type=str,
        default=None,
        help="Optional dataset quality JSON",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_report.html", help="Output HTML path"
    )
    parser.add_argument(
        "--title", type=str, default="Stick-Gen Evaluation Report", help="Report title"
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_html_report(
        model_results_path=args.model_results,
        dataset_results_path=args.dataset_results,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    main()
