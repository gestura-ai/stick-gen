"""Simple HTML reporting utilities for Stick-Gen evaluation.

This module combines JSON outputs from scripts/evaluate.py and
scripts/eval_dataset_quality.py into a compact HTML report suitable for
RunPod jobs or local inspection.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return html.escape(str(x))


def build_html_report(
    model_results: dict[str, Any],
    dataset_results: dict[str, Any] | None = None,
    title: str = "Stick-Gen Evaluation Report",
) -> str:
    """Render a minimal HTML report from evaluation JSON blobs."""

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"

    def _section_heading(name: str) -> str:
        return f"<h2>{html.escape(name)}</h2>\n"

    parts = [
        "<html>",
        "<head>",
        f"<title>{html.escape(title)}</title>",
        "<meta charset='utf-8'>",
        "<style>body{font-family:Arial, sans-serif;}table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:4px 8px;}</style>",  # noqa: E501
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p><em>Generated at {html.escape(ts)}</em></p>",
    ]

    # Model evaluation summary
    parts.append(_section_heading("Model Evaluation"))
    mse = model_results.get("mse", {})
    temporal = model_results.get("temporal_consistency", {})
    physics = model_results.get("physics", {})
    camera = model_results.get("camera", {})

    parts.append("<table>")
    rows = [
        (
            "MSE (mean ± std)",
            f"{_fmt(mse.get('mean', 'n/a'))} ± {_fmt(mse.get('std', 'n/a'))}",
        ),
        ("Smoothness score", _fmt(temporal.get("smoothness_score", "n/a"))),
        ("Physics validator score", _fmt(physics.get("validator_score", "n/a"))),
        ("Camera stability", _fmt(camera.get("mean_stability_score", "n/a"))),
    ]
    parts.append("<tr><th>Metric</th><th>Value</th></tr>")
    for name, value in rows:
        parts.append(f"<tr><td>{html.escape(name)}</td><td>{value}</td></tr>")
    parts.append("</table>")

    # Optional dataset-level section
    if dataset_results is not None:
        parts.append(_section_heading("Dataset Quality"))
        motion = dataset_results.get("motion", {})
        phys = dataset_results.get("physics", {})
        cam = dataset_results.get("camera", {})
        meta = dataset_results.get("metadata", {})

        parts.append("<table>")
        parts.append("<tr><th>Field</th><th>Value</th></tr>")
        rows = [
            ("Num samples", dataset_results.get("num_samples", "n/a")),
            ("Motion smoothness (mean)", motion.get("smoothness_score_mean", "n/a")),
            ("Physics validator score (mean)", phys.get("validator_score_mean", "n/a")),
            ("Camera stability (mean)", cam.get("mean_stability_score", "n/a")),
            ("Data path", meta.get("data", "n/a")),
        ]
        for name, value in rows:
            parts.append(f"<tr><td>{html.escape(name)}</td><td>{_fmt(value)}</td></tr>")
        parts.append("</table>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def write_html_report(
    model_results_path: str,
    output_path: str,
    dataset_results_path: str | None = None,
    title: str = "Stick-Gen Evaluation Report",
) -> None:
    """Load JSON files and write an HTML report to disk."""

    model_results = json.loads(Path(model_results_path).read_text())
    dataset_results: dict[str, Any] | None = None
    if dataset_results_path is not None:
        dataset_results = json.loads(Path(dataset_results_path).read_text())

    html_str = build_html_report(model_results, dataset_results, title=title)
    Path(output_path).write_text(html_str, encoding="utf-8")
