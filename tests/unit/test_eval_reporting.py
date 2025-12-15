from pathlib import Path

from src.eval.reporting import build_html_report


def test_build_html_report_minimal(tmp_path: Path) -> None:
    model_results = {
        "mse": {"mean": 0.1, "std": 0.01},
        "temporal_consistency": {"smoothness_score": 0.9},
        "physics": {"validator_score": 0.8},
        "camera": {"mean_stability_score": 0.7},
    }
    dataset_results = {
        "num_samples": 10,
        "motion": {"smoothness_score_mean": 0.85},
        "physics": {"validator_score_mean": 0.75},
        "camera": {"mean_stability_score": 0.65},
        "metadata": {"data": "dummy.pt"},
    }

    html_str = build_html_report(model_results, dataset_results)

    assert "Stick-Gen Evaluation Report" in html_str
    assert "Model Evaluation" in html_str
    assert "Dataset Quality" in html_str

    out_path = tmp_path / "report.html"
    out_path.write_text(html_str, encoding="utf-8")

    assert out_path.read_text(encoding="utf-8").startswith("<html>")

