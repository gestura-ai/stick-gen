import json
from pathlib import Path

import torch

from scripts.eval_dataset_quality import evaluate_dataset


def test_eval_dataset_quality_smoke(tmp_path: Path) -> None:
    """Tiny smoke test for scripts.eval_dataset_quality.evaluate_dataset.

    Creates a small synthetic canonical dataset on disk and verifies that the
    dataset-level evaluator runs and returns the expected top-level keys.
    """
    data = []
    for _ in range(3):
        T, D = 8, 20
        motion = torch.zeros(T, D)
        physics = torch.zeros(T, 6)
        camera = torch.zeros(T, 3)
        data.append({"motion": motion, "physics": physics, "camera": camera})

    dataset_path = tmp_path / "synthetic_dataset.pt"
    torch.save(data, dataset_path)

    results = evaluate_dataset(str(dataset_path), max_samples=2)

    assert results["num_samples"] == 2
    assert "motion" in results
    assert "camera" in results
    assert "physics" in results

    # Also check that the results dict is JSON-serialisable
    out_path = tmp_path / "report.json"
    out_path.write_text(json.dumps(results))

