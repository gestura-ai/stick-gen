import torch
from torch.utils.data import DataLoader, Dataset


def test_evaluate_model_uses_eval_metrics():
    """Smoke test that scripts.evaluate.evaluate_model runs end-to-end.

    We use a tiny dummy model and dataset to ensure the evaluation pipeline
    executes without errors and returns the expected metric keys, including
    those backed by src.eval.metrics.
    """
    from scripts.evaluate import evaluate_model  # noqa: WPS433

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, motion, embedding, return_all_outputs=False, camera_data=None):  # noqa: D401
            seq, batch, dim = motion.shape
            pose = torch.zeros_like(motion)
            physics = torch.zeros(seq, batch, 6, device=motion.device)
            action_logits = torch.zeros(seq, batch, 4, device=motion.device)
            return {
                "pose": pose,
                "physics": physics,
                "action_logits": action_logits,
            }

    class DummyDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):  # noqa: D401
            T, D = 8, 20
            motion = torch.zeros(T, D)
            embedding = torch.zeros(1024)
            targets = torch.zeros(T, D)
            actions = torch.zeros(T, dtype=torch.long)
            physics = torch.zeros(T, 6)
            camera = torch.zeros(T, 3)
            return motion, embedding, targets, actions, physics, camera

    device = torch.device("cpu")
    model = DummyModel().to(device)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = evaluate_model(model, loader, device)

    # Core metrics from original script
    assert "mse" in results
    assert "temporal_consistency" in results

    # Physics metrics should include validator-backed fields
    assert "physics" in results
    assert "validator_score" in results["physics"]

    # Camera metrics should be present and provide stability information
    assert "camera" in results
    assert "mean_stability_score" in results["camera"], results

