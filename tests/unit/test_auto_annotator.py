import torch

from src.data_gen.auto_annotator import (
    infer_shot_type,
    infer_camera_motion,
    summarize_actions,
    summarize_physics,
    compute_quality,
    annotate_sample,
)


def test_infer_shot_type_wide_medium_close():
    cam_wide = torch.zeros((10, 3)); cam_wide[:, 2] = 0.5
    cam_medium = torch.zeros((10, 3)); cam_medium[:, 2] = 1.0
    cam_close = torch.zeros((10, 3)); cam_close[:, 2] = 1.8

    assert infer_shot_type(cam_wide) == "wide"
    assert infer_shot_type(cam_medium) == "medium"
    assert infer_shot_type(cam_close) == "close"


def test_infer_camera_motion_static_pan_zoom_tracking():
    T = 20
    cam_static = torch.zeros((T, 3))

    cam_pan = torch.zeros((T, 3))
    cam_pan[:, 0] = torch.linspace(0, 5, T)

    cam_zoom = torch.zeros((T, 3))
    cam_zoom[:, 2] = torch.linspace(1.0, 2.0, T)

    motion = torch.zeros((T, 1, 20))
    motion[:, 0, 0] = torch.linspace(0, 5, T)

    assert infer_camera_motion(cam_static) == "static"
    assert infer_camera_motion(cam_pan) == "pan"
    assert infer_camera_motion(cam_zoom) == "zoom"
    assert infer_camera_motion(cam_pan, motion) in {"pan", "tracking"}


def test_summarize_actions_and_quality():
    actions = torch.tensor([[1, 1, 2, 2, 2, 0]])
    summary = summarize_actions(actions)
    assert "dominant" in summary and len(summary["dominant"]) >= 1
    assert "distribution" in summary and len(summary["distribution"]) == 3

    physics = torch.zeros((10, 1, 6))
    phys_summary = summarize_physics(physics)
    ann = {"physics": phys_summary, "action_summary": summary}
    quality = compute_quality(ann)
    assert 0.0 <= quality["score"] <= 1.0


def test_annotate_sample_end_to_end():
    T, A = 10, 2
    sample = {
        "description": "test scene",
        "motion": torch.zeros((T, A, 20)),
        "actions": torch.zeros((T, A), dtype=torch.long),
        "physics": torch.zeros((T, A, 6)),
        "camera": torch.zeros((T, 3)),
    }

    out = annotate_sample(sample, {"enabled": True})
    assert "annotations" in out
    assert "shot_type" in out
    assert "camera_motion" in out
    assert "quality_score" in out

