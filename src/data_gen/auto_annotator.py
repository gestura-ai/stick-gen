import math
from typing import Any, Dict, Optional

import torch

from .schema import IDX_TO_ACTION, ActionType
from .validator import DataValidator


DEFAULT_ANNOTATION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    # Per-feature toggles
    "shot_type": True,
    "camera_motion": True,
    "action_summary": True,
    "physics": True,
    "quality": True,
}


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user config with defaults.

    Supports either:

    data_generation:
      annotation:
        enabled: true
        shot_type: true
        camera_motion: true
        ...

    or nested under an "annotators" key:

      annotation:
        enabled: true
        annotators:
          shot_type: false
    """

    if config is None:
        return DEFAULT_ANNOTATION_CONFIG.copy()

    merged = DEFAULT_ANNOTATION_CONFIG.copy()

    # Flatten any nested "annotators" section
    flat: Dict[str, Any] = {}
    for k, v in config.items():
        if k == "annotators" and isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v

    for key in merged.keys():
        if key in flat:
            merged[key] = bool(flat[key]) if isinstance(merged[key], bool) else flat[key]

    return merged


def infer_shot_type(camera: torch.Tensor) -> str:
    """Classify shot type (wide/medium/close) from camera zoom.

    Simple heuristic based on mean zoom value. Assumes ``camera`` is [T, 3]
    with columns [x, y, zoom].
    """

    cam = torch.as_tensor(camera, dtype=torch.float32)
    if cam.ndim != 2 or cam.shape[1] < 3 or cam.numel() == 0:
        return "unknown"

    zoom = cam[:, 2]
    mean_zoom = float(zoom.mean().item())

    # Heuristic thresholds tuned for typical generator ranges (zoom≈1.0 base)
    if mean_zoom < 0.8:
        return "wide"
    if mean_zoom < 1.4:
        return "medium"
    return "close"


def infer_camera_motion(camera: torch.Tensor, motion: Optional[torch.Tensor] = None) -> str:
    """Classify camera movement pattern.

    Returns one of: "static", "pan", "zoom", "tracking", "complex", "unknown".
    """

    cam = torch.as_tensor(camera, dtype=torch.float32)
    if cam.ndim != 2 or cam.shape[1] < 3 or cam.numel() == 0:
        return "unknown"

    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]

    dx_range = float(x.max().item() - x.min().item())
    dy_range = float(y.max().item() - y.min().item())
    dz_range = float(z.max().item() - z.min().item())
    pos_range = max(dx_range, dy_range)

    # Thresholds in world/zoom units – empirical but stable
    eps_pos = 1e-3
    eps_zoom = 1e-3
    small_pos = 0.1
    small_zoom = 0.05
    large_pos = 0.5
    large_zoom = 0.2

    if pos_range < eps_pos and dz_range < eps_zoom:
        return "static"

    if dz_range >= large_zoom and pos_range < small_pos:
        return "zoom"

    if pos_range >= large_pos and dz_range < small_zoom:
        return "pan"

    # Mixed position + zoom changes: try to detect tracking vs generic complex
    if motion is not None:
        try:
            mot = torch.as_tensor(motion, dtype=torch.float32)
            if mot.ndim == 2:
                mot = mot.unsqueeze(1)  # [F, 1, 20]
            if mot.ndim != 3 or mot.shape[-1] != 20:
                return "complex"

            F, A, _ = mot.shape
            if F < 2 or A < 1:
                return "complex"

            segments = mot.view(F, A, 5, 4)
            starts = segments[..., 0:2]
            ends = segments[..., 2:4]
            points = torch.cat([starts, ends], dim=2)  # [F, A, 10, 2]
            centers = points.mean(dim=2)               # [F, A, 2]

            # Average actor center across actors
            actor_centers = centers.mean(dim=1)        # [F, 2]
            cam_pos = torch.stack([x, y], dim=1)       # [F, 2]

            # Correlation along x between actor path and camera path
            vx = actor_centers[:, 0] - actor_centers[:, 0].mean()
            cx = cam_pos[:, 0] - cam_pos[:, 0].mean()
            denom = float(vx.norm().item() * cx.norm().item())
            if denom > 1e-6:
                corr = float((vx * cx).sum().item() / denom)
                if corr > 0.7:
                    return "tracking"
        except Exception:
            # Fall back to generic complex classification
            return "complex"

    return "complex"


def summarize_actions(actions: torch.Tensor) -> Dict[str, Any]:
    """Summarize dominant actions over the sequence.

    Returns a dict with:

    - ``dominant``: list of top action names (up to 3)
    - ``distribution``: mapping from action name → frequency fraction
    """

    acts = torch.as_tensor(actions, dtype=torch.long)
    if acts.ndim == 0:
        acts = acts.unsqueeze(0)
    if acts.ndim == 1:
        flat = acts
    else:
        flat = acts.reshape(-1)

    if flat.numel() == 0:
        return {"dominant": [], "distribution": {}}

    unique, counts = torch.unique(flat, return_counts=True)
    total = float(flat.numel())

    distribution: Dict[str, float] = {}
    for idx, c in zip(unique.tolist(), counts.tolist()):
        action_enum = IDX_TO_ACTION.get(idx)
        if isinstance(action_enum, ActionType):
            name = action_enum.value
        else:
            name = str(action_enum)
        distribution[name] = float(c) / total

    dominant = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    top = [name for name, _ in dominant[:3]]
    return {"dominant": top, "distribution": distribution}


def summarize_physics(physics: torch.Tensor, validator: Optional[DataValidator] = None) -> Dict[str, Any]:
    """Compute basic physics statistics and violation ratios."""

    phys = torch.as_tensor(physics, dtype=torch.float32)
    if phys.ndim == 2:
        phys = phys.unsqueeze(1)  # [F, 1, 6]
    if phys.ndim != 3 or phys.shape[-1] < 4:
        return {
            "valid": False,
            "reason": f"Unexpected physics shape: {tuple(phys.shape)}",
        }

    if validator is None:
        validator = DataValidator()

    valid, _, reason = validator.check_physics_consistency(phys)

    velocity = torch.linalg.norm(phys[..., 0:2], dim=-1)
    acceleration = torch.linalg.norm(phys[..., 2:4], dim=-1)

    max_v = float(velocity.max().item())
    mean_v = float(velocity.mean().item())
    max_a = float(acceleration.max().item())
    mean_a = float(acceleration.mean().item())

    v_ratio = max_v / float(validator.max_velocity) if validator.max_velocity > 0 else 0.0
    a_ratio = max_a / float(validator.max_acceleration) if validator.max_acceleration > 0 else 0.0
    overall_ratio = max(v_ratio, a_ratio)

    return {
        "valid": bool(valid),
        "reason": reason,
        "max_velocity": max_v,
        "mean_velocity": mean_v,
        "max_acceleration": max_a,
        "mean_acceleration": mean_a,
        "velocity_ratio": v_ratio,
        "acceleration_ratio": a_ratio,
        "overall_ratio": overall_ratio,
    }


def compute_quality(annotations: Dict[str, Any]) -> Dict[str, Any]:
    """Compute an overall quality score from available annotations.

    Combines physics safety, camera behavior, and action diversity into a
    single scalar in [0, 1], plus diagnostic flags.
    """

    physics_info = annotations.get("physics", {}) or {}
    physics_ratio = float(physics_info.get("overall_ratio", 0.0))

    # Map physics ratio → [0, 1]
    # - ratio ≤ 0.5 → 1.0
    # - ratio = 1.0 → 0.5
    # - ratio ≥ 1.5 → 0.0
    if physics_ratio <= 0.5:
        physics_score = 1.0
    elif physics_ratio <= 1.0:
        physics_score = 1.5 - physics_ratio
    else:
        physics_score = max(0.0, 1.0 - 0.5 * (physics_ratio - 1.0))

    cam_motion = annotations.get("camera_motion", "unknown")
    if cam_motion == "static":
        cam_score = 0.9
    elif cam_motion in ("pan", "zoom", "tracking"):
        cam_score = 1.0
    elif cam_motion == "complex":
        cam_score = 0.8
    else:
        cam_score = 0.8

    action_summary = annotations.get("action_summary", {}) or {}
    dist = action_summary.get("distribution", {}) or {}
    if not dist:
        diversity_score = 0.8
    else:
        max_p = max(dist.values())
        # Encourage some diversity but don't penalize single-action clips too hard
        # max_p in [0.5, 1.0] → diversity in [1.0, 0.6]
        max_p = float(max(0.0, min(1.0, max_p)))
        if max_p <= 0.5:
            diversity_score = 1.0
        else:
            diversity_score = 1.0 - 0.4 * (max_p - 0.5) / 0.5

    score = max(0.0, min(1.0, (physics_score + cam_score + diversity_score) / 3.0))

    flags = []
    if physics_ratio > 1.0:
        flags.append("physics_threshold_exceeded")
    elif physics_ratio > 0.8:
        flags.append("high_speed_or_acceleration")

    if cam_motion == "static":
        flags.append("static_camera")
    elif cam_motion == "complex":
        flags.append("complex_camera_motion")

    return {
        "score": score,
        "physics_component": physics_score,
        "camera_component": cam_score,
        "diversity_component": diversity_score,
        "flags": flags,
    }


def annotate_sample(sample: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Annotate a motion sample with camera/motion/physics labels.

    The returned sample is a shallow copy with an ``"annotations"`` dict
    attached and a few convenience top-level fields (``shot_type``,
    ``camera_motion``, ``quality_score``).
    """

    cfg = _normalize_config(config)
    if not cfg.get("enabled", True):
        return sample

    motion = sample.get("motion")
    camera = sample.get("camera")
    physics = sample.get("physics")
    actions = sample.get("actions")

    annotations: Dict[str, Any] = dict(sample.get("annotations", {}))

    if cfg.get("shot_type", True) and camera is not None:
        annotations["shot_type"] = infer_shot_type(camera)

    if cfg.get("camera_motion", True) and camera is not None:
        annotations["camera_motion"] = infer_camera_motion(camera, motion)

    if cfg.get("action_summary", True) and actions is not None:
        annotations["action_summary"] = summarize_actions(actions)

    if cfg.get("physics", True) and physics is not None:
        annotations["physics"] = summarize_physics(physics)

    if cfg.get("quality", True):
        annotations["quality"] = compute_quality(annotations)

    out = dict(sample)
    out["annotations"] = annotations

    # Convenience top-level fields
    if "shot_type" in annotations:
        out["shot_type"] = annotations["shot_type"]
    if "camera_motion" in annotations:
        out["camera_motion"] = annotations["camera_motion"]
    if "quality" in annotations:
        out["quality_score"] = annotations["quality"].get("score")

    return out
