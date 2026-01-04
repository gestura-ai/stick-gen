"""Utilities for the v3 12-segment stick-figure skeleton.

This module defines shared helpers for constructing the v3 12-segment,
48-dimensional stick-figure representation from canonical 2D joints and
for validating exact joint connectivity.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


# Mapping type used throughout the data pipeline for canonical joints
CanonicalJoints2D = Mapping[str, np.ndarray]

# Fixed segment ordering used across the project for v3
V3_SEGMENT_NAMES: tuple[str, ...] = (
    "head",  # neck -> head_center
    "upper_torso",  # neck -> chest
    "lower_torso",  # chest -> pelvis_center
    "l_upper_arm",  # l_shoulder -> l_elbow
    "l_forearm",  # l_elbow -> l_wrist
    "r_upper_arm",  # r_shoulder -> r_elbow
    "r_forearm",  # r_elbow -> r_wrist
    "l_thigh",  # l_hip -> l_knee
    "l_shin",  # l_knee -> l_ankle
    "r_thigh",  # r_hip -> r_knee
    "r_shin",  # r_knee -> r_ankle
    "pelvis_width",  # l_hip -> r_hip
)


def joints_to_v3_segments_2d(
    joints_2d: CanonicalJoints2D,
    *,
    flatten: bool = True,
) -> np.ndarray:
    """Convert canonical 2D joints into v3 stick-figure segments.

    Args:
        joints_2d: Mapping from canonical joint name to array of shape [T, 2].
        flatten: If True, return array of shape [T, 48]; otherwise [T, 12, 4].

    Returns:
        Array of v3 segments with endpoints encoded as [x1, y1, x2, y2].

    Raises:
        KeyError: If any required canonical joint is missing.
        ValueError: If joint arrays have inconsistent shapes.
    """
    required = {
        "pelvis_center",
        "chest",
        "neck",
        "head_center",
        "l_shoulder",
        "r_shoulder",
        "l_elbow",
        "r_elbow",
        "l_wrist",
        "r_wrist",
        "l_hip",
        "r_hip",
        "l_knee",
        "r_knee",
        "l_ankle",
        "r_ankle",
    }
    missing = sorted(required.difference(joints_2d.keys()))
    if missing:
        raise KeyError(f"Missing canonical joints for v3 conversion: {missing}")

    ref = joints_2d["neck"]
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"Expected [T, 2] array for joints, got shape {ref.shape}")
    T = ref.shape[0]

    for name in required:
        arr = joints_2d[name]
        if arr.shape != ref.shape:
            raise ValueError(
                f"Inconsistent joint shape for {name}: expected {ref.shape}, got {arr.shape}"
            )

    segs = np.empty((T, 12, 4), dtype=np.float32)

    def _stack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([a, b], axis=-1)

    segs[:, 0] = _stack(joints_2d["neck"], joints_2d["head_center"])
    segs[:, 1] = _stack(joints_2d["neck"], joints_2d["chest"])
    segs[:, 2] = _stack(joints_2d["chest"], joints_2d["pelvis_center"])
    segs[:, 3] = _stack(joints_2d["l_shoulder"], joints_2d["l_elbow"])
    segs[:, 4] = _stack(joints_2d["l_elbow"], joints_2d["l_wrist"])
    segs[:, 5] = _stack(joints_2d["r_shoulder"], joints_2d["r_elbow"])
    segs[:, 6] = _stack(joints_2d["r_elbow"], joints_2d["r_wrist"])
    segs[:, 7] = _stack(joints_2d["l_hip"], joints_2d["l_knee"])
    segs[:, 8] = _stack(joints_2d["l_knee"], joints_2d["l_ankle"])
    segs[:, 9] = _stack(joints_2d["r_hip"], joints_2d["r_knee"])
    segs[:, 10] = _stack(joints_2d["r_knee"], joints_2d["r_ankle"])
    segs[:, 11] = _stack(joints_2d["l_hip"], joints_2d["r_hip"])

    if flatten:
        return segs.reshape(T, 48)
    return segs


def _as_segments_array(segments: np.ndarray) -> np.ndarray:
    """Normalize v3 segments to [T, 12, 4] shape.

    Args:
        segments: Array with shape [T, 48] or [T, 12, 4].

    Returns:
        Array with shape [T, 12, 4].
    """
    if segments.ndim == 2:
        T, dim = segments.shape
        if dim != 48:
            raise ValueError(f"Expected flattened shape [T, 48], got {segments.shape}")
        return segments.reshape(T, 12, 4)

    if segments.ndim == 3:
        T, segs, dims = segments.shape
        if segs != 12 or dims != 4:
            raise ValueError(f"Expected [T, 12, 4] segments, got {segments.shape}")
        return segments

    raise ValueError(f"Expected 2D or 3D array for segments, got ndim={segments.ndim}")


def validate_v3_connectivity(segments: np.ndarray, atol: float = 1e-5) -> None:
    """Validate that v3 stick-figure segments form a connected kinematic chain.

    This checks that shared joints (neck, chest, pelvis_center, elbows, knees,
    wrists, ankles, hips) have identical coordinates wherever they appear and
    that all coordinates are finite.

    Args:
        segments: V3 segments with shape [T, 48] or [T, 12, 4].
        atol: Absolute tolerance for coordinate comparisons.

    Raises:
        ValueError: If any connectivity constraint is violated or if non-finite
            coordinates are detected.
    """
    segs = _as_segments_array(segments)

    # Fast pre-check: require all coordinates to be finite. This avoids
    # confusing connectivity errors when upstream data contains NaNs or Infs.
    if not np.isfinite(segs).all():
        raise ValueError("v3 connectivity violated: non-finite coordinates in segments")

    def _start(idx: int) -> np.ndarray:
        return segs[:, idx, 0:2]

    def _end(idx: int) -> np.ndarray:
        return segs[:, idx, 2:4]

    # Neck consistency: head.start == upper_torso.start
    if not np.allclose(_start(0), _start(1), atol=atol):
        raise ValueError(
            "v3 connectivity violated: neck mismatch between head and upper_torso"
        )

    # Chest consistency: upper_torso.end == lower_torso.start
    if not np.allclose(_end(1), _start(2), atol=atol):
        raise ValueError(
            "v3 connectivity violated: chest mismatch between upper_torso and lower_torso"
        )

    # Pelvis center consistency: lower_torso.end == midpoint of pelvis_width
    pelvis_center = _end(2)
    pelvis_mid = 0.5 * (_start(11) + _end(11))
    if not np.allclose(pelvis_center, pelvis_mid, atol=atol):
        raise ValueError(
            "v3 connectivity violated: pelvis_center inconsistent with pelvis_width"
        )

    # Hips: left and right hips consistent with pelvis_width segment
    if not np.allclose(_start(7), _start(11), atol=atol):
        raise ValueError("v3 connectivity violated: left hip mismatch")
    if not np.allclose(_start(9), _end(11), atol=atol):
        raise ValueError("v3 connectivity violated: right hip mismatch")

    # Left arm: l_upper_arm.end == l_forearm.start
    if not np.allclose(_end(3), _start(4), atol=atol):
        raise ValueError("v3 connectivity violated: left elbow mismatch")

    # Right arm: r_upper_arm.end == r_forearm.start
    if not np.allclose(_end(5), _start(6), atol=atol):
        raise ValueError("v3 connectivity violated: right elbow mismatch")

    # Left leg: l_thigh.end == l_shin.start
    if not np.allclose(_end(7), _start(8), atol=atol):
        raise ValueError("v3 connectivity violated: left knee mismatch")

    # Right leg: r_thigh.end == r_shin.start
    if not np.allclose(_end(9), _start(10), atol=atol):
        raise ValueError("v3 connectivity violated: right knee mismatch")


def v3_segments_to_joints_2d(
    segments: np.ndarray,
    *,
    validate: bool = False,
) -> dict[str, np.ndarray]:
    """Reconstruct canonical 2D joints from v3 stick-figure segments.

    This is the inverse of :func:`joints_to_v3_segments_2d` for the 12-segment,
    48-dimensional schema. It takes either flattened ``[T, 48]`` arrays or
    ``[T, 12, 4]`` segment arrays and returns the full canonical joint set.

    Args:
        segments: V3 segments with shape ``[T, 48]`` or ``[T, 12, 4]``.
        validate: If ``True``, run :func:`validate_v3_connectivity` before
            reconstructing joints to catch malformed inputs early.

    Returns:
        A dictionary mapping canonical joint names to arrays of shape
        ``[T, 2]``. The returned joints follow the same naming convention as
        :func:`joints_to_v3_segments_2d`.

    Raises:
        ValueError: If ``segments`` has an invalid shape or fails
            connectivity validation.
    """
    segs = _as_segments_array(segments)

    if validate:
        validate_v3_connectivity(segs)

    # Axial chain
    neck = segs[:, 0, 0:2]
    head_center = segs[:, 0, 2:4]
    chest = segs[:, 1, 2:4]
    pelvis_center = segs[:, 2, 2:4]

    # Arms
    l_shoulder = segs[:, 3, 0:2]
    l_elbow = segs[:, 3, 2:4]
    l_wrist = segs[:, 4, 2:4]
    r_shoulder = segs[:, 5, 0:2]
    r_elbow = segs[:, 5, 2:4]
    r_wrist = segs[:, 6, 2:4]

    # Hips and legs
    l_hip = segs[:, 11, 0:2]
    r_hip = segs[:, 11, 2:4]
    l_knee = segs[:, 7, 2:4]
    l_ankle = segs[:, 8, 2:4]
    r_knee = segs[:, 9, 2:4]
    r_ankle = segs[:, 10, 2:4]

    return {
        "pelvis_center": pelvis_center.copy(),
        "chest": chest.copy(),
        "neck": neck.copy(),
        "head_center": head_center.copy(),
        "l_shoulder": l_shoulder.copy(),
        "r_shoulder": r_shoulder.copy(),
        "l_elbow": l_elbow.copy(),
        "r_elbow": r_elbow.copy(),
        "l_wrist": l_wrist.copy(),
        "r_wrist": r_wrist.copy(),
        "l_hip": l_hip.copy(),
        "r_hip": r_hip.copy(),
        "l_knee": l_knee.copy(),
        "r_knee": r_knee.copy(),
        "l_ankle": l_ankle.copy(),
        "r_ankle": r_ankle.copy(),
    }


def clean_canonical_joints(joints: CanonicalJoints2D) -> CanonicalJoints2D:
    """Apply geometric fixes to canonical joints to ensure valid humanoid structure.
    
    Performs the following cleaning operations:
    1. **Alignment**: Rotates the skeleton so the Spine (Pelvis->Neck) points Up (0, +Y).
    2. **Degeneracy Separation**: Forces coincident left/right joints (e.g. knees) apart.
    3. **Anthropometric Clamping**: Enforces strict limb length limits (Arm < 1.3x Torso, Leg < 1.5x Torso).
    4. **Head Rectification**: Ensures Head is strictly above the Neck.
    
    Args:
        joints: Canonical joints dict [T, 2]
        
    Returns:
        Cleaned joints dict [T, 2]
    """
    cleaned = {k: v.copy() for k, v in joints.items()}
    
    # Helper for vector magnitude
    def vec_len(v): return np.linalg.norm(v, axis=-1, keepdims=True)
    
    # --- 1. Alignment (Spine Upright) ---
    neck = cleaned["neck"]
    pelvis = cleaned["pelvis_center"]
    
    dx = neck[:, 0] - pelvis[:, 0]
    dy = neck[:, 1] - pelvis[:, 1]
    
    # Calculate rotation angle to bring spine to (0, 1) [Up]
    # Current angle in standard grid (0=Right, PI/2=Up)
    current_angles = np.arctan2(dy, dx)
    target_angle = np.pi / 2
    rotations = target_angle - current_angles
    
    cos_r = np.cos(rotations)
    sin_r = np.sin(rotations)
    
    # Rotate all joints around Pelvis
    for name in cleaned:
        if name == "pelvis_center": 
            continue
        
        # Relative to pelvis
        rel_x = cleaned[name][:, 0] - pelvis[:, 0]
        rel_y = cleaned[name][:, 1] - pelvis[:, 1]
        
        # Apply rotation
        rot_x = rel_x * cos_r - rel_y * sin_r
        rot_y = rel_x * sin_r + rel_y * cos_r
        
        # Restore position
        cleaned[name][:, 0] = pelvis[:, 0] + rot_x
        cleaned[name][:, 1] = pelvis[:, 1] + rot_y

    # --- Re-calculate Body Scale (Torso Length) ---
    # After rotation, dx should be ~0, dy should be positive
    neck = cleaned["neck"]
    pelvis = cleaned["pelvis_center"]
    
    torso_vec = neck - pelvis
    torso_len = vec_len(torso_vec)
    # Safety floor
    safe_torso = np.maximum(torso_len, 0.01)

    # --- 2. Degenerate Separation (Paired L/R Joints) ---
    # Separate coincident left/right joints horizontally along the body's lateral axis.
    # After upright alignment, the lateral axis is X (left = -X, right = +X).
    # This is critical for 2.5D parallax rendering where L/R joints have different
    # Z-depths; collapsed joints would break occlusion ordering and parallax motion.
    DEGENERACY_THRESHOLD = 0.05
    DEGENERACY_OFFSET = 0.1

    joint_pairs = [
        ("l_knee", "r_knee"),
        ("l_hip", "r_hip"),
        ("l_elbow", "r_elbow"),
        ("l_shoulder", "r_shoulder"),
    ]

    for left_name, right_name in joint_pairs:
        if left_name not in cleaned or right_name not in cleaned:
            continue

        left_joint = cleaned[left_name]
        right_joint = cleaned[right_name]

        dist = vec_len(left_joint - right_joint)
        mask_degenerate = (dist < DEGENERACY_THRESHOLD).flatten()

        if np.any(mask_degenerate):
            # Compute center point of the collapsed pair
            cx = (left_joint[mask_degenerate, 0] + right_joint[mask_degenerate, 0]) * 0.5
            cy = (left_joint[mask_degenerate, 1] + right_joint[mask_degenerate, 1]) * 0.5

            # Force separation: left goes to -X, right goes to +X
            cleaned[left_name][mask_degenerate, 0] = cx - DEGENERACY_OFFSET
            cleaned[left_name][mask_degenerate, 1] = cy
            cleaned[right_name][mask_degenerate, 0] = cx + DEGENERACY_OFFSET
            cleaned[right_name][mask_degenerate, 1] = cy

    # --- 3. Anthropometric Clamping (Parent-Relative) ---
    MAX_SHOULDER_RATIO = 0.6
    MAX_ARM_RATIO = 1.3
    MAX_LEG_RATIO = 1.5

    def clamp_rel(joint_name, parent_name, ratio):
        if joint_name not in cleaned or parent_name not in cleaned: return
        j = cleaned[joint_name]
        p = cleaned[parent_name]
        
        limit_dist = safe_torso * ratio
        curr_dist = vec_len(j - p)
        
        # Find violations
        mask_out = (curr_dist > limit_dist).flatten()
        if np.any(mask_out):
            scale = limit_dist[mask_out] / curr_dist[mask_out]
            
            # Vector J->P
            vec = j[mask_out] - p[mask_out]
            scaled_vec = vec * scale
            
            cleaned[joint_name][mask_out] = p[mask_out] + scaled_vec

    # Shoulders (relative to Neck)
    clamp_rel("l_shoulder", "neck", MAX_SHOULDER_RATIO)
    clamp_rel("r_shoulder", "neck", MAX_SHOULDER_RATIO)
    
    # Arms (relative to Shoulders)
    clamp_rel("l_wrist", "l_shoulder", MAX_ARM_RATIO)
    clamp_rel("r_wrist", "r_shoulder", MAX_ARM_RATIO)
    clamp_rel("l_elbow", "l_shoulder", MAX_ARM_RATIO * 0.6)
    clamp_rel("r_elbow", "r_shoulder", MAX_ARM_RATIO * 0.6)
    
    # Legs (relative to Hips)
    clamp_rel("l_ankle", "l_hip", MAX_LEG_RATIO)
    clamp_rel("r_ankle", "r_hip", MAX_LEG_RATIO)
    clamp_rel("l_knee", "l_hip", MAX_LEG_RATIO * 0.6)
    clamp_rel("r_knee", "r_hip", MAX_LEG_RATIO * 0.6)

    # --- 4. Head Rectification ---
    # Ensure Head is above Neck
    head = cleaned["head_center"]
    neck = cleaned["neck"]
    
    min_head_height = safe_torso * 0.2
    
    # Axis alignment: Y is Up.
    # Condition: Head Y < Neck Y + min_height
    target_y = neck[:, 1] + min_head_height[:, 0]
    mask_low_head = (head[:, 1] < target_y)
    
    if np.any(mask_low_head):
        # Snap head Y to minimum height
        # Preserve X unless it's wandering wildly? 
        # For simplicity, just fix Y, keeping X allows 'looking down' (if X is correct)
        # But if head is in chest, X might be weird too.
        # Let's enforce Y and clamp X to be close to Neck X
        
        cleaned["head_center"][mask_low_head, 1] = target_y[mask_low_head]
        
        # Clamp X to Neck X +/- 0.3*Torso
        max_x_err = safe_torso[mask_low_head, 0] * 0.3
        hx = head[mask_low_head, 0]
        nx = neck[mask_low_head, 0]
        dx_head = hx - nx
        
        clamp_mask = np.abs(dx_head) > max_x_err
        if np.any(clamp_mask):
             # Hard snap X to neck X for extreme outliers
             # Indices within mask_low_head
             # This double masking is tricky in numpy.
             # Simplified: just set X to Neck X for all low heads.
             cleaned["head_center"][mask_low_head, 0] = nx
    
    return cleaned


def normalize_skeleton_height(
    joints_2d: CanonicalJoints2D,
    *,
    target_height: float = 1.0,
) -> dict[str, np.ndarray]:
    """Normalize canonical joints so overall body height matches ``target_height``.

    This helper operates in canonical joint space and is intended for data
    preprocessing. It estimates body height from the distance between
    ``head_center`` and the ankle joints and rescales all joints around the
    mean pelvis position.

    Args:
        joints_2d: Mapping from canonical joint name to array of shape
            ``[T, 2]``.
        target_height: Desired approximate head-to-foot height in world units.

    Returns:
        A new dictionary containing scaled joint coordinates.
    """
    required = {"pelvis_center", "head_center", "l_ankle", "r_ankle"}
    missing = sorted(required.difference(joints_2d.keys()))
    if missing:
        raise KeyError(
            f"Missing canonical joints for height normalization: {missing}"
        )

    pelvis = joints_2d["pelvis_center"]
    head = joints_2d["head_center"]
    l_ankle = joints_2d["l_ankle"]
    r_ankle = joints_2d["r_ankle"]

    for name, arr in (
        ("pelvis_center", pelvis),
        ("head_center", head),
        ("l_ankle", l_ankle),
        ("r_ankle", r_ankle),
    ):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected [T, 2] array for joint '{name}', got shape {arr.shape}"
            )

    # Estimate per-frame height from head to each ankle and take the maximum.
    head_to_l = np.linalg.norm(head - l_ankle, axis=1)
    head_to_r = np.linalg.norm(head - r_ankle, axis=1)
    heights = np.maximum(head_to_l, head_to_r)
    finite = np.isfinite(heights) & (heights > 1e-6)
    if not np.any(finite):
        # Degenerate skeleton; return a shallow copy without scaling.
        return {name: np.array(arr, copy=True) for name, arr in joints_2d.items()}

    height = float(np.median(heights[finite]))
    if height <= 0.0:
        return {name: np.array(arr, copy=True) for name, arr in joints_2d.items()}

    scale = float(target_height) / height
    pelvis_ref = pelvis.mean(axis=0)

    scaled: dict[str, np.ndarray] = {}
    for name, arr in joints_2d.items():
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected [T, 2] array for joint '{name}', got shape {arr.shape}"
            )
        centered = arr - pelvis_ref
        scaled_arr = centered * scale + pelvis_ref
        scaled[name] = scaled_arr.astype(np.float32, copy=False)

    return scaled


def smooth_joints_over_time(
    joints_2d: CanonicalJoints2D,
    *,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Apply simple temporal smoothing to canonical 2D joints.

    Uses a per-coordinate moving average with ``mode='same'`` along the time
    axis. This is intended for light denoising; it does not enforce any
    physical constraints beyond preserving joint connectivity implicit in the
    input.

    Args:
        joints_2d: Mapping from canonical joint name to array of shape
            ``[T, 2]``.
        window_size: Size of the symmetric moving-average window. Values ``<= 1``
            disable smoothing.

    Returns:
        A new dictionary containing smoothed joint coordinates.
    """
    if window_size <= 1:
        return {name: np.array(arr, copy=True) for name, arr in joints_2d.items()}

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    pad = window_size // 2
    smoothed: dict[str, np.ndarray] = {}

    for name, arr in joints_2d.items():
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected [T, 2] array for joint '{name}', got shape {arr.shape}"
            )
        if arr.shape[0] < 2 or pad == 0:
            smoothed[name] = np.array(arr, copy=True)
            continue

        out = np.empty_like(arr, dtype=np.float32)
        for d in range(2):
            padded = np.pad(arr[:, d], pad_width=pad, mode="edge")
            out[:, d] = np.convolve(padded, kernel, mode="valid")
        smoothed[name] = out

    return smoothed


def mirror_v3(segments: np.ndarray) -> np.ndarray:
    """Mirror v3 segments horizontally around the vertical axis.

    This flips the x-coordinates of all segment endpoints (``x -> -x``) while
    preserving segment connectivity. It does **not** swap left/right semantics
    (e.g. ``l_arm`` remains the left arm in the canonical frame), making it
    suitable as a simple geometric augmentation.

    Args:
        segments: V3 segments with shape ``[T, 48]`` or ``[T, 12, 4]``.

    Returns:
        Mirrored segments with the same shape as the input.
    """
    segs = _as_segments_array(segments)
    mirrored = segs.copy()
    # Flip x coordinates for both endpoints.
    mirrored[:, :, 0] *= -1.0
    mirrored[:, :, 2] *= -1.0

    if segments.ndim == 2:
        return mirrored.reshape(segments.shape[0], 48)
    return mirrored


def time_stretch_v3(segments: np.ndarray, factor: float) -> np.ndarray:
    """Resample v3 segments along the time axis by a given factor.

    A factor ``> 1`` produces a longer (slower) sequence, while ``< 1``
    shortens (speeds up) the motion. Uses linear interpolation per coordinate
    and preserves segment connectivity.

    Args:
        segments: V3 segments with shape ``[T, 48]`` or ``[T, 12, 4]``.
        factor: Positive resampling factor.

    Returns:
        Resampled segments with shape ``[T', 48]`` or ``[T', 12, 4]``.
    """
    if factor <= 0.0:
        raise ValueError(f"time_stretch_v3 requires factor > 0, got {factor}")

    segs = _as_segments_array(segments)
    T = segs.shape[0]
    if T == 0 or np.isclose(factor, 1.0):
        return np.array(segments, copy=True)

    new_T = max(int(round(T * factor)), 1)
    src_idx = np.arange(T, dtype=np.float32)
    dst_idx = np.linspace(0.0, float(T - 1), new_T, dtype=np.float32)

    flat = segs.reshape(T, -1)
    flat_out = np.empty((new_T, flat.shape[1]), dtype=flat.dtype)
    for j in range(flat.shape[1]):
        flat_out[:, j] = np.interp(dst_idx, src_idx, flat[:, j])

    stretched = flat_out.reshape(new_T, 12, 4)
    if segments.ndim == 2:
        return stretched.reshape(new_T, 48)
    return stretched


def add_pose_noise_v3(
    segments: np.ndarray,
    *,
    std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add small Gaussian noise to v3 joints while preserving connectivity.

    Noise is sampled in canonical joint space so that all shared joints receive
    the same offset wherever they appear in the segment graph. This guarantees
    that :func:`validate_v3_connectivity` continues to pass after augmentation.

    Args:
        segments: V3 segments with shape ``[T, 48]`` or ``[T, 12, 4]``.
        std: Noise standard deviation as a fraction of the overall skeleton
            extent. For example, ``std=0.01`` adds noise at ~1% of body height.
        rng: Optional NumPy random generator. When ``None``,
            ``np.random.default_rng()`` is used.

    Returns:
        New segments array with the same shape as the input.
    """
    segs = _as_segments_array(segments)
    joints = v3_segments_to_joints_2d(segs, validate=True)

    all_xy = np.concatenate(list(joints.values()), axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    extent = float(np.max(max_xy - min_xy))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 1.0

    noise_scale = float(std) * extent
    if rng is None:
        rng = np.random.default_rng()

    noisy_joints: dict[str, np.ndarray] = {}
    for name, arr in joints.items():
        noise = rng.normal(loc=0.0, scale=noise_scale, size=arr.shape).astype(
            np.float32,
        )
        noisy_joints[name] = (arr + noise).astype(np.float32, copy=False)

    # Re-impose derived joints that must satisfy stricter geometric relations.
    # In the canonical layout, pelvis_center is defined as the midpoint between
    # the left and right hip joints.
    if "l_hip" in noisy_joints and "r_hip" in noisy_joints:
        pelvis_center = 0.5 * (
            noisy_joints["l_hip"] + noisy_joints["r_hip"]
        )
        noisy_joints["pelvis_center"] = pelvis_center.astype(np.float32, copy=False)

    flattened = segments.ndim == 2
    return joints_to_v3_segments_2d(noisy_joints, flatten=flattened)


def validate_segment_lengths_v3(
    segments: np.ndarray,
    *,
    min_length: float = 1e-5,
) -> None:
    """Validate that all v3 segment lengths are finite and non-degenerate.

    This helper is optional and intended for data-quality checks. It ensures
    that no segment has NaN/inf coordinates and that segment lengths exceed a
    small threshold.

    Args:
        segments: V3 segments with shape ``[T, 48]`` or ``[T, 12, 4]``.
        min_length: Minimum allowed length for any segment. Values at or below
            this threshold are treated as degenerate.

    Raises:
        ValueError: If any segment has non-finite coordinates or is shorter
            than ``min_length``.
    """
    segs = _as_segments_array(segments)
    deltas = segs[:, :, 2:4] - segs[:, :, 0:2]
    lengths = np.linalg.norm(deltas, axis=-1)

    if not np.isfinite(lengths).all():
        raise ValueError("v3 segment lengths contain non-finite values")

    if np.any(lengths <= float(min_length)):
        raise ValueError(
            f"v3 segment lengths contain values below minimum {min_length}",
        )
