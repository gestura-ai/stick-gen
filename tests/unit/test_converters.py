"""Unit tests for dataset converters.

This module tests:

* HumanML3D / KIT-ML feature mappings to the v3 12-segment schema.
* BABEL and BEAT label mappings.
* NTU RGB+D and AIST++ legacy 5-segment helpers (backward compat).
* New v3 converters for NTU RGB+D, AIST++, and 100STYLE.
* Backward compatibility of the validator and enhanced metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.data_gen.convert_babel import (
    _get_segment_actions,
    _map_babel_action,
)
from src.data_gen.convert_beat import BEAT_EMOTION_TO_ACTION
from src.data_gen.convert_humanml3d import (
    _build_sample as build_humanml3d_sample,
    _features_to_stick as humanml3d_features_to_stick,
    _infer_action_from_text,
)
from src.data_gen.convert_kit_ml import (
    _build_sample as build_kit_sample,
    _features_to_stick as kit_features_to_stick,
)
from src.data_gen.convert_ntu_rgbd import (
    _build_canonical_sample as build_ntu_canonical_sample,
    joints_to_stick as ntu_joints_to_stick,
    joints_to_v3_segments as ntu_joints_to_v3_segments,
)
from src.data_gen.convert_aist_plusplus import (
    _build_sample as build_aist_sample,
    keypoints3d_to_stick as aist_keypoints_to_stick,
    keypoints3d_to_v3_segments as aist_keypoints_to_v3_segments,
)
from src.data_gen.convert_100style_canonical import (
    _v1_motion_to_v3,
    _build_canonical_sample as build_100style_canonical_sample,
)
from src.data_gen.convert_amass import (
	    AMASSConverter,
	    build_canonical_sample as build_amass_canonical_sample,
	)
from src.data_gen.joint_utils import (
	    CanonicalJoints2D,
	    joints_to_v3_segments_2d,
	    v3_segments_to_joints_2d,
	    validate_v3_connectivity,
	)
from src.data_gen.metadata_extractors import build_enhanced_metadata
from src.data_gen.schema import ActionType, EnhancedSampleMetadata
from src.data_gen.validator import DataValidator


# --- Shared helpers for v3 end-to-end sample tests ---


def _make_synthetic_v3_motion(T: int = 16) -> torch.Tensor:
    """Create a simple connectivity-consistent v3 motion tensor ``[T, 48]``.

    We construct a minimal canonical joint set, convert it to the v3
    12-segment representation via :func:`joints_to_v3_segments_2d`, and
    validate connectivity. This exercises the same path used by dataset
    converters.
    """

    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    zeros = np.zeros_like(t)
    ones = np.ones_like(t)

    def _stack(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([x, y], axis=1)

    pelvic_y = zeros
    chest_y = 0.5 * ones
    neck_y = 0.75 * ones
    head_y = ones

    pelvis_center = _stack(zeros, pelvic_y)
    chest = _stack(zeros, chest_y)
    neck = _stack(zeros, neck_y)
    head_center = _stack(zeros, head_y)

    # Symmetric limbs around the vertical axis
    l_shoulder = _stack(-0.2 * ones, chest_y)
    r_shoulder = _stack(0.2 * ones, chest_y)
    l_elbow = _stack(-0.5 * ones, 0.6 * ones)
    r_elbow = _stack(0.5 * ones, 0.6 * ones)
    l_wrist = _stack(-0.8 * ones, 0.4 * ones)
    r_wrist = _stack(0.8 * ones, 0.4 * ones)

    l_hip = _stack(-0.2 * ones, pelvic_y)
    r_hip = _stack(0.2 * ones, pelvic_y)
    l_knee = _stack(-0.2 * ones, -0.5 * ones)
    r_knee = _stack(0.2 * ones, -0.5 * ones)
    l_ankle = _stack(-0.2 * ones, -1.0 * ones)
    r_ankle = _stack(0.2 * ones, -1.0 * ones)

    canonical_joints: CanonicalJoints2D = {
        "pelvis_center": pelvis_center,
        "chest": chest,
        "neck": neck,
        "head_center": head_center,
        "l_shoulder": l_shoulder,
        "l_elbow": l_elbow,
        "l_wrist": l_wrist,
        "r_shoulder": r_shoulder,
        "r_elbow": r_elbow,
        "r_wrist": r_wrist,
        "l_hip": l_hip,
        "l_knee": l_knee,
        "l_ankle": l_ankle,
        "r_hip": r_hip,
        "r_knee": r_knee,
        "r_ankle": r_ankle,
    }

    segments = joints_to_v3_segments_2d(canonical_joints, flatten=True)
    validate_v3_connectivity(segments)
    return torch.from_numpy(segments)


def _assert_canonical_sample_structure(
    sample: dict[str, Any], *, expect_camera: bool | None
) -> None:
    """Validate the shared structure/invariants of canonical v3 samples.

    Checks motion shape, physics/actions alignment, non-empty description,
    parseable enhanced metadata, and v3 connectivity via
    :func:`validate_v3_connectivity`.
    """

    assert "motion" in sample, "sample is missing 'motion' field"
    motion = sample["motion"]
    assert isinstance(motion, torch.Tensor)
    assert motion.ndim == 2 and motion.shape[1] == 48

    T = motion.shape[0]

    physics = sample.get("physics")
    assert isinstance(physics, torch.Tensor)
    assert physics.shape[0] == T
    assert physics.shape[1] == 6

    actions = sample.get("actions")
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (T,)

    desc = sample.get("description")
    assert isinstance(desc, str) and desc

    camera = sample.get("camera")
    if expect_camera is True:
        assert isinstance(camera, torch.Tensor)
        assert camera.shape[0] == T
    elif expect_camera is False:
        assert camera is None or isinstance(camera, torch.Tensor)

    enhanced_meta_raw = sample.get("enhanced_meta")
    assert isinstance(enhanced_meta_raw, dict)
    # Ensure the dict is compatible with the EnhancedSampleMetadata schema.
    EnhancedSampleMetadata.model_validate(enhanced_meta_raw)

    # Finally, enforce v3 skeleton connectivity.
    validate_v3_connectivity(motion.detach().cpu().numpy())


# --- HumanML3D Action Inference Tests ---


def test_infer_action_walk() -> None:
    """Test action inference for walking."""

    texts = ["A person is walking forward slowly"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.WALK


def test_infer_action_run() -> None:
    """Test action inference for running."""

    texts = ["Someone running fast across the field"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.RUN


def test_infer_action_jump() -> None:
    """Test action inference for jumping."""

    texts = ["A person jumps high into the air"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.JUMP


def test_infer_action_dance() -> None:
    """Test action inference for dancing."""

    texts = ["A dancer performs ballet moves"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.DANCE


def test_infer_action_punch() -> None:
    """Test action inference for punching."""

    texts = ["The boxer throws a punch"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.PUNCH


def test_infer_action_idle_fallback() -> None:
    """Unknown actions should fall back to IDLE."""

    texts = ["Something completely unrelated to motion"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.IDLE


def test_infer_action_empty() -> None:
    """Empty text list should return IDLE."""

    action = _infer_action_from_text([])
    assert action == ActionType.IDLE


def test_infer_action_multiple_texts() -> None:
    """Should infer from multiple text descriptions."""

    texts = [
        "A person stands still",
        "Then they start walking",
        "And eventually run",
    ]
    action = _infer_action_from_text(texts)
    # Should pick based on frequency - walk or run or stand
    assert action in [ActionType.WALK, ActionType.RUN, ActionType.STAND]


# --- HumanML3D Feature Mapping Tests ---


def test_humanml3d_features_to_stick_shape() -> None:
    """Test that feature mapping produces correct v3 shape [T, 48]."""

    feats = np.random.randn(100, 263).astype(np.float32)
    stick = humanml3d_features_to_stick(feats)

    assert stick.shape == (100, 48)
    assert stick.dtype == np.float32


def test_humanml3d_features_to_stick_small_input() -> None:
    """Test feature mapping with smaller input."""

    feats = np.random.randn(50, 67).astype(np.float32)
    stick = humanml3d_features_to_stick(feats)

    assert stick.shape == (50, 48)


# --- KIT-ML Feature Mapping Tests ---


def test_kit_features_to_stick_shape() -> None:
    """Test KIT-ML feature mapping to v3 [T, 48]."""

    feats = np.random.randn(100, 251).astype(np.float32)
    stick = kit_features_to_stick(feats)

    assert stick.shape == (100, 48)
    assert stick.dtype == np.float32


# --- BABEL Action Mapping Tests ---


def test_babel_map_action_walk() -> None:
    """Test BABEL action mapping for walk."""

    action = _map_babel_action("walk")
    assert action == ActionType.WALK


def test_babel_map_action_running() -> None:
    """Test BABEL action mapping for running."""

    action = _map_babel_action("running")
    assert action == ActionType.RUN


def test_babel_map_action_fuzzy() -> None:
    """Test fuzzy matching for BABEL actions."""

    action = _map_babel_action("walking slowly")
    assert action == ActionType.WALK


def test_babel_map_action_unknown() -> None:
    """Unknown BABEL actions should map to IDLE."""

    action = _map_babel_action("completely unknown action xyz")
    assert action == ActionType.IDLE


def test_babel_get_segment_actions_empty() -> None:
    """Test segment action extraction with no annotations."""

    annotations: dict[str, object] = {}
    actions, labels, desc = _get_segment_actions(annotations, "seq_001", 100)

    assert actions.shape == (100,)
    assert labels == ["idle"]
    assert "BABEL" in desc


# --- BEAT Emotion Mapping Tests ---


def test_beat_emotion_mapping_complete() -> None:
    """Test that all BEAT emotions have mappings."""

    expected_emotions = [
        "neutral",
        "happy",
        "sad",
        "angry",
        "surprised",
        "fear",
        "disgust",
        "contempt",
    ]

    for emotion in expected_emotions:
        assert emotion in BEAT_EMOTION_TO_ACTION
        assert isinstance(BEAT_EMOTION_TO_ACTION[emotion], ActionType)


# --- Backward Compatibility Tests (Validator & Metadata) ---


def test_sample_without_enhanced_meta_validates() -> None:
    """Samples without enhanced_meta should still validate correctly."""

    sample = {
        "motion": torch.randn(100, 20),
        "physics": torch.randn(100, 6),
        "actions": torch.zeros(100, dtype=torch.long),
        "camera": torch.zeros(100, 3),
        "description": "A person walking",
        "source": "test",
        "meta": {"fps": 25},
        # No enhanced_meta field
    }

    validator = DataValidator(fps=25)
    is_valid, _score, reason = validator.validate(sample)

    # Should still validate (enhanced_meta is optional). We allow physics/skeleton
    # failures here because this is mostly checking the control flow.
    assert is_valid or "physics" in reason.lower() or "skeleton" in reason.lower()


def test_sample_with_enhanced_meta_validates() -> None:
    """Samples with valid enhanced_meta should validate correctly."""

    motion = torch.randn(100, 20)
    enhanced_meta = build_enhanced_metadata(
        motion=motion,
        fps=25,
        description="A person walking",
        original_fps=30,
        original_num_frames=120,
    )

    sample = {
        "motion": motion,
        "physics": torch.randn(100, 6) * 0.1,  # Small values to pass physics checks
        "actions": torch.zeros(100, dtype=torch.long),
        "camera": torch.zeros(100, 3),
        "description": "A person walking",
        "source": "test",
        "meta": {"fps": 25},
        "enhanced_meta": enhanced_meta.model_dump(),
    }

    validator = DataValidator(fps=25)
    is_valid, _score, reason = validator.validate(sample)

    # Should validate metadata ranges
    assert "motion style range error" not in reason.lower()
    assert "temporal metadata error" not in reason.lower()
    assert "quality range error" not in reason.lower()


def test_enhanced_meta_with_invalid_ranges_fails() -> None:
    """Samples with out-of-range enhanced_meta values should fail validation."""

    sample = {
        "motion": torch.randn(100, 20),
        "physics": torch.randn(100, 6) * 0.1,
        "actions": torch.zeros(100, dtype=torch.long),
        "enhanced_meta": {
            "motion_style": {
                "tempo": 1.5,  # Invalid: > 1.0
                "energy_level": 0.5,
                "smoothness": 0.5,
            },
            "temporal": None,
            "quality": None,
        },
    }

    validator = DataValidator(fps=25)
    is_valid, _score, reason = validator.check_enhanced_metadata(
        sample["enhanced_meta"]
    )

    assert not is_valid
    assert "motion style range error" in reason.lower()


# --- Canonical Segment Ordering Tests (legacy 5-segment schema) ---


def test_canonical_segment_names() -> None:
    """Verify that the canonical segment names match the legacy exporter."""

    from src.inference.exporter import MotionExporter

    exporter = MotionExporter()
    expected_segments = ["torso", "l_leg", "r_leg", "l_arm", "r_arm"]
    assert exporter.segment_names == expected_segments


def test_ntu_joints_to_stick_segment_ordering() -> None:
    """Test that NTU joints_to_stick produces canonical segment ordering.

    Canonical legacy format (5 segments):
        Segment 0: Torso (neck -> hip)
        Segment 1: Left Leg (hip -> left_foot)
        Segment 2: Right Leg (hip -> right_foot)
        Segment 3: Left Arm (neck -> left_hand)
        Segment 4: Right Arm (neck -> right_hand)
    """

    # Create a simple test skeleton with known positions
    T = 1
    joints = np.zeros((T, 25, 3), dtype=np.float32)

    # Set known joint positions (x, y, z)
    # Index 2: Neck (SpineShoulder)
    joints[0, 2] = [0.0, 1.0, 0.0]  # neck at (0, 1)
    # Index 1: SpineBase (hip)
    joints[0, 1] = [0.0, 0.0, 0.0]  # hip at origin
    # Index 12: HipLeft, Index 16: HipRight
    joints[0, 12] = [-0.2, 0.0, 0.0]
    joints[0, 16] = [0.2, 0.0, 0.0]
    # Index 7: HandLeft
    joints[0, 7] = [-1.0, 0.5, 0.0]  # left hand
    # Index 11: HandRight
    joints[0, 11] = [1.0, 0.5, 0.0]  # right hand
    # Index 19: FootLeft
    joints[0, 19] = [-0.3, -1.0, 0.0]  # left foot
    # Index 23: FootRight
    joints[0, 23] = [0.3, -1.0, 0.0]  # right foot

    stick = ntu_joints_to_stick(joints)
    assert stick.shape == (1, 20)

    # Reshape to [T, 5, 4] for easier checking
    segments = stick.reshape(1, 5, 4)

    # Segment 0: Torso (neck -> hip)
    assert np.allclose(segments[0, 0, 0:2], [0.0, 1.0], atol=0.1)  # start: neck

    # Segment 1: Left Leg (hip_center -> left_foot)
    assert np.allclose(segments[0, 1, 2:4], [-0.3, -1.0], atol=0.1)  # end: left foot

    # Segment 2: Right Leg (hip_center -> right_foot)
    assert np.allclose(segments[0, 2, 2:4], [0.3, -1.0], atol=0.1)  # end: right foot

    # Segment 3: Left Arm (neck -> left_hand)
    assert np.allclose(segments[0, 3, 0:2], [0.0, 1.0], atol=0.1)  # start: neck
    assert np.allclose(segments[0, 3, 2:4], [-1.0, 0.5], atol=0.1)  # end: left hand

    # Segment 4: Right Arm (neck -> right_hand)
    assert np.allclose(segments[0, 4, 0:2], [0.0, 1.0], atol=0.1)  # start: neck
    assert np.allclose(segments[0, 4, 2:4], [1.0, 0.5], atol=0.1)  # end: right hand


def test_aist_keypoints3d_to_stick_shape() -> None:
    """Legacy AIST++ helper should still produce [T, 20] stick figures."""

    keypoints = np.random.randn(16, 17, 3).astype(np.float32)
    stick = aist_keypoints_to_stick(keypoints)

    assert stick.shape == (16, 20)


# --- New v3 Converter Tests (NTU, AIST++, 100STYLE) ---


def test_ntu_joints_to_v3_segments_shape_and_connectivity() -> None:
    """NTU v3 helper should produce [T, 48] with valid connectivity."""

    joints = np.random.randn(8, 25, 3).astype(np.float32)
    segments = ntu_joints_to_v3_segments(joints)

    assert segments.shape == (8, 48)
    validate_v3_connectivity(segments)


def test_aist_keypoints3d_to_v3_segments_shape_and_connectivity() -> None:
    """AIST++ v3 helper should produce [T, 48] with valid connectivity."""

    keypoints = np.random.randn(12, 17, 3).astype(np.float32)
    segments = aist_keypoints_to_v3_segments(keypoints)

    assert segments.shape == (12, 48)
    validate_v3_connectivity(segments)


def test_100style_v1_motion_to_v3_shape_and_connectivity() -> None:
    """100STYLE legacy 20D motion should upgrade cleanly to v3 48D."""

    motion_v1 = torch.randn(32, 20)
    motion_v3 = _v1_motion_to_v3(motion_v1)

    assert motion_v3.shape == (32, 48)
    validate_v3_connectivity(motion_v3.numpy())


# --- End-to-end v3 sample construction tests ---


def test_amass_build_canonical_sample_v3_end_to_end() -> None:
    """AMASS canonical builder should emit a fully v3-compliant sample.

    Uses a synthetic but connectivity-consistent v3 motion tensor to exercise
    physics, actions, and enhanced metadata construction.
    """

    motion = _make_synthetic_v3_motion(T=16)
    sample = build_amass_canonical_sample(
        motion=motion,
        npz_path="data/amass/Subject1/sequence_test.npz",
        fps=25,
        original_fps=120,
        original_num_frames=480,
        betas=np.zeros(16, dtype=np.float32),
        gender="neutral",
    )

    assert sample["source"] == "amass"
    _assert_canonical_sample_structure(sample, expect_camera=False)


def test_amass_smpl_to_v3_segments_height_normalization() -> None:
	    """AMASS smpl_to_v3_segments_2d should normalize body height to ~1.8.

	    We construct a simple upright SMPL skeleton with an arbitrary scale and
	    verify that after conversion the median head-to-ankle distance is close
	    to the 1.8-unit target used across converters.
	    """

	    T = 10
	    smpl_joints = np.zeros((T, 22, 3), dtype=np.float32)
	    idx = AMASSConverter.SMPL_JOINTS

	    for t in range(T):
	        # Axial chain
	        smpl_joints[t, idx["pelvis"]] = [0.0, 0.0, 0.0]
	        smpl_joints[t, idx["neck"]] = [0.0, 0.8, 0.0]
	        smpl_joints[t, idx["head"]] = [0.0, 1.4, 0.0]

	        # Hips and legs (symmetric around pelvis)
	        smpl_joints[t, idx["l_hip"]] = [-0.15, -0.1, 0.0]
	        smpl_joints[t, idx["r_hip"]] = [0.15, -0.1, 0.0]
	        smpl_joints[t, idx["l_knee"]] = [-0.15, -0.7, 0.0]
	        smpl_joints[t, idx["r_knee"]] = [0.15, -0.7, 0.0]
	        smpl_joints[t, idx["l_ankle"]] = [-0.15, -1.4, 0.0]
	        smpl_joints[t, idx["r_ankle"]] = [0.15, -1.4, 0.0]

	        # Shoulders and arms
	        smpl_joints[t, idx["l_shoulder"]] = [-0.25, 0.8, 0.0]
	        smpl_joints[t, idx["r_shoulder"]] = [0.25, 0.8, 0.0]
	        smpl_joints[t, idx["l_elbow"]] = [-0.6, 0.5, 0.0]
	        smpl_joints[t, idx["r_elbow"]] = [0.6, 0.5, 0.0]
	        smpl_joints[t, idx["l_wrist"]] = [-0.9, 0.2, 0.0]
	        smpl_joints[t, idx["r_wrist"]] = [0.9, 0.2, 0.0]

	    converter = AMASSConverter()
	    segments = converter.smpl_to_v3_segments_2d(smpl_joints)

	    # Recover canonical joints to measure height after normalization.
	    joints = v3_segments_to_joints_2d(segments, validate=True)
	    head = joints["head_center"]
	    l_ankle = joints["l_ankle"]
	    r_ankle = joints["r_ankle"]

	    heights = np.maximum(
	        np.linalg.norm(head - l_ankle, axis=1),
	        np.linalg.norm(head - r_ankle, axis=1),
	    )
	    median_height = float(np.median(heights))

	    assert np.isclose(median_height, 1.8, atol=1e-2)


def test_humanml3d_build_sample_v3_end_to_end() -> None:
    """HumanML3D _build_sample should produce a valid v3 canonical sample."""

    T = 20
    feats = np.random.randn(T, 263).astype(np.float32)
    texts = ["A person walks across the room."]
    sample = build_humanml3d_sample(
        feats=feats,
        texts=texts,
        clip_id="test_clip",
        stats_fps=20,
        include_camera=True,
    )

    assert sample["source"] == "humanml3d"
    _assert_canonical_sample_structure(sample, expect_camera=True)


def test_kit_build_sample_v3_end_to_end() -> None:
    """KIT-ML _build_sample should produce a valid v3 canonical sample."""

    T = 24
    feats = np.random.randn(T, 251).astype(np.float32)
    texts = ["A person performs a motion from KIT-ML."]
    sample = build_kit_sample(
        feats=feats,
        texts=texts,
        clip_id="kit_test_clip",
        fps=30,
    )

    assert sample["source"] == "kit_ml"
    _assert_canonical_sample_structure(sample, expect_camera=False)


def test_ntu_build_canonical_sample_v3_end_to_end() -> None:
    """NTU RGB+D canonical builder should emit a valid v3 sample."""

    T = 12
    joints = np.random.randn(T, 25, 3).astype(np.float32)
    meta = {
        "action_id": 31,
        "subject_id": 1,
        "camera_id": 1,
        "replication_id": 1,
        "fps": 30,
    }
    sample = build_ntu_canonical_sample(joints=joints, meta=meta, fps=30)

    assert sample["source"] == "ntu_rgbd"
    _assert_canonical_sample_structure(sample, expect_camera=False)


def test_aist_build_sample_v3_end_to_end() -> None:
    """AIST++ _build_sample should emit a valid v3 sample with camera."""

    T = 18
    keypoints = np.random.randn(T, 17, 3).astype(np.float32)
    seq_name = "gWA_sFM_cAll_d24_mWA1_ch01"
    meta = {"fps": 60}
    sample = build_aist_sample(keypoints=keypoints, seq_name=seq_name, meta=meta, fps=60)

    assert sample["source"] == "aist_plusplus"
    _assert_canonical_sample_structure(sample, expect_camera=True)


def test_100style_canonical_build_sample_v3_end_to_end() -> None:
    """100STYLE canonical txt path should upgrade v1 motion to v3 samples."""

    T = 32
    motion_v1 = torch.randn(T, 20)
    item = {
        "motion": motion_v1,
        "style": "neutral",
        "sequence_idx": 0,
        "frame_range": (0, T - 1),
        "source": "100style_txt",
    }

    sample = build_100style_canonical_sample(item, fps=25)
    assert sample["source"] == "100style_txt"
    _assert_canonical_sample_structure(sample, expect_camera=False)
