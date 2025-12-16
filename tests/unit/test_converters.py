"""Unit tests for dataset converters.

Tests the converter modules for HumanML3D, KIT-ML, BABEL, and BEAT datasets.
"""

import torch
import numpy as np

from src.data_gen.schema import ActionType
from src.data_gen.convert_humanml3d import (
    _infer_action_from_text,
    _features_to_stick as humanml3d_features_to_stick,
)
from src.data_gen.convert_kit_ml import (
    _features_to_stick as kit_features_to_stick,
)
from src.data_gen.convert_babel import (
    _map_babel_action,
    _get_segment_actions,
)
from src.data_gen.convert_beat import (
    BEAT_EMOTION_TO_ACTION,
    _parse_bvh_to_stick,
    _load_text_annotation,
)


# --- HumanML3D Action Inference Tests ---


def test_infer_action_walk():
    """Test action inference for walking."""
    texts = ["A person is walking forward slowly"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.WALK


def test_infer_action_run():
    """Test action inference for running."""
    texts = ["Someone running fast across the field"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.RUN


def test_infer_action_jump():
    """Test action inference for jumping."""
    texts = ["A person jumps high into the air"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.JUMP


def test_infer_action_dance():
    """Test action inference for dancing."""
    texts = ["A dancer performs ballet moves"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.DANCE


def test_infer_action_punch():
    """Test action inference for punching."""
    texts = ["The boxer throws a punch"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.PUNCH


def test_infer_action_idle_fallback():
    """Unknown actions should fall back to IDLE."""
    texts = ["Something completely unrelated to motion"]
    action = _infer_action_from_text(texts)
    assert action == ActionType.IDLE


def test_infer_action_empty():
    """Empty text list should return IDLE."""
    action = _infer_action_from_text([])
    assert action == ActionType.IDLE


def test_infer_action_multiple_texts():
    """Should infer from multiple text descriptions."""
    texts = [
        "A person stands still",
        "Then they start walking",
        "And eventually run",
    ]
    action = _infer_action_from_text(texts)
    # Should pick based on frequency - walk or run
    assert action in [ActionType.WALK, ActionType.RUN, ActionType.STAND]


# --- HumanML3D Feature Mapping Tests ---


def test_humanml3d_features_to_stick_shape():
    """Test that feature mapping produces correct shape."""
    # HumanML3D has 263 dimensions
    feats = np.random.randn(100, 263).astype(np.float32)
    stick = humanml3d_features_to_stick(feats)

    assert stick.shape == (100, 20)
    assert stick.dtype == np.float32


def test_humanml3d_features_to_stick_small_input():
    """Test feature mapping with smaller input."""
    feats = np.random.randn(50, 67).astype(np.float32)
    stick = humanml3d_features_to_stick(feats)

    assert stick.shape == (50, 20)


# --- KIT-ML Feature Mapping Tests ---


def test_kit_features_to_stick_shape():
    """Test KIT-ML feature mapping."""
    feats = np.random.randn(100, 251).astype(np.float32)
    stick = kit_features_to_stick(feats)

    assert stick.shape == (100, 20)
    assert stick.dtype == np.float32


# --- BABEL Action Mapping Tests ---


def test_babel_map_action_walk():
    """Test BABEL action mapping for walk."""
    action = _map_babel_action("walk")
    assert action == ActionType.WALK


def test_babel_map_action_running():
    """Test BABEL action mapping for running."""
    action = _map_babel_action("running")
    assert action == ActionType.RUN


def test_babel_map_action_fuzzy():
    """Test fuzzy matching for BABEL actions."""
    action = _map_babel_action("walking slowly")
    assert action == ActionType.WALK


def test_babel_map_action_unknown():
    """Unknown BABEL actions should map to IDLE."""
    action = _map_babel_action("completely unknown action xyz")
    assert action == ActionType.IDLE


def test_babel_get_segment_actions_empty():
    """Test segment action extraction with no annotations."""
    annotations = {}
    actions, labels, desc = _get_segment_actions(annotations, "seq_001", 100)

    assert actions.shape == (100,)
    assert labels == ["idle"]
    assert "BABEL" in desc


# --- BEAT Emotion Mapping Tests ---


def test_beat_emotion_mapping_complete():
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
