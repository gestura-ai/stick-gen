

# Import centralized paths config
try:
    from ..config.paths import get_path
    DEFAULT_OUTPUT_PATH = str(get_path("babel_canonical"))
except ImportError:
    DEFAULT_OUTPUT_PATH = "data/motions_processed/babel/canonical.pt"

"""Convert BABEL dataset to Stick-Gen canonical format.

BABEL (Bodies, Action and Behavior with English Labels) provides dense
action annotations for AMASS sequences. It maps frame-level action labels
to continuous motion capture data.

Dataset: https://babel.is.tue.mpg.de/
Paper: "BABEL: Bodies, Action and Behavior with English Labels" (CVPR 2021)

Requirements:
- AMASS dataset (provides motion data)
- BABEL annotations (provides action labels)
"""

import json
import logging
import os
from typing import Any

import torch

from .convert_amass import AMASSConverter, compute_basic_physics
from .metadata_extractors import build_enhanced_metadata
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator

logger = logging.getLogger(__name__)

# BABEL action categories mapped to Stick-Gen ActionType
# BABEL has ~250+ action categories, we map common ones
BABEL_TO_STICKGEN = {
    # Locomotion
    "walk": ActionType.WALK,
    "walking": ActionType.WALK,
    "run": ActionType.RUN,
    "running": ActionType.RUN,
    "jog": ActionType.RUN,
    "jogging": ActionType.RUN,
    "sprint": ActionType.SPRINT,
    "jump": ActionType.JUMP,
    "jumping": ActionType.JUMP,
    "hop": ActionType.JUMP,
    "skip": ActionType.JUMP,
    # Sitting/Standing
    "sit": ActionType.SIT,
    "sit down": ActionType.SIT,
    "sitting": ActionType.SIT,
    "stand": ActionType.STAND,
    "stand up": ActionType.STAND,
    "standing": ActionType.STAND,
    "kneel": ActionType.KNEEL,
    "kneeling": ActionType.KNEEL,
    "lie": ActionType.LIE_DOWN,
    "lie down": ActionType.LIE_DOWN,
    "lying": ActionType.LIE_DOWN,
    # Dancing
    "dance": ActionType.DANCE,
    "dancing": ActionType.DANCE,
    "spin": ActionType.DANCE,
    "twirl": ActionType.DANCE,
    # Gestures
    "wave": ActionType.WAVE,
    "waving": ActionType.WAVE,
    "point": ActionType.POINT,
    "pointing": ActionType.POINT,
    "clap": ActionType.CLAP,
    "clapping": ActionType.CLAP,
    # Fighting
    "punch": ActionType.PUNCH,
    "punching": ActionType.PUNCH,
    "kick": ActionType.KICK,
    "kicking": ActionType.KICK,
    "fight": ActionType.FIGHT,
    "fighting": ActionType.FIGHT,
    "dodge": ActionType.DODGE,
    # Sports
    "throw": ActionType.THROWING,
    "throwing": ActionType.THROWING,
    "catch": ActionType.CATCHING,
    "catching": ActionType.CATCHING,
    "kick ball": ActionType.KICKING,
    # Movement
    "climb": ActionType.CLIMBING,
    "climbing": ActionType.CLIMBING,
    "crawl": ActionType.CRAWLING,
    "crawling": ActionType.CRAWLING,
    "swim": ActionType.SWIMMING,
    "swimming": ActionType.SWIMMING,
    # Emotions
    "celebrate": ActionType.CELEBRATE,
    "celebrating": ActionType.CELEBRATE,
    "laugh": ActionType.LAUGH,
    "cry": ActionType.CRY,
    "crying": ActionType.CRY,
    # Daily activities
    "eat": ActionType.EATING,
    "eating": ActionType.EATING,
    "drink": ActionType.DRINKING,
    "drinking": ActionType.DRINKING,
    # Communication
    "talk": ActionType.TALK,
    "talking": ActionType.TALK,
}


def _map_babel_action(babel_action: str) -> ActionType:
    """Map BABEL action label to Stick-Gen ActionType."""
    action_lower = babel_action.lower().strip()

    # Direct lookup
    if action_lower in BABEL_TO_STICKGEN:
        return BABEL_TO_STICKGEN[action_lower]

    # Fuzzy matching - check if any key is substring
    for key, action_type in BABEL_TO_STICKGEN.items():
        if key in action_lower or action_lower in key:
            return action_type

    return ActionType.IDLE


def _load_babel_annotations(babel_path: str) -> dict[str, Any]:
    """Load BABEL annotation JSON file."""
    with open(babel_path, encoding="utf-8") as f:
        return json.load(f)


def _get_segment_actions(
    annotations: dict[str, Any],
    seq_name: str,
    num_frames: int,
    fps: float = 30.0,
) -> tuple[torch.Tensor, list[str], str]:
    """Extract per-frame actions from BABEL annotations.

    Returns:
        - actions tensor [T]
        - list of action labels (strings)
        - primary description
    """
    # Initialize all frames as IDLE
    actions = torch.full(
        (num_frames,), ACTION_TO_IDX[ActionType.IDLE], dtype=torch.long
    )
    action_labels = []
    descriptions = []

    # BABEL annotations are keyed by sequence name
    if seq_name not in annotations:
        return actions, ["idle"], "Motion sequence from BABEL dataset."

    seq_annots = annotations[seq_name]

    # Handle None or malformed annotation entries
    if seq_annots is None:
        return actions, ["idle"], "Motion sequence from BABEL dataset."

    # BABEL has frame_ann (per-frame) and seq_ann (sequence-level)
    # These can be None in some entries, so we need to handle that
    frame_ann = seq_annots.get("frame_ann") if isinstance(seq_annots, dict) else None
    seq_ann = seq_annots.get("seq_ann") if isinstance(seq_annots, dict) else None

    # Process frame-level annotations (segments)
    if frame_ann is not None and isinstance(frame_ann, dict):
        labels = frame_ann.get("labels") or []
        for segment in labels:
            if segment is None or not isinstance(segment, dict):
                continue
            act_cat = segment.get("act_cat", [])
            raw_label = segment.get("raw_label", "")
            start_t = segment.get("start_t", 0)
            end_t = segment.get("end_t", 0)

            # Convert time to frames
            start_frame = int(start_t * fps)
            end_frame = min(int(end_t * fps), num_frames)

            # Use first action category if available
            if act_cat:
                action_str = act_cat[0] if isinstance(act_cat, list) else act_cat
                action_type = _map_babel_action(action_str)
                action_idx = ACTION_TO_IDX[action_type]

                # Assign to frame range
                actions[start_frame:end_frame] = action_idx
                action_labels.append(action_type.value)

            if raw_label:
                descriptions.append(raw_label)

    # Also get sequence-level description
    if seq_ann is not None and isinstance(seq_ann, dict):
        seq_labels = seq_ann.get("labels") or []
        for label in seq_labels:
            if label is None or not isinstance(label, dict):
                continue
            raw = label.get("raw_label", "")
            if raw:
                descriptions.append(raw)

    # Build primary description
    if descriptions:
        primary_desc = descriptions[0]
    else:
        primary_desc = f"Motion from BABEL: {seq_name}"

    if not action_labels:
        action_labels = ["idle"]

    return actions, list(set(action_labels)), primary_desc


def _build_sample(
    motion: torch.Tensor,
    actions: torch.Tensor,
    action_labels: list[str],
    description: str,
    seq_name: str,
    fps: int = 30,
    original_fps: int | None = None,
    original_num_frames: int | None = None,
) -> dict[str, Any]:
    """Build canonical sample from BABEL data."""
    physics = compute_basic_physics(motion, fps=fps)

    # Get dominant action
    mode_idx = int(torch.mode(actions).values.item())
    if 0 <= mode_idx < len(ActionType):
        dominant_action = list(ActionType)[mode_idx].value
    else:
        dominant_action = "idle"

    # Build enhanced metadata
    enhanced_meta = build_enhanced_metadata(
        motion=motion,
        fps=fps,
        description=description,
        original_fps=original_fps,
        original_num_frames=original_num_frames,
    )

    return {
        "description": description,
        "all_descriptions": [description],
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "action_label": dominant_action,
        "action_labels": action_labels,
        "camera": None,
        "source": "babel",
        "meta": {
            "sequence_name": seq_name,
            "fps": fps,
            "num_frames": motion.shape[0],
        },
        "enhanced_meta": enhanced_meta.model_dump(),
    }


def convert_babel(
    amass_root: str,
    babel_path: str,
    output_path: str,
    smpl_model_path: str = "data/smpl_models",
    fps: int = 30,
    max_sequences: int = -1,
    physics_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Convert BABEL-annotated AMASS sequences to canonical format.

    Args:
        amass_root: Root directory of AMASS dataset
        babel_path: Path to BABEL annotations JSON (train.json or val.json)
        output_path: Output .pt file path
        smpl_model_path: Path to SMPL models for AMASS conversion
        fps: Target frame rate
        max_sequences: Maximum sequences to process (-1 for all)
        physics_threshold: Physics validation threshold

    Returns:
        List of converted samples
    """
    logger.info(f"Loading BABEL annotations from {babel_path}")
    annotations = _load_babel_annotations(babel_path)
    logger.info(f"Found {len(annotations)} annotated sequences")

    # Initialize AMASS converter
    converter = AMASSConverter(smpl_model_path=smpl_model_path)
    validator = DataValidator(fps=fps)
    validator.max_velocity *= physics_threshold
    validator.max_acceleration *= physics_threshold

    samples: list[dict[str, Any]] = []
    skipped = 0

    seq_names = list(annotations.keys())
    if max_sequences > 0:
        seq_names = seq_names[:max_sequences]

    for i, seq_name in enumerate(seq_names):
        if i % 500 == 0:
            logger.info(f"Processing {i}/{len(seq_names)}...")

        try:
            # Parse sequence path from BABEL annotation
            seq_info = annotations[seq_name]

            # Handle None or malformed annotation entries
            if seq_info is None:
                logger.debug(f"Skipping {seq_name}: annotation entry is None")
                skipped += 1
                continue

            if not isinstance(seq_info, dict):
                logger.debug(f"Skipping {seq_name}: annotation entry is not a dict")
                skipped += 1
                continue

            feat_p = seq_info.get("feat_p", "")
            if not feat_p:
                logger.debug(f"Skipping {seq_name}: no feat_p path in annotation")
                skipped += 1
                continue

            # Construct AMASS path
            npz_path = os.path.join(amass_root, feat_p)
            if not os.path.exists(npz_path):
                logger.debug(f"AMASS file not found: {npz_path}")
                skipped += 1
                continue

            # Convert AMASS to stick figure
            motion = converter.convert_sequence(npz_path, target_fps=fps)
            if motion is None or motion.shape[0] < 25:
                skipped += 1
                continue

            # Get BABEL actions
            actions, action_labels, description = _get_segment_actions(
                annotations, seq_name, motion.shape[0], fps
            )

            # Build sample
            sample = _build_sample(
                motion, actions, action_labels, description, seq_name, fps
            )

            # Validate physics
            ok, _, reason = validator.check_physics_consistency(sample["physics"])
            if not ok:
                logger.debug(f"Skipping {seq_name}: {reason}")
                skipped += 1
                continue

            samples.append(sample)

        except Exception as e:
            logger.warning(f"Error processing {seq_name}: {e}")
            skipped += 1

    logger.info(
        f"Converted {len(samples)}/{len(seq_names)} sequences ({skipped} skipped)"
    )

    # Report action distribution
    action_counts: dict[str, int] = {}
    for s in samples:
        for label in s.get("action_labels", ["idle"]):
            action_counts[label] = action_counts.get(label, 0) + 1
    logger.info(f"Action distribution: {action_counts}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)
    logger.info(f"Saved to {output_path}")

    return samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BABEL-annotated AMASS to Stick-Gen format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.data_gen.convert_babel \\
        --amass-root data/amass \\
        --babel-path data/babel/train.json \\
        --output data/babel_train.pt

Prerequisites:
    1. Download AMASS: https://amass.is.tue.mpg.de/
    2. Download BABEL: https://babel.is.tue.mpg.de/
    3. Ensure SMPL models are in data/smpl_models/
        """,
    )
    parser.add_argument(
        "--amass-root", type=str, required=True, help="Root directory of AMASS dataset"
    )
    parser.add_argument(
        "--babel-path",
        type=str,
        required=True,
        help="Path to BABEL annotations (train.json or val.json)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file path"
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default="data/smpl_models",
        help="Path to SMPL models",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-sequences", type=int, default=-1)
    parser.add_argument("--physics-threshold", type=float, default=2.0)
    args = parser.parse_args()

    convert_babel(
        args.amass_root,
        args.babel_path,
        args.output,
        smpl_model_path=args.smpl_model_path,
        fps=args.fps,
        max_sequences=args.max_sequences,
        physics_threshold=args.physics_threshold,
    )


if __name__ == "__main__":
    main()
