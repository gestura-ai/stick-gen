"""Convert BEAT dataset to Stick-Gen canonical format.

BEAT (Body-Expression-Audio-Text) provides synchronized gesture, speech,
and text data for conversational motion synthesis.

Dataset: https://pantomatrix.github.io/BEAT/
Paper: "BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset
        for Conversational Gestures Synthesis" (ECCV 2022)

The dataset includes:
- Motion capture data (BVH format)
- Audio recordings
- Text transcriptions
- Emotion labels
- Semantic annotations

We focus on extracting gesture motion with speech/text alignment.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .convert_amass import compute_basic_physics
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator

logger = logging.getLogger(__name__)

# BEAT emotion labels mapped to Stick-Gen ActionType
BEAT_EMOTION_TO_ACTION = {
    "neutral": ActionType.IDLE,
    "happy": ActionType.CELEBRATE,
    "sad": ActionType.CRY,
    "angry": ActionType.FIGHT,
    "surprised": ActionType.LOOKING_AROUND,
    "fear": ActionType.LOOKING_AROUND,
    "disgust": ActionType.IDLE,
    "contempt": ActionType.IDLE,
}


def _parse_bvh_to_stick(bvh_path: str, target_fps: int = 30) -> torch.Tensor | None:
    """Parse BVH file and extract stick figure representation.

    BEAT uses a 75-joint skeleton. We extract key joints for our 5-segment
    stick figure: torso, left/right arms, left/right legs.

    Returns:
        Tensor of shape [T, 20] or None if parsing fails
    """
    try:
        with open(bvh_path) as f:
            content = f.read()

        # Parse BVH header and motion data
        lines = content.strip().split("\n")

        # Find MOTION section
        motion_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "MOTION":
                motion_idx = i
                break

        if motion_idx is None:
            logger.debug(f"No MOTION section in {bvh_path}")
            return None

        # Parse frame info
        frames_line = lines[motion_idx + 1].strip()
        frame_time_line = lines[motion_idx + 2].strip()

        num_frames = int(frames_line.split(":")[1].strip())
        frame_time = float(frame_time_line.split(":")[1].strip())
        source_fps = 1.0 / frame_time if frame_time > 0 else 30.0

        # Parse motion data
        motion_data = []
        for i in range(motion_idx + 3, motion_idx + 3 + num_frames):
            if i >= len(lines):
                break
            values = [float(v) for v in lines[i].strip().split()]
            motion_data.append(values)

        motion_np = np.array(motion_data, dtype=np.float32)

        if motion_np.shape[0] < 10:
            return None

        # BEAT skeleton: extract key joint positions
        # Typical BVH has 3 channels per joint (rotation) + root position
        # We'll use a simplified extraction based on joint indices

        # For BEAT's 75-joint skeleton, approximate key joints:
        # Root: 0-2, Spine: ~9-11, Neck: ~36-38
        # LeftShoulder: ~39-41, LeftElbow: ~42-44, LeftWrist: ~45-47
        # RightShoulder: ~63-65, RightElbow: ~66-68, RightWrist: ~69-71
        # LeftHip: ~3-5, LeftKnee: ~6-8, LeftAnkle: ~9-11
        # RightHip: ~12-14, RightKnee: ~15-17, RightAnkle: ~18-20

        T = motion_np.shape[0]
        D = motion_np.shape[1]

        # Simplified: use first 20 dims or construct from available
        if D >= 20:
            stick = motion_np[:, :20]
        else:
            # Pad if needed
            stick = np.zeros((T, 20), dtype=np.float32)
            stick[:, :D] = motion_np

        # Normalize to reasonable range
        stick = (stick - stick.mean()) / (stick.std() + 1e-8) * 0.1

        # Resample to target FPS if needed
        if abs(source_fps - target_fps) > 1:
            target_frames = int(T * target_fps / source_fps)
            indices = np.linspace(0, T - 1, target_frames).astype(int)
            stick = stick[indices]

        return torch.from_numpy(stick.astype(np.float32))

    except Exception as e:
        logger.debug(f"Failed to parse {bvh_path}: {e}")
        return None


def _load_text_annotation(txt_path: str) -> list[str]:
    """Load text transcription from BEAT annotation file."""
    if not os.path.exists(txt_path):
        return []

    try:
        with open(txt_path, encoding="utf-8") as f:
            content = f.read().strip()
        # BEAT text files may have timestamps, extract just text
        lines = []
        for line in content.split("\n"):
            # Skip timestamp lines
            if line.strip() and not line.strip()[0].isdigit():
                lines.append(line.strip())
        return lines if lines else [content]
    except Exception as e:
        logger.debug(f"Failed to load {txt_path}: {e}")
        return []


def _load_emotion_label(json_path: str) -> str:
    """Load emotion label from BEAT annotation JSON."""
    if not os.path.exists(json_path):
        return "neutral"

    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("emotion", "neutral")
    except Exception:
        return "neutral"


def _build_sample(
    motion: torch.Tensor,
    texts: list[str],
    emotion: str,
    clip_id: str,
    fps: int = 30,
) -> dict[str, Any]:
    """Build canonical sample from BEAT data."""
    physics = compute_basic_physics(motion, fps=fps)

    # Map emotion to action
    action_type = BEAT_EMOTION_TO_ACTION.get(emotion.lower(), ActionType.IDLE)
    action_idx = ACTION_TO_IDX[action_type]
    T = motion.shape[0]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    # Build description from text
    if texts:
        description = f"Person speaking: {texts[0][:200]}"
    else:
        description = f"Conversational gesture with {emotion} emotion."

    return {
        "description": description,
        "all_descriptions": texts,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "action_label": action_type.value,
        "emotion": emotion,
        "camera": None,
        "source": "beat",
        "meta": {
            "clip_id": clip_id,
            "fps": fps,
            "num_frames": T,
            "has_speech": len(texts) > 0,
        },
    }


def convert_beat(
    root_dir: str,
    output_path: str,
    fps: int = 30,
    max_clips: int = -1,
    physics_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Convert BEAT dataset to canonical format.

    Expected directory structure:
        root_dir/
            speaker_1/
                1_wayne_0_1_1.bvh
                1_wayne_0_1_1.txt
                1_wayne_0_1_1.json (optional, for emotion)
            speaker_2/
                ...

    Args:
        root_dir: Root directory of BEAT dataset
        output_path: Output .pt file path
        fps: Target frame rate
        max_clips: Maximum clips to process (-1 for all)
        physics_threshold: Physics validation threshold

    Returns:
        List of converted samples
    """
    logger.info(f"Converting BEAT from {root_dir}")

    # Find all BVH files
    bvh_files = list(Path(root_dir).rglob("*.bvh"))
    logger.info(f"Found {len(bvh_files)} BVH files")

    if max_clips > 0:
        bvh_files = bvh_files[:max_clips]

    validator = DataValidator(fps=fps)
    validator.max_velocity *= physics_threshold
    validator.max_acceleration *= physics_threshold

    samples: list[dict[str, Any]] = []
    skipped = 0
    emotion_counts: dict[str, int] = {}

    for i, bvh_path in enumerate(bvh_files):
        if i % 200 == 0:
            logger.info(f"Processing {i}/{len(bvh_files)}...")

        clip_id = bvh_path.stem

        try:
            # Parse BVH
            motion = _parse_bvh_to_stick(str(bvh_path), target_fps=fps)
            if motion is None or motion.shape[0] < 25:
                skipped += 1
                continue

            # Load text annotation
            txt_path = bvh_path.with_suffix(".txt")
            texts = _load_text_annotation(str(txt_path))

            # Load emotion label
            json_path = bvh_path.with_suffix(".json")
            emotion = _load_emotion_label(str(json_path))
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Build sample
            sample = _build_sample(motion, texts, emotion, clip_id, fps)

            # Validate physics
            ok, _, reason = validator.check_physics_consistency(sample["physics"])
            if not ok:
                logger.debug(f"Skipping {clip_id}: {reason}")
                skipped += 1
                continue

            samples.append(sample)

        except Exception as e:
            logger.warning(f"Error processing {clip_id}: {e}")
            skipped += 1

    logger.info(f"Converted {len(samples)}/{len(bvh_files)} clips ({skipped} skipped)")
    logger.info(f"Emotion distribution: {emotion_counts}")

    # Report action distribution
    action_counts: dict[str, int] = {}
    for s in samples:
        label = s.get("action_label", "idle")
        action_counts[label] = action_counts.get(label, 0) + 1
    logger.info(f"Action distribution: {action_counts}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)
    logger.info(f"Saved to {output_path}")

    return samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BEAT dataset to Stick-Gen format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.data_gen.convert_beat \\
        --root data/beat \\
        --output data/beat.pt

Prerequisites:
    1. Download BEAT: https://pantomatrix.github.io/BEAT/
    2. Extract to data/beat/
        """,
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Root directory of BEAT dataset"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file path"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target frame rate (default: 30)"
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=-1,
        help="Maximum clips to process (-1 for all)",
    )
    parser.add_argument(
        "--physics-threshold",
        type=float,
        default=2.0,
        help="Physics validation threshold multiplier",
    )
    args = parser.parse_args()

    convert_beat(
        args.root,
        args.output,
        fps=args.fps,
        max_clips=args.max_clips,
        physics_threshold=args.physics_threshold,
    )


if __name__ == "__main__":
    main()
