import glob
import os
import pickle
from typing import Any

import numpy as np
import torch

from .convert_amass import AMASSConverter, compute_basic_physics
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator


def _person_dict_to_motion(
    person: dict[str, Any], converter: AMASSConverter
) -> torch.Tensor:
    """Convert a single person's SMPL-like parameters into [T, 20] stick motion.

    The InterHuman motions store SMPL-style parameters per person:

      - trans: [T, 3]
      - root_orient: [T, 3]
      - pose_body: [T, 63]
      - betas: [10]

    We reuse the SMPL-X model and stick-figure mapping from ``AMASSConverter``
    to obtain 3D joints and then 2D stick segments.
    """

    # Ensure SMPL-X model is loaded (will no-op if already present)
    converter._load_smpl_model("smplx")
    model = converter.smplx_model
    if model is None:
        raise RuntimeError("SMPL-X model not available for InterHuman conversion")

    trans = torch.tensor(person["trans"], dtype=torch.float32)
    root_orient = torch.tensor(person["root_orient"], dtype=torch.float32)
    body_pose = torch.tensor(person["pose_body"], dtype=torch.float32)

    # Normalize body pose to [T, 63] if provided as [T, 21, 3]
    if body_pose.ndim == 3:
        T = body_pose.shape[0]
        body_pose = body_pose.view(T, -1)
    else:
        T = trans.shape[0]

    betas = person.get("betas")
    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    betas_tensor_1 = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(
        0
    )  # [1, 10]

    # Single-frame hand poses from the model's default parameters. The current
    # smplx version in this environment does not reliably handle batched
    # sequences when some pose components use learned defaults, so we run the
    # model one frame at a time.
    if hasattr(model, "left_hand_pose"):
        left_hand_pose_1 = model.left_hand_pose.clone()  # [1, 45]
    else:
        left_hand_pose_1 = torch.zeros(1, 45, dtype=torch.float32)

    if hasattr(model, "right_hand_pose"):
        right_hand_pose_1 = model.right_hand_pose.clone()  # [1, 45]
    else:
        right_hand_pose_1 = torch.zeros(1, 45, dtype=torch.float32)

    joints_list = []
    with torch.no_grad():
        for i in range(T):
            trans_i = trans[i : i + 1]
            root_i = root_orient[i : i + 1]
            body_i = body_pose[i : i + 1]

            output = model(
                body_pose=body_i,
                global_orient=root_i,
                transl=trans_i,
                left_hand_pose=left_hand_pose_1,
                right_hand_pose=right_hand_pose_1,
                betas=betas_tensor_1,
            )
            joints_list.append(output.joints[:, :22, :].cpu().numpy())  # [1, 22, 3]

    joints = np.concatenate(joints_list, axis=0)  # [T, 22, 3]

    stick = converter.smpl_to_stick_figure(joints)  # [T, 5, 4]
    motion = torch.from_numpy(stick.reshape(T, 20).astype(np.float32))
    return motion


def _load_motion_pair_from_pkl(
    pkl_path: str, converter: AMASSConverter
) -> torch.Tensor:
    """Load an InterHuman motion .pkl and pack two actors into [T, 2, 20]."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    person1 = obj["person1"]
    person2 = obj["person2"]

    m1 = _person_dict_to_motion(person1, converter)
    m2 = _person_dict_to_motion(person2, converter)

    # Ensure equal length (clip to shortest if needed)
    T = min(m1.shape[0], m2.shape[0])
    if m1.shape[0] != T:
        m1 = m1[:T]
    if m2.shape[0] != T:
        m2 = m2[:T]

    stacked = torch.stack([m1, m2], dim=1)  # [T, 2, 20]
    return stacked


def _load_texts(annots_dir: str, clip_id: str) -> list[str]:
    """Load textual annotations for a given clip.

    The InterHuman dataset typically stores one text file per clip in
    `annots/CLIP_ID.txt`, each line being a separate natural language
    description.
    """
    path = os.path.join(annots_dir, f"{clip_id}.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def _build_sample(
    motion: torch.Tensor, texts: list[str], meta: dict[str, Any], fps: int = 30
) -> dict[str, Any]:
    # motion: [T, 2, 20]
    physics = compute_basic_physics(motion, fps=fps)  # [T, 2, 6]

    # For now treat all InterHuman clips as generic interactions.
    action_enum = ActionType.FIGHT
    action_idx = ACTION_TO_IDX[action_enum]
    T = motion.shape[0]
    actions = torch.full((T, 2), action_idx, dtype=torch.long)

    description = (
        texts[0] if texts else "Two people interacting in the InterHuman dataset."
    )

    return {
        "description": description,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "camera": None,
        "source": "interhuman",
        "meta": meta,
    }


def convert_interhuman(
    data_root: str, output_path: str, fps: int = 30, max_clips: int = -1
) -> None:
    """Convert InterHuman motions into canonical schema.

    Supports the official InterHuman release layout under ``data_root``:

      - motions/*.pkl              (per-clip SMPL-style parameters)
      - annots/*.txt               (per-clip text descriptions)
      - split/train.txt, val.txt, test.txt (optional)
    """

    motions_dir = os.path.join(data_root, "motions")
    annots_dir = os.path.join(data_root, "annots")

    motion_files = sorted(glob.glob(os.path.join(motions_dir, "*.pkl")))
    if max_clips > 0:
        motion_files = motion_files[:max_clips]

    validator = DataValidator(fps=fps)
    converter = AMASSConverter()  # Reuse SMPL-X + stick-figure utilities
    samples: list[dict[str, Any]] = []

    for motion_path in motion_files:
        clip_id = os.path.splitext(os.path.basename(motion_path))[0]
        # Load metadata once to retrieve mocap framerate for physics
        with open(motion_path, "rb") as f:
            obj = pickle.load(f)
        mocap_fps = obj.get("mocap_framerate", fps)
        try:
            fps_clip = int(round(float(mocap_fps)))
        except Exception:
            fps_clip = fps

        motion = _load_motion_pair_from_pkl(motion_path, converter)  # [T, 2, 20]
        texts = _load_texts(annots_dir, clip_id)
        meta = {
            "clip_id": clip_id,
            "motion_path": os.path.relpath(motion_path, data_root),
            "mocap_framerate": mocap_fps,
            "frames": obj.get("frames"),
        }
        sample = _build_sample(motion, texts, meta, fps=fps_clip)
        # Multi-actor motions already satisfy a structural layout; here we only
        # enforce basic physics bounds to avoid discarding valid interactions
        # due to 2D projection artifacts.
        ok, score, reason = validator.check_physics_consistency(sample["physics"])
        if not ok:
            continue
        samples.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert InterHuman to canonical schema"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of InterHuman Dataset",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file path"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-clips", type=int, default=-1)
    args = parser.parse_args()

    convert_interhuman(
        args.data_root, args.output, fps=args.fps, max_clips=args.max_clips
    )


if __name__ == "__main__":
    main()
