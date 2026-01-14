
import os
import shutil
import subprocess

import numpy as np
import torch

from src.data_gen.schema import IDX_TO_ACTION
from src.inference.exporter import MotionExporter


def _has_node_runtime() -> bool:
    """Return True if a Node.js runtime is available on PATH."""

    return shutil.which("node") is not None


def _export_actor_motion_to_file(
    sample: dict,
    actor_idx: int,
    out_path: str,
    fps: int = 25,
) -> None:
    """Export a single actor's motion sequence to a ``.motion`` JSON file.

    This helper targets the **renderer/export schema** and is compatible with
    both the legacy v1 representation (5 segments, ``[T, 20]``) and the
    canonical v3 representation (12 segments, ``[T, 48]``). The motion can be
    provided as either:

    * ``[T, D]`` for a single actor (``D`` in ``{20, 48}``), or
    * ``[T, A, D]`` for multiple actors, where this function selects the
      slice for ``actor_idx``.
    """

    motion = sample.get("motion")
    if motion is None:
        return

    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if motion.dim() == 2:
        # [T, D] single-actor, D in {20, 48}
        motion_actor = motion
    elif motion.dim() == 3:
        # [T, actors, D] multi-actor
        if actor_idx >= motion.shape[1]:
            return
        motion_actor = motion[:, actor_idx, :]
    else:
        # Unexpected shape; skip this sample for parallax export.
        return

    action_tensor = sample.get("actions")
    action_names = None
    import torch
    if action_tensor is not None:
        # Ensure it's a tensor to access .dim()
        if not torch.is_tensor(action_tensor):
            action_tensor = torch.tensor(action_tensor)
        
        if action_tensor.dim() == 2:
            ids = action_tensor[:, actor_idx].tolist()
            action_names = []
            for i in ids:
                idx = int(i)
                action = IDX_TO_ACTION.get(idx)
                action_names.append(action.value if action is not None else "unknown")

    description = sample.get("description", f"sample_actor_{actor_idx}")

    exporter = MotionExporter(fps=fps)
    json_str = exporter.export_to_json(
        motion_tensor=motion_actor,
        action_names=action_names,
        description=description,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def generate_parallax_for_dataset(
    dataset_path: str,
    output_dir: str,
    views_per_motion: int = 1000,
    node_script: str = "src/data_gen/renderers/threejs_parallax_renderer.js",
    max_samples: int | None = None,
    fps: int = 25,
    frames_per_view: int = 1,
    minimal: bool = True,
) -> None:
    """Generate 2.5D parallax PNG frames for a canonical ``.pt`` dataset.

    Workflow:
      1. Load the saved list[dict] dataset from ``dataset_path``.
      2. For each sample and actor, export a temporary ``.motion`` file.
      3. Invoke the Node.js Three.js renderer to create PNG frames.

    Parameters
    ----------
    dataset_path:
        Path to a ``torch.save``-d list of samples from data_gen.
    output_dir:
        Root directory where PNG frames (and temporary .motion files) are written.
    views_per_motion:
        Number of parallax views (frames) to render per actor motion.
    node_script:
        Path to ``threejs_parallax_renderer.js``.
    max_samples:
        Optional cap on number of samples to process (for debugging).
    fps:
        Frame rate metadata to embed in exported ``.motion`` files.
    frames_per_view:
        Number of rendered PNG frames per camera trajectory ("view").
        When >1, filenames are suffixed with the intra-view frame index and
        the Node renderer emits a metadata JSON sidecar per sample/actor.
    minimal:
        If True (default), pass --minimal to renderer for clean backgrounds
        without distracting scene elements. Recommended for training data.
    """

    if views_per_motion <= 0:
        print("[parallax] views_per_motion <= 0, skipping Three.js augmentation.")
        return

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not os.path.exists(node_script):
        print(f"[parallax] Node renderer not found at {node_script}, skipping.")
        return

    if not _has_node_runtime():
        print(
            "[parallax] Node.js runtime not found in PATH, skipping Three.js augmentation."
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    tmp_motion_root = os.path.join(output_dir, "_motion_tmp")
    os.makedirs(tmp_motion_root, exist_ok=True)

    print(f"[parallax] Loading dataset from {dataset_path} ...")
    samples = torch.load(dataset_path)
    print(
        f"[parallax] Loaded {len(samples)} samples. Generating parallax frames into {output_dir}"
    )

    for sample_idx, sample in enumerate(samples):
        if max_samples is not None and sample_idx >= max_samples:
            break

        motion = sample.get("motion")
        if motion is None:
            continue

        # Normalize to tensor
        if isinstance(motion, list):
            motion = torch.tensor(motion, dtype=torch.float32)
            sample["motion"] = motion
        elif isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)
            sample["motion"] = motion

        # Single-actor [T, D] → [T, 1, D]
        if motion.dim() == 2:
            motion = motion.unsqueeze(1)
            sample["motion"] = motion

        if motion.dim() != 3:
            continue

        # Skip samples with all-zero motion (corrupted/invalid data)
        if motion.abs().sum() < 1e-6:
            print(f"[parallax] Skipping sample {sample_idx}: all-zero motion")
            continue

        _, num_actors, _ = motion.shape
        for actor_idx in range(num_actors):
            # Skip actors whose motion is effectively all zeros (unused slots in
            # multi-actor sequences). Rendering these would produce collapsed or
            # invisible stick-figures, which looks like a rendering bug.
            motion_actor = motion[:, actor_idx, :]
            if motion_actor.abs().sum() < 1e-6:
                print(
                    f"[parallax] Skipping sample {sample_idx}, actor {actor_idx}: zero-motion actor",
                )
                continue

            motion_path = os.path.join(
                tmp_motion_root,
                f"sample{sample_idx:06d}_actor{actor_idx}.motion",
            )
            # Pass the modified sample
            _export_actor_motion_to_file(sample, actor_idx, motion_path, fps=fps)

            actor_output = os.path.join(
                output_dir,
                f"sample_{sample_idx:06d}",
                f"actor_{actor_idx}",
            )
            os.makedirs(actor_output, exist_ok=True)

            cmd = [
                "node",
                node_script,
                "--input",
                motion_path,
                "--output-dir",
                actor_output,
                "--views",
                str(views_per_motion),
                "--frames-per-view",
                str(frames_per_view),
                "--width",
                "256",
                "--height",
                "256",
                "--sample-id",
                f"sample_{sample_idx:06d}",
                "--actor-id",
                str(actor_idx),
            ]
            if minimal:
                cmd.append("--minimal")
            print(
                f"[parallax] Rendering {os.path.basename(motion_path)} -> {actor_output}"
            )
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[parallax] Node renderer failed for {motion_path}: {exc}")
                continue
