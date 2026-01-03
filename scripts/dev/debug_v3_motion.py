import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Add project root to path so local imports work when running as a script.
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data_gen.joint_utils import (  # noqa: E402
    V3_SEGMENT_NAMES,
    validate_v3_connectivity,
    v3_segments_to_joints_2d,
)
from src.data_gen.validator import DataValidator  # noqa: E402


def _load_v3_motion_generic(path: Path) -> np.ndarray:
    """Load a single-actor v3 motion clip into a ``[T, 48]`` array.

    This helper understands three common formats:

    * ``.motion`` JSON files produced by :class:`MotionExporter`.
    * ``.npy`` files containing a NumPy array of shape ``[T, 48]`` or
      ``[T, A, 48]`` (first actor is selected).
    * ``.pt`` files containing a PyTorch tensor or a dict/list with a
      ``"motion"`` tensor field.
    """

    suffix = path.suffix.lower()
    if suffix == ".motion" or suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        skel = data.get("skeleton", {}) or {}
        input_dim = int(skel.get("input_dim", 0))
        if input_dim != 48:
            raise ValueError(
                f"Expected v3 motion with input_dim=48, found input_dim={input_dim}"
            )

        flat = np.asarray(data.get("motion", []), dtype=np.float32)
        if flat.ndim != 1 or flat.size == 0:
            raise ValueError(
                "Expected 1D flattened motion array in .motion file, "
                f"got shape {flat.shape}"
            )
        if flat.size % input_dim != 0:
            raise ValueError(
                f"Flattened motion length {flat.size} is not divisible by "
                f"input_dim={input_dim}"
            )
        frames = flat.size // input_dim
        motion = flat.reshape(frames, input_dim)
        return motion

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] == 48:
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[2] == 48:
            # [T, A, 48] -> first actor
            return arr[:, 0, :].astype(np.float32, copy=False)
        raise ValueError(f"Unsupported .npy motion shape {arr.shape}, expected [T, 48] or [T, A, 48]")

    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        tensor: torch.Tensor | None = None
        if isinstance(obj, torch.Tensor):
            tensor = obj
        elif isinstance(obj, dict) and "motion" in obj:
            maybe_motion = obj["motion"]
            if isinstance(maybe_motion, torch.Tensor):
                tensor = maybe_motion
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            first = obj[0]
            maybe_motion = first.get("motion")
            if isinstance(maybe_motion, torch.Tensor):
                tensor = maybe_motion

        if tensor is None:
            raise ValueError(".pt/.pth file does not contain a recognisable 'motion' tensor")

        if tensor.dim() == 2 and tensor.shape[1] == 48:
            arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            return arr
        if tensor.dim() == 3 and tensor.shape[2] == 48:
            # [T, A, 48] -> first actor
            arr = tensor[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            return arr

        raise ValueError(
            f"Unsupported motion tensor shape {tuple(tensor.shape)}, expected [T, 48] or [T, A, 48]"
        )

    raise ValueError(f"Unsupported motion file extension '{suffix}' for {path}")


def _compute_segment_lengths(motion: np.ndarray) -> np.ndarray:
    """Compute per-segment lengths for a v3 motion clip.

    Args:
        motion: Array of shape ``[T, 48]``.

    Returns:
        Array of shape ``[T, 12]`` containing segment lengths per frame.
    """

    if motion.ndim != 2 or motion.shape[1] != 48:
        raise ValueError(f"Expected motion with shape [T, 48], got {motion.shape}")

    segs = motion.reshape(motion.shape[0], 12, 4)
    diffs = segs[:, :, 2:4] - segs[:, :, 0:2]
    lengths = np.linalg.norm(diffs, axis=-1)
    return lengths


def _print_summary(motion: np.ndarray) -> None:
    """Print a brief textual summary of a v3 motion clip.

    This includes basic shape information, per-segment length statistics, and
    outputs from key :class:`DataValidator` checks.
    """

    frames, dim = motion.shape
    print(f"Loaded motion: shape={motion.shape}, frames={frames}, dim={dim}")

    # Validate connectivity and basic structure before running physics checks.
    validate_v3_connectivity(motion)
    print("- v3 connectivity: OK")

    lengths = _compute_segment_lengths(motion)
    mean_lengths = lengths.mean(axis=0)
    min_lengths = lengths.min(axis=0)
    max_lengths = lengths.max(axis=0)

    print("- Segment length statistics:")
    for idx, name in enumerate(V3_SEGMENT_NAMES):
        print(
            f"  [{idx:02d}] {name:12s} | mean={mean_lengths[idx]:.4f} "
            f"min={min_lengths[idx]:.4f} max={max_lengths[idx]:.4f}"
        )

    validator = DataValidator(fps=25, environment_type="earth_normal")
    motion_t = torch.from_numpy(motion)

    skel_ok, skel_score, skel_reason = validator.check_skeleton_consistency(motion_t)
    print(f"- Skeleton consistency: ok={skel_ok} score={skel_score:.3f} reason='{skel_reason}'")

    angles_ok, angles_score, angles_reason = validator.check_joint_angles_v3(motion_t)
    print(f"- Joint angles:        ok={angles_ok} score={angles_score:.3f} reason='{angles_reason}'")


def _interactive_view(motion: np.ndarray) -> None:
    """Launch a simple interactive Matplotlib viewer for a v3 motion clip.

    Use the left/right arrow keys or ``n``/``p`` to step through frames. This
    viewer overlays segment indices and lengths to make it easier to debug
    connectivity and local distortions.
    """

    import matplotlib.pyplot as plt  # Imported lazily to avoid GUI overhead in tests

    if motion.ndim != 2 or motion.shape[1] != 48:
        raise ValueError(f"Expected motion with shape [T, 48], got {motion.shape}")

    frames = motion.shape[0]
    segs = motion.reshape(frames, 12, 4)
    lengths = _compute_segment_lengths(motion)

    joints = v3_segments_to_joints_2d(motion, validate=True)
    all_xy = np.concatenate(list(joints.values()), axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.tight_layout()

    state = {"frame": 0}

    def _draw(idx: int) -> None:
        ax.clear()
        ax.set_title(f"v3 debug viewer – frame {idx+1}/{frames}")
        ax.set_aspect("equal")
        pad = 0.1 * float(np.max(max_xy - min_xy)) if np.all(np.isfinite(max_xy - min_xy)) else 0.1
        ax.set_xlim(min_xy[0] - pad, max_xy[0] + pad)
        ax.set_ylim(min_xy[1] - pad, max_xy[1] + pad)
        ax.invert_yaxis()

        seg = segs[idx]
        seg_len = lengths[idx]
        for s in range(12):
            x1, y1, x2, y2 = seg[s]
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=2)
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            ax.text(
                mx,
                my,
                f"{s}:{seg_len[s]:.3f}",
                fontsize=8,
                ha="center",
                va="center",
                color="blue",
            )

        # Draw joints as small circles on top.
        for name, arr in joints.items():
            xy = arr[idx]
            ax.plot(xy[0], xy[1], "ro", markersize=3)

        fig.canvas.draw_idle()

    def _on_key(event: Any) -> None:
        if event.key in {"right", "n"}:
            state["frame"] = (state["frame"] + 1) % frames
            _draw(state["frame"])
        elif event.key in {"left", "p"}:
            state["frame"] = (state["frame"] - 1) % frames
            _draw(state["frame"])

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _draw(state["frame"])
    print("Interactive viewer controls: 'n'/'right' = next frame, 'p'/'left' = prev frame")
    plt.show()


def main(argv: list[str] | None = None) -> None:
    """Entry point for the v3 motion debugging CLI.

    This helper is intended for manual inspection and debugging of v3
    12-segment motion clips. Typical usage::

        python scripts/dev/debug_v3_motion.py --input path/to/clip.motion

    Use ``--no-gui`` to restrict the script to textual diagnostics, which is
    safer to run in headless environments or automated jobs.
    """

    parser = argparse.ArgumentParser(description="Interactive debugger for v3 [T,48] motion")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a v3 motion file (.motion, .npy, .pt)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Only print diagnostics; do not open an interactive viewer",
    )

    args = parser.parse_args(argv)
    motion_path = Path(args.input)
    if not motion_path.exists():
        raise SystemExit(f"Input motion file does not exist: {motion_path}")

    motion = _load_v3_motion_generic(motion_path)
    _print_summary(motion)

    if args.no_gui:
        return

    _interactive_view(motion)


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    main()
