import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # Optional dependency for PIL-based image loading
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - handled lazily
    Image = None  # type: ignore

logger = logging.getLogger(__name__)


class MultimodalParallaxDataset(Dataset):
    """Multimodal dataset for 2.5D parallax PNG frames + motion metadata.

    Each item corresponds to a single rendered PNG frame and returns::

        (image_tensor, motion_frame_data, camera_pose, text_prompt,
         action_label, environment_type)

    - ``image_tensor``: [3, H, W] float32 in [0, 1]
    - ``motion_frame_data``: [D_motion] motion vector for the selected
      actor/frame, where canonical training uses ``D_motion = 48`` for the
      v3 12-segment schema (12 × 4 coords). Legacy export pipelines may still
      use ``D_motion = 20`` for the compact 5-segment v1 renderer format.
    - ``camera_pose``: [7] (pos_x, pos_y, pos_z, tgt_x, tgt_y, tgt_z, fov)
    - ``text_prompt``: str (sample description)
    - ``action_label``: LongTensor scalar or ``None``
    - ``environment_type``: str environment type (e.g., "earth_normal", "underwater")
    """

    def __init__(
        self,
        parallax_root: str,
        motion_data_path: str,
        image_size: tuple[int, int] = (256, 256),
        image_backend: str = "pil",
        image_transform: Any | None = None,
    ) -> None:
        super().__init__()
        self.parallax_root = parallax_root
        self.motion_data_path = motion_data_path
        self.image_size = image_size
        self.image_backend = image_backend.lower()
        self.image_transform = image_transform

        if self.image_backend not in {"pil", "torchvision"}:
            raise ValueError("image_backend must be 'pil' or 'torchvision'")

        if self.image_backend == "pil" and Image is None:
            raise RuntimeError(
                "PIL is required for image_backend='pil' but is not installed."
            )

        self._read_image = None
        if self.image_backend == "torchvision":
            try:
                from torchvision.io import read_image  # type: ignore

                self._read_image = read_image
            except Exception as exc:  # pragma: no cover - optional path
                raise ImportError(
                    "image_backend='torchvision' requires torchvision to be installed"
                ) from exc

        # Load motion / text / labels dataset (list[dict])
        self.samples: list[dict[str, Any]] = torch.load(self.motion_data_path)
        if not isinstance(self.samples, list):
            raise ValueError(
                f"Expected a list of samples in {self.motion_data_path}, "
                f"got {type(self.samples)}"
            )

        # Build flat index over all PNG frames across samples/actors
        self.index: list[dict[str, Any]] = []
        self._build_index()

    def _build_index(self) -> None:
        for root, _dirs, files in os.walk(self.parallax_root):
            if "metadata.json" not in files:
                continue
            meta_path = os.path.join(root, "metadata.json")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            sample_id = meta.get("sample_id") or os.path.basename(os.path.dirname(root))
            actor_id = meta.get("actor_id") or os.path.basename(root)
            sample_idx = self._parse_trailing_int(str(sample_id))
            actor_idx = self._parse_trailing_int(str(actor_id))

            for frame in meta.get("frames", []):
                image_path = os.path.join(root, frame["file"])
                motion_frame_index = int(frame.get("motion_frame_index", 0))
                view_id = int(frame.get("view_id", frame.get("view_index", 0)))
                camera = frame.get("camera") or {}

                self.index.append(
                    {
                        "image_path": image_path,
                        "sample_idx": sample_idx,
                        "actor_idx": actor_idx,
                        "motion_frame_index": motion_frame_index,
                        "view_id": view_id,
                        "camera": camera,
                    }
                )

        if not self.index:
            raise ValueError(
                f"No metadata.json files found under parallax_root={self.parallax_root}"
            )

    @staticmethod
    def _parse_trailing_int(value: str) -> int:
        match = re.search(r"(\d+)$", value)
        return int(match.group(1)) if match else 0

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.index)

    def __getitem__(self, idx: int):  # type: ignore[override]
        meta = self.index[idx]
        sample = self.samples[meta["sample_idx"]]
        motion = sample["motion"]
        frame_idx = int(meta["motion_frame_index"])
        actor_idx = int(meta["actor_idx"])

        # Normalize motion to numpy array if stored as nested lists
        if isinstance(motion, list):
            motion = np.array(motion, dtype=np.float32)
        elif isinstance(motion, torch.Tensor):
            motion = motion.numpy()

        # Select motion frame for this actor
        if motion.ndim == 2:
            t = motion.shape[0]
            frame_idx = max(0, min(frame_idx, t - 1))
            motion_frame = motion[frame_idx]
        elif motion.ndim == 3:
            t, a, _d = motion.shape
            frame_idx = max(0, min(frame_idx, t - 1))
            actor_idx = max(0, min(actor_idx, a - 1))
            motion_frame = motion[frame_idx, actor_idx]
        else:
            raise ValueError(f"Unexpected motion shape: {tuple(motion.shape)}")

        # Action label granularity: per-frame or clip-level
        actions = sample.get("actions")
        action_label = self._select_action_label(actions, frame_idx, actor_idx)
        # Ensure action_label is never None for collation compatibility
        if action_label is None:
            action_label = torch.tensor(0, dtype=torch.long)  # Default to "idle" action

        # Camera pose tensor [pos_x, pos_y, pos_z, tgt_x, tgt_y, tgt_z, fov]
        camera = meta["camera"] or {}
        pos = camera.get("position", {})
        tgt = camera.get("target", {})
        fov = camera.get("fov", 0.0)
        camera_pose = torch.tensor(
            [
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
                float(tgt.get("x", 0.0)),
                float(tgt.get("y", 0.0)),
                float(tgt.get("z", 0.0)),
                float(fov),
            ],
            dtype=torch.float32,
        )

        image_tensor = self._load_image(meta["image_path"])

        text_prompt = sample.get("description") or ""
        environment_type = sample.get("environment_type") or "earth_normal"

        return image_tensor, motion_frame, camera_pose, text_prompt, action_label, environment_type

    def _select_action_label(
        self, actions: Any, frame_idx: int, actor_idx: int
    ) -> torch.Tensor | None:
        """Handle both per-frame and clip-level action labels.

        - [T] or [T, A]: per-frame indexing
        - scalar / 0-D tensor / single-element tensor: clip-level label
        """

        if actions is None:
            return None

        if isinstance(actions, torch.Tensor):
            if actions.ndim == 1:  # [T]
                t = actions.shape[0]
                idx = max(0, min(frame_idx, t - 1))
                return actions[idx].long()
            if actions.ndim == 2:  # [T, A]
                t, a = actions.shape
                fi = max(0, min(frame_idx, t - 1))
                ai = max(0, min(actor_idx, a - 1))
                return actions[fi, ai].long()
            if actions.ndim == 0:  # scalar tensor
                return actions.long()
            if actions.numel() == 1:  # single value per clip
                return actions.view(-1)[0].long()
            # Fallback: treat as flattened per-frame sequence
            flat = actions.view(-1)
            idx = max(0, min(frame_idx, flat.shape[0] - 1))
            return flat[idx].long()

        # Non-tensor: best-effort handling (clip-level or simple list)
        if isinstance(actions, (int, float)):
            return torch.tensor(int(actions), dtype=torch.long)
        if isinstance(actions, (list, tuple)) and actions:
            # Handle nested list: actions[frame_idx][actor_idx]
            if isinstance(actions[0], (list, tuple)):
                fi = max(0, min(frame_idx, len(actions) - 1))
                frame_actions = actions[fi]
                if isinstance(frame_actions, (list, tuple)) and frame_actions:
                    ai = max(0, min(actor_idx, len(frame_actions) - 1))
                    return torch.tensor(int(frame_actions[ai]), dtype=torch.long)
                elif isinstance(frame_actions, (int, float)):
                    return torch.tensor(int(frame_actions), dtype=torch.long)
                return None
            # Simple 1D list
            if len(actions) == 1:
                return torch.tensor(int(actions[0]), dtype=torch.long)
            idx = max(0, min(frame_idx, len(actions) - 1))
            return torch.tensor(int(actions[idx]), dtype=torch.long)

        return None

    def _load_image(self, path: str) -> torch.Tensor:
        h, w = self.image_size

        if self.image_backend == "torchvision":
            assert self._read_image is not None
            img = self._read_image(path).float() / 255.0
            if img.shape[1] != h or img.shape[2] != w:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        else:  # PIL backend
            assert Image is not None
            with Image.open(path) as im:  # type: ignore[attr-defined]
                im = im.convert("RGB").resize((w, h))
                arr = np.array(im, dtype="float32") / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)

        if self.image_transform is not None:
            img = self.image_transform(img)

        return img


class MultimodalParallaxSequenceDataset(Dataset):
    """Multimodal dataset returning temporally contiguous frame sequences.

    Unlike ``MultimodalParallaxDataset`` which returns single frames, this
    dataset groups consecutive frames by (sample_id, actor_id, view_id) and
    returns sequences for temporal learning.

    Each item returns::

        (image_sequence, motion_sequence, camera_trajectory, text_prompt,
         action_sequence, environment_type)

    - ``image_sequence``: [seq_len, 3, H, W] float32 in [0, 1]
    - ``motion_sequence``: [seq_len, D_motion] motion vectors
    - ``camera_trajectory``: [seq_len, 7] camera poses over time
    - ``text_prompt``: str (sample description)
    - ``action_sequence``: [seq_len] LongTensor or scalar
    - ``environment_type``: str environment type

    This enables temporal consistency loss and smoothness metrics during training.
    """

    def __init__(
        self,
        parallax_root: str,
        motion_data_path: str,
        sequence_length: int = 4,
        stride: int = 1,
        image_size: tuple[int, int] = (256, 256),
        image_backend: str = "pil",
        image_transform: Any | None = None,
        conditioning_mode: str = "first_frame",
        min_sequence_length: int | None = None,
    ) -> None:
        """Initialize the sequence dataset.

        Args:
            parallax_root: Root directory containing parallax PNG frames
            motion_data_path: Path to the .pt file with motion/text/labels
            sequence_length: Number of frames per sequence (default: 4)
            stride: Step size for sliding window over sequences (default: 1)
            image_size: (H, W) to resize images to
            image_backend: "pil" or "torchvision"
            image_transform: Optional transform to apply to images
            conditioning_mode: How to use images for conditioning:
                - "first_frame": Return only first image for conditioning
                - "all_frames": Return all images in sequence
                - "random_frame": Return random frame for conditioning
            min_sequence_length: Minimum frames to form a sequence (defaults to sequence_length)
        """
        super().__init__()
        self.parallax_root = parallax_root
        self.motion_data_path = motion_data_path
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        self.image_backend = image_backend.lower()
        self.image_transform = image_transform
        self.conditioning_mode = conditioning_mode
        self.min_sequence_length = min_sequence_length or sequence_length

        if self.image_backend not in {"pil", "torchvision"}:
            raise ValueError("image_backend must be 'pil' or 'torchvision'")

        if self.image_backend == "pil" and Image is None:
            raise RuntimeError(
                "PIL is required for image_backend='pil' but is not installed."
            )

        self._read_image = None
        if self.image_backend == "torchvision":
            try:
                from torchvision.io import read_image  # type: ignore

                self._read_image = read_image
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "image_backend='torchvision' requires torchvision to be installed"
                ) from exc

        # Load motion / text / labels dataset (list[dict])
        self.samples: list[dict[str, Any]] = torch.load(self.motion_data_path)
        if not isinstance(self.samples, list):
            raise ValueError(
                f"Expected a list of samples in {self.motion_data_path}, "
                f"got {type(self.samples)}"
            )

        # Build flat index first, then group into sequences
        self._flat_index: list[dict[str, Any]] = []
        self.sequence_index: list[dict[str, Any]] = []
        self._build_index()

        logger.info(
            f"MultimodalParallaxSequenceDataset: {len(self.sequence_index)} sequences "
            f"(seq_len={sequence_length}, stride={stride}) from {len(self._flat_index)} frames"
        )

    def _build_index(self) -> None:
        """Build flat index, then group by view for sequence extraction."""
        # Step 1: Build flat index (same as MultimodalParallaxDataset)
        for root, _dirs, files in os.walk(self.parallax_root):
            if "metadata.json" not in files:
                continue
            meta_path = os.path.join(root, "metadata.json")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            sample_id = meta.get("sample_id") or os.path.basename(os.path.dirname(root))
            actor_id = meta.get("actor_id") or os.path.basename(root)
            sample_idx = self._parse_trailing_int(str(sample_id))
            actor_idx = self._parse_trailing_int(str(actor_id))

            for frame in meta.get("frames", []):
                image_path = os.path.join(root, frame["file"])
                motion_frame_index = int(frame.get("motion_frame_index", 0))
                view_id = int(frame.get("view_id", frame.get("view_index", 0)))
                step_index = int(frame.get("step_index", 0))
                camera = frame.get("camera") or {}

                self._flat_index.append(
                    {
                        "image_path": image_path,
                        "sample_idx": sample_idx,
                        "actor_idx": actor_idx,
                        "motion_frame_index": motion_frame_index,
                        "view_id": view_id,
                        "step_index": step_index,
                        "camera": camera,
                    }
                )

        if not self._flat_index:
            raise ValueError(
                f"No metadata.json files found under parallax_root={self.parallax_root}"
            )

        # Step 2: Group frames by (sample_idx, actor_idx, view_id)
        view_groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
        for entry in self._flat_index:
            key = (entry["sample_idx"], entry["actor_idx"], entry["view_id"])
            view_groups[key].append(entry)

        # Step 3: Create sequence indices using sliding window
        for (sample_idx, actor_idx, view_id), frames in view_groups.items():
            # Sort by step_index to ensure temporal order
            frames = sorted(frames, key=lambda x: x["step_index"])

            if len(frames) < self.min_sequence_length:
                continue  # Skip views with insufficient frames

            # Sliding window with stride
            for start in range(0, len(frames) - self.sequence_length + 1, self.stride):
                seq_frames = frames[start : start + self.sequence_length]
                self.sequence_index.append(
                    {
                        "frames": seq_frames,
                        "sample_idx": sample_idx,
                        "actor_idx": actor_idx,
                        "view_id": view_id,
                        "start_step": start,
                    }
                )

    @staticmethod
    def _parse_trailing_int(value: str) -> int:
        match = re.search(r"(\d+)$", value)
        return int(match.group(1)) if match else 0

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int):
        seq_meta = self.sequence_index[idx]
        frames = seq_meta["frames"]
        sample_idx = seq_meta["sample_idx"]
        actor_idx = seq_meta["actor_idx"]

        # Get the motion sample
        sample = self.samples[sample_idx]
        motion = sample["motion"]

        # Normalize motion to numpy array
        if isinstance(motion, list):
            motion = np.array(motion, dtype=np.float32)
        elif isinstance(motion, torch.Tensor):
            motion = motion.numpy()

        # Extract motion sequence for all frames
        motion_frames = []
        camera_poses = []
        action_labels = []

        for frame_meta in frames:
            frame_idx = int(frame_meta["motion_frame_index"])

            # Get motion frame
            if motion.ndim == 2:
                t = motion.shape[0]
                fi = max(0, min(frame_idx, t - 1))
                motion_frame = motion[fi]
            elif motion.ndim == 3:
                t, a, _d = motion.shape
                fi = max(0, min(frame_idx, t - 1))
                ai = max(0, min(actor_idx, a - 1))
                motion_frame = motion[fi, ai]
            else:
                raise ValueError(f"Unexpected motion shape: {tuple(motion.shape)}")

            motion_frames.append(torch.from_numpy(motion_frame.copy()))

            # Get camera pose
            camera = frame_meta["camera"] or {}
            pos = camera.get("position", {})
            tgt = camera.get("target", {})
            fov = camera.get("fov", 0.0)
            camera_pose = torch.tensor(
                [
                    float(pos.get("x", 0.0)),
                    float(pos.get("y", 0.0)),
                    float(pos.get("z", 0.0)),
                    float(tgt.get("x", 0.0)),
                    float(tgt.get("y", 0.0)),
                    float(tgt.get("z", 0.0)),
                    float(fov),
                ],
                dtype=torch.float32,
            )
            camera_poses.append(camera_pose)

            # Get action label for this frame
            actions = sample.get("actions")
            action_label = self._select_action_label(actions, frame_idx, actor_idx)
            if action_label is None:
                action_label = torch.tensor(0, dtype=torch.long)
            action_labels.append(action_label)

        # Stack into sequences
        motion_sequence = torch.stack(motion_frames)  # [seq_len, D]
        camera_trajectory = torch.stack(camera_poses)  # [seq_len, 7]
        action_sequence = torch.stack(action_labels)  # [seq_len]

        # Load images based on conditioning mode
        if self.conditioning_mode == "first_frame":
            image_sequence = self._load_image(frames[0]["image_path"]).unsqueeze(0)
        elif self.conditioning_mode == "random_frame":
            rand_idx = torch.randint(0, len(frames), (1,)).item()
            image_sequence = self._load_image(frames[rand_idx]["image_path"]).unsqueeze(0)
        else:  # all_frames
            images = [self._load_image(f["image_path"]) for f in frames]
            image_sequence = torch.stack(images)  # [seq_len, 3, H, W]

        text_prompt = sample.get("description") or ""
        environment_type = sample.get("environment_type") or "earth_normal"

        return (
            image_sequence,
            motion_sequence,
            camera_trajectory,
            text_prompt,
            action_sequence,
            environment_type,
        )

    def _select_action_label(
        self, actions: Any, frame_idx: int, actor_idx: int
    ) -> torch.Tensor | None:
        """Handle both per-frame and clip-level action labels."""
        if actions is None:
            return None

        if isinstance(actions, torch.Tensor):
            if actions.ndim == 1:  # [T]
                t = actions.shape[0]
                idx = max(0, min(frame_idx, t - 1))
                return actions[idx].long()
            if actions.ndim == 2:  # [T, A]
                t, a = actions.shape
                fi = max(0, min(frame_idx, t - 1))
                ai = max(0, min(actor_idx, a - 1))
                return actions[fi, ai].long()
            if actions.ndim == 0:
                return actions.long()
            if actions.numel() == 1:
                return actions.view(-1)[0].long()
            flat = actions.view(-1)
            idx = max(0, min(frame_idx, flat.shape[0] - 1))
            return flat[idx].long()

        if isinstance(actions, (int, float)):
            return torch.tensor(int(actions), dtype=torch.long)
        if isinstance(actions, (list, tuple)) and actions:
            if isinstance(actions[0], (list, tuple)):
                fi = max(0, min(frame_idx, len(actions) - 1))
                frame_actions = actions[fi]
                if isinstance(frame_actions, (list, tuple)) and frame_actions:
                    ai = max(0, min(actor_idx, len(frame_actions) - 1))
                    return torch.tensor(int(frame_actions[ai]), dtype=torch.long)
                elif isinstance(frame_actions, (int, float)):
                    return torch.tensor(int(frame_actions), dtype=torch.long)
                return None
            if len(actions) == 1:
                return torch.tensor(int(actions[0]), dtype=torch.long)
            idx = max(0, min(frame_idx, len(actions) - 1))
            return torch.tensor(int(actions[idx]), dtype=torch.long)

        return None

    def _load_image(self, path: str) -> torch.Tensor:
        h, w = self.image_size

        if self.image_backend == "torchvision":
            assert self._read_image is not None
            img = self._read_image(path).float() / 255.0
            if img.shape[1] != h or img.shape[2] != w:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        else:  # PIL backend
            assert Image is not None
            with Image.open(path) as im:
                im = im.convert("RGB").resize((w, h))
                arr = np.array(im, dtype="float32") / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)

        if self.image_transform is not None:
            img = self.image_transform(img)

        return img
