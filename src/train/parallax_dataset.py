import json
import os
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # Optional dependency for PIL-based image loading
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - handled lazily
    Image = None  # type: ignore


class MultimodalParallaxDataset(Dataset):
    """Multimodal dataset for 2.5D parallax PNG frames + motion metadata.

    Each item corresponds to a single rendered PNG frame and returns:

        (image_tensor, motion_frame_data, camera_pose, text_prompt, action_label, environment_type)

    - ``image_tensor``: [3, H, W] float32 in [0, 1]
    - ``motion_frame_data``: [20] motion vector for the selected actor/frame
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

        text_prompt = sample.get("description", "")
        environment_type = sample.get("environment_type", "earth_normal")

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
