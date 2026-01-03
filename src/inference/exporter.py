import datetime
import json

import numpy as np
import torch

from src.data_gen.joint_utils import V3_SEGMENT_NAMES


class MotionExporter:
    """Export stick-figure motion to a standardized ``.motion`` JSON schema.

    This exporter supports **both** the legacy v1 renderer schema and the
    canonical v3 training schema:

    * **v1 (legacy)**: 5 segments, 20D (``[T, 20]``) used by older web
      renderers. The skeleton metadata uses ``type="stick_figure_5_segment"``.
    * **v3 (canonical)**: 12 segments, 48D (``[T, 48]``) matching the
      connectivity-validated v3 schema used throughout the data pipeline. The
      skeleton metadata uses ``type="stick_figure_12_segment_v3"`` and the
      segment ordering from :data:`V3_SEGMENT_NAMES`.
    """

    def __init__(self, fps: int = 25) -> None:
        """Initialize the exporter.

        Args:
            fps: Frames-per-second metadata to embed in exported files.
        """

        self.fps = fps
        # Canonical 5-segment topology corresponding to the 20D legacy output:
        # [x1, y1, x2, y2] * 5 segments
        #   1. Torso (Neck -> Hip)
        #   2. Left Leg (Hip -> L Foot)
        #   3. Right Leg (Hip -> R Foot)
        #   4. Left Arm (Neck -> L Hand)
        #   5. Right Arm (Neck -> R Hand)
        self.segment_names = ["torso", "l_leg", "r_leg", "l_arm", "r_arm"]

    def export_to_json(
        self,
        motion_tensor: torch.Tensor,
        action_names: list | None = None,
        description: str = "",
        physics_data: torch.Tensor | None = None,
        camera_data: torch.Tensor | None = None,
    ) -> str:
        """Convert a motion tensor to a ``.motion`` JSON string.

        Args:
            motion_tensor: Motion array with shape ``[T, D]`` where ``D`` is
                either 20 (legacy v1 renderer schema) or 48 (canonical v3
                schema).
            action_names: Optional list of per-frame action label strings.
            description: Optional description or prompt to embed in metadata.
            physics_data: Optional physics tensor to be flattened and stored
                under the ``"physics"`` key.
            camera_data: Optional camera tensor ``[T, 3]`` (x, y, zoom) to be
                flattened and stored under the ``"camera"`` key.

        Raises:
            ValueError: If ``motion_tensor`` does not have shape ``[T, 20]`` or
                ``[T, 48]``.
        """

        if isinstance(motion_tensor, torch.Tensor):
            motion_np = motion_tensor.detach().cpu().numpy()
        else:
            motion_np = motion_tensor

        if motion_np.ndim != 2:
            raise ValueError(
                f"MotionExporter expects [T, D] motion, got shape {motion_np.shape}"
            )

        seq_len, dim = motion_np.shape
        if dim not in (20, 48):
            raise ValueError(
                f"MotionExporter only supports D in {20, 48}, got D={dim}"
            )

        duration = seq_len / self.fps

        # Round decimals for smaller file size
        motion_flat = np.round(motion_np.flatten(), 4).tolist()

        # Skeleton metadata depends on representation dimensionality.
        if dim == 20:
            skeleton_type = "stick_figure_5_segment"
            segments = self.segment_names
        else:  # dim == 48, canonical v3 schema
            skeleton_type = "stick_figure_12_segment_v3"
            segments = list(V3_SEGMENT_NAMES)

        data = {
            "meta": {
                "version": "1.0",
                "generator": "Stick-Gen v2",
                "created_at": datetime.datetime.now().isoformat(),
                "fps": self.fps,
                "duration": duration,
                "total_frames": seq_len,
                "description": description,
            },
            "skeleton": {
                "type": skeleton_type,
                "joint_format": "xy_lines_flat",
                "input_dim": dim,
                "segments": segments,
            },
            "motion": motion_flat,
            "actions": action_names if action_names else [],
        }

        if physics_data is not None:
            if isinstance(physics_data, torch.Tensor):
                phys_np = physics_data.detach().cpu().numpy()
            else:
                phys_np = physics_data
            data["physics"] = np.round(phys_np.flatten(), 4).tolist()

        if camera_data is not None:
            if isinstance(camera_data, torch.Tensor):
                cam_np = camera_data.detach().cpu().numpy()
            else:
                cam_np = camera_data
            data["camera"] = np.round(cam_np.flatten(), 4).tolist()

        return json.dumps(data, indent=None)  # Minified for network transfer

    def save(self, data: str, filepath: str):
        with open(filepath, "w") as f:
            f.write(data)
