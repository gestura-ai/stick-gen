import datetime
import json

import numpy as np
import torch


class MotionExporter:
    """
    Exports stick figure animations to a standardized JSON schema (.motion)
    optimized for web rendering (Three.js / React-Three-Fiber).
    """

    def __init__(self, fps: int = 25):
        self.fps = fps
        # Canonical 5-segment topology corresponding to the 20-dim output:
        # [x1,y1, x2,y2] * 5 segments
        # 1. Torso (Neck -> Hip)
        # 2. Left Leg (Hip -> L Foot)
        # 3. Right Leg (Hip -> R Foot)
        # 4. Left Arm (Neck -> L Hand)
        # 5. Right Arm (Neck -> R Hand)
        self.segment_names = ["torso", "l_leg", "r_leg", "l_arm", "r_arm"]

    def export_to_json(
        self,
        motion_tensor: torch.Tensor,
        action_names: list | None = None,
        description: str = "",
        physics_data: torch.Tensor | None = None,
        camera_data: torch.Tensor | None = None,
    ) -> str:
        """
        Convert motion tensor to JSON string.

        Args:
            motion_tensor: [seq_len, input_dim] (e.g., [250, 20])
            action_names: List of action labels per frame
            description: Text prompt used to generate this
            physics_data: Optional physics tensor
            camera_data: Optional camera tensor [seq_len, 3] (x, y, zoom)
        """
        if isinstance(motion_tensor, torch.Tensor):
            motion_np = motion_tensor.detach().cpu().numpy()
        else:
            motion_np = motion_tensor

        seq_len, dim = motion_np.shape
        duration = seq_len / self.fps

        # Round decimals for smaller file size
        motion_flat = np.round(motion_np.flatten(), 4).tolist()

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
                "type": "stick_figure_5_segment",
                "joint_format": "xy_lines_flat",
                "input_dim": dim,
                "segments": self.segment_names,
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
