"""
Phase 1: Action-Conditioned Generation - Interactive Generation API

This template shows how to use the action-conditioned model for interactive
generation with frame-by-frame action control.

INTEGRATION INSTRUCTIONS:
1. Add generate_with_actions() method to src/inference/generator.py
2. Create interactive demo script
3. Add CLI support for action sequences
"""

import torch
from typing import List, Optional
from src.data_gen.schema import ActionType, ACTION_TO_IDX

# ============================================================================
# STEP 1: Add to StickFigureGenerator class (src/inference/generator.py)
# ============================================================================


class StickFigureGenerator:
    """Generator with action-conditioned generation support"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.eval()

    def generate_with_actions(
        self,
        text_prompt: str,
        action_sequence: List[ActionType],
        output_path: str = "output.mp4",
        fps: int = 25,
    ) -> str:
        """
        Generate animation with explicit action control.

        Args:
            text_prompt: Text description of the scene
            action_sequence: List of actions, one per frame (length 250)
            output_path: Path to save output video
            fps: Frames per second (default: 25)

        Returns:
            Path to generated video

        Example:
            >>> generator = StickFigureGenerator('model.pth')
            >>>
            >>> # Create action sequence: WALK → RUN → JUMP
            >>> actions = (
            ...     [ActionType.WALK] * 100 +
            ...     [ActionType.RUN] * 100 +
            ...     [ActionType.JUMP] * 50
            ... )
            >>>
            >>> generator.generate_with_actions(
            ...     text_prompt="A person walking, then running, then jumping",
            ...     action_sequence=actions,
            ...     output_path="walk_run_jump.mp4"
            ... )
        """
        # Validate action sequence length
        expected_frames = 250  # 10s @ 25fps
        if len(action_sequence) != expected_frames:
            raise ValueError(
                f"Action sequence must have {expected_frames} frames, "
                f"got {len(action_sequence)}"
            )

        # Convert actions to indices
        action_indices = torch.tensor(
            [ACTION_TO_IDX[action] for action in action_sequence],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(
            0
        )  # [1, seq_len]

        # Get text embedding
        text_embedding = self.embed_text(text_prompt)  # [1, 1024]

        # Initialize motion sequence (can be zeros or previous frame)
        motion_sequence = torch.zeros(1, expected_frames, 20, device=self.device)

        # Generate with action conditioning
        with torch.no_grad():
            outputs = self.model(
                motion_sequence, text_embedding, action_sequence=action_indices
            )

        # Extract pose predictions
        predicted_poses = outputs["pose"].cpu().numpy()[0]  # [250, 20]

        # Render to video
        self.render_to_video(predicted_poses, output_path, fps=fps)

        return output_path

    def generate_interactive(
        self,
        text_prompt: str,
        initial_action: ActionType = ActionType.IDLE,
        output_path: str = "interactive.mp4",
    ) -> str:
        """
        Generate animation with interactive action transitions.

        This is a simplified version that allows changing actions at key frames.

        Example:
            >>> generator = StickFigureGenerator('model.pth')
            >>> generator.generate_interactive(
            ...     text_prompt="A person performing various actions",
            ...     initial_action=ActionType.WALK
            ... )
        """
        # Define action timeline (frame: action)
        action_timeline = {
            0: ActionType.IDLE,
            50: ActionType.WALK,
            125: ActionType.RUN,
            175: ActionType.JUMP,
            200: ActionType.WALK,
            225: ActionType.IDLE,
        }

        # Build action sequence
        action_sequence = []
        current_action = initial_action

        for frame in range(250):
            if frame in action_timeline:
                current_action = action_timeline[frame]
            action_sequence.append(current_action)

        return self.generate_with_actions(text_prompt, action_sequence, output_path)


# ============================================================================
# STEP 2: CLI interface for action-conditioned generation
# ============================================================================


def parse_action_sequence(action_string: str) -> List[ActionType]:
    """
    Parse action sequence from string format.

    Format: "ACTION1:duration1,ACTION2:duration2,..."
    Duration in seconds, will be converted to frames @ 25fps

    Example:
        >>> parse_action_sequence("WALK:5.0,RUN:3.0,JUMP:2.0")
        [ActionType.WALK] * 125 + [ActionType.RUN] * 75 + [ActionType.JUMP] * 50
    """
    actions = []

    for segment in action_string.split(","):
        action_name, duration = segment.split(":")
        action = ActionType[action_name.upper()]
        num_frames = int(float(duration) * 25)  # 25 fps
        actions.extend([action] * num_frames)

    # Pad or truncate to 250 frames
    if len(actions) < 250:
        actions.extend([actions[-1]] * (250 - len(actions)))
    else:
        actions = actions[:250]

    return actions


# ============================================================================
# STEP 3: Example usage scripts
# ============================================================================


def example_smooth_transition():
    """Example: Smooth action transitions"""
    generator = StickFigureGenerator("checkpoint_epoch_30_action_conditioned.pth")

    # WALK → RUN transition over 10 seconds
    actions = [ActionType.WALK] * 125 + [ActionType.RUN] * 125

    generator.generate_with_actions(
        text_prompt="A person walking then running",
        action_sequence=actions,
        output_path="walk_to_run.mp4",
    )


def example_complex_sequence():
    """Example: Complex action sequence"""
    generator = StickFigureGenerator("checkpoint_epoch_30_action_conditioned.pth")

    # Complex sequence: IDLE → WALK → RUN → JUMP → LAND → WALK → IDLE
    actions = (
        [ActionType.IDLE] * 25
        + [ActionType.WALK] * 50
        + [ActionType.RUN] * 50
        + [ActionType.JUMP] * 25
        + [ActionType.WALK] * 75
        + [ActionType.IDLE] * 25
    )

    generator.generate_with_actions(
        text_prompt="A person performing a sequence of actions",
        action_sequence=actions,
        output_path="complex_sequence.mp4",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Action-conditioned generation")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument(
        "--actions", required=True, help='Action sequence (e.g., "WALK:5.0,RUN:5.0")'
    )
    parser.add_argument("--output", default="output.mp4", help="Output video path")

    args = parser.parse_args()

    # Parse action sequence
    action_sequence = parse_action_sequence(args.actions)

    # Generate
    generator = StickFigureGenerator(args.model)
    output_path = generator.generate_with_actions(
        text_prompt=args.prompt,
        action_sequence=action_sequence,
        output_path=args.output,
    )

    print(f"✓ Generated: {output_path}")
