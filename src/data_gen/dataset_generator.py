import argparse
import os
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .auto_annotator import annotate_sample
from .llm_story_engine import LLMStoryGenerator
from .renderer import StickFigure
from .schema import (
    ACTION_TO_IDX,
    FacialExpression,
    MouthShape,
)
from .story_engine import StoryGenerator
from .validator import DataValidator


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


# Mapping for Face Features
EXPRESSION_TO_IDX = {e: i for i, e in enumerate(FacialExpression)}
MOUTH_SHAPE_TO_IDX = {s: i for i, s in enumerate(MouthShape)}
EYE_TYPE_TO_IDX = {"dots": 0, "curves": 1, "wide": 2, "closed": 3}


def augment_motion_sequence(motion_tensor, augmentation_type="speed"):
    """
    Apply data augmentation to motion sequence

    Args:
        motion_tensor: [frames, num_actors, 20] tensor
        augmentation_type: Type of augmentation
    """
    if augmentation_type == "speed":
        # Speed variation: ±20%
        speed_factor = random.uniform(0.8, 1.2)
        if speed_factor < 1.0:
            # Slow down: interpolate
            num_frames = motion_tensor.shape[0]
            new_num_frames = int(num_frames / speed_factor)
            indices = torch.linspace(0, num_frames - 1, new_num_frames)

            augmented = []
            for idx in indices:
                idx_floor = int(idx.floor())
                idx_ceil = min(int(idx.ceil()), num_frames - 1)
                alpha = idx - idx_floor
                # Interpolate for all actors and lines
                frame = (1 - alpha) * motion_tensor[idx_floor] + alpha * motion_tensor[
                    idx_ceil
                ]
                augmented.append(frame)
            return torch.stack(augmented)
        else:
            # Speed up: subsample
            num_frames = motion_tensor.shape[0]
            new_num_frames = int(num_frames / speed_factor)
            indices = torch.linspace(0, num_frames - 1, new_num_frames).long()
            return motion_tensor[indices]

    elif augmentation_type == "position":
        # Position jitter: ±0.5 units (apply same jitter to all actors to preserve relative interaction)
        jitter_x = random.uniform(-0.5, 0.5)
        jitter_y = random.uniform(-0.5, 0.5)
        augmented = motion_tensor.clone()
        # [frames, actors, 20] -> [frames, actors, 5 lines, 4 coords]
        frames, actors, _ = augmented.shape
        reshaped = augmented.view(frames, actors, 5, 4)

        reshaped[:, :, :, 0] += jitter_x  # x1
        reshaped[:, :, :, 1] += jitter_y  # y1
        reshaped[:, :, :, 2] += jitter_x  # x2
        reshaped[:, :, :, 3] += jitter_y  # y2

        return augmented

    elif augmentation_type == "scale":
        # Scale variation: ±10%
        scale_factor = random.uniform(0.9, 1.1)
        return motion_tensor * scale_factor

    elif augmentation_type == "mirror":
        # Horizontal flip
        augmented = motion_tensor.clone()
        frames, actors, _ = augmented.shape
        reshaped = augmented.view(frames, actors, 5, 4)

        reshaped[:, :, :, 0] *= -1  # x1
        reshaped[:, :, :, 2] *= -1  # x2

        return augmented

    else:
        return motion_tensor


def generate_dataset(
    config_path: str = "configs/base.yaml",
    num_samples: int = None,
    output_path: str = None,
    augment: bool = None,
):
    """
    Generate enhanced training dataset with new features.

    Args:
        config_path: Path to YAML configuration file
        num_samples: Override number of samples (uses config if None)
        output_path: Override output path (uses config if None)
        augment: Override augmentation setting (uses config if None)
    """
    # Load configuration
    config = load_config(config_path)
    gen_config = config.get("data_generation", {})

    # Annotation settings
    annotation_config = gen_config.get("annotation", {})
    annotation_enabled = annotation_config.get("enabled", True)

    # Get settings from config with fallbacks
    if num_samples is None:
        num_samples = gen_config.get("num_samples", 50000)
    if output_path is None:
        output_path = gen_config.get("output_path", "data/train_data.pt")
    if augment is None:
        augment = gen_config.get("augmentation", {}).get("enabled", True)

    # Sequence settings from config
    seq_config = gen_config.get("sequence", {})
    sequence_duration = seq_config.get("duration_seconds", 10.0)
    fps = seq_config.get("fps", 25)
    max_actors = seq_config.get("max_actors", 3)
    target_frames = int(sequence_duration * fps)

    # LLM settings from config
    llm_config = gen_config.get("llm", {})
    use_mock_llm = llm_config.get("use_mock", True)
    llm_ratio = llm_config.get("llm_ratio", 0.2)

    # Augmentation multiplier from config
    aug_multiplier = gen_config.get("augmentation", {}).get("multiplier", 4)

    print(f"Generating {num_samples} base samples with enhanced diversity...")
    print(f"Config: {config_path}")
    if augment:
        print(
            f"With {aug_multiplier}x augmentation, total effective samples: {num_samples * (aug_multiplier + 1)}"
        )

    story_engine = StoryGenerator()

    # Initialize LLM story engine with proper configuration
    llm_provider = "mock" if use_mock_llm else "grok"
    llm_story_engine = LLMStoryGenerator(
        provider=llm_provider,
        fallback_to_mock=True,  # Always allow fallback for robustness
        verbose=True,  # Enable logging to see what's happening
    )

    validator = DataValidator()

    # Print LLM configuration
    if llm_ratio > 0:
        print(f"\n{'='*60}")
        print("LLM Configuration:")
        print(f"  Provider: {llm_provider}")
        print(f"  LLM Ratio: {llm_ratio*100:.0f}% of samples")
        print(f"  Use Mock: {use_mock_llm}")
        if not use_mock_llm:
            import os

            api_key = os.getenv("GROK_API_KEY")
            if api_key:
                print(f"  GROK_API_KEY: {api_key[:10]}...{api_key[-4:]}")
            else:
                print("  ⚠️  WARNING: GROK_API_KEY not set! Will use mock data.")
        print(f"{'='*60}\n")

    data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Pre-calculate time indices for manual interpolation
    frame_indices = torch.arange(target_frames, dtype=torch.float32)

    print(
        f"Sequence: {sequence_duration}s ({target_frames} frames), Max Actors: {max_actors}"
    )

    rejections = 0
    generated_count = 0

    pbar = tqdm(total=num_samples, desc="Generating base samples")

    while generated_count < num_samples:
        # Retry loop for validation
        valid_sample = False
        attempts = 0
        max_attempts = 10

        while not valid_sample and attempts < max_attempts:
            attempts += 1

            # Mix procedural and LLM based on config ratio
            use_llm = random.random() < llm_ratio
            if use_llm:
                prompts = [
                    "heist",
                    "dance battle",
                    "adventure",
                    "sports",
                    "conversation",
                ]
                prompt = random.choice(prompts)
                try:
                    if use_mock_llm:
                        scene = story_engine.generate_random_scene()
                    else:
                        script = llm_story_engine.generate_script(prompt)
                        scenes = llm_story_engine.script_to_scenes(script)
                        scene = (
                            scenes[0]
                            if scenes
                            else story_engine.generate_random_scene()
                        )
                except Exception:
                    scene = story_engine.generate_random_scene()
            else:
                scene = story_engine.generate_random_scene()

            # Initialize tensors
            # Motion: [frames, actors, 20]
            motion_tensor = torch.zeros(
                (target_frames, max_actors, 20), dtype=torch.float32
            )
            # Actions: [frames, actors]
            action_tensor = torch.zeros((target_frames, max_actors), dtype=torch.long)
            # Face: [frames, actors, 7] -> [exp_idx, eye_idx, eyebrow, mouth_idx, openness, speak, speed]
            face_tensor = torch.zeros(
                (target_frames, max_actors, 7), dtype=torch.float32
            )

            actors = [StickFigure(a) for a in scene.actors]
            num_generated_frames = int(scene.duration * 25)

            # Limit processed actors to max_actors
            active_actors = actors[:max_actors]

            for f in range(min(num_generated_frames, target_frames)):
                t = f * 0.04

                for actor_idx, actor in enumerate(active_actors):
                    # 1. Update Motion
                    lines, _ = actor.get_pose(t)

                    # Flatten lines (5 lines * 4 coords)
                    actor_flat = []
                    for li, (start, end) in enumerate(lines):
                        if li >= 5:
                            break
                        actor_flat.extend([start[0], start[1], end[0], end[1]])
                    while len(actor_flat) < 20:
                        actor_flat.extend([0.0, 0.0, 0.0, 0.0])

                    motion_tensor[f, actor_idx] = torch.tensor(actor_flat)

                    # 2. Update Action
                    current_action = actor.get_current_action(t)
                    action_tensor[f, actor_idx] = ACTION_TO_IDX.get(current_action, 0)

                    # 3. Update Face
                    face_feats = actor.get_interpolated_features(t)
                    if face_feats:
                        face_vector = [
                            float(EXPRESSION_TO_IDX.get(face_feats.expression, 0)),
                            float(EYE_TYPE_TO_IDX.get(face_feats.eye_type, 0)),
                            face_feats.eyebrow_angle,
                            float(MOUTH_SHAPE_TO_IDX.get(face_feats.mouth_shape, 0)),
                            face_feats.mouth_openness,
                            1.0 if face_feats.is_speaking else 0.0,
                            face_feats.speech_cycle_speed,
                        ]
                        face_tensor[f, actor_idx] = torch.tensor(face_vector)

            # Pad remaining frames if any
            if num_generated_frames < target_frames:
                # Replicate last valid frame content
                if num_generated_frames > 0:
                    last_f = num_generated_frames - 1
                    motion_tensor[last_f + 1 :] = motion_tensor[last_f]
                    action_tensor[last_f + 1 :] = action_tensor[last_f]
                    face_tensor[last_f + 1 :] = face_tensor[last_f]

            # 4. Compute Physics (Vectorized)
            # [frames, actors, 6] -> [vx, vy, ax, ay, px, py]
            # Use head position (lines 0, 0-1 indices) as proxy for actor position
            head_pos = motion_tensor[:, :, 0:2]  # [frames, actors, 2]

            dt = 0.04
            velocity = torch.zeros_like(head_pos)
            velocity[:-1] = (head_pos[1:] - head_pos[:-1]) / dt
            velocity[-1] = velocity[-2]

            acceleration = torch.zeros_like(velocity)
            acceleration[:-1] = (velocity[1:] - velocity[:-1]) / dt
            acceleration[-1] = acceleration[-2]

            momentum = velocity.clone()  # Unit mass

            physics_tensor = torch.cat(
                [velocity, acceleration, momentum], dim=2
            )  # [frames, actors, 6]

            # 5. Camera (Interpolation)
            camera_tensor = torch.zeros((target_frames, 3))  # [x, y, zoom]
            camera_tensor[:, 2] = 1.0  # default zoom

            if scene.camera_keyframes:
                keyframes = sorted(scene.camera_keyframes, key=lambda k: k.frame)
                k_times = np.array([min(k.frame, target_frames - 1) for k in keyframes])
                k_x = np.array([k.x for k in keyframes])
                k_y = np.array([k.y for k in keyframes])
                k_zoom = np.array([k.zoom for k in keyframes])

                if len(keyframes) == 1:
                    camera_tensor[:] = torch.tensor(
                        [k_x[0], k_y[0], k_zoom[0]], dtype=torch.float32
                    )
                else:
                    # Interpolate using numpy
                    t_np = frame_indices.numpy()
                    x_interp = np.interp(t_np, k_times, k_x)
                    y_interp = np.interp(t_np, k_times, k_y)
                    z_interp = np.interp(t_np, k_times, k_zoom)

                    camera_tensor[:, 0] = torch.from_numpy(x_interp)
                    camera_tensor[:, 1] = torch.from_numpy(y_interp)
                    camera_tensor[:, 2] = torch.from_numpy(z_interp)

            # Store Base Sample Candidate
            candidate_sample = {
                "description": scene.description,
                "motion": motion_tensor,  # [250, 3, 20]
                "actions": action_tensor,  # [250, 3]
                "physics": physics_tensor,  # [250, 3, 6]
                "face": face_tensor,  # [250, 3, 7]
                "camera": camera_tensor,  # [250, 3]
                "augmented": False,
            }

            # Automated annotations (shot type, camera motion, physics, etc.)
            if annotation_enabled:
                candidate_sample = annotate_sample(candidate_sample, annotation_config)

            # --- VALIDATION STEP ---
            is_valid, score, reason = validator.validate(candidate_sample)
            if is_valid:
                valid_sample = True
                data.append(candidate_sample)

                # Augmentation (only for valid samples)
                if augment:
                    aug_types = ["speed", "position", "scale", "mirror"]
                    for aug in aug_types:
                        aug_motion = augment_motion_sequence(motion_tensor.clone(), aug)

                        # Fix length if speed changed
                        curr_len = aug_motion.shape[0]
                        if curr_len > target_frames:
                            aug_motion = aug_motion[:target_frames]
                        elif curr_len < target_frames:
                            padding = aug_motion[-1:].repeat(
                                target_frames - curr_len, 1, 1
                            )
                            aug_motion = torch.cat([aug_motion, padding], dim=0)

                        # Recompute physics for augmented motion
                        aug_head = aug_motion[:, :, 0:2]
                        aug_vel = torch.zeros_like(aug_head)
                        aug_vel[:-1] = (aug_head[1:] - aug_head[:-1]) / dt
                        aug_vel[-1] = aug_vel[-2]
                        aug_acc = torch.zeros_like(aug_vel)
                        aug_acc[:-1] = (aug_vel[1:] - aug_vel[:-1]) / dt
                        aug_acc[-1] = aug_acc[-2]

                        aug_physics = torch.cat([aug_vel, aug_acc, aug_vel], dim=2)

                        # Clone other tensors
                        aug_action = action_tensor.clone()
                        aug_face = face_tensor.clone()

                        aug_sample = {
                            "description": scene.description,
                            "motion": aug_motion,
                            "actions": aug_action,
                            "physics": aug_physics,
                            "face": aug_face,
                            "camera": camera_tensor,
                            "augmented": True,
                            "aug_type": aug,
                        }

                        if annotation_enabled:
                            aug_sample = annotate_sample(aug_sample, annotation_config)

                        data.append(aug_sample)
            else:
                rejections += 1

        if valid_sample:
            generated_count += 1
            pbar.update(1)

    pbar.close()

    print("\nFinal Dataset Stats:")
    print(f"  Samples Generated: {len(data)}")
    print(f"  Rejections: {rejections}")
    if len(data) > 0:
        print(f"  Motion Shape: {data[0]['motion'].shape}")
        print(f"  Face Shape: {data[0]['face'].shape}")
        print(f"  Physics Shape: {data[0]['physics'].shape}")

    print(f"Saving to {output_path}...")
    torch.save(data, output_path)
    print("Done.")

    # Return the resolved output path so callers (e.g., CLI tools) can
    # discover where the dataset was written without re-parsing configs.
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of samples to generate",
    )
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    generate_dataset(
        config_path=args.config, num_samples=args.num_samples, output_path=args.output
    )
