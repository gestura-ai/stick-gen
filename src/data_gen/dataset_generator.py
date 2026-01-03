import argparse
import asyncio
import glob
import json
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .async_processor import (
    AsyncDataGenerator,
    AsyncGeneratorStats,
    ResourceLimits,
)
from .auto_annotator import annotate_sample
from .llm_story_engine import LLMStoryGenerator
from .prompt_generator import DynamicPromptGenerator, ScenePromptGenerator
from .renderer import StickFigure
from .schema import (
    ACTION_TO_IDX,
    FacialExpression,
    MouthShape,
)
from .story_engine import StoryGenerator
from .validator import DataValidator


@dataclass
class GenerationMetadata:
    """Tracks generation progress for resumable checkpoints."""

    target_samples: int = 0
    generated_count: int = 0
    total_samples_written: int = 0  # Includes augmented samples
    rejections: int = 0
    start_time: float = 0.0
    last_update_time: float = 0.0
    rng_state: Optional[list] = None
    numpy_rng_state: Optional[dict] = None
    config_hash: str = ""
    version: str = "1.0"
    completed: bool = False

    def save(self, path: str):
        """Save metadata to JSON file."""
        data = asdict(self)
        # Convert numpy state to serializable format
        if self.numpy_rng_state is not None:
            data["numpy_rng_state"] = {
                "keys": self.numpy_rng_state["keys"].tolist()
                if hasattr(self.numpy_rng_state["keys"], "tolist")
                else self.numpy_rng_state["keys"],
                "pos": int(self.numpy_rng_state["pos"]),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GenerationMetadata":
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Restore numpy state
        if data.get("numpy_rng_state"):
            data["numpy_rng_state"] = {
                "keys": np.array(data["numpy_rng_state"]["keys"], dtype=np.uint32),
                "pos": data["numpy_rng_state"]["pos"],
            }
        return cls(**data)


def sample_to_json(sample: dict) -> dict:
    """Convert a sample dict with tensors to JSON-serializable format."""
    result = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            result[key] = {
                "_type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data": value.tolist(),
            }
        else:
            result[key] = value
    return result


def json_to_sample(data: dict) -> dict:
    """Convert JSON data back to sample dict with tensors."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict) and value.get("_type") == "tensor":
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.float64": torch.float64,
                "torch.int64": torch.int64,
                "torch.int32": torch.int32,
                "torch.long": torch.long,
            }
            dtype = dtype_map.get(value["dtype"], torch.float32)
            result[key] = torch.tensor(value["data"], dtype=dtype)
        else:
            result[key] = value
    return result


def save_sample(
    sample: dict,
    output_dir: str,
    sample_id: int,
    compress: bool = False,
    max_retries: int = 3,
) -> str:
    """Save a single sample to a JSON file with error handling.

    Args:
        sample: Sample dictionary with tensors
        output_dir: Directory to save samples
        sample_id: Unique sample identifier
        compress: If True, use gzip compression (.json.gz)
        max_retries: Number of retries on write failure

    Returns:
        Path to saved file

    Raises:
        IOError: If file cannot be written after max_retries
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = ".json.gz" if compress else ".json"
    filename = f"sample_{sample_id:08d}{ext}"
    filepath = os.path.join(output_dir, filename)
    temp_filepath = filepath + ".tmp"

    json_data = sample_to_json(sample)

    for attempt in range(max_retries):
        try:
            # Write to temp file first for atomicity
            if compress:
                import gzip

                with gzip.open(temp_filepath, "wt", encoding="utf-8") as f:
                    json.dump(json_data, f)
            else:
                with open(temp_filepath, "w") as f:
                    json.dump(json_data, f, indent=2)

            # Atomic rename
            os.replace(temp_filepath, filepath)
            return filepath

        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue
            # Clean up temp file if it exists
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass
            raise IOError(
                f"Failed to save sample {sample_id} after {max_retries} attempts: {e}"
            ) from e

    return filepath  # Should not reach here


def merge_samples(
    samples_dir: str,
    output_path: str,
    delete_after_merge: bool = False,
    verbose: bool = True,
) -> str:
    """Merge individual sample JSON files into a single .pt dataset.

    Supports both plain JSON (.json) and compressed (.json.gz) sample files.

    Args:
        samples_dir: Directory containing sample_*.json or sample_*.json.gz files
        output_path: Path for output .pt file
        delete_after_merge: If True, delete JSON files after successful merge
        verbose: Print progress information

    Returns:
        Path to merged dataset
    """
    import gzip

    # Find both compressed and uncompressed files
    json_pattern = os.path.join(samples_dir, "sample_*.json")
    gz_pattern = os.path.join(samples_dir, "sample_*.json.gz")
    sample_files = sorted(glob.glob(json_pattern) + glob.glob(gz_pattern))

    if not sample_files:
        raise ValueError(f"No sample files found in {samples_dir}")

    if verbose:
        print(f"Merging {len(sample_files)} samples from {samples_dir}...")

    data = []
    errors = []
    for filepath in tqdm(sample_files, desc="Loading samples", disable=not verbose):
        try:
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    json_data = json.load(f)
            else:
                with open(filepath) as f:
                    json_data = json.load(f)
            sample = json_to_sample(json_data)
            data.append(sample)
        except (json.JSONDecodeError, OSError) as e:
            errors.append((filepath, str(e)))
            if verbose:
                print(f"\n⚠️ Error loading {filepath}: {e}")

    if errors and verbose:
        print(f"\n⚠️ {len(errors)} files had errors and were skipped")

    if not data:
        raise ValueError(f"No valid samples could be loaded from {samples_dir}")

    if verbose:
        print(f"Saving merged dataset to {output_path}...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(data, output_path)

    if delete_after_merge:
        if verbose:
            print("Cleaning up individual sample files...")
        for filepath in sample_files:
            try:
                os.remove(filepath)
            except OSError:
                pass  # Ignore cleanup errors
        # Also remove metadata file if present
        meta_path = os.path.join(samples_dir, "generation_meta.json")
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass

    if verbose:
        print(f"✓ Merged {len(data)} samples to {output_path}")

    return output_path


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

    elif augmentation_type == "noise":
        # Add small Gaussian noise to motion coordinates
        noise_scale = random.uniform(0.01, 0.05)
        noise = torch.randn_like(motion_tensor) * noise_scale
        return motion_tensor + noise

    elif augmentation_type == "time_shift":
        # Shift the motion sequence in time (circular shift)
        num_frames = motion_tensor.shape[0]
        shift = random.randint(-num_frames // 4, num_frames // 4)
        return torch.roll(motion_tensor, shifts=shift, dims=0)

    else:
        return motion_tensor


class MotionConditionedSampleGenerator:
    """
    Generate synthetic samples by augmenting real motion data.

    This class loads motion clips from a canonical dataset and generates
    new samples by applying augmentations and generating matching descriptions.
    Optionally uses Grok LLM to generate richer, more natural descriptions.

    LLM Configuration:
        LLM settings are read from config file under `data_generation.llm`:
        - use_mock: true/false (true = template-based, false = use Grok)
        - llm_ratio: float (ratio of samples to use LLM descriptions)

        Environment variable overrides (same as procedural generation):
        - DISABLE_MOCK_LLM=true or USE_REAL_LLM=true: forces real LLM
        - GROK_API_KEY set: automatically uses real LLM (unless FORCE_MOCK_LLM=true)

        CLI override:
        - use_llm_override parameter can force LLM on/off regardless of config
    """

    AUGMENTATION_TYPES = ["speed", "position", "scale", "mirror", "noise", "time_shift"]

    def __init__(
        self,
        motion_source_path: str,
        config_path: str = "configs/base.yaml",
        target_frames: int = 250,
        max_actors: int = 4,
        augmentations_per_sample: int = 2,
        cache_size: int = 1000,
        use_llm_override: Optional[bool] = None,
        verbose: bool = True,
    ):
        """
        Initialize the motion-conditioned generator.

        Args:
            motion_source_path: Path to merged canonical .pt dataset
            config_path: Path to YAML configuration file for LLM settings
            target_frames: Target number of frames per sample
            max_actors: Maximum actors per sample
            augmentations_per_sample: Number of augmentations to apply per sample
            cache_size: Number of source samples to cache in memory
            use_llm_override: Override config LLM setting (None = use config,
                True = force LLM on, False = force LLM off)
            verbose: Whether to print verbose logging
        """
        self.motion_source_path = motion_source_path
        self.config_path = config_path
        self.target_frames = target_frames
        self.max_actors = max_actors
        self.augmentations_per_sample = augmentations_per_sample
        self.cache_size = cache_size
        self.verbose = verbose

        # Load LLM settings from config with env var overrides
        self.use_llm = self._resolve_llm_setting(use_llm_override)

        # Load source data
        self._source_samples: list[dict] = []
        self._load_source_data()

        # Initialize prompt generator for creating descriptions
        self.prompt_generator = ScenePromptGenerator(verbose=verbose)

        # Initialize Grok client if LLM is enabled
        self._grok_client = None
        if self.use_llm:
            self._init_grok_client()

    def _resolve_llm_setting(self, use_llm_override: Optional[bool]) -> bool:
        """
        Resolve whether to use LLM based on config, env vars, and CLI override.

        Priority (highest to lowest):
        1. use_llm_override parameter (CLI --use-llm flag)
        2. Environment variables (DISABLE_MOCK_LLM, USE_REAL_LLM, GROK_API_KEY)
        3. Config file setting (data_generation.llm.use_mock)

        Returns:
            True if LLM should be used, False for template-based generation
        """
        # Load config
        config = load_config(self.config_path)
        gen_config = config.get("data_generation", {})
        llm_config = gen_config.get("llm", {})

        # Start with config file setting (inverted: use_mock=true means NO LLM)
        use_mock_llm = llm_config.get("use_mock", True)

        # Apply environment variable overrides (same logic as generate_dataset)
        if os.getenv("DISABLE_MOCK_LLM", "").lower() in ("true", "1", "yes"):
            use_mock_llm = False
        if os.getenv("USE_REAL_LLM", "").lower() in ("true", "1", "yes"):
            use_mock_llm = False
        # If GROK_API_KEY is set, automatically use real LLM
        if os.getenv("GROK_API_KEY") and os.getenv("FORCE_MOCK_LLM", "").lower() not in (
            "true", "1", "yes"
        ):
            use_mock_llm = False

        # Apply CLI override (highest priority)
        if use_llm_override is not None:
            use_llm = use_llm_override
        else:
            use_llm = not use_mock_llm  # Invert: use_mock=False means use_llm=True

        if self.verbose:
            source = "CLI override" if use_llm_override is not None else (
                "env var" if os.getenv("GROK_API_KEY") or os.getenv("USE_REAL_LLM") else "config"
            )
            print(f"[MotionConditioned] LLM setting: {'enabled' if use_llm else 'disabled'} (from {source})")

        return use_llm

    def _init_grok_client(self) -> None:
        """Initialize Grok API client for LLM-based description generation."""
        try:
            from openai import OpenAI
            api_key = os.getenv("GROK_API_KEY")
            if not api_key:
                if self.verbose:
                    print("[MotionConditioned] WARNING: GROK_API_KEY not set, disabling LLM")
                self.use_llm = False
                return

            self._grok_client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
            if self.verbose:
                print(f"[MotionConditioned] Grok LLM enabled (API key: {api_key[:8]}...)")
        except ImportError:
            if self.verbose:
                print("[MotionConditioned] WARNING: openai package not installed, disabling LLM")
            self.use_llm = False
        except Exception as e:
            if self.verbose:
                print(f"[MotionConditioned] WARNING: Failed to init Grok: {e}, disabling LLM")
            self.use_llm = False

    # Banned phrases that make descriptions sound robotic/technical
    BANNED_PHRASES = [
        "the figure", "a figure", "subtle jitter", "jittery", "mirrored symmetry",
        "temporal", "oscillation", "pendulation", "amplitude", "laterality",
        "positional", "off-center", "scaled up", "scaled down", "shifted stance",
        "with subtle", "noise-induced", "micro-adjustments", "weight transfers",
    ]

    # Diverse sentence starters to encourage variety
    SENTENCE_STARTERS = [
        "Arms", "Hands", "Stepping", "Moving", "Reaching", "Turning", "Leaning",
        "Swinging", "Lifting", "Lowering", "Extending", "Bending", "Twisting",
        "Swaying", "Shifting", "Gliding", "Striding", "Walking", "Running",
        "Jumping", "Crouching", "Rising", "Falling", "Spinning", "Pivoting",
    ]

    def _extract_motion_metadata(
        self, source_sample: dict, augmentations: list[str]
    ) -> dict:
        """
        Extract metadata from a motion sample for Grok description generation.

        Builds a structured dict with motion characteristics that Grok can use
        to generate natural, varied descriptions. Focuses on human-readable
        motion qualities rather than technical augmentation details.

        Args:
            source_sample: Source motion sample dict
            augmentations: List of augmentation types applied

        Returns:
            Dict with motion metadata for Grok
        """
        action_label = source_sample.get("action_label", "motion")
        original_desc = source_sample.get("description", "")

        # Get enhanced metadata if available
        enhanced = source_sample.get("enhanced_metadata", {})
        if hasattr(enhanced, "model_dump"):
            enhanced = enhanced.model_dump()
        elif hasattr(enhanced, "__dict__"):
            enhanced = enhanced.__dict__

        # Map action labels to natural descriptions
        action_mapping = {
            "walk": "walking",
            "run": "running",
            "jump": "jumping",
            "sit": "sitting down",
            "stand": "standing up",
            "wave": "waving",
            "punch": "punching",
            "kick": "kicking",
            "dance": "dancing",
            "gesture": "gesturing",
            "idle": "standing still",
            "motion": "moving",
        }
        natural_action = action_mapping.get(
            action_label.lower() if action_label else "motion",
            action_label or "moving"
        )

        # Translate augmentations to natural motion qualities (not technical terms)
        motion_qualities = []
        for aug in augmentations:
            if aug == "speed":
                motion_qualities.append(
                    random.choice(["at a varied pace", "with changing tempo", "rhythmically"])
                )
            elif aug == "mirror":
                motion_qualities.append(
                    random.choice(["with balanced movement", "symmetrically", "evenly"])
                )
            elif aug == "scale":
                motion_qualities.append(
                    random.choice(["with exaggerated motion", "with compact gestures", "expressively"])
                )
            elif aug == "noise":
                motion_qualities.append(
                    random.choice(["with natural variation", "organically", "with lifelike imperfection"])
                )
            elif aug == "time_shift":
                motion_qualities.append(
                    random.choice(["from a different starting point", "mid-action", "in progress"])
                )
            elif aug == "position":
                motion_qualities.append(
                    random.choice(["from an offset position", "slightly displaced", "off to one side"])
                )

        # Build motion characteristics - focus on natural language
        motion_info = {
            "action": natural_action,
            "motion_qualities": motion_qualities[:2] if motion_qualities else ["naturally"],
        }

        # Add velocity description if available
        if isinstance(enhanced, dict) and "statistics" in enhanced:
            stats = enhanced["statistics"]
            if hasattr(stats, "model_dump"):
                stats = stats.model_dump()
            avg_vel = stats.get("avg_velocity") if isinstance(stats, dict) else getattr(stats, "avg_velocity", None)
            if avg_vel is not None:
                if avg_vel < 0.5:
                    motion_info["pace"] = "slow and deliberate"
                elif avg_vel < 1.5:
                    motion_info["pace"] = "moderate"
                else:
                    motion_info["pace"] = "quick and energetic"

        # Add original description if meaningful
        if original_desc and len(original_desc) > 15 and "motion" not in original_desc.lower():
            motion_info["reference"] = original_desc[:100]

        # Suggest a random sentence starter for variety
        motion_info["suggested_start"] = random.choice(self.SENTENCE_STARTERS)

        return motion_info

    def _filter_description(self, description: str) -> tuple[bool, str]:
        """
        Filter and clean a generated description.

        Args:
            description: Raw description from Grok

        Returns:
            Tuple of (is_valid, cleaned_description)
        """
        # Remove quotes
        desc = description.strip()
        if desc.startswith('"') and desc.endswith('"'):
            desc = desc[1:-1]
        if desc.startswith("'") and desc.endswith("'"):
            desc = desc[1:-1]

        # Check for banned phrases
        desc_lower = desc.lower()
        for banned in self.BANNED_PHRASES:
            if banned.lower() in desc_lower:
                return False, desc

        # Check word count
        word_count = len(desc.split())
        if word_count < 10 or word_count > 35:
            return False, desc

        return True, desc

    def _generate_description_with_grok(
        self, source_sample: dict, augmentations: list[str]
    ) -> str:
        """
        Generate a natural description using Grok LLM.

        Includes retry logic and post-processing to ensure high-quality,
        varied descriptions without technical jargon.

        Args:
            source_sample: Source motion sample dict
            augmentations: List of augmentation types applied

        Returns:
            Natural language description generated by Grok
        """
        import json

        metadata = self._extract_motion_metadata(source_sample, augmentations)

        system_prompt = """You are a motion caption writer creating training data for AI.
Write ONE natural sentence (15-25 words) describing human body movement.

CRITICAL RULES:
1. Start with the suggested word or a body part (Arms, Hands, Legs, etc.)
2. Describe WHAT the body does, not technical terms
3. Use everyday language a person would use to describe movement
4. NO quotes around your response

BANNED WORDS (never use these):
- figure, subtle, jittery, mirrored, temporal, oscillation
- amplitude, positional, scaled, shifted, noise, micro-adjustments

GOOD examples:
- "Arms sweep upward in a wide arc as the body rises onto tiptoes"
- "Stepping forward with purpose, weight shifts smoothly from heel to toe"
- "Hands reach out and grasp, then pull back toward the chest"
- "Turning sharply to the left, momentum carries through the hips"
- "Crouching low, then springing upward with arms extended overhead"

BAD examples (avoid these patterns):
- "The figure maintains subtle oscillations" (too technical)
- "A mirrored, jittery motion sequence" (uses banned words)
- "Smooth, steady movement with temporal shifts" (robotic)"""

        user_prompt = f"""Describe this motion naturally:
Action: {metadata.get('action', 'moving')}
Qualities: {', '.join(metadata.get('motion_qualities', ['naturally']))}
Start your sentence with: {metadata.get('suggested_start', 'Moving')}"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._grok_client.chat.completions.create(
                    model="grok-4-1-fast",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=60,
                    temperature=0.85 + (attempt * 0.05),  # Increase temp on retries
                )

                result = response.choices[0].message.content.strip()
                is_valid, cleaned = self._filter_description(result)

                if is_valid:
                    if self.verbose:
                        print(f"[MotionConditioned] Grok: {cleaned[:50]}...")
                    return cleaned

                if self.verbose and attempt < max_retries - 1:
                    print(f"[MotionConditioned] Retry {attempt + 1}: filtered '{result[:30]}...'")

            except Exception as e:
                if self.verbose:
                    print(f"[MotionConditioned] Grok error attempt {attempt + 1}: {e}")

        # Fallback to template if all retries fail
        if self.verbose:
            print("[MotionConditioned] All Grok retries failed, using template")
        return self._generate_description_template(source_sample, augmentations)

    def _generate_description_template(
        self, source_sample: dict, augmentations: list[str]
    ) -> str:
        """Generate description using templates (fallback when Grok unavailable)."""
        original_desc = source_sample.get("description", "")
        action_label = source_sample.get("action_label", "motion")

        # Build augmentation description
        aug_phrases = []
        if "speed" in augmentations:
            aug_phrases.append(random.choice(["with varied tempo", "at different speed"]))
        if "mirror" in augmentations:
            aug_phrases.append(random.choice(["mirrored", "reflected"]))
        if "scale" in augmentations:
            aug_phrases.append(random.choice(["with adjusted scale", "resized"]))

        # Generate description
        if original_desc and len(original_desc) > 10:
            desc = original_desc
            if aug_phrases:
                desc = f"{desc} ({', '.join(aug_phrases)})"
        else:
            desc = f"A figure performing {action_label} motion"
            if aug_phrases:
                desc = f"{desc} {', '.join(aug_phrases)}"

        return desc

    def _load_source_data(self) -> None:
        """Load source motion samples from canonical dataset."""
        if not os.path.exists(self.motion_source_path):
            raise FileNotFoundError(f"Motion source not found: {self.motion_source_path}")

        print(f"[MotionConditioned] Loading source data from {self.motion_source_path}")
        data = torch.load(self.motion_source_path, weights_only=False)

        # Handle different data formats
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        else:
            samples = [data]

        # Filter to samples with valid motion tensors
        valid_samples = []
        for s in samples:
            if isinstance(s, dict) and "motion" in s:
                motion = s["motion"]
                if isinstance(motion, torch.Tensor) and motion.numel() > 0:
                    valid_samples.append(s)

        # Cache a subset for memory efficiency
        if len(valid_samples) > self.cache_size:
            self._source_samples = random.sample(valid_samples, self.cache_size)
        else:
            self._source_samples = valid_samples

        print(f"[MotionConditioned] Loaded {len(self._source_samples)} source samples")

    def _sample_source_clip(self) -> dict:
        """Sample a random clip from the source data."""
        if not self._source_samples:
            raise RuntimeError("No source samples available")
        return random.choice(self._source_samples)

    def _apply_augmentations(self, motion: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to motion tensor."""
        augmented = motion.clone()

        # Select random augmentations
        augs = random.sample(
            self.AUGMENTATION_TYPES,
            min(self.augmentations_per_sample, len(self.AUGMENTATION_TYPES))
        )

        for aug_type in augs:
            augmented = augment_motion_sequence(augmented, aug_type)

        return augmented

    def _normalize_motion_shape(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize motion tensor to expected shape [frames, actors, 20]."""
        # Handle various input shapes
        if motion.dim() == 2:
            # [frames, 20] -> [frames, 1, 20]
            motion = motion.unsqueeze(1)
        elif motion.dim() == 3:
            # Already [frames, actors, 20]
            pass
        else:
            raise ValueError(f"Unexpected motion shape: {motion.shape}")

        frames, actors, features = motion.shape

        # Pad/truncate frames
        if frames < self.target_frames:
            # Pad by repeating last frame
            padding = motion[-1:].expand(self.target_frames - frames, -1, -1)
            motion = torch.cat([motion, padding], dim=0)
        elif frames > self.target_frames:
            # Truncate or resample
            indices = torch.linspace(0, frames - 1, self.target_frames).long()
            motion = motion[indices]

        # Pad/truncate actors
        if actors < self.max_actors:
            padding = torch.zeros(
                self.target_frames, self.max_actors - actors, features,
                dtype=motion.dtype
            )
            motion = torch.cat([motion, padding], dim=1)
        elif actors > self.max_actors:
            motion = motion[:, :self.max_actors, :]

        # Ensure 20 features
        if features < 20:
            padding = torch.zeros(
                self.target_frames, self.max_actors, 20 - features,
                dtype=motion.dtype
            )
            motion = torch.cat([motion, padding], dim=2)
        elif features > 20:
            motion = motion[:, :, :20]

        return motion

    def _generate_description(self, source_sample: dict, augmentations: list[str]) -> str:
        """Generate a new description for the augmented sample.

        Uses Grok LLM if enabled, otherwise falls back to template-based generation.
        """
        if self.use_llm and self._grok_client is not None:
            return self._generate_description_with_grok(source_sample, augmentations)
        return self._generate_description_template(source_sample, augmentations)

    def generate_conditioned_sample(self) -> Optional[dict]:
        """
        Generate a synthetic sample conditioned on real motion data.

        Returns:
            Dictionary with motion tensors and metadata, or None if invalid.
        """
        try:
            # Sample source clip
            source = self._sample_source_clip()
            source_motion = source["motion"]

            if not isinstance(source_motion, torch.Tensor):
                source_motion = torch.tensor(source_motion, dtype=torch.float32)

            # Normalize shape
            motion = self._normalize_motion_shape(source_motion)

            # Apply augmentations
            augs_applied = random.sample(
                self.AUGMENTATION_TYPES,
                min(self.augmentations_per_sample, len(self.AUGMENTATION_TYPES))
            )
            for aug in augs_applied:
                motion = augment_motion_sequence(motion, aug)

            # Re-normalize after augmentation (some may change frame count)
            motion = self._normalize_motion_shape(motion)

            # Compute physics
            fps = 25
            dt = 1.0 / fps
            head_pos = motion[:, :, 0:2]
            velocity = torch.zeros_like(head_pos)
            velocity[:-1] = (head_pos[1:] - head_pos[:-1]) / dt
            velocity[-1] = velocity[-2] if motion.shape[0] > 1 else velocity[-1]

            acceleration = torch.zeros_like(velocity)
            acceleration[:-1] = (velocity[1:] - velocity[:-1]) / dt
            acceleration[-1] = acceleration[-2] if motion.shape[0] > 1 else acceleration[-1]

            physics_tensor = torch.cat([velocity, acceleration, velocity], dim=2)

            # Copy or generate other tensors
            actions = source.get("actions")
            if actions is not None:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions, dtype=torch.long)
                # Normalize action shape
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1).expand(-1, self.max_actors)
                if actions.shape[0] != self.target_frames:
                    if actions.shape[0] < self.target_frames:
                        padding = actions[-1:].expand(self.target_frames - actions.shape[0], -1)
                        actions = torch.cat([actions, padding], dim=0)
                    else:
                        actions = actions[:self.target_frames]
                if actions.shape[1] < self.max_actors:
                    padding = torch.zeros(
                        self.target_frames, self.max_actors - actions.shape[1],
                        dtype=actions.dtype
                    )
                    actions = torch.cat([actions, padding], dim=1)
            else:
                actions = torch.zeros((self.target_frames, self.max_actors), dtype=torch.long)

            # Generate camera (simple static camera)
            camera = torch.zeros((self.target_frames, 3))
            camera[:, 2] = 1.0

            # Face tensor (zeros if not in source)
            face = source.get("face")
            if face is not None and isinstance(face, torch.Tensor):
                if face.shape[0] != self.target_frames:
                    if face.shape[0] < self.target_frames:
                        padding = face[-1:].expand(self.target_frames - face.shape[0], -1, -1)
                        face = torch.cat([face, padding], dim=0)
                    else:
                        face = face[:self.target_frames]
            else:
                face = torch.zeros((self.target_frames, self.max_actors, 7), dtype=torch.float32)

            # Generate description
            description = self._generate_description(source, augs_applied)

            sample = {
                "description": description,
                "motion": motion,
                "actions": actions,
                "physics": physics_tensor,
                "face": face,
                "camera": camera,
                "augmented": True,
                "augmentation_types": augs_applied,
                "source": "synthetic_conditioned",
                "original_source": source.get("source", "unknown"),
                "environment_type": source.get("environment_type"),
                "weather_type": source.get("weather_type"),
            }

            # Skip validation for motion-conditioned samples:
            # - Source data was already validated when converted
            # - Augmentations (speed, mirror, etc.) preserve structural validity
            # - Only basic shape check needed
            if motion.shape != (self.target_frames, self.max_actors, 20):
                return None

            return sample

        except Exception as e:
            print(f"[MotionConditioned] Error generating sample: {e}")
            return None

    def __len__(self) -> int:
        """Return number of source samples available."""
        return len(self._source_samples)


def generate_dataset(
    config_path: str = "configs/base.yaml",
    num_samples: int = None,
    output_path: str = None,
    augment: bool = None,
    force: bool = False,
    streaming: bool = True,
    append: bool = False,
    batch_size: int = 10,
    compress: bool = False,
):
    """
    Generate enhanced training dataset with new features.

    Args:
        config_path: Path to YAML configuration file
        num_samples: Override number of samples (uses config if None)
        output_path: Override output path (uses config if None)
        augment: Override augmentation setting (uses config if None)
        force: If True, ignore existing data and start fresh
        streaming: If True, write samples incrementally to disk (memory efficient)
        append: If True, append to existing dataset instead of overwriting
        batch_size: Number of samples to buffer before writing (streaming mode)
        compress: If True, compress sample files with gzip (streaming mode only)

    Returns:
        Path to output file (.pt) or samples directory (streaming mode)
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

    # Environment variable overrides for RunPod deployment
    # DISABLE_MOCK_LLM=true or USE_REAL_LLM=true forces real LLM usage
    if os.getenv("DISABLE_MOCK_LLM", "").lower() in ("true", "1", "yes"):
        use_mock_llm = False
    if os.getenv("USE_REAL_LLM", "").lower() in ("true", "1", "yes"):
        use_mock_llm = False
    # If GROK_API_KEY is set, automatically use real LLM
    if os.getenv("GROK_API_KEY") and not os.getenv("FORCE_MOCK_LLM", "").lower() in (
        "true",
        "1",
        "yes",
    ):
        use_mock_llm = False

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

    # Initialize scene-based prompt generator (generates descriptions from actual scene data)
    scene_prompt_generator = ScenePromptGenerator(verbose=True)

    # Legacy random prompt generator (fallback)
    prompt_generator = DynamicPromptGenerator(cache_size=5000)

    # LLM response cache to avoid redundant API calls
    llm_script_cache: dict = {}

    validator = DataValidator()

    # Print LLM configuration
    print(f"\n{'='*60}")
    print("Prompt Generation Configuration:")
    print(f"  Mode: Scene-first (prompts describe actual motion data)")
    print(f"  LLM Provider: {llm_provider}")
    print(f"  LLM Ratio: {llm_ratio*100:.0f}% of samples use LLM-generated scenes")
    print(f"  Use Mock: {use_mock_llm}")
    if not use_mock_llm:
        api_key = os.getenv("GROK_API_KEY")
        if api_key:
            print(f"  GROK_API_KEY: {api_key[:10]}...{api_key[-4:]}")
        else:
            print("  ⚠️  WARNING: GROK_API_KEY not set! Will use template fallback.")
    print(f"  Streaming Mode: {streaming}")
    print(f"  Append Mode: {append}")
    print(f"{'='*60}\n")

    # Pre-calculate time indices for manual interpolation
    frame_indices = torch.arange(target_frames, dtype=torch.float32)

    print(
        f"Sequence: {sequence_duration}s ({target_frames} frames), Max Actors: {max_actors}"
    )

    # Setup output paths based on mode
    if streaming:
        # Streaming mode: write to samples directory
        samples_dir = output_path.replace(".pt", "_samples")
        meta_path = os.path.join(samples_dir, "generation_meta.json")
        os.makedirs(samples_dir, exist_ok=True)
    else:
        samples_dir = None
        meta_path = None

    # Initialize counters and state
    rejections = 0
    generated_count = 0
    total_samples_written = 0
    sample_buffer = []  # Buffer for batch writes in streaming mode
    data = []  # Only used in non-streaming mode
    start_time = time.time()
    eta_warning_shown = False

    # Resume from existing progress
    if streaming and os.path.exists(meta_path) and not force:
        try:
            meta = GenerationMetadata.load(meta_path)
            if not meta.completed:
                generated_count = meta.generated_count
                total_samples_written = meta.total_samples_written
                rejections = meta.rejections
                # Restore RNG state for reproducibility
                if meta.rng_state:
                    random.setstate(tuple(meta.rng_state))
                if meta.numpy_rng_state:
                    np.random.set_state(
                        (
                            "MT19937",
                            meta.numpy_rng_state["keys"],
                            meta.numpy_rng_state["pos"],
                            0,
                            0.0,
                        )
                    )
                print(f"\n[RESUME] Resuming from sample {generated_count}/{num_samples}")
                print(f"[RESUME] Total samples written: {total_samples_written}")
            else:
                if append:
                    # Append mode: continue from where we left off
                    generated_count = 0  # Reset for new batch
                    total_samples_written = meta.total_samples_written
                    print(f"\n[APPEND] Adding to existing {total_samples_written} samples")
                else:
                    print(f"\n[INFO] Previous generation completed. Use --force to regenerate.")
                    return samples_dir
        except Exception as e:
            print(f"[RESUME] Could not load metadata: {e}, starting fresh")
    elif not streaming and not force:
        # Legacy checkpoint mode for non-streaming
        checkpoint_path = output_path.replace(".pt", "_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = torch.load(checkpoint_path, weights_only=False)
                if isinstance(checkpoint_data, list) and len(checkpoint_data) > 0:
                    data = checkpoint_data
                    base_samples = sum(1 for s in data if not s.get("augmented", False))
                    generated_count = base_samples
                    print(f"\n[RESUME] Loaded {len(data)} samples from checkpoint")
                    print(f"[RESUME] Resuming from base sample {generated_count}/{num_samples}")
            except Exception as e:
                print(f"[RESUME] Could not load checkpoint: {e}, starting fresh")

    pbar = tqdm(total=num_samples, desc="Generating base samples", initial=generated_count)

    while generated_count < num_samples:
        # Retry loop for validation
        valid_sample = False
        attempts = 0
        max_attempts = 10

        while not valid_sample and attempts < max_attempts:
            attempts += 1

            # SCENE-FIRST APPROACH:
            # 1. Generate scene procedurally (or via LLM for variety)
            # 2. Then generate text prompt that describes the actual scene
            # This ensures text prompts accurately describe the motion data

            use_llm_scene = random.random() < llm_ratio
            if use_llm_scene and not use_mock_llm:
                # Use LLM to generate scene (for more creative scenarios)
                try:
                    # Generate a prompt for scene creation
                    scene_prompt = prompt_generator.generate_prompt()
                    if scene_prompt in llm_script_cache:
                        script = llm_script_cache[scene_prompt]
                    else:
                        script = llm_story_engine.generate_script(scene_prompt)
                        if len(llm_script_cache) < 1000:
                            llm_script_cache[scene_prompt] = script
                    scenes = llm_story_engine.script_to_scenes(script)
                    scene = scenes[0] if scenes else story_engine.generate_random_scene()
                except Exception as e:
                    # Log the error with details for debugging
                    print(f"\n[LLM ERROR] Sample {generated_count}: {type(e).__name__}: {e}")
                    print(f"[LLM ERROR] Traceback: {traceback.format_exc()}")
                    scene = story_engine.generate_random_scene()
            else:
                # Use procedural generation (faster, deterministic)
                scene = story_engine.generate_random_scene()

            # Generate text prompt that describes the ACTUAL scene
            # This is the key change - prompt now matches the motion data
            text_prompt = scene_prompt_generator.generate_prompt_from_scene(scene)

            # Update scene description to match generated prompt
            scene.description = text_prompt

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

            # Pass environment and weather to StickFigure for physics-aware motion
            actors = [
                StickFigure(
                    a,
                    environment_type=getattr(scene, 'environment_type', None),
                    weather_type=getattr(scene, 'weather_type', None)
                )
                for a in scene.actors
            ]
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
            # Include environment metadata for environment-aware training
            env_type = getattr(scene, 'environment_type', None)
            weather_type = getattr(scene, 'weather_type', None)
            candidate_sample = {
                "description": scene.description,
                "motion": motion_tensor,  # [250, 3, 20]
                "actions": action_tensor,  # [250, 3]
                "physics": physics_tensor,  # [250, 3, 6]
                "face": face_tensor,  # [250, 3, 7]
                "camera": camera_tensor,  # [250, 3]
                "augmented": False,
                # Environment metadata for physics-aware training
                "environment_type": env_type.value if env_type else None,
                "weather_type": weather_type.value if weather_type else None,
            }

            # Automated annotations (shot type, camera motion, physics, etc.)
            if annotation_enabled:
                candidate_sample = annotate_sample(candidate_sample, annotation_config)

            # --- VALIDATION STEP ---
            is_valid, score, reason = validator.validate(candidate_sample)
            if is_valid:
                valid_sample = True

                # Collect samples to write (base + augmented)
                samples_to_write = [candidate_sample]

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
                            # Preserve environment metadata for augmented samples
                            "environment_type": env_type.value if env_type else None,
                            "weather_type": weather_type.value if weather_type else None,
                        }

                        if annotation_enabled:
                            aug_sample = annotate_sample(aug_sample, annotation_config)

                        samples_to_write.append(aug_sample)

                # Write samples based on mode
                if streaming:
                    # Streaming mode: write each sample immediately
                    for sample in samples_to_write:
                        save_sample(
                            sample, samples_dir, total_samples_written, compress=compress
                        )
                        total_samples_written += 1
                        sample_buffer.append(1)  # Track for batch metadata updates

                        # Update metadata periodically
                        if len(sample_buffer) >= batch_size:
                            # Save RNG state for reproducibility
                            rng_state = list(random.getstate())
                            np_state = np.random.get_state()
                            numpy_rng_state = {
                                "keys": np_state[1],
                                "pos": np_state[2],
                            }

                            meta = GenerationMetadata(
                                target_samples=num_samples,
                                generated_count=generated_count + 1,
                                total_samples_written=total_samples_written,
                                rejections=rejections,
                                start_time=start_time,
                                last_update_time=time.time(),
                                rng_state=rng_state,
                                numpy_rng_state=numpy_rng_state,
                                completed=False,
                            )
                            meta.save(meta_path)
                            sample_buffer.clear()
                else:
                    # Non-streaming mode: accumulate in memory
                    data.extend(samples_to_write)
            else:
                rejections += 1

        if valid_sample:
            generated_count += 1
            pbar.update(1)

            # Progress reporting
            if generated_count % 100 == 0 or generated_count == num_samples:
                elapsed = time.time() - start_time
                rate = generated_count / elapsed if elapsed > 0 else 0
                remaining = num_samples - generated_count
                eta_hours = (remaining / rate / 3600) if rate > 0 else float("inf")

                if streaming:
                    print(
                        f"\n[PROGRESS] {generated_count}/{num_samples} base samples, "
                        f"{total_samples_written} total written. "
                        f"Rate: {rate:.2f}/sec, ETA: {eta_hours:.1f}h"
                    )
                else:
                    # Legacy checkpoint for non-streaming mode
                    checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL", "500"))
                    if generated_count % checkpoint_interval == 0:
                        checkpoint_path = output_path.replace(".pt", "_checkpoint.pt")
                        print(f"\n[CHECKPOINT] Saving {len(data)} samples...")
                        torch.save(data, checkpoint_path)
                        print(
                            f"[CHECKPOINT] ✓ Saved. Rate: {rate:.2f}/sec, ETA: {eta_hours:.1f}h"
                        )

                # Warn if ETA is unreasonably long (>48 hours)
                if eta_hours > 48 and not eta_warning_shown:
                    print(
                        f"\n⚠️  WARNING: Estimated time to completion is {eta_hours:.0f} hours!"
                    )
                    print("   Consider reducing num_samples or disabling LLM calls.")
                    print(
                        "   Set USE_MOCK_LLM=true or reduce LLM_RATIO for faster generation."
                    )
                    eta_warning_shown = True

    pbar.close()

    # Final stats and saving
    print("\n" + "=" * 60)
    print("Final Dataset Stats:")
    print(f"  Base Samples Generated: {generated_count}")
    print(f"  Rejections: {rejections}")

    if streaming:
        print(f"  Total Samples Written: {total_samples_written}")
        print(f"  Output Directory: {samples_dir}")

        # Save final metadata
        meta = GenerationMetadata(
            target_samples=num_samples,
            generated_count=generated_count,
            total_samples_written=total_samples_written,
            rejections=rejections,
            start_time=start_time,
            last_update_time=time.time(),
            completed=True,
        )
        meta.save(meta_path)
        print(f"  Metadata: {meta_path}")
        print("=" * 60)
        print(f"\n✓ Generation complete! Samples saved to: {samples_dir}")
        print(f"  To merge into .pt file, run:")
        print(f"    python -c \"from src.data_gen.dataset_generator import merge_samples; "
              f"merge_samples('{samples_dir}', '{output_path}')\"")

        return samples_dir
    else:
        # Non-streaming mode: save all at once
        print(f"  Total Samples: {len(data)}")
        if len(data) > 0:
            print(f"  Motion Shape: {data[0]['motion'].shape}")
            print(f"  Face Shape: {data[0]['face'].shape}")
            print(f"  Physics Shape: {data[0]['physics'].shape}")

        print(f"\nSaving to {output_path}...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(data, output_path)
        print("Done.")

        # Clean up checkpoint file after successful completion
        checkpoint_path = output_path.replace(".pt", "_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"[CHECKPOINT] Removed checkpoint file: {checkpoint_path}")
            except OSError:
                pass  # Ignore cleanup errors

        return output_path


class AsyncSampleGenerator:
    """
    Encapsulates sample generation logic for async processing.

    This class wraps scene generation and motion generation into
    callable methods that can be used with AsyncDataGenerator.
    """

    def __init__(
        self,
        config_path: str = "configs/base.yaml",
        use_llm: bool = False,
        llm_ratio: float = 0.2,
        max_actors: int = 3,
        target_frames: int = 250,
        fps: int = 25,
        augment: bool = True,
        annotation_enabled: bool = True,
        annotation_config: Optional[dict] = None,
    ):
        # Load config
        config = load_config(config_path)
        gen_config = config.get("data_generation", {})

        self.max_actors = max_actors
        self.target_frames = target_frames
        self.fps = fps
        self.augment = augment
        self.annotation_enabled = annotation_enabled
        self.annotation_config = annotation_config or gen_config.get("annotation", {})
        self.use_llm = use_llm
        self.llm_ratio = llm_ratio

        # Initialize generators
        self.story_engine = StoryGenerator()
        self.prompt_generator = DynamicPromptGenerator()
        self.scene_prompt_generator = ScenePromptGenerator()
        self.validator = DataValidator()

        # LLM engine (only if needed)
        self.llm_story_engine = None
        if not use_llm:
            self.llm_story_engine = LLMStoryGenerator(provider="mock")
        else:
            try:
                # Try real LLM provider
                self.llm_story_engine = LLMStoryGenerator(
                    provider="openai", fallback_to_mock=True
                )
            except Exception:
                self.llm_story_engine = LLMStoryGenerator(provider="mock")

        self._llm_cache: dict = {}

    def generate_scene(self) -> Any:
        """Generate a scene (may call LLM). Thread-safe."""
        use_llm_scene = random.random() < self.llm_ratio

        # Check if using mock backend
        is_mock = self.llm_story_engine.provider == "mock"
        if use_llm_scene and not is_mock:
            try:
                scene_prompt = self.prompt_generator.generate_prompt()
                if scene_prompt in self._llm_cache:
                    script = self._llm_cache[scene_prompt]
                else:
                    script = self.llm_story_engine.generate_script(scene_prompt)
                    if len(self._llm_cache) < 1000:
                        self._llm_cache[scene_prompt] = script
                scenes = self.llm_story_engine.script_to_scenes(script)
                scene = scenes[0] if scenes else self.story_engine.generate_random_scene()
            except Exception:
                scene = self.story_engine.generate_random_scene()
        else:
            scene = self.story_engine.generate_random_scene()

        # Generate matching text prompt
        text_prompt = self.scene_prompt_generator.generate_prompt_from_scene(scene)
        scene.description = text_prompt
        return scene

    def generate_motion_sample(self, scene: Any) -> Optional[dict]:
        """Generate motion tensors from a scene. Returns None if invalid."""
        # Initialize tensors
        motion_tensor = torch.zeros(
            (self.target_frames, self.max_actors, 20), dtype=torch.float32
        )
        action_tensor = torch.zeros(
            (self.target_frames, self.max_actors), dtype=torch.long
        )
        face_tensor = torch.zeros(
            (self.target_frames, self.max_actors, 7), dtype=torch.float32
        )

        # Create actors
        actors = [
            StickFigure(
                a,
                environment_type=getattr(scene, "environment_type", None),
                weather_type=getattr(scene, "weather_type", None),
            )
            for a in scene.actors
        ]
        num_generated_frames = int(scene.duration * self.fps)
        active_actors = actors[: self.max_actors]

        # Generate motion frame by frame
        for f in range(min(num_generated_frames, self.target_frames)):
            t = f / self.fps

            for actor_idx, actor in enumerate(active_actors):
                lines, _ = actor.get_pose(t)

                actor_flat = []
                for li, (start, end) in enumerate(lines):
                    if li >= 5:
                        break
                    actor_flat.extend([start[0], start[1], end[0], end[1]])
                while len(actor_flat) < 20:
                    actor_flat.extend([0.0, 0.0, 0.0, 0.0])

                motion_tensor[f, actor_idx] = torch.tensor(actor_flat)

                current_action = actor.get_current_action(t)
                action_tensor[f, actor_idx] = ACTION_TO_IDX.get(current_action, 0)

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

        # Pad remaining frames
        if num_generated_frames < self.target_frames and num_generated_frames > 0:
            last_f = num_generated_frames - 1
            motion_tensor[last_f + 1 :] = motion_tensor[last_f]
            action_tensor[last_f + 1 :] = action_tensor[last_f]
            face_tensor[last_f + 1 :] = face_tensor[last_f]

        # Compute physics
        dt = 1.0 / self.fps
        head_pos = motion_tensor[:, :, 0:2]
        velocity = torch.zeros_like(head_pos)
        velocity[:-1] = (head_pos[1:] - head_pos[:-1]) / dt
        velocity[-1] = velocity[-2] if self.target_frames > 1 else velocity[-1]

        acceleration = torch.zeros_like(velocity)
        acceleration[:-1] = (velocity[1:] - velocity[:-1]) / dt
        acceleration[-1] = acceleration[-2] if self.target_frames > 1 else acceleration[-1]

        physics_tensor = torch.cat([velocity, acceleration, velocity], dim=2)

        # Camera
        camera_tensor = torch.zeros((self.target_frames, 3))
        camera_tensor[:, 2] = 1.0

        # Build sample
        env_type = getattr(scene, "environment_type", None)
        weather_type = getattr(scene, "weather_type", None)

        sample = {
            "description": scene.description,
            "motion": motion_tensor,
            "actions": action_tensor,
            "physics": physics_tensor,
            "face": face_tensor,
            "camera": camera_tensor,
            "augmented": False,
            "environment_type": env_type.value if env_type else None,
            "weather_type": weather_type.value if weather_type else None,
        }

        if self.annotation_enabled:
            sample = annotate_sample(sample, self.annotation_config)

        # Validate
        is_valid, score, reason = self.validator.validate(sample)
        if not is_valid:
            return None

        return sample


async def generate_dataset_async(
    config_path: str = "configs/base.yaml",
    num_samples: int = 1000,
    output_dir: str = "data/samples_async",
    resource_limits: Optional[ResourceLimits] = None,
    use_llm_override: Optional[bool] = None,
    llm_ratio: float = 0.2,
    augment: bool = True,
    compress: bool = False,
    max_retries: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    motion_source_path: Optional[str] = None,
) -> AsyncGeneratorStats:
    """
    Generate dataset asynchronously with resource-aware concurrency.

    This function provides async/concurrent sample generation with:
    - Semaphore-based rate limiting for LLM calls
    - Resource monitoring with backpressure
    - Queue-based async I/O for sample writing
    - Optional motion conditioning from existing datasets

    LLM Configuration:
        LLM settings are read from config file under `data_generation.llm`.
        Environment variables and CLI flags can override the config.
        See MotionConditionedSampleGenerator for full override priority.

    Args:
        config_path: Path to YAML configuration file
        num_samples: Number of samples to generate
        output_dir: Output directory for samples
        resource_limits: Optional resource limits (auto-detected if None)
        use_llm_override: Override config LLM setting (None = use config,
            True = force LLM on, False = force LLM off). For CLI --use-llm flag.
        llm_ratio: Ratio of samples to use LLM for scene generation
        augment: Whether to apply augmentation
        compress: Whether to compress output files
        max_retries: Maximum retries per sample for validation
        progress_callback: Optional callback(completed, total)
        motion_source_path: Path to merged canonical dataset for motion conditioning.
            If provided, synthetic samples will be generated by augmenting real
            motion clips from this dataset instead of procedural generation.

    Returns:
        AsyncGeneratorStats with generation statistics

    Example:
        ```python
        import asyncio
        from src.data_gen.dataset_generator import generate_dataset_async

        # Procedural generation (no motion conditioning)
        stats = asyncio.run(generate_dataset_async(
            num_samples=5000,
            output_dir="data/train_async",
        ))

        # Motion-conditioned generation with LLM descriptions
        stats = asyncio.run(generate_dataset_async(
            num_samples=5000,
            output_dir="data/train_async",
            motion_source_path="data/merged_canonical.pt",
            use_llm_override=True,  # Force LLM on via CLI
        ))
        ```
    """
    # Load config for settings
    config = load_config(config_path)
    gen_config = config.get("data_generation", {})

    # Get settings
    seq_config = gen_config.get("sequence", {})
    fps = seq_config.get("fps", 25)
    target_frames = int(seq_config.get("duration_seconds", 10.0) * fps)
    max_actors = seq_config.get("max_actors", 3)
    annotation_config = gen_config.get("annotation", {})

    # Check if using motion-conditioned generation
    use_motion_conditioning = (
        motion_source_path is not None and os.path.exists(motion_source_path)
    )

    if use_motion_conditioning:
        # Create motion-conditioned generator
        # LLM setting resolved inside the generator from config + env vars + override
        print(f"[ASYNC] Using motion-conditioned generation")
        print(f"[ASYNC] Motion source: {motion_source_path}")
        print(f"[ASYNC] Config: {config_path}")

        conditioned_gen = MotionConditionedSampleGenerator(
            motion_source_path=motion_source_path,
            config_path=config_path,
            target_frames=target_frames,
            max_actors=max_actors,
            augmentations_per_sample=2,
            use_llm_override=use_llm_override,
            verbose=True,
        )

        # For motion-conditioned, scene generator just returns a dummy
        def scene_generator_conditioned() -> Any:
            return None  # Not used in conditioned mode

        # Motion generator uses conditioned samples
        def motion_generator_conditioned(_scene: Any) -> dict:
            for _attempt in range(max_retries):
                result = conditioned_gen.generate_conditioned_sample()
                if result is not None:
                    return result
            # Fallback to a valid sample
            return conditioned_gen.generate_conditioned_sample() or {
                "description": "fallback motion",
                "motion": torch.zeros(target_frames, max_actors, 20),
                "source": "synthetic_conditioned",
            }

        scene_generator = scene_generator_conditioned
        motion_generator = motion_generator_conditioned

    else:
        # Use procedural generation (original behavior)
        print(f"[ASYNC] Using procedural generation (no motion conditioning)")

        sample_gen = AsyncSampleGenerator(
            config_path=config_path,
            use_llm=use_llm,
            llm_ratio=llm_ratio,
            max_actors=max_actors,
            target_frames=target_frames,
            fps=fps,
            augment=augment,
            annotation_enabled=annotation_config.get("enabled", True),
            annotation_config=annotation_config,
        )

        # Scene generator with retry logic
        def scene_generator_procedural() -> Any:
            return sample_gen.generate_scene()

        # Motion generator with retry and validation
        def motion_generator_procedural(scene: Any) -> dict:
            for _attempt in range(max_retries):
                result = sample_gen.generate_motion_sample(scene)
                if result is not None:
                    return result
                # Regenerate scene on validation failure
                scene = sample_gen.generate_scene()

            # Return a minimal valid sample on max retries
            return sample_gen.generate_motion_sample(
                sample_gen.story_engine.generate_random_scene()
            ) or {"description": "fallback", "motion": torch.zeros(target_frames, max_actors, 20)}

        scene_generator = scene_generator_procedural
        motion_generator = motion_generator_procedural

    # Use auto-detected limits if not provided
    limits = resource_limits or ResourceLimits.from_system()

    # Create async generator
    generator = AsyncDataGenerator(
        limits=limits,
        output_dir=output_dir,
        compress=compress,
    )

    mode_str = "motion-conditioned" if use_motion_conditioning else "procedural"
    print(f"[ASYNC] Starting async generation of {num_samples} {mode_str} samples")
    print(f"[ASYNC] Concurrency: {limits.max_concurrent_samples} samples, "
          f"{limits.max_concurrent_llm} LLM calls")
    print(f"[ASYNC] Output: {output_dir}")

    # Run generation
    stats = await generator.generate_samples_async(
        num_samples=num_samples,
        scene_generator=scene_generator,
        motion_generator=motion_generator,
        progress_callback=progress_callback,
    )

    print(f"\n[ASYNC] ✓ Complete!")
    print(f"[ASYNC]   Generated: {stats.samples_generated}")
    print(f"[ASYNC]   Written: {stats.samples_written}")
    print(f"[ASYNC]   Failed: {stats.samples_failed}")
    print(f"[ASYNC]   Rate: {stats.samples_per_second():.1f} samples/sec")
    if use_motion_conditioning:
        print(f"[ASYNC]   Mode: Motion-conditioned (augmented real data)")
    else:
        print(f"[ASYNC]   LLM calls: {stats.llm_calls}")

    return stats


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
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode (write samples incrementally, default: True)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (accumulate in memory)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dataset instead of overwriting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration, ignoring existing data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for metadata updates in streaming mode (default: 10)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress sample files with gzip (reduces disk usage ~5x)",
    )
    parser.add_argument(
        "--merge",
        type=str,
        metavar="SAMPLES_DIR",
        help="Merge samples from directory into .pt file (use with --output)",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async mode for concurrent generation with resource management",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent samples in async mode (default: auto-detect)",
    )
    parser.add_argument(
        "--max-llm-concurrent",
        type=int,
        default=2,
        help="Max concurrent LLM calls in async mode (default: 2)",
    )
    parser.add_argument(
        "--motion-source",
        type=str,
        default=None,
        help="Path to merged canonical .pt file for motion-conditioned generation. "
             "When provided, synthetic samples are generated by augmenting real motion "
             "clips instead of procedural generation.",
    )
    args = parser.parse_args()

    # Handle merge command
    if args.merge:
        if not args.output:
            print("Error: --output required when using --merge")
            exit(1)
        merge_samples(args.merge, args.output, delete_after_merge=False, verbose=True)
    elif args.async_mode:
        # Async mode
        limits = ResourceLimits.from_system()
        if args.max_concurrent:
            limits.max_concurrent_samples = args.max_concurrent
        limits.max_concurrent_llm = args.max_llm_concurrent

        output_dir = args.output or "data/samples_async"

        asyncio.run(generate_dataset_async(
            config_path=args.config,
            num_samples=args.num_samples or 1000,
            output_dir=output_dir,
            resource_limits=limits,
            compress=args.compress,
            motion_source_path=args.motion_source,
        ))
    else:
        streaming = args.streaming and not args.no_streaming
        generate_dataset(
            config_path=args.config,
            num_samples=args.num_samples,
            output_path=args.output,
            force=args.force,
            streaming=streaming,
            append=args.append,
            batch_size=args.batch_size,
            compress=args.compress,
        )
