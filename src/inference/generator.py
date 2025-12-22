import logging
from typing import Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data_gen.llm_story_engine import LLMStoryGenerator
from src.data_gen.renderer import Renderer
from src.data_gen.schema import ACTION_TO_IDX, NUM_ACTIONS, ActionType
from src.data_gen.story_engine import StoryGenerator
from src.inference.exporter import MotionExporter
from src.model.transformer import StickFigureTransformer

# Environment-specific LoRA adapter support
try:
    from src.model.lora import (
        inject_lora_adapters,
        load_environment_adapter,
        load_environment_adapter_from_file,
        get_registered_environments,
    )
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

try:  # Optional dependency for image loading
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - handled lazily
    Image = None  # type: ignore

# Phase 3: Diffusion refinement (optional)
try:
    from src.model.diffusion import (
        DDPMScheduler,
        DiffusionRefinementModule,
        PoseRefinementUNet,
    )

    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False

# Safety critic for robustness checking
try:
    from src.eval.safety_critic import (
        SafetyCritic,
        SafetyCriticConfig,
        SafetyCriticResult,
    )

    SAFETY_CRITIC_AVAILABLE = True
except ImportError:
    SAFETY_CRITIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceGenerator:
    def __init__(
        self,
        model_path="model_checkpoint.pth",
        use_diffusion=False,
        diffusion_model_path="diffusion_unet.pth",
        diffusion_steps=20,
        enable_safety_check=False,
        safety_config: SafetyCriticConfig | None = None,
    ):
        """
        Initialize inference generator

        Args:
            model_path: Path to transformer checkpoint
            use_diffusion: Enable diffusion refinement (Phase 3)
            diffusion_model_path: Path to diffusion UNet checkpoint
            diffusion_steps: Number of denoising steps (10-50)
            enable_safety_check: Enable safety critic for output validation
            safety_config: Optional custom SafetyCriticConfig
        """
        self.input_dim = 20
        # Updated to match 11M+ parameter model architecture
        self.d_model = 384  # Increased to 384
        self.nhead = 12  # Increased to 12
        self.num_layers = 8  # Increased to 8
        self.dropout = 0.1
        self.device = torch.device("cpu")
        # Default image configuration for multimodal inference
        self.image_size: Tuple[int, int] = (256, 256)
        self.use_diffusion = use_diffusion and DIFFUSION_AVAILABLE
        self.diffusion_steps = diffusion_steps

        print(f"Loading 11M+ parameter model from {model_path}...")
        # Initialize with 11M architecture parameters + Phase 1 action conditioning
        self.model = StickFigureTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            output_dim=self.input_dim,
            embedding_dim=1024,  # BAAI/bge-large-en-v1.5
            dropout=self.dropout,
            num_actions=NUM_ACTIONS,  # Phase 1: Action conditioning
            # Enable multimodal image conditioning by default so that
            # image-to-motion inference is always available when images
            # are provided at inference time.
            enable_image_conditioning=True,
            image_encoder_arch="lightweight_cnn",
            image_size=self.image_size,
            fusion_strategy="gated",
        )
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            incompatible = self.model.load_state_dict(state_dict, strict=False)

            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(
                    "⚠️  Loaded checkpoint with key mismatches. "
                    f"Missing: {len(incompatible.missing_keys)}, "
                    f"Unexpected: {len(incompatible.unexpected_keys)}."
                )

            print("✓ 11M+ parameter model loaded successfully")
        except (FileNotFoundError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "size mismatch" in str(e):
                print(
                    "⚠️  Model architecture mismatch (old model). Using procedural animations only."
                )
            else:
                print(
                    "⚠️  Model checkpoint not found. Using procedural animations only."
                )

        self.model.to(self.device)
        self.model.eval()

        print("Loading High-Quality Embedding Model (BAAI/bge-large-en-v1.5)...")
        self.embed_model_name = "BAAI/bge-large-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_name).to("cpu")
        self.embed_model.eval()

        self.story_generator = StoryGenerator()  # Changed from self.story_engine
        self.llm_story_generator = LLMStoryGenerator()  # Added LLM story generator

        # Phase 3: Initialize diffusion refinement module (optional)
        self.diffusion_module = None
        if self.use_diffusion:
            print("\nLoading Diffusion Refinement Module (Phase 3)...")
            print(f"  - Denoising steps: {self.diffusion_steps}")
            try:
                unet = PoseRefinementUNet(
                    pose_dim=self.input_dim,
                    hidden_dims=[64, 128, 256],
                    time_emb_dim=128,
                )
                unet.load_state_dict(
                    torch.load(diffusion_model_path, map_location=self.device)
                )
                scheduler = DDPMScheduler(num_train_timesteps=1000)
                self.diffusion_module = DiffusionRefinementModule(
                    unet, scheduler, device=str(self.device)
                )
                self.diffusion_module.unet.eval()
                print("✓ Diffusion refinement module loaded successfully")
            except FileNotFoundError:
                print("⚠️  Diffusion model not found. Disabling refinement.")
                self.use_diffusion = False
            except Exception as e:
                print(f"⚠️  Error loading diffusion model: {e}. Disabling refinement.")
                self.use_diffusion = False

        # Safety critic for robustness checking
        self.enable_safety_check = enable_safety_check and SAFETY_CRITIC_AVAILABLE
        self.safety_critic = None
        if self.enable_safety_check:
            print("Initializing Safety Critic...")
            self.safety_critic = SafetyCritic(safety_config)
            print("✓ Safety critic enabled")
        elif enable_safety_check and not SAFETY_CRITIC_AVAILABLE:
            print("⚠️  Safety critic requested but not available. Disabling.")

    def check_motion_safety(
        self,
        motion: torch.Tensor,
        physics: torch.Tensor | None = None,
        quality_score: float | None = None,
        environment_type: str | None = None,
    ) -> tuple[bool, SafetyCriticResult | None]:
        """
        Check if generated motion passes safety/quality checks.

        Args:
            motion: Motion tensor [T, D] or [T, A, D]
            physics: Optional physics tensor
            quality_score: Optional pre-computed quality score
            environment_type: Optional environment type for physics-aware thresholds
                              (e.g., "underwater" uses lower velocity thresholds)

        Returns:
            Tuple of (is_safe, SafetyCriticResult or None if critic disabled)
        """
        if not self.enable_safety_check or self.safety_critic is None:
            return True, None

        result = self.safety_critic.evaluate(
            motion, physics, quality_score, environment_type=environment_type
        )
        if not result.is_safe:
            logger.warning(
                f"Motion failed safety check: {result.get_rejection_reasons()}"
            )
        return result.is_safe, result

    def load_environment_lora(
        self, environment_type: str | None = None, adapter_path: str | None = None
    ) -> tuple[bool, int]:
        """Load environment-specific LoRA adapter for specialized motion generation.

        This allows the model to generate motion that is better suited for specific
        environments (e.g., underwater motion with slower, floatier movements).

        Args:
            environment_type: Environment type to load adapter for (uses registry)
            adapter_path: Direct path to adapter file (overrides environment_type)

        Returns:
            Tuple of (success: bool, num_params_loaded: int)
        """
        if not LORA_AVAILABLE:
            logger.warning("LoRA module not available. Cannot load environment adapter.")
            return False, 0

        # First, ensure LoRA adapters are injected into the model
        # (This is idempotent if already injected)
        try:
            inject_lora_adapters(self.model, target_modules=["q_proj", "v_proj"])
        except Exception as e:
            logger.warning(f"Failed to inject LoRA adapters: {e}")
            return False, 0

        # Load adapter from path or registry
        if adapter_path:
            env_type, loaded = load_environment_adapter_from_file(
                self.model, adapter_path, device=str(self.device)
            )
            if loaded > 0:
                logger.info(
                    f"Loaded environment adapter from {adapter_path} "
                    f"(env: {env_type}, params: {loaded})"
                )
            return loaded > 0, loaded
        elif environment_type:
            success, loaded = load_environment_adapter(
                self.model, environment_type, device=str(self.device)
            )
            if success:
                logger.info(
                    f"Loaded {environment_type} environment adapter ({loaded} params)"
                )
            else:
                logger.warning(
                    f"No adapter registered for environment: {environment_type}. "
                    f"Available: {get_registered_environments()}"
                )
            return success, loaded
        else:
            logger.warning("No environment_type or adapter_path specified.")
            return False, 0

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        """Load an RGB image from disk and preprocess for the image encoder.

        The preprocessing matches the MultimodalParallaxDataset convention:
        - Convert to RGB
        - Resize to ``self.image_size``
        - Normalize to [0, 1]
        - Return a float32 tensor of shape [3, H, W]
        """

        if Image is None:
            raise RuntimeError(
                "Pillow is required for image-conditioned inference but is not installed."
            )

        h, w = self.image_size

        with Image.open(image_path) as im:  # type: ignore[attr-defined]
            im = im.convert("RGB").resize((w, h))
            arr = np.array(im, dtype="float32") / 255.0

        img = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        return img

    def _default_image_camera_pose(self, batch_size: int = 1) -> torch.Tensor:
        """Construct a simple default camera pose for single-image conditioning.

        Format: [pos_x, pos_y, pos_z, tgt_x, tgt_y, tgt_z, fov]
        """

        pose = torch.tensor(
            [0.0, 1.5, 3.0, 0.0, 1.0, 0.0, 45.0], dtype=torch.float32
        )
        return pose.unsqueeze(0).repeat(batch_size, 1)

    def generate(
        self,
        prompt: str,
        output_path: str,
        style: str = "normal",
        camera_mode: str = "static",
        use_llm: bool = False,
        environment_type: str | None = None,
    ):
        """
        Generate animation from prompt

        Args:
            prompt: Text description
            output_path: Output video path
            style: Rendering style (normal, sketch, ink, neon)
            camera_mode: Camera behavior (static, dynamic)
            use_llm: Whether to use LLM for script generation
            environment_type: Optional environment type override (e.g., "underwater", "space").
                              If None, the story engine determines the environment from the prompt.
        """
        print(
            f"Generating scene for: '{prompt}' (Style: {style}, Camera: {camera_mode})"
        )

        # 1. Generate Scene
        if use_llm:
            print("Using LLM Story Engine...")
            scene = self.llm_story_generator.generate_script(prompt)
        else:
            # Use standard story engine
            scene = self.story_generator.generate_scene_from_prompt(prompt)

        # Override environment_type if specified
        if environment_type is not None:
            from src.data_gen.schema import EnvironmentType
            try:
                scene.environment_type = EnvironmentType(environment_type)
                print(f"Using environment override: {environment_type}")
            except ValueError:
                print(f"⚠️  Unknown environment type '{environment_type}', using scene default")

        # 2. Render Scene
        print(f"Rendering {scene.duration}s animation...")
        renderer = Renderer(style=style)
        renderer.render_scene(scene, output_path, camera_mode=camera_mode)

        print(f"Done! Saved to {output_path}")

    def generate_with_actions(
        self,
        prompt: str,
        action_sequence: list[ActionType],
        output_path: str = "output.mp4",
        refine: bool | None = None,
        style: str = "normal",  # Added style
        camera_mode: str = "static",  # Added camera_mode
        environment_type: str | None = None,  # Environment for physics-aware safety checks
    ) -> str:
        """
        Phase 1: Generate animation with explicit action control.
        Phase 3: Optional diffusion refinement for smoother motion.

        Args:
            prompt: Text description of the scene
            action_sequence: List of actions, one per frame (length 250)
            output_path: Path to save output video
            refine: Apply diffusion refinement (None = use default, True/False = override)
            style: Rendering style (normal, sketch, ink, neon)
            camera_mode: Camera behavior (static, dynamic)
            environment_type: Environment type for physics-aware safety checks
                              (e.g., "underwater", "space", "moon"). If None, uses Earth-normal.

        Returns:
            Path to generated video

        Example:
            >>> generator = InferenceGenerator('checkpoint.pth', use_diffusion=True)
            >>> actions = [ActionType.WALK] * 125 + [ActionType.RUN] * 125
            >>> generator.generate_with_actions("person moving", actions, refine=True)
        """
        print(f"Generating action-conditioned animation for: '{prompt}'")
        print(f"Action sequence length: {len(action_sequence)}")

        # Determine if refinement should be applied
        apply_refinement = refine if refine is not None else self.use_diffusion

        # Validate action sequence length
        if len(action_sequence) != 250:
            raise ValueError(
                f"Action sequence must be 250 frames, got {len(action_sequence)}"
            )

        # Get text embedding
        text_embedding = self._get_text_embedding(prompt)

        # Convert actions to indices
        action_indices = torch.tensor(
            [ACTION_TO_IDX[action] for action in action_sequence],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(
            0
        )  # [1, 250]

        # Initialize with zeros (autoregressive generation)
        motion_sequence = torch.zeros(1, 250, self.input_dim, device=self.device)

        # Generate motion autoregressively
        print("Generating motion sequence...")
        with torch.no_grad():
            for t in range(1, 250):
                # Prepare input: [seq_len, batch, dim]
                motion_input = motion_sequence[:, :t, :].permute(1, 0, 2)
                action_input = action_indices[:, :t].permute(1, 0)

                # Forward pass with action conditioning
                output = self.model(
                    motion_input,
                    text_embedding,
                    return_all_outputs=False,
                    action_sequence=action_input,
                )

                # Get next frame prediction
                next_frame = output[-1, 0, :]  # Last timestep, first batch
                motion_sequence[0, t, :] = next_frame

        # Phase 3: Apply diffusion refinement if enabled
        if apply_refinement and self.diffusion_module is not None:
            print(f"Applying diffusion refinement ({self.diffusion_steps} steps)...")
            motion_sequence = self.diffusion_module.refine_poses(
                motion_sequence,
                text_embedding=text_embedding,
                num_inference_steps=self.diffusion_steps,
            )
            print("✓ Refinement complete")

        # Safety check on generated motion (environment-aware thresholds)
        if self.enable_safety_check:
            is_safe, safety_result = self.check_motion_safety(
                motion_sequence[0], environment_type=environment_type
            )
            if not is_safe:
                print(
                    f"⚠️  Motion failed safety check: {safety_result.get_rejection_reasons()}"
                )
                # Log but continue - caller can check safety_result if needed
            else:
                print("✓ Motion passed safety check")

        # Render animation (using procedural renderer for now)
        print(f"Rendering to {output_path}...")
        # TODO: Implement ML-based rendering with generated motion
        # For now, fall back to procedural rendering
        scene = self.story_generator.generate_scene_from_prompt(
            prompt
        )  # Changed from self.story_engine
        # The generated motion sequence needs to be integrated into the scene for rendering.
        # For now, we'll just pass the motion sequence to the renderer directly if it supports it,
        # or fall back to procedural if not.
        # Assuming the renderer can take a motion sequence for the main actor.
        # This part needs careful integration based on how `render_scene` expects motion data.
        # For now, let's assume `render_scene` can take a `motion_sequence` argument.
        # If not, the `TODO` comment implies this is still a placeholder.

        # For the purpose of this edit, we'll update the renderer call to match the new `generate` method's style.
        if output_path.endswith(".mp4"):
            motion_path = output_path.replace(".mp4", ".motion")
            print(f"Exporting motion data to {motion_path}...")
            exporter = MotionExporter(fps=25)
            # action_sequence is a list of Enums, we probably want strings for JSON
            action_names = [a.value for a in action_sequence]

            motion_json = exporter.export_to_json(
                motion_tensor=motion_sequence[0],  # [250, 20]
                action_names=action_names,
                description=prompt,
            )
            exporter.save(motion_json, motion_path)
            print(f"✓ Motion data saved ({len(motion_json)/1024:.1f} KB)")

        # Render animation
        print(f"Rendering to {output_path}...")
        renderer = Renderer(style=style)
        renderer.render_scene(
            scene, output_path, camera_mode=camera_mode
        )  # Added style, camera_mode

        print("Done!")
        return output_path

    def generate_with_timeline(
        self,
        prompt: str,
        action_timeline: dict[int, ActionType],
        initial_action: ActionType = ActionType.IDLE,
        output_path: str = "output.mp4",
        refine: bool | None = None,
    ) -> str:
        """
        Phase 1: Generate animation with action timeline (frame → action mapping).
        Phase 3: Optional diffusion refinement for smoother motion.

        Args:
            prompt: Text description
            action_timeline: Dict mapping frame numbers to actions
            initial_action: Action for frames not in timeline
            output_path: Output video path
            refine: Apply diffusion refinement (None = use default, True/False = override)

        Returns:
            Path to generated video

        Example:
            >>> timeline = {0: ActionType.IDLE, 50: ActionType.WALK, 150: ActionType.RUN}
            >>> generator.generate_with_timeline("person moving", timeline, refine=True)
        """
        # Build action sequence from timeline
        action_sequence = []
        current_action = initial_action

        for frame in range(250):
            if frame in action_timeline:
                current_action = action_timeline[frame]
            action_sequence.append(current_action)

        return self.generate_with_actions(
            prompt, action_sequence, output_path, refine=refine
        )

    def generate_from_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str = "output.mp4",
        refine: bool | None = None,
        style: str = "normal",
        camera_mode: str = "static",
        environment_type: str | None = None,
    ) -> str:
        """Generate an animation conditioned on a single 2.5D stick-figure image.

        This uses the StickFigureTransformer with ``enable_image_conditioning=True``
        and a default camera pose, matching the multimodal training setup.

        Args:
            image_path: Path to an RGB stick-figure image.
            prompt: Optional text description used for joint text+image conditioning.
            output_path: Path for the rendered video (and derived .motion file).
            refine: Apply diffusion refinement (None = use default, True/False = override).
            style: Rendering style (normal, sketch, ink, neon).
            camera_mode: Camera behavior (static, dynamic).
            environment_type: Environment type for physics-aware safety checks
                              (e.g., "underwater", "space", "moon"). If None, uses Earth-normal.

        Returns:
            Path to the generated video (or motion file if non-MP4).
        """

        print(
            f"Generating image-conditioned animation for '{prompt}' "
            f"from image '{image_path}'"
        )

        # Determine if refinement should be applied
        apply_refinement = refine if refine is not None else self.use_diffusion

        # Encode text prompt
        text_embedding = self._get_text_embedding(prompt)

        # Load and preprocess image -> [1, 3, H, W]
        image_tensor = self._load_image_tensor(image_path).unsqueeze(0).to(self.device)

        # Default camera pose for the conditioning image -> [1, 7]
        image_camera_pose = self._default_image_camera_pose(batch_size=1).to(
            self.device
        )

        # Initialize motion sequence [1, 250, 20]
        seq_len = 250
        motion_sequence = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        print("Generating motion sequence with image conditioning...")
        with torch.no_grad():
            for t in range(1, seq_len):
                motion_input = motion_sequence[:, :t, :].permute(1, 0, 2)

                output = self.model(
                    motion_input,
                    text_embedding,
                    return_all_outputs=False,
                    image_tensor=image_tensor,
                    image_camera_pose=image_camera_pose,
                )

                next_frame = output[-1, 0, :]
                motion_sequence[0, t, :] = next_frame

        # Optional diffusion refinement
        if apply_refinement and self.diffusion_module is not None:
            print(f"Applying diffusion refinement ({self.diffusion_steps} steps)...")
            motion_sequence = self.diffusion_module.refine_poses(
                motion_sequence,
                text_embedding=text_embedding,
                num_inference_steps=self.diffusion_steps,
            )
            print("✓ Refinement complete")

        # Safety check on generated motion (environment-aware thresholds)
        if self.enable_safety_check:
            is_safe, safety_result = self.check_motion_safety(
                motion_sequence[0], environment_type=environment_type
            )
            if not is_safe:
                print(
                    "⚠️  Motion failed safety check: "
                    f"{safety_result.get_rejection_reasons()}"
                )
            else:
                print("✓ Motion passed safety check")

        # Export motion sequence to .motion JSON
        if output_path.endswith(".mp4"):
            motion_path = output_path.replace(".mp4", ".motion")
        else:
            # If not an MP4 path, treat output_path itself as the motion file
            motion_path = output_path

        print(f"Exporting motion data to {motion_path}...")
        exporter = MotionExporter(fps=25)
        motion_json = exporter.export_to_json(
            motion_tensor=motion_sequence[0],
            action_names=None,
            description=prompt,
        )
        exporter.save(motion_json, motion_path)
        print(f"✓ Motion data saved ({len(motion_json) / 1024:.1f} KB)")

        # Render animation video if an MP4 path is provided
        if output_path.endswith(".mp4"):
            print(f"Rendering to {output_path}...")
            scene = self.story_generator.generate_scene_from_prompt(prompt)
            renderer = Renderer(style=style)
            renderer.render_scene(scene, output_path, camera_mode=camera_mode)
            print("Done!")

        return output_path

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get text embedding using BAAI/bge-large-en-v1.5"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]

        return embedding
