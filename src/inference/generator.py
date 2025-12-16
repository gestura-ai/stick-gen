import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.model.transformer import StickFigureTransformer
from src.data_gen.schema import ActionType
from src.data_gen.renderer import Renderer, StickFigure, RenderStyle
from src.data_gen.story_engine import StoryGenerator
from src.data_gen.llm_story_engine import LLMStoryGenerator
from src.data_gen.schema import ActionType, ACTION_TO_IDX, NUM_ACTIONS
from transformers import AutoTokenizer, AutoModel
from src.inference.exporter import MotionExporter

# Phase 3: Diffusion refinement (optional)
try:
    from src.model.diffusion import (
        PoseRefinementUNet,
        DDPMScheduler,
        DiffusionRefinementModule,
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
        safety_config: Optional[SafetyCriticConfig] = None,
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
        )
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
            print(f"\nLoading Diffusion Refinement Module (Phase 3)...")
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
        physics: Optional[torch.Tensor] = None,
        quality_score: Optional[float] = None,
    ) -> Tuple[bool, Optional[SafetyCriticResult]]:
        """
        Check if generated motion passes safety/quality checks.

        Args:
            motion: Motion tensor [T, D] or [T, A, D]
            physics: Optional physics tensor
            quality_score: Optional pre-computed quality score

        Returns:
            Tuple of (is_safe, SafetyCriticResult or None if critic disabled)
        """
        if not self.enable_safety_check or self.safety_critic is None:
            return True, None

        result = self.safety_critic.evaluate(motion, physics, quality_score)
        if not result.is_safe:
            logger.warning(
                f"Motion failed safety check: {result.get_rejection_reasons()}"
            )
        return result.is_safe, result

    def generate(
        self,
        prompt: str,
        output_path: str,
        style: str = "normal",
        camera_mode: str = "static",
        use_llm: bool = False,
    ):
        """
        Generate animation from prompt

        Args:
            prompt: Text description
            output_path: Output video path
            style: Rendering style (normal, sketch, ink, neon)
            camera_mode: Camera behavior (static, dynamic)
            use_llm: Whether to use LLM for script generation
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

        # 2. Render Scene
        print(f"Rendering {scene.duration}s animation...")
        renderer = Renderer(style=style)
        renderer.render_scene(scene, output_path, camera_mode=camera_mode)

        print(f"Done! Saved to {output_path}")

    def generate_with_actions(
        self,
        prompt: str,
        action_sequence: List[ActionType],
        output_path: str = "output.mp4",
        refine: Optional[bool] = None,
        style: str = "normal",  # Added style
        camera_mode: str = "static",  # Added camera_mode
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

        # Safety check on generated motion
        if self.enable_safety_check:
            is_safe, safety_result = self.check_motion_safety(motion_sequence[0])
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
        action_timeline: Dict[int, ActionType],
        initial_action: ActionType = ActionType.IDLE,
        output_path: str = "output.mp4",
        refine: Optional[bool] = None,
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
