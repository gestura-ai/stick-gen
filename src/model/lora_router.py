"""
MultiLoRARouter: Dynamic text-conditioned routing between LoRA experts.

This module provides:
- MultiLoRARouter: Routes between multiple LoRA experts based on text embeddings
- ExpertConfig: Configuration for individual experts (style vs orthogonal)
- Support for cosine similarity routing with temperature-controlled softmax

The router supports two expert types:
1. Style Experts: Mutually exclusive, routed via softmax (dramatic, action, expressive_body, multi_actor)
2. Orthogonal Experts: Always-on with learned gating (camera, timing)

Reference: Mixture of LoRA Experts (MoLE) pattern
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for a single LoRA expert.

    Attributes:
        name: Expert identifier (e.g., "dramatic_style", "camera")
        checkpoint_path: Path to the LoRA checkpoint file
        expert_type: "style" (routed via softmax) or "orthogonal" (always active with gating)
        target_phase: Which model phase this expert targets ("both", "transformer_only", "diffusion_only")
        prototype_prompts: Representative prompts used to compute domain prototype embedding
    """

    name: str
    checkpoint_path: str
    expert_type: str = "style"  # "style" or "orthogonal"
    target_phase: str = "both"
    prototype_prompts: list[str] = field(default_factory=list)


# Default prototype prompts for each expert domain
DEFAULT_PROTOTYPES: dict[str, list[str]] = {
    "dramatic_style": [
        "A slow, emotional scene with a character expressing deep sadness",
        "Dramatic pause before revealing important information",
        "Intense confrontation with measured, deliberate movements",
        "Romantic slow dance under moonlight",
        "Character processes grief with restrained body language",
    ],
    "action_style": [
        "Fast-paced chase sequence through an urban environment",
        "Explosive fight scene with martial arts moves",
        "Athletic character performing parkour jumps",
        "High-energy sports competition with dynamic movements",
        "Quick action sequence with rapid cuts and movements",
    ],
    "expressive_body": [
        "Character uses exaggerated gestures while telling a story",
        "Animated conversation with natural body language",
        "Expressive dance with full body movements",
        "Character acts out a pantomime scene",
        "Physical comedy with exaggerated reactions",
    ],
    "multi_actor": [
        "Two characters engaged in synchronized dance",
        "Group conversation with multiple speakers",
        "Team performing coordinated action sequence",
        "Pair of characters in a physical altercation",
        "Multiple actors moving together in formation",
    ],
    "camera": [
        "Cinematic wide shot establishing the scene",
        "Close-up on character's face during emotional moment",
        "Dynamic tracking shot following the action",
        "Sweeping pan across the landscape",
        "Dutch angle for dramatic tension",
    ],
    "timing": [
        "Dramatic pause before the reveal",
        "Quick cuts during action sequence",
        "Slow motion moment of impact",
        "Comedic timing with beat pause",
        "Rhythmic editing matching the music",
    ],
}


class MultiLoRARouter(nn.Module):
    """Dynamic router for multiple LoRA experts based on text conditioning.

    Routes input prompts to appropriate LoRA experts using cosine similarity
    between the prompt embedding and pre-computed domain prototype embeddings.

    Style experts (dramatic, action, expressive_body, multi_actor) are routed
    via temperature-controlled softmax for soft blending.

    Orthogonal experts (camera, timing) use learned gating that's always active.

    Example:
        >>> router = MultiLoRARouter(embed_dim=1024)
        >>> router.register_expert(ExpertConfig(name="dramatic_style", ...))
        >>> router.compute_prototype_embeddings(text_encoder)
        >>> weights = router(text_embedding)  # {"dramatic_style": 0.7, "action_style": 0.3, ...}
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        temperature: float = 0.1,
        min_expert_weight: float = 0.0,
    ):
        """Initialize the MultiLoRARouter.

        Args:
            embed_dim: Dimension of text embeddings (1024 for BAAI/bge-large-en-v1.5)
            temperature: Temperature for softmax routing (lower = sharper)
            min_expert_weight: Minimum weight threshold for expert activation
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.min_expert_weight = min_expert_weight

        # Expert registries
        self.style_experts: dict[str, ExpertConfig] = {}
        self.orthogonal_experts: dict[str, ExpertConfig] = {}

        # Prototype embeddings (computed later)
        self.register_buffer("style_prototypes", torch.zeros(0, embed_dim))
        self.style_expert_names: list[str] = []

        # Orthogonal expert gating (learned)
        self.orthogonal_gates: nn.ModuleDict = nn.ModuleDict()

        # Track if prototypes are initialized
        self._prototypes_initialized = False

    def register_expert(self, config: ExpertConfig) -> None:
        """Register a new LoRA expert with the router.

        Args:
            config: Expert configuration
        """
        if config.expert_type == "style":
            self.style_experts[config.name] = config
            self.style_expert_names.append(config.name)
            logger.info(f"Registered style expert: {config.name}")
        elif config.expert_type == "orthogonal":
            self.orthogonal_experts[config.name] = config
            # Add learned gating for orthogonal expert
            self.orthogonal_gates[config.name] = nn.Sequential(
                nn.Linear(self.embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            logger.info(f"Registered orthogonal expert: {config.name}")
        else:
            raise ValueError(f"Unknown expert_type: {config.expert_type}")

    @torch.no_grad()
    def compute_prototype_embeddings(
        self,
        text_encoder: nn.Module,
        tokenizer,
        device: str = "cpu",
    ) -> None:
        """Compute prototype embeddings for all style experts.

        Uses the text encoder to embed representative prompts for each domain,
        then averages them to create the domain prototype.

        Args:
            text_encoder: The text encoder model (e.g., BAAI/bge-large-en-v1.5)
            tokenizer: Tokenizer for the text encoder
            device: Device to run encoding on
        """
        if not self.style_experts:
            logger.warning("No style experts registered, skipping prototype computation")
            return

        text_encoder.eval()
        prototypes = []

        for expert_name in self.style_expert_names:
            config = self.style_experts[expert_name]
            prompts = config.prototype_prompts or DEFAULT_PROTOTYPES.get(expert_name, [])

            if not prompts:
                logger.warning(f"No prototype prompts for {expert_name}, using zero vector")
                prototypes.append(torch.zeros(self.embed_dim, device=device))
                continue

            # Encode all prototype prompts
            prompt_embeddings = []
            for prompt in prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = text_encoder(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                prompt_embeddings.append(embedding)

            # Average embeddings to get prototype
            prototype = torch.stack(prompt_embeddings).mean(dim=0)
            prototype = F.normalize(prototype, dim=-1)  # L2 normalize
            prototypes.append(prototype)

            logger.info(f"Computed prototype for {expert_name} from {len(prompts)} prompts")

        # Store as buffer
        self.style_prototypes = torch.stack(prototypes)  # [num_experts, embed_dim]
        self._prototypes_initialized = True
        logger.info(f"Initialized {len(prototypes)} style expert prototypes")

    def _compute_style_weights(self, text_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute routing weights for style experts via cosine similarity.

        Args:
            text_embedding: Input text embedding [batch, embed_dim]

        Returns:
            Dictionary mapping expert names to routing weights [batch, 1]
        """
        if not self._prototypes_initialized or len(self.style_expert_names) == 0:
            return {}

        # Normalize input embedding
        text_norm = F.normalize(text_embedding, dim=-1)  # [batch, embed_dim]

        # Cosine similarity with all prototypes
        # [batch, embed_dim] @ [embed_dim, num_experts] -> [batch, num_experts]
        similarities = text_norm @ self.style_prototypes.T

        # Temperature-scaled softmax
        weights = F.softmax(similarities / self.temperature, dim=-1)

        # Convert to dictionary
        result = {}
        for i, name in enumerate(self.style_expert_names):
            weight = weights[:, i : i + 1]  # [batch, 1]
            if weight.mean() >= self.min_expert_weight:
                result[name] = weight

        return result

    def _compute_orthogonal_weights(
        self, text_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute gating weights for orthogonal experts.

        Args:
            text_embedding: Input text embedding [batch, embed_dim]

        Returns:
            Dictionary mapping expert names to gating weights [batch, 1]
        """
        result = {}
        for name, gate in self.orthogonal_gates.items():
            weight = gate(text_embedding)  # [batch, 1]
            result[name] = weight
        return result

    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor | None = None,  # noqa: ARG002 - reserved for future multimodal routing
    ) -> dict[str, torch.Tensor]:
        """Compute routing weights for all experts.

        Args:
            text_embedding: Input text embedding [batch, embed_dim]
            image_embedding: Optional image embedding [batch, embed_dim] (reserved for future use)

        Returns:
            Dictionary mapping expert names to routing weights [batch, 1]
            Style expert weights sum to 1.0, orthogonal weights are independent [0, 1]
        """
        weights = {}

        # Style experts: mutually exclusive via softmax
        style_weights = self._compute_style_weights(text_embedding)
        weights.update(style_weights)

        # Orthogonal experts: independent gating
        orthogonal_weights = self._compute_orthogonal_weights(text_embedding)
        weights.update(orthogonal_weights)

        return weights

    def get_active_experts(
        self, weights: dict[str, torch.Tensor], threshold: float = 0.1
    ) -> list[str]:
        """Get list of experts with weight above threshold.

        Args:
            weights: Routing weights from forward()
            threshold: Minimum weight to consider expert active

        Returns:
            List of active expert names
        """
        active = []
        for name, weight in weights.items():
            if weight.mean().item() >= threshold:
                active.append(name)
        return active

    def get_expert_config(self, name: str) -> ExpertConfig | None:
        """Get configuration for a specific expert.

        Args:
            name: Expert name

        Returns:
            ExpertConfig or None if not found
        """
        if name in self.style_experts:
            return self.style_experts[name]
        if name in self.orthogonal_experts:
            return self.orthogonal_experts[name]
        return None

    @property
    def all_experts(self) -> dict[str, ExpertConfig]:
        """Get all registered experts."""
        return {**self.style_experts, **self.orthogonal_experts}

    @property
    def num_experts(self) -> int:
        """Total number of registered experts."""
        return len(self.style_experts) + len(self.orthogonal_experts)

    def state_dict_with_prototypes(self) -> dict:
        """Get state dict including prototype embeddings.

        Returns:
            State dict with model parameters and prototype metadata
        """
        state = self.state_dict()
        state["_style_expert_names"] = self.style_expert_names
        state["_prototypes_initialized"] = self._prototypes_initialized
        return state

    def load_state_dict_with_prototypes(self, state: dict) -> None:
        """Load state dict including prototype embeddings.

        Args:
            state: State dict from state_dict_with_prototypes()
        """
        self.style_expert_names = state.pop("_style_expert_names", [])
        self._prototypes_initialized = state.pop("_prototypes_initialized", False)
        self.load_state_dict(state)
