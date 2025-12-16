"""
Feature fusion module for multimodal conditioning.

Combines text embeddings, image features, and camera pose into a unified
conditioning signal for the motion transformer.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Concatenation-based feature fusion.

    Concatenates text, image, and camera features, then projects
    to the target dimension.
    """

    def __init__(
        self,
        text_dim: int = 1024,
        image_dim: int = 1024,
        camera_dim: int = 7,
        output_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        total_dim = text_dim + image_dim + camera_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Learned modality weights for weighted averaging fallback
        self.text_weight = nn.Parameter(torch.ones(1))
        self.image_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multimodal features.

        Args:
            text_features: [batch, text_dim] text embedding
            image_features: Optional [batch, image_dim] image features
            camera_pose: Optional [batch, 7] camera pose (xyz pos, xyz target, fov)

        Returns:
            Fused features [batch, output_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device

        # Handle missing modalities with zeros
        if image_features is None:
            image_features = torch.zeros(
                batch_size, self.projection[0].in_features - 1024 - 7,
                device=device, dtype=text_features.dtype
            )

        if camera_pose is None:
            camera_pose = torch.zeros(
                batch_size, 7, device=device, dtype=text_features.dtype
            )

        # Concatenate all features
        combined = torch.cat([text_features, image_features, camera_pose], dim=-1)
        return self.projection(combined)


class GatedFusion(nn.Module):
    """Gated feature fusion with learned modality gates.

    Uses sigmoid gates to dynamically weight each modality's contribution.
    """

    def __init__(
        self,
        text_dim: int = 1024,
        image_dim: int = 1024,
        camera_dim: int = 7,
        output_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        # Project each modality to output_dim
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.camera_proj = nn.Sequential(
            nn.Linear(camera_dim, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, output_dim),
        )

        # Gating networks
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, 1),
            nn.Sigmoid(),
        )
        self.image_gate = nn.Sequential(
            nn.Linear(image_dim, 1),
            nn.Sigmoid(),
        )
        self.camera_gate = nn.Sequential(
            nn.Linear(camera_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multimodal features with gating.

        Args:
            text_features: [batch, text_dim]
            image_features: Optional [batch, image_dim]
            camera_pose: Optional [batch, 7]

        Returns:
            Fused features [batch, output_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        dtype = text_features.dtype

        # Project and gate text (always present)
        text_proj = self.text_proj(text_features)
        text_gate = self.text_gate(text_features)
        fused = text_proj * text_gate

        # Project and gate image if present
        if image_features is not None:
            image_proj = self.image_proj(image_features)
            image_gate = self.image_gate(image_features)
            fused = fused + image_proj * image_gate

        # Project and gate camera if present
        if camera_pose is not None:
            camera_proj = self.camera_proj(camera_pose)
            camera_gate = self.camera_gate(camera_pose)
            fused = fused + camera_proj * camera_gate

        return self.dropout(self.norm(fused))


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) fusion.

    Image and camera features modulate text features via learned
    scale (gamma) and shift (beta) parameters.

    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(
        self,
        text_dim: int = 1024,
        image_dim: int = 1024,
        camera_dim: int = 7,
        output_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        # Project text to output_dim
        self.text_proj = nn.Linear(text_dim, output_dim)

        # FiLM generators from image features
        self.image_gamma = nn.Linear(image_dim, output_dim)
        self.image_beta = nn.Linear(image_dim, output_dim)

        # FiLM generators from camera pose
        conditioning_dim = camera_dim
        self.camera_gamma = nn.Sequential(
            nn.Linear(conditioning_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
        )
        self.camera_beta = nn.Sequential(
            nn.Linear(conditioning_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize FiLM parameters to identity transform
        nn.init.ones_(self.image_gamma.weight.data[:, :min(image_dim, output_dim)])
        nn.init.zeros_(self.image_gamma.bias.data)
        nn.init.zeros_(self.image_beta.weight.data)
        nn.init.zeros_(self.image_beta.bias.data)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            text_features: [batch, text_dim]
            image_features: Optional [batch, image_dim]
            camera_pose: Optional [batch, 7]

        Returns:
            Modulated features [batch, output_dim]
        """
        # Project text features
        x = self.text_proj(text_features)

        # Apply image FiLM if present
        if image_features is not None:
            gamma_img = self.image_gamma(image_features)
            beta_img = self.image_beta(image_features)
            x = gamma_img * x + beta_img

        # Apply camera FiLM if present
        if camera_pose is not None:
            gamma_cam = self.camera_gamma(camera_pose)
            beta_cam = self.camera_beta(camera_pose)
            x = gamma_cam * x + beta_cam

        return self.dropout(self.norm(x))


class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion.

    Text features attend to image and camera features via cross-attention.
    """

    def __init__(
        self,
        text_dim: int = 1024,
        image_dim: int = 1024,
        camera_dim: int = 7,
        output_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        # Project all modalities to output_dim
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.camera_proj = nn.Sequential(
            nn.Linear(camera_dim, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, output_dim),
        )

        # Cross-attention: text queries, image+camera as key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention fusion.

        Args:
            text_features: [batch, text_dim]
            image_features: Optional [batch, image_dim]
            camera_pose: Optional [batch, 7]

        Returns:
            Fused features [batch, output_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device

        # Project text to query
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [batch, 1, output_dim]

        # Build key/value sequence from available modalities
        kv_list = [text_proj]  # Always include text as self-reference

        if image_features is not None:
            image_proj = self.image_proj(image_features).unsqueeze(1)
            kv_list.append(image_proj)

        if camera_pose is not None:
            camera_proj = self.camera_proj(camera_pose).unsqueeze(1)
            kv_list.append(camera_proj)

        kv = torch.cat(kv_list, dim=1)  # [batch, num_modalities, output_dim]

        # Cross-attention
        attn_out, _ = self.cross_attn(text_proj, kv, kv)
        x = self.norm1(text_proj + attn_out)

        # FFN
        x = self.norm2(x + self.ffn(x))

        return x.squeeze(1)  # [batch, output_dim]


# Registry of fusion strategies
FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "concat": ConcatFusion,
    "gated": GatedFusion,
    "film": FiLMFusion,
    "cross_attention": CrossAttentionFusion,
}


def create_fusion_module(
    strategy: Literal["concat", "gated", "film", "cross_attention"] = "gated",
    text_dim: int = 1024,
    image_dim: int = 1024,
    camera_dim: int = 7,
    output_dim: int = 1024,
    **kwargs,
) -> nn.Module:
    """Factory function to create fusion modules.

    Args:
        strategy: Fusion strategy name.
        text_dim: Text embedding dimension.
        image_dim: Image feature dimension.
        camera_dim: Camera pose dimension.
        output_dim: Output dimension.
        **kwargs: Additional strategy-specific arguments.

    Returns:
        Instantiated fusion module.

    Example:
        >>> fusion = create_fusion_module("gated", output_dim=1024)
        >>> fused = fusion(text_emb, image_features, camera_pose)
    """
    if strategy not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion strategy: {strategy}. "
            f"Available: {list(FUSION_REGISTRY.keys())}"
        )

    fusion_cls = FUSION_REGISTRY[strategy]
    return fusion_cls(
        text_dim=text_dim,
        image_dim=image_dim,
        camera_dim=camera_dim,
        output_dim=output_dim,
        **kwargs,
    )

