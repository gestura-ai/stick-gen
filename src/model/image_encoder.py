"""
Image encoder for 2.5D parallax frames in multimodal training.

Provides lightweight CNN and ViT-based encoders for processing stick-figure
PNG renders into feature vectors compatible with text embeddings.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class LightweightCNN(nn.Module):
    """Lightweight CNN encoder for stick-figure images.

    A simple 5-layer CNN that progressively downsamples the input image
    and produces a feature vector. Designed to be fast and parameter-efficient.

    Architecture:
        - 4 conv blocks with stride-2 downsampling
        - Global average pooling
        - Linear projection to output_dim

    For 256x256 input: 256 -> 128 -> 64 -> 32 -> 16 -> GAP -> output_dim
    """

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 1024,
        base_channels: int = 32,
    ) -> None:
        """Initialize the CNN encoder.

        Args:
            in_channels: Number of input channels (3 for RGB).
            output_dim: Output feature dimension (matches text embedding dim).
            base_channels: Base number of channels (doubled at each layer).
        """
        super().__init__()

        c = base_channels
        self.encoder = nn.Sequential(
            # 256 -> 128
            ConvBlock(in_channels, c, kernel_size=7, stride=2, padding=3),
            # 128 -> 64
            ConvBlock(c, c * 2, kernel_size=3, stride=2, padding=1),
            # 64 -> 32
            ConvBlock(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            # 32 -> 16
            ConvBlock(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            # 16 -> 8
            ConvBlock(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 16, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor [batch, 3, H, W]

        Returns:
            Feature vector [batch, output_dim]
        """
        features = self.encoder(x)
        pooled = self.pool(features)
        return self.projection(pooled)


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class ResNetEncoder(nn.Module):
    """ResNet-style encoder for stick-figure images.

    A deeper encoder using residual connections for better gradient flow.
    Based on ResNet-18 architecture but with configurable width.
    """

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 1024,
        base_channels: int = 64,
    ) -> None:
        """Initialize the ResNet encoder.

        Args:
            in_channels: Number of input channels (3 for RGB).
            output_dim: Output feature dimension.
            base_channels: Base number of channels.
        """
        super().__init__()

        c = base_channels

        # Initial convolution (256 -> 128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, 2, 3, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1),  # 128 -> 64
        )

        # Residual stages
        self.stage1 = self._make_stage(c, c, 2, stride=1)  # 64 -> 64
        self.stage2 = self._make_stage(c, c * 2, 2, stride=2)  # 64 -> 32
        self.stage3 = self._make_stage(c * 2, c * 4, 2, stride=2)  # 32 -> 16
        self.stage4 = self._make_stage(c * 4, c * 8, 2, stride=2)  # 16 -> 8

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor [batch, 3, H, W]

        Returns:
            Feature vector [batch, output_dim]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.projection(x)


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, C, H, W] -> [batch, num_patches, embed_dim]
        x = self.projection(x)  # [batch, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x


class MiniViT(nn.Module):
    """Minimal Vision Transformer encoder for stick-figure images.

    A lightweight ViT with fewer layers and smaller dimensions,
    suitable for encoding simple stick-figure renders.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        output_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Mini ViT encoder.

        Args:
            img_size: Input image size (assumes square).
            patch_size: Size of each patch.
            in_channels: Number of input channels.
            embed_dim: Transformer embedding dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            output_dim: Output feature dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, output_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor [batch, 3, H, W]

        Returns:
            Feature vector [batch, output_dim]
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches+1, embed_dim]

        # Add position embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)

        # Use [CLS] token output
        cls_output = x[:, 0]
        return self.projection(cls_output)


# Registry of available encoders
ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "lightweight_cnn": LightweightCNN,
    "resnet": ResNetEncoder,
    "mini_vit": MiniViT,
}


def create_image_encoder(
    architecture: Literal["lightweight_cnn", "resnet", "mini_vit"] = "lightweight_cnn",
    output_dim: int = 1024,
    **kwargs,
) -> nn.Module:
    """Factory function to create image encoders.

    Args:
        architecture: Encoder architecture name.
        output_dim: Output feature dimension (should match text embedding dim).
        **kwargs: Additional architecture-specific arguments.

    Returns:
        Instantiated image encoder module.

    Example:
        >>> encoder = create_image_encoder("resnet", output_dim=1024)
        >>> features = encoder(images)  # [batch, 1024]
    """
    if architecture not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )

    encoder_cls = ENCODER_REGISTRY[architecture]
    return encoder_cls(output_dim=output_dim, **kwargs)
