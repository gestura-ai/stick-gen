"""
Phase 3: Diffusion Refinement Module

Lightweight diffusion model to refine transformer outputs for smoother motion.
Based on DDPM (Denoising Diffusion Probabilistic Models) with temporal convolutions.

Architecture:
- Input: Noisy poses [batch, seq_len, 20] (5 lines × 4 coords)
- Output: Noise prediction [batch, seq_len, 20]
- Model: 1D UNet with temporal convolutions
- Conditioning: Optional style/emotion metadata for controlled generation
- Parameters: <5M for efficiency

References:
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Style Conditioning
# =============================================================================


@dataclass
class StyleCondition:
    """Container for style conditioning signals.

    All fields are optional - missing values use learned defaults.
    """

    tempo: float | None = None  # 0-1, motion speed
    energy_level: float | None = None  # 0-1, motion intensity
    smoothness: float | None = None  # 0-1, jerk-based smoothness
    valence: float | None = None  # -1 to 1, emotional positivity
    arousal: float | None = None  # 0-1, emotional activation

    @classmethod
    def from_enhanced_meta(cls, enhanced_meta: dict[str, Any] | None) -> "StyleCondition":
        """Create StyleCondition from enhanced_meta dict.

        Args:
            enhanced_meta: Enhanced metadata dict from sample, or None.

        Returns:
            StyleCondition with available fields populated.
        """
        if enhanced_meta is None:
            return cls()

        motion_style = enhanced_meta.get("motion_style") or {}
        emotion = enhanced_meta.get("emotion") or {}

        return cls(
            tempo=motion_style.get("tempo"),
            energy_level=motion_style.get("energy_level"),
            smoothness=motion_style.get("smoothness"),
            valence=emotion.get("valence"),
            arousal=emotion.get("arousal"),
        )

    def to_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Convert to tensor, using -1 for missing values.

        Returns:
            Tensor of shape [5] with values or -1 for missing.
        """
        values = [
            self.tempo if self.tempo is not None else -1.0,
            self.energy_level if self.energy_level is not None else -1.0,
            self.smoothness if self.smoothness is not None else -1.0,
            self.valence if self.valence is not None else -1.0,
            self.arousal if self.arousal is not None else -1.0,
        ]
        return torch.tensor(values, dtype=torch.float32, device=device)


class StyleConditioningModule(nn.Module):
    """Encode style metadata into conditioning embeddings.

    Handles missing values gracefully using learned default embeddings.
    Uses FiLM (Feature-wise Linear Modulation) for conditioning.
    """

    STYLE_DIM = 5  # tempo, energy, smoothness, valence, arousal

    def __init__(self, output_dim: int = 128, dropout: float = 0.1):
        """Initialize style conditioning module.

        Args:
            output_dim: Dimension of output conditioning embedding.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.output_dim = output_dim

        # Learned default embedding for missing/null conditioning
        self.null_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

        # MLP to project style features to embedding
        self.style_mlp = nn.Sequential(
            nn.Linear(self.STYLE_DIM, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Mask embedding for handling missing values
        self.mask_mlp = nn.Sequential(
            nn.Linear(self.STYLE_DIM, output_dim // 2),
            nn.SiLU(),
            nn.Linear(output_dim // 2, output_dim),
        )

    def forward(
        self,
        style_conditions: list[StyleCondition] | None = None,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Encode style conditions to embeddings.

        Args:
            style_conditions: List of StyleCondition, one per batch item.
                If None, returns null embeddings for entire batch.
            batch_size: Batch size (used when style_conditions is None).
            device: Device for output tensor.

        Returns:
            Style embeddings [batch, output_dim].
        """
        if style_conditions is None:
            # Return null embedding for entire batch
            return self.null_embedding.unsqueeze(0).expand(batch_size, -1)

        # Stack style tensors
        style_tensors = torch.stack(
            [sc.to_tensor(device) for sc in style_conditions], dim=0
        )  # [batch, 5]

        # Create mask for valid (non-missing) values
        valid_mask = (style_tensors >= 0).float()  # [batch, 5]

        # Replace -1 with 0.5 (neutral default) for MLP input
        style_input = torch.where(
            style_tensors >= 0, style_tensors, torch.full_like(style_tensors, 0.5)
        )

        # Compute style embedding
        style_emb = self.style_mlp(style_input)  # [batch, output_dim]

        # Compute mask-aware adjustment
        mask_emb = self.mask_mlp(valid_mask)  # [batch, output_dim]

        # Combine: use mask embedding to weight between style and null
        has_any_valid = valid_mask.sum(dim=-1, keepdim=True) > 0  # [batch, 1]
        combined = torch.where(
            has_any_valid,
            style_emb + mask_emb,
            self.null_embedding.unsqueeze(0).expand(len(style_conditions), -1),
        )

        return combined


class DDPMScheduler:
    """
    DDPM noise scheduler for diffusion process

    Implements linear beta schedule for adding/removing noise
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Timesteps for inference (will be set by set_timesteps)
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        """
        Set timesteps for inference (DDIM-style uniform spacing)

        Args:
            num_inference_steps: Number of denoising steps
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step_ratio).flip(0)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to timestep

        Args:
            original_samples: [batch, seq_len, dim]
            noise: [batch, seq_len, dim]
            timesteps: [batch] - timestep indices

        Returns:
            Noisy samples [batch, seq_len, dim]
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting: [batch, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion step: predict x_{t-1} from x_t

        Args:
            model_output: Predicted noise [batch, seq_len, dim]
            timestep: Current timestep
            sample: Current noisy sample x_t [batch, seq_len, dim]

        Returns:
            Previous sample x_{t-1} [batch, seq_len, dim]
        """
        t = timestep

        # Compute coefficients
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t

        # Predict original sample from noise
        pred_original_sample = (
            sample - torch.sqrt(beta_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)

        # Compute variance
        variance = self.posterior_variance[t]

        # Compute previous sample mean
        pred_sample_direction = (
            torch.sqrt(1 - alpha_prod_t_prev - variance) * model_output
        )
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        # Add noise if not final step
        if t > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + torch.sqrt(variance) * noise

        return prev_sample


class TemporalConvBlock(nn.Module):
    """
    1D Temporal Convolution Block with residual connection
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, channels, seq_len]
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return self.activation(out + residual)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch] - timestep indices
        Returns:
            [batch, dim] - time embeddings
        """
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # MLP projection
        emb = self.mlp(emb)
        return emb


class PoseRefinementUNet(nn.Module):
    """
    Lightweight 1D UNet for pose sequence refinement with style conditioning.

    Architecture:
    - Encoder: 3 downsampling blocks (250 -> 125 -> 62 -> 31 frames)
    - Bottleneck: 2 conv blocks with FiLM conditioning
    - Decoder: 3 upsampling blocks with skip connections
    - Style Conditioning: FiLM modulation from StyleConditioningModule
    - Total params: ~3-4M
    """

    def __init__(
        self,
        pose_dim: int = 20,  # 5 lines × 4 coords
        hidden_dims: list | None = None,
        time_emb_dim: int = 128,
        style_emb_dim: int = 128,
        use_style_conditioning: bool = True,
    ):
        """Initialize PoseRefinementUNet.

        Args:
            pose_dim: Dimension of pose features (default 20 for stick figures).
            hidden_dims: Hidden dimensions for encoder/decoder blocks.
            time_emb_dim: Dimension of time embeddings.
            style_emb_dim: Dimension of style conditioning embeddings.
            use_style_conditioning: Whether to enable style conditioning.
        """
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        super().__init__()
        self.pose_dim = pose_dim
        self.use_style_conditioning = use_style_conditioning
        self.style_emb_dim = style_emb_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Style conditioning module (optional)
        if use_style_conditioning:
            self.style_conditioning = StyleConditioningModule(
                output_dim=style_emb_dim, dropout=0.1
            )
            # Combined conditioning dimension
            cond_dim = time_emb_dim + style_emb_dim
        else:
            self.style_conditioning = None
            cond_dim = time_emb_dim

        # Initial projection
        self.init_conv = nn.Conv1d(pose_dim, hidden_dims[0], kernel_size=3, padding=1)

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        in_ch = hidden_dims[0]
        for out_ch in hidden_dims:
            self.encoder_blocks.append(TemporalConvBlock(in_ch, out_ch))
            self.downsample_blocks.append(
                nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            TemporalConvBlock(hidden_dims[-1], hidden_dims[-1]),
            TemporalConvBlock(hidden_dims[-1], hidden_dims[-1]),
        )

        # FiLM modulation for bottleneck (style-aware)
        if use_style_conditioning:
            self.bottleneck_film_gamma = nn.Linear(cond_dim, hidden_dims[-1])
            self.bottleneck_film_beta = nn.Linear(cond_dim, hidden_dims[-1])

        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims)):
            in_ch = reversed_dims[i]
            out_ch = (
                reversed_dims[i + 1] if i + 1 < len(reversed_dims) else hidden_dims[0]
            )

            # Skip connection doubles input channels
            self.decoder_blocks.append(TemporalConvBlock(in_ch * 2, out_ch))
            self.upsample_blocks.append(
                nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
            )

        # Final projection
        self.final_conv = nn.Conv1d(hidden_dims[0], pose_dim, kernel_size=3, padding=1)

        # Time + style conditioning (inject into each encoder block)
        self.cond_mlps = nn.ModuleList(
            [nn.Linear(cond_dim, dim) for dim in hidden_dims]
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        style_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional style conditioning.

        Args:
            x: Noisy poses [batch, seq_len, pose_dim].
            timesteps: Timestep indices [batch].
            style_embedding: Optional style embedding [batch, style_emb_dim].
                If None and use_style_conditioning=True, uses null embedding.

        Returns:
            Predicted noise [batch, seq_len, pose_dim].
        """
        batch_size = x.shape[0]

        # Transpose to [batch, pose_dim, seq_len] for Conv1d
        x = x.transpose(1, 2)

        # Time embedding
        t_emb = self.time_embedding(timesteps)  # [batch, time_emb_dim]

        # Combine time and style conditioning
        if self.use_style_conditioning:
            if style_embedding is None:
                # Use null embedding from conditioning module
                style_embedding = self.style_conditioning(
                    style_conditions=None, batch_size=batch_size, device=x.device
                )
            cond_emb = torch.cat([t_emb, style_embedding], dim=-1)  # [batch, cond_dim]
        else:
            cond_emb = t_emb

        # Initial conv
        x = self.init_conv(x)

        # Encoder with skip connections
        skip_connections = []
        for i, (enc_block, downsample) in enumerate(
            zip(self.encoder_blocks, self.downsample_blocks)
        ):
            x = enc_block(x)

            # Add conditioning (time + style)
            cond = self.cond_mlps[i](cond_emb)[:, :, None]  # [batch, dim, 1]
            x = x + cond

            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck with FiLM modulation
        x = self.bottleneck(x)

        if self.use_style_conditioning:
            # Apply FiLM: gamma * x + beta
            gamma = self.bottleneck_film_gamma(cond_emb)[:, :, None]  # [batch, dim, 1]
            beta = self.bottleneck_film_beta(cond_emb)[:, :, None]
            x = gamma * x + beta

        # Decoder with skip connections
        for i, (dec_block, upsample) in enumerate(
            zip(self.decoder_blocks, self.upsample_blocks)
        ):
            x = upsample(x)

            # Concatenate skip connection
            skip = skip_connections[-(i + 1)]
            # Handle size mismatch due to downsampling/upsampling
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(
                    x, size=skip.shape[2], mode="linear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

            x = dec_block(x)

        # Final conv
        x = self.final_conv(x)

        # Transpose back to [batch, seq_len, pose_dim]
        x = x.transpose(1, 2)

        return x


class DiffusionRefinementModule:
    """
    Phase 3: Diffusion-based pose refinement with style conditioning.

    Refines transformer outputs using iterative denoising for smoother motion.
    Supports classifier-free guidance for controllable style-conditioned generation.
    """

    def __init__(
        self,
        unet: PoseRefinementUNet,
        scheduler: DDPMScheduler,
        device: str = "cpu",
        cfg_dropout_prob: float = 0.1,
    ):
        """Initialize diffusion refinement module.

        Args:
            unet: PoseRefinementUNet model.
            scheduler: DDPMScheduler for noise scheduling.
            device: Device to run inference on.
            cfg_dropout_prob: Probability of dropping conditioning during training
                for classifier-free guidance. 0.1 is typical.
        """
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.device = device
        self.cfg_dropout_prob = cfg_dropout_prob

    @torch.no_grad()
    def refine_poses(
        self,
        transformer_output: torch.Tensor,
        style_conditions: list[StyleCondition] | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Refine transformer output using style-conditioned diffusion denoising.

        Args:
            transformer_output: [batch, seq_len, 20] - raw transformer predictions.
            style_conditions: Optional list of StyleCondition, one per batch item.
                If None, uses unconditional generation.
            num_inference_steps: Number of denoising steps (10-50).
            guidance_scale: Classifier-free guidance scale.
                1.0 = no guidance (unconditional).
                > 1.0 = stronger adherence to style condition.
                Typical values: 1.5 - 7.5.

        Returns:
            Refined poses [batch, seq_len, 20].
        """
        batch_size, seq_len, pose_dim = transformer_output.shape

        # Start from pure noise
        sample = torch.randn_like(transformer_output).to(self.device)

        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Compute style embeddings if conditioning is provided and UNet supports it
        style_embedding = None
        if self.unet.use_style_conditioning and style_conditions is not None:
            style_embedding = self.unet.style_conditioning(
                style_conditions=style_conditions,
                batch_size=batch_size,
                device=self.device,
            )

        # Iterative denoising
        for t in timesteps:
            # Prepare timestep tensor
            timestep_tensor = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )

            # Classifier-free guidance
            if guidance_scale > 1.0 and style_embedding is not None:
                # Conditional prediction
                noise_pred_cond = self.unet(sample, timestep_tensor, style_embedding)

                # Unconditional prediction (null embedding)
                noise_pred_uncond = self.unet(sample, timestep_tensor, None)

                # CFG interpolation: uncond + scale * (cond - uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # Standard prediction (conditional or unconditional)
                noise_pred = self.unet(sample, timestep_tensor, style_embedding)

            # Denoise step
            sample = self.scheduler.step(noise_pred, t, sample)

        return sample

    def train_step(
        self,
        clean_poses: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        style_conditions: list[StyleCondition] | None = None,
    ) -> dict[str, float]:
        """Single training step for diffusion model with optional style conditioning.

        Implements classifier-free guidance training by randomly dropping
        conditioning with probability cfg_dropout_prob.

        Args:
            clean_poses: [batch, seq_len, 20] - ground truth poses.
            optimizer: Optimizer for UNet parameters.
            style_conditions: Optional list of StyleCondition, one per batch item.

        Returns:
            Dictionary with loss value.
        """
        batch_size = clean_poses.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,), device=self.device
        ).long()

        # Sample noise
        noise = torch.randn_like(clean_poses).to(self.device)

        # Add noise to clean poses
        noisy_poses = self.scheduler.add_noise(clean_poses, noise, timesteps)

        # Compute style embeddings with CFG dropout
        style_embedding = None
        if self.unet.use_style_conditioning and style_conditions is not None:
            # Randomly drop conditioning for CFG training
            if torch.rand(1).item() > self.cfg_dropout_prob:
                style_embedding = self.unet.style_conditioning(
                    style_conditions=style_conditions,
                    batch_size=batch_size,
                    device=self.device,
                )
            # else: keep None for unconditional training

        # Predict noise
        noise_pred = self.unet(noisy_poses, timesteps, style_embedding)

        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_style_conditions_from_batch(
    batch: dict[str, Any],
) -> list[StyleCondition] | None:
    """Extract StyleCondition list from a training batch.

    Args:
        batch: Dictionary with optional 'enhanced_meta' key containing
            list of enhanced metadata dicts.

    Returns:
        List of StyleCondition, or None if no metadata available.
    """
    enhanced_metas = batch.get("enhanced_meta")
    if enhanced_metas is None:
        return None

    # Handle both list of dicts and dict of lists
    if isinstance(enhanced_metas, list):
        return [StyleCondition.from_enhanced_meta(meta) for meta in enhanced_metas]
    elif isinstance(enhanced_metas, dict):
        # Single sample case (shouldn't happen in batched training)
        return [StyleCondition.from_enhanced_meta(enhanced_metas)]

    return None
