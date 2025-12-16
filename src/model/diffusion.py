"""
Phase 3: Diffusion Refinement Module

Lightweight diffusion model to refine transformer outputs for smoother motion.
Based on DDPM (Denoising Diffusion Probabilistic Models) with temporal convolutions.

Architecture:
- Input: Noisy poses [batch, seq_len, 20] (5 lines × 4 coords)
- Output: Noise prediction [batch, seq_len, 20]
- Model: 1D UNet with temporal convolutions
- Parameters: <5M for efficiency

References:
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Lightweight 1D UNet for pose sequence refinement

    Architecture:
    - Encoder: 3 downsampling blocks (250 -> 125 -> 62 -> 31 frames)
    - Bottleneck: 2 conv blocks
    - Decoder: 3 upsampling blocks with skip connections
    - Total params: ~3-4M
    """

    def __init__(
        self,
        pose_dim: int = 20,  # 5 lines × 4 coords
        hidden_dims: list = None,
        time_emb_dim: int = 128,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        super().__init__()
        self.pose_dim = pose_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)

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

        # Time conditioning (inject time embedding into each block)
        self.time_mlps = nn.ModuleList(
            [nn.Linear(time_emb_dim, dim) for dim in hidden_dims]
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy poses [batch, seq_len, pose_dim]
            timesteps: Timestep indices [batch]

        Returns:
            Predicted noise [batch, seq_len, pose_dim]
        """
        # Transpose to [batch, pose_dim, seq_len] for Conv1d
        x = x.transpose(1, 2)

        # Time embedding
        t_emb = self.time_embedding(timesteps)  # [batch, time_emb_dim]

        # Initial conv
        x = self.init_conv(x)

        # Encoder with skip connections
        skip_connections = []
        for i, (enc_block, downsample) in enumerate(
            zip(self.encoder_blocks, self.downsample_blocks)
        ):
            x = enc_block(x)

            # Add time conditioning
            t_cond = self.time_mlps[i](t_emb)[:, :, None]  # [batch, dim, 1]
            x = x + t_cond

            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

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
    Phase 3: Diffusion-based pose refinement

    Refines transformer outputs using iterative denoising for smoother motion
    """

    def __init__(
        self, unet: PoseRefinementUNet, scheduler: DDPMScheduler, device: str = "cpu"
    ):
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def refine_poses(
        self,
        transformer_output: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Refine transformer output using diffusion denoising

        Args:
            transformer_output: [batch, seq_len, 20] - raw transformer predictions
            text_embedding: Optional text conditioning (not used in basic version)
            num_inference_steps: Number of denoising steps (10-50)
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            Refined poses [batch, seq_len, 20]
        """
        batch_size, seq_len, pose_dim = transformer_output.shape

        # Start from pure noise
        sample = torch.randn_like(transformer_output).to(self.device)

        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Iterative denoising
        for t in timesteps:
            # Prepare timestep tensor
            timestep_tensor = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )

            # Predict noise
            noise_pred = self.unet(sample, timestep_tensor)

            # Denoise step
            sample = self.scheduler.step(noise_pred, t, sample)

        return sample

    def train_step(
        self, clean_poses: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        Single training step for diffusion model

        Args:
            clean_poses: [batch, seq_len, 20] - ground truth poses
            optimizer: Optimizer for UNet parameters

        Returns:
            Dictionary with loss value
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

        # Predict noise
        noise_pred = self.unet(noisy_poses, timesteps)

        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
