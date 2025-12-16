import math
from typing import Optional, Dict, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A drop-in replacement for nn.LayerNorm without mean-centering.
    Following modern LLM best practices (Qwen/Llama standards).

    Formula: output = x * (1 / sqrt(mean(x^2) + epsilon)) * gamma
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        """Initialize RMSNorm.

        Args:
            normalized_shape: Input shape from the last dimension (matches LayerNorm interface)
            eps: Small constant for numerical stability
            elementwise_affine: If True, learn a per-element scale parameter gamma
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., *normalized_shape]

        Returns:
            Normalized tensor of the same shape
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize
        x_normed = x / rms
        # Apply learnable scale
        if self.weight is not None:
            return x_normed * self.weight
        return x_normed


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    A gated linear unit with Swish (SiLU) activation, following modern
    LLM architectures (Qwen, Llama, etc.).

    Architecture:
        - Gate projection: Linear(d_model, hidden_dim, bias=False)
        - Value projection: Linear(d_model, hidden_dim, bias=False)
        - Output projection: Linear(hidden_dim, d_model, bias=False)

    Forward pass:
        output = Swish(x @ W_gate) ⊙ (x @ W_value) @ W_output
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        """Initialize SwiGLU.

        Args:
            d_model: Model dimension (input and output dimension)
            hidden_dim: Hidden layer dimension (intermediate size)
            dropout: Dropout rate applied after the gated projection
        """
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.value_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape [..., d_model]

        Returns:
            Output tensor of shape [..., d_model]
        """
        # Swish(gate) ⊙ value
        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x)
        hidden = gate * value
        # Apply dropout and output projection
        return self.output_proj(self.dropout(hidden))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len

        if self.cached_cos is None or self.cached_cos.size(0) < seq_len:
            t = torch.arange(
                self.max_seq_len, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()
            self.cached_sin = emb.sin()

        return self.cached_cos[:seq_len], self.cached_sin[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [seq_len, batch, nhead, head_dim]
    # cos, sin: [seq_len, head_dim]

    # Reshape cos/sin for broadcasting: [seq_len, 1, 1, head_dim]
    cos = cos.unsqueeze(1).unsqueeze(1)
    sin = sin.unsqueeze(1).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        self.nhead = nhead
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.batch_first = False

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, key_padding_mask=None, need_weights=False, attn_mask=None):
        # x: [seq_len, batch, d_model]
        seq_len, batch_size, _ = x.shape

        q = self.q_proj(x).view(seq_len, batch_size, self.nhead, self.head_dim)
        k = self.k_proj(x).view(seq_len, batch_size, self.nhead, self.head_dim)
        v = self.v_proj(x).view(seq_len, batch_size, self.nhead, self.head_dim)

        # Apply RoPE
        cos, sin = self.rope(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose to [batch, nhead, seq_len, head_dim] for SDPA
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # [batch, nhead, seq, head_dim] -> [seq, batch, nhead, head_dim] -> [seq, batch, d_model]
        output = output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.d_model)
        return self.out_proj(output)


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE, RMSNorm, and SwiGLU.

    Follows modern LLM best practices (Qwen/Llama standards):
    - Pre-Norm architecture: x = x + Block(Norm(x))
    - RMSNorm instead of LayerNorm for faster computation
    - SwiGLU activation for better training dynamics
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        """Initialize the encoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Hidden dimension for the feed-forward network
            dropout: Dropout rate
            activation: Ignored (kept for backward compatibility, SwiGLU is always used)
        """
        super().__init__()
        # Self-attention with RoPE
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)

        # SwiGLU feed-forward network (replaces Linear → GELU → Linear)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout=dropout)

        # RMSNorm (replaces LayerNorm) - Pre-Norm architecture
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Dropout after attention (dropout in SwiGLU handles FFN dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        """Forward pass with Pre-Norm architecture.

        Pre-Norm pattern:
            x = x + Attention(RMSNorm(x))
            x = x + FFN(RMSNorm(x))

        Args:
            src: Source tensor [seq_len, batch, d_model]
            src_mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask

        Returns:
            Output tensor [seq_len, batch, d_model]
        """
        # Self-attention block with Pre-Norm
        src2 = self.self_attn(
            self.norm1(src), key_padding_mask=src_key_padding_mask, attn_mask=src_mask
        )
        src = src + self.dropout1(src2)

        # Feed-forward block with Pre-Norm (SwiGLU)
        src = src + self.ffn(self.norm2(src))

        return src


class StickFigureTransformer(nn.Module):
    """Main Transformer model for stick-figure motion generation.

    This is the ~11M parameter model used across training, evaluation,
    and inference. It supports:

    - Text conditioning via a projected 1024-dim embedding
    - Optional action conditioning (Phase 1)
    - Optional camera conditioning
    - Multiple decoder heads (pose, position, velocity, physics, environment)
    """

    def __init__(
        self,
        input_dim: int = 20,
        d_model: int = 384,
        nhead: int = 12,
        num_layers: int = 8,
        output_dim: int = 20,
        embedding_dim: int = 1024,
        dropout: float = 0.1,
        num_actions: int = 60,
    ) -> None:
        """Initialize the Transformer.

               Args:
                   input_dim: Input motion dimension. Default 20 for the canonical
                       stick-figure schema (5 segments
        4 coordinates). Legacy
                       configs may use 10 (5 joints
        2 coords).
                   d_model: Transformer model dimension (384 for ~11M params).
                   nhead: Number of attention heads.
                   num_layers: Number of Transformer encoder layers.
                   output_dim: Output motion dimension (usually equal to ``input_dim``).
                   embedding_dim: Text embedding dimension
                       (1024 for BAAI/bge-large-en-v1.5).
                   dropout: Dropout rate.
                   num_actions: Number of action types for action conditioning (Phase 1).
        """

        super().__init__()

        self.d_model = d_model
        self.output_dim = output_dim
        self.num_actions = num_actions

        # Project pre-trained text embedding to d_model (e.g. 1024 2 384)
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Motion embedding: project input_dim 2 d_model with a small MLP
        self.motion_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Phase 1: Action conditioning (per-frame action indices)
        self.action_embedding = nn.Embedding(num_actions, 64)
        self.action_projection = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Camera conditioning: camera state (x, y, zoom) per frame
        # Input shape: [seq_len, batch, 3] 2 projects to d_model
        self.camera_projection = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Partner conditioning: motion of a second actor (for multi-actor interactions)
        # Input shape: [seq_len, batch, input_dim] -> projects to d_model
        self.partner_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Transformer encoder (using RoPE-aware layers)
        encoder_layer = RoPETransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Head 1: Joint positions (main task)
        self.pose_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        # Head 2: Actor position in scene (x, y)
        self.position_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2),
        )

        # Head 3: Movement velocity (vx, vy)
        self.velocity_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2),
        )

        # Phase 1: Action prediction head
        self.action_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions),
        )

        # Phase 2: Physics decoder heads
        # Head 4: Physics state (vx, vy, ax, ay, momentum_x, momentum_y)
        self.physics_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
        )

        # Head 5: Environment context (ground_level, obstacles, context features)
        self.environment_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 32),
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"Model initialized with {total_params:,} parameters "
            f"({total_params/1e6:.1f}M)"
        )

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        motion: torch.Tensor,
        text_embedding: torch.Tensor,
        return_all_outputs: bool = False,
        action_sequence: Optional[torch.Tensor] = None,
        camera_data: Optional[torch.Tensor] = None,
        partner_motion: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with multi-task outputs, action & camera conditioning.

        Args:
            motion: [seq_len, batch_size, input_dim]
            text_embedding: [batch_size, embedding_dim]
            return_all_outputs: If True, return dict with all outputs.
            action_sequence: Optional [seq_len, batch_size] action indices.
            camera_data: Optional [seq_len, batch_size, 3] camera state
                (x, y, zoom) per frame.
            partner_motion: Optional [seq_len, batch_size, input_dim] motion of interacting partner.
        """

        # Embed motion
        motion_emb = self.motion_embedding(motion)  # [seq, batch, d_model]
        # No absolute positional encoding (RoPE is applied in attention)

        # Project text embedding and broadcast as first token
        text_emb = self.text_projection(text_embedding)  # [batch, d_model]
        text_emb = text_emb.unsqueeze(0)  # [1, batch, d_model]

        conditioned_motion = motion_emb

        # Phase 1: Action conditioning
        if action_sequence is not None:
            action_emb = self.action_embedding(action_sequence)  # [seq, batch, 64]
            action_features = self.action_projection(
                action_emb
            )  # [seq, batch, d_model]
            conditioned_motion = conditioned_motion + action_features

        # Camera conditioning
        if camera_data is not None:
            camera_features = self.camera_projection(
                camera_data
            )  # [seq, batch, d_model]
            conditioned_motion = conditioned_motion + camera_features

        # Partner conditioning (Multi-Actor)
        if partner_motion is not None:
            partner_features = self.partner_projection(
                partner_motion
            )  # [seq, batch, d_model]
            conditioned_motion = conditioned_motion + partner_features

        # Combine text token with conditioned motion
        combined_input = torch.cat([text_emb, conditioned_motion], dim=0)

        # Transformer encoding
        output = self.transformer_encoder(combined_input)

        # Skip the text token, keep motion representations
        motion_output = output[1:, :, :]  # [seq_len, batch, d_model]

        # Main pose output
        pose_output = self.pose_decoder(motion_output)  # [seq_len, batch, output_dim]

        if not return_all_outputs:
            return pose_output

        # For training / evaluation: return all outputs
        pooled = motion_output.mean(dim=0)  # [batch, d_model]

        position_output = self.position_decoder(pooled)  # [batch, 2]
        velocity_output = self.velocity_decoder(pooled)  # [batch, 2]

        # Action prediction over sequence
        action_logits = self.action_predictor(
            motion_output
        )  # [seq_len, batch, num_actions]

        # Physics and environment prediction
        physics_output = self.physics_decoder(motion_output)  # [seq_len, batch, 6]
        environment_output = self.environment_decoder(pooled)  # [batch, 32]

        return {
            "pose": pose_output,
            "position": position_output,
            "velocity": velocity_output,
            "action_logits": action_logits,
            "physics": physics_output,
            "environment": environment_output,
        }
