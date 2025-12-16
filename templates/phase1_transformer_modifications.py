"""
Phase 1: Action-Conditioned Generation - Transformer Modifications

This template shows the modifications needed for src/model/transformer.py
to support action-conditioned generation.

INTEGRATION INSTRUCTIONS:
1. Add action embedding layer to __init__
2. Add action prediction head to __init__
3. Modify forward() to accept and process action sequences
4. Return action predictions in output dict
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

# ============================================================================
# STEP 1: Add to StickFigureTransformer.__init__ (after line 55)
# ============================================================================


class StickFigureTransformer(nn.Module):
    def __init__(
        self,
        input_dim=10,
        d_model=384,
        nhead=12,
        num_layers=8,
        output_dim=10,
        embedding_dim=1024,
        dropout=0.1,
        num_actions=60,  # NEW PARAMETER
    ):
        super().__init__()

        # Existing layers...
        self.text_projection = nn.Sequential(...)
        self.motion_embedding = nn.Sequential(...)
        self.transformer_encoder = nn.TransformerEncoder(...)

        # ====================================================================
        # NEW: Action embedding layer (Phase 1)
        # ====================================================================
        self.num_actions = num_actions

        # Action embedding: 60 actions → 64-dim → 384-dim
        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions, embedding_dim=64
        )

        # Project action embeddings to model dimension
        self.action_projection = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ====================================================================
        # NEW: Action prediction head (Phase 1)
        # ====================================================================
        # Predicts next action from current state
        self.action_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions),
        )

        # Existing decoder heads...
        self.pose_decoder = nn.Sequential(...)
        self.position_decoder = nn.Sequential(...)
        self.velocity_decoder = nn.Sequential(...)

    # ============================================================================
    # STEP 2: Modify forward() method (replace existing forward)
    # ============================================================================

    def forward(
        self,
        motion_sequence: torch.Tensor,
        text_embedding: torch.Tensor,
        action_sequence: Optional[torch.Tensor] = None,  # NEW PARAMETER
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional action conditioning.

        Args:
            motion_sequence: [batch, seq_len, input_dim] - Motion data
            text_embedding: [batch, embedding_dim] - Text embeddings
            action_sequence: [batch, seq_len] - Action indices (optional)
                If provided, conditions generation on these actions.
                If None, runs without action conditioning (backward compatible).

        Returns:
            Dictionary containing:
                - pose: [batch, seq_len, output_dim] - Predicted poses
                - position: [batch, seq_len, 2] - Predicted positions
                - velocity: [batch, seq_len, 2] - Predicted velocities
                - action_logits: [batch, seq_len, num_actions] - Action predictions
        """
        batch_size, seq_len, _ = motion_sequence.shape

        # Project text embedding
        text_features = self.text_projection(text_embedding)  # [batch, d_model]
        text_features = text_features.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq_len, d_model]

        # Embed motion sequence
        motion_features = self.motion_embedding(
            motion_sequence
        )  # [batch, seq_len, d_model]

        # ====================================================================
        # NEW: Action conditioning (Phase 1)
        # ====================================================================
        if action_sequence is not None:
            # Embed actions
            action_emb = self.action_embedding(action_sequence)  # [batch, seq_len, 64]
            action_features = self.action_projection(
                action_emb
            )  # [batch, seq_len, d_model]

            # Combine: motion + text + action
            combined_features = motion_features + text_features + action_features
        else:
            # Original behavior: motion + text only
            combined_features = motion_features + text_features

        # Transformer encoding
        # Note: PyTorch expects [seq_len, batch, d_model]
        combined_features = combined_features.transpose(
            0, 1
        )  # [seq_len, batch, d_model]
        encoded = self.transformer_encoder(
            combined_features
        )  # [seq_len, batch, d_model]
        encoded = encoded.transpose(0, 1)  # [batch, seq_len, d_model]

        # Multi-task decoding
        pose_output = self.pose_decoder(encoded)  # [batch, seq_len, output_dim]
        position_output = self.position_decoder(encoded)  # [batch, seq_len, 2]
        velocity_output = self.velocity_decoder(encoded)  # [batch, seq_len, 2]

        # ====================================================================
        # NEW: Action prediction (Phase 1)
        # ====================================================================
        action_logits = self.action_predictor(encoded)  # [batch, seq_len, num_actions]

        return {
            "pose": pose_output,
            "position": position_output,
            "velocity": velocity_output,
            "action_logits": action_logits,  # NEW OUTPUT
        }


# ============================================================================
# STEP 3: Example usage
# ============================================================================

if __name__ == "__main__":
    # Create model with action conditioning
    model = StickFigureTransformer(
        input_dim=20,
        d_model=384,
        nhead=12,
        num_layers=8,
        output_dim=20,
        embedding_dim=1024,
        num_actions=60,  # NEW
    )

    # Example inputs
    batch_size = 4
    seq_len = 250
    motion = torch.randn(batch_size, seq_len, 20)
    text_emb = torch.randn(batch_size, 1024)
    actions = torch.randint(0, 60, (batch_size, seq_len))  # Random action sequence

    # Forward pass with action conditioning
    outputs = model(motion, text_emb, action_sequence=actions)

    print("Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
