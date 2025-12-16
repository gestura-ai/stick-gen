"""
Phase 1: Action-Conditioned Generation - Training Loop Updates

This template shows the modifications needed for src/train/train.py
to support action-conditioned training with action prediction loss.

INTEGRATION INSTRUCTIONS:
1. Update dataset to include action sequences
2. Add action loss to multi_task_loss function
3. Update training loop to pass action sequences
4. Add action accuracy metric tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# ============================================================================
# STEP 1: Update multi_task_loss function (around line 30)
# ============================================================================


def multi_task_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    target_actions: torch.Tensor = None,  # NEW PARAMETER
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task loss with action prediction.

    Args:
        outputs: Model outputs dict with keys:
            - pose: [batch, seq_len, output_dim]
            - position: [batch, seq_len, 2]
            - velocity: [batch, seq_len, 2]
            - action_logits: [batch, seq_len, num_actions] (NEW)
        target: Ground truth poses [batch, seq_len, output_dim]
        target_actions: Ground truth actions [batch, seq_len] (NEW)

    Returns:
        total_loss: Weighted sum of all losses
        loss_dict: Individual loss values for logging
    """
    # Existing losses
    pose_loss = F.mse_loss(outputs["pose"], target)
    temporal_loss = compute_temporal_consistency_loss(outputs["pose"])

    # ====================================================================
    # NEW: Action prediction loss (Phase 1)
    # ====================================================================
    if target_actions is not None and "action_logits" in outputs:
        # Cross-entropy loss for action classification
        action_logits = outputs["action_logits"]  # [batch, seq_len, num_actions]

        # Reshape for cross-entropy: [batch * seq_len, num_actions]
        action_logits_flat = action_logits.reshape(-1, action_logits.size(-1))
        target_actions_flat = target_actions.reshape(-1)

        action_loss = F.cross_entropy(action_logits_flat, target_actions_flat)

        # Calculate action prediction accuracy
        action_preds = torch.argmax(action_logits_flat, dim=-1)
        action_accuracy = (action_preds == target_actions_flat).float().mean()
    else:
        action_loss = torch.tensor(0.0, device=target.device)
        action_accuracy = torch.tensor(0.0, device=target.device)

    # ====================================================================
    # Weighted loss combination
    # ====================================================================
    # Loss weights (can be tuned)
    POSE_WEIGHT = 1.0
    TEMPORAL_WEIGHT = 0.3
    ACTION_WEIGHT = 0.15  # NEW

    total_loss = (
        POSE_WEIGHT * pose_loss
        + TEMPORAL_WEIGHT * temporal_loss
        + ACTION_WEIGHT * action_loss  # NEW
    )

    loss_dict = {
        "pose_loss": pose_loss.item(),
        "temporal_loss": temporal_loss.item(),
        "action_loss": action_loss.item(),  # NEW
        "action_accuracy": action_accuracy.item(),  # NEW
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


# ============================================================================
# STEP 2: Update training loop (modify train_epoch function)
# ============================================================================


def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch with action conditioning.
    """
    model.train()
    total_loss = 0
    total_action_acc = 0
    num_batches = 0

    for batch in dataloader:
        # Unpack batch
        motion_sequence = batch["motion"].to(device)  # [batch, seq_len, input_dim]
        text_embedding = batch["text_embedding"].to(device)  # [batch, embedding_dim]
        target_motion = batch["target_motion"].to(
            device
        )  # [batch, seq_len, output_dim]

        # ====================================================================
        # NEW: Load action sequences (Phase 1)
        # ====================================================================
        action_sequence = batch.get("actions", None)  # [batch, seq_len]
        if action_sequence is not None:
            action_sequence = action_sequence.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            motion_sequence, text_embedding, action_sequence=action_sequence  # NEW
        )

        # Compute loss
        loss, loss_dict = multi_task_loss(
            outputs, target_motion, target_actions=action_sequence  # NEW
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_action_acc += loss_dict["action_accuracy"]
        num_batches += 1

        # Log progress
        if num_batches % 100 == 0:
            print(
                f"Batch {num_batches}: "
                f"Loss={loss.item():.4f}, "
                f"Action Acc={loss_dict['action_accuracy']:.2%}"
            )

    avg_loss = total_loss / num_batches
    avg_action_acc = total_action_acc / num_batches

    return avg_loss, avg_action_acc


# ============================================================================
# STEP 3: Update dataset to include action sequences
# ============================================================================


class StickFigureDataset(torch.utils.data.Dataset):
    """
    Dataset with action sequences for action-conditioned training.
    """

    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            "motion": sample["motion"],
            "text_embedding": sample["text_embedding"],
            "target_motion": sample["motion"],  # For autoregressive training
            "actions": sample.get("actions", None),  # NEW: Per-frame actions
            "description": sample.get("description", ""),
        }

    def __len__(self):
        return len(self.data)


# ============================================================================
# STEP 4: Example training script
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = torch.device("cpu")  # or 'cuda'
    model = StickFigureTransformer(num_actions=60).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Load dataset with action sequences
    dataset = StickFigureDataset("data/train_data_embedded.pt")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        avg_loss, avg_action_acc = train_epoch(model, dataloader, optimizer, device)

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Loss={avg_loss:.4f}, "
            f"Action Accuracy={avg_action_acc:.2%}"
        )

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "action_accuracy": avg_action_acc,
                },
                f"checkpoint_epoch_{epoch+1}_action_conditioned.pth",
            )
