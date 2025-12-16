import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.data_gen.schema import NUM_ACTIONS
from src.model.physics_layer import DifferentiablePhysicsLoss
from src.model.transformer import StickFigureTransformer
from src.train.config import TrainingConfig

# Optional multimodal dataset support
try:
    from src.train.parallax_dataset import MultimodalParallaxDataset

    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# Configure logging
# Set to DEBUG for verbose output, INFO for normal output
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose debugging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_verbose.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized at level: {logging.getLevelName(LOG_LEVEL)}")

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
    print("⚠️  Diffusion module not available - training without refinement")

# LoRA support (optional)
try:
    from src.model.lora import (
        count_lora_parameters,
        freeze_base_model,
        get_lora_parameters,
        get_lora_state_dict,
        inject_lora_adapters,
    )

    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️  LoRA module not available - training without LoRA support")


class StickFigureDataset(Dataset):
    def __init__(self, data_path="data/train_data_embedded.pt"):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Item is dict {"description": str, "motion": tensor, "embedding": tensor, "actions": tensor, "physics": tensor}
        item = self.data[idx]
        motion = item["motion"]
        embedding = item["embedding"]  # [1024]
        actions = item.get("actions", None)  # [num_frames] - Phase 1
        physics = item.get("physics", None)  # [num_frames, 6] - Phase 2

        # Input: Frame t
        # Target: Frame t+1
        return (
            motion[:-1],
            embedding,
            motion[1:],
            actions[:-1] if actions is not None else None,
            physics[:-1] if physics is not None else None,
        )


def temporal_consistency_loss(predictions):
    """
    Compute temporal consistency loss to encourage smooth motion

    Args:
        predictions: [seq_len, batch, dim] tensor of predictions

    Returns:
        Scalar loss value
    """
    logger.debug("    ENTER temporal_consistency_loss()")
    logger.debug(f"      predictions shape: {predictions.shape}")

    # Penalize large frame-to-frame changes
    frame_diff = predictions[1:] - predictions[:-1]
    logger.debug(f"      frame_diff shape: {frame_diff.shape}")

    loss = torch.mean(frame_diff**2)
    logger.debug(f"      temporal loss: {loss.item():.6f}")
    logger.debug("    EXIT temporal_consistency_loss()")
    return loss


def physics_loss(physics_output, physics_targets):
    """
    Phase 2: Physics-aware loss function

    Enforces physical constraints:
    - Gravity: vertical acceleration should be ~-9.8 m/s^2 when airborne
    - Momentum conservation: momentum should be continuous
    - Velocity-acceleration consistency: v(t+1) = v(t) + a(t) * dt

    Args:
        physics_output: [seq_len, batch, 6] - predicted (vx, vy, ax, ay, momentum_x, momentum_y)
        physics_targets: [seq_len, batch, 6] - ground truth physics

    Returns:
        Total physics loss (scalar), dict of loss components
    """
    # Extract components
    pred_vx, pred_vy = physics_output[:, :, 0], physics_output[:, :, 1]
    pred_ax, pred_ay = physics_output[:, :, 2], physics_output[:, :, 3]
    pred_mx, pred_my = physics_output[:, :, 4], physics_output[:, :, 5]

    _target_vx, _target_vy = physics_targets[:, :, 0], physics_targets[:, :, 1]
    _target_ax, _target_ay = physics_targets[:, :, 2], physics_targets[:, :, 3]
    _target_mx, _target_my = physics_targets[:, :, 4], physics_targets[:, :, 5]

    # 1. Basic MSE loss for all physics parameters
    mse_loss = nn.MSELoss()(physics_output, physics_targets)

    # 2. Gravity constraint: ay should be close to -9.8 when airborne
    # (simplified: penalize deviation from expected gravity)
    gravity_constant = -9.8
    gravity_loss = torch.mean((pred_ay - gravity_constant) ** 2)

    # 3. Momentum conservation: momentum should change smoothly
    momentum_diff_x = pred_mx[1:] - pred_mx[:-1]
    momentum_diff_y = pred_my[1:] - pred_my[:-1]
    momentum_loss = torch.mean(momentum_diff_x**2 + momentum_diff_y**2)

    # 4. Velocity-acceleration consistency: v(t+1) ≈ v(t) + a(t) * dt
    dt = 1.0 / 25.0  # 25 FPS
    expected_vx = pred_vx[:-1] + pred_ax[:-1] * dt
    expected_vy = pred_vy[:-1] + pred_ay[:-1] * dt
    consistency_loss = torch.mean(
        (pred_vx[1:] - expected_vx) ** 2 + (pred_vy[1:] - expected_vy) ** 2
    )

    # Combine losses with weights
    total_loss = (
        mse_loss + 0.1 * gravity_loss + 0.1 * momentum_loss + 0.2 * consistency_loss
    )

    # Phase 2: Physics-aware loss function (Standard)

    # Phase 2: Physics-aware loss function (Standard)
    # physics_output is already passed in argument

    # physics_targets would be needed here, currently reusing what we have or skipping if not available
    # For now, we keep the existing logic but wrap the new layer

    # NEW: Differentiable Physics Layer (Brax)
    # We instantiate it once outside the loop in a real scenario, but for minimal diff:
    # (In practice, pass this instance from main())

    return total_loss, {
        "physics_mse": mse_loss.item(),
        "gravity_loss": gravity_loss.item(),
        "momentum_loss": momentum_loss.item(),
        "consistency_loss": consistency_loss.item(),
    }


def compute_evaluation_metrics(predictions, targets):
    """
    Compute additional evaluation metrics beyond MSE loss

    Returns:
        dict of metrics
    """
    with torch.no_grad():
        # Smoothness: variance of frame-to-frame changes
        pred_diff = predictions[1:] - predictions[:-1]
        target_diff = targets[1:] - targets[:-1]
        smoothness_error = torch.mean((pred_diff - target_diff) ** 2)

        # Position accuracy (first 2 coords of each frame)
        position_error = torch.mean((predictions[:, :, :2] - targets[:, :, :2]) ** 2)

        return {
            "smoothness_error": smoothness_error.item(),
            "position_error": position_error.item(),
        }


def train(
    config_path="configs/base.yaml",
    data_path_override=None,
    checkpoint_dir_override=None,
    resume_from_cli=None,
    init_from_cli=None,
    seed=42,
):
    """Train the stick figure generation model.

    Args:
        config_path: Path to YAML configuration file (default: configs/base.yaml)
        data_path_override: Override path to training data directory (for RunPod deployments)
        checkpoint_dir_override: Override path to checkpoint output directory (for RunPod deployments)
        resume_from_cli: Optional checkpoint path passed explicitly via CLI (--resume_from)
            Resumes training from checkpoint (loads model, optimizer, scheduler, epoch counter)
        init_from_cli: Optional checkpoint path passed explicitly via CLI (--init_from)
            Initializes model weights only (for SFT), does NOT restore optimizer/scheduler/epoch
        seed: Random seed for reproducibility (default: 42)
    """

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")

    # Load configuration
    config = TrainingConfig(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    config.print_config()

    # Determine data and checkpoint paths (CLI overrides take precedence)
    data_path = data_path_override or config.get(
        "data.train_data", "data/train_data_final.pt"
    )
    checkpoint_dir = checkpoint_dir_override or config.get("data.checkpoint_dir", ".")

    # If data_path is a directory, look for train_data_final.pt inside it
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, "train_data_final.pt")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Data path: {data_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Extract hyperparameters from config
    INPUT_DIM = config.get("model.input_dim", 20)
    D_MODEL = config.get("model.d_model", 384)
    NHEAD = config.get("model.nhead", 12)
    NUM_LAYERS = config.get("model.num_layers", 8)
    DROPOUT = config.get("model.dropout", 0.1)

    BATCH_SIZE = config.get("training.batch_size", 2)
    # Match YAML key `training.grad_accum_steps` (see configs/*.yaml)
    GRAD_ACCUM_STEPS = config.get("training.grad_accum_steps", 32)
    EPOCHS = config.get("training.epochs", 50)
    LEARNING_RATE = config.get("training.learning_rate", 0.0003)
    WARMUP_EPOCHS = config.get("training.warmup_epochs", 10)
    MAX_GRAD_NORM = config.get("training.max_grad_norm", 1.0)

    # Loss weights now come from the documented `loss_weights` section
    TEMPORAL_LOSS_WEIGHT = config.get("loss_weights.temporal", 0.1)
    ACTION_LOSS_WEIGHT = config.get("loss_weights.action", 0.15)
    PHYSICS_LOSS_WEIGHT = config.get("loss_weights.physics", 0.2)
    DIFF_PHYSICS_LOSS_WEIGHT = config.get("loss_weights.diff_physics", 0.1)

    USE_DIFFUSION = config.get("diffusion.enabled", False)
    # Diffusion loss weight is also controlled under loss_weights
    DIFFUSION_LOSS_WEIGHT = config.get("loss_weights.diffusion", 0.0)
    DIFFUSION_LR = config.get("diffusion.learning_rate", 1e-4)

    if USE_DIFFUSION and not DIFFUSION_AVAILABLE:
        print(
            "⚠️  Diffusion enabled in config but diffusion module not available; continuing without diffusion."
        )
        USE_DIFFUSION = False

    # LoRA settings
    USE_LORA = config.get("lora.enabled", False)
    LORA_RANK = config.get("lora.rank", 8)
    LORA_ALPHA = config.get("lora.alpha", 16.0)
    LORA_DROPOUT = config.get("lora.dropout", 0.05)
    LORA_TARGET_MODULES = config.get(
        "lora.target_modules", ["transformer_encoder", "pose_decoder"]
    )

    if USE_LORA and not LORA_AVAILABLE:
        print(
            "⚠️  LoRA enabled in config but LoRA module not available; continuing without LoRA."
        )
        USE_LORA = False

    # Multimodal (2.5D parallax) settings
    USE_PARALLAX = config.get("data.use_parallax_augmentation", False)
    PARALLAX_ROOT = config.get("data.parallax_root", "data/2.5d_parallax")
    PARALLAX_IMAGE_SIZE = tuple(config.get("data.parallax_image_size", [256, 256]))
    IMAGE_ENCODER_ARCH = config.get("model.image_encoder_arch", "lightweight_cnn")
    FUSION_STRATEGY = config.get("model.fusion_strategy", "gated")
    IMAGE_BACKEND = config.get("data.image_backend", "pil")

    if USE_PARALLAX and not MULTIMODAL_AVAILABLE:
        print(
            "⚠️  Parallax augmentation enabled but MultimodalParallaxDataset not available; "
            "continuing with motion-only training."
        )
        USE_PARALLAX = False

    if USE_PARALLAX and not os.path.isdir(PARALLAX_ROOT):
        print(
            f"⚠️  Parallax root directory not found: {PARALLAX_ROOT}; "
            "continuing with motion-only training."
        )
        USE_PARALLAX = False

    # Training stage (pretraining vs sft)
    TRAINING_STAGE = config.get("training.stage", "pretraining")

    # Device selection honoring config.device.type
    device_type = config.get("device.type", "auto")
    device = torch.device("cpu")
    if device_type == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("⚠️  Requested CUDA but no GPU is available. Falling back to CPU.")
    elif device_type == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print(
                "⚠️  Requested MPS but no MPS device is available. Falling back to CPU."
            )
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:  # "auto"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    elif device.type == "mps":
        print(f"Using device: {device} (Apple Silicon Acceleration)")
    else:
        print(f"Using device: {device}")
        print("⚠️  Training on CPU - this will take significant time")
        print("   Consider using a smaller dataset or cloud GPU for faster training")
        # Optimize for CPU
        torch.set_num_threads(torch.get_num_threads())  # Use all CPU cores
        print(f"   Using {torch.get_num_threads()} CPU threads")

    print("\n" + "=" * 60)
    print("INITIALIZING 11M+ PARAMETER MODEL")
    print("=" * 60)
    print(f"  - d_model: {D_MODEL}")
    print(f"  - num_layers: {NUM_LAYERS}")
    print(f"  - nhead: {NHEAD}")
    print(f"  - dropout: {DROPOUT}")
    print(f"  - batch_size: {BATCH_SIZE}")
    print(
        f"  - gradient_accumulation: {GRAD_ACCUM_STEPS} (effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS})"
    )
    if USE_PARALLAX:
        print(f"  - multimodal: ENABLED (image encoder: {IMAGE_ENCODER_ARCH})")
        print(f"  - fusion_strategy: {FUSION_STRATEGY}")
        print(f"  - parallax_image_size: {PARALLAX_IMAGE_SIZE}")
    else:
        print("  - multimodal: DISABLED (motion-only training)")

    # Initialize model with optional multimodal support
    model = StickFigureTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        output_dim=INPUT_DIM,
        embedding_dim=1024,  # Updated for BAAI/bge-large-en-v1.5
        dropout=DROPOUT,
        num_actions=NUM_ACTIONS,  # Phase 1: Action conditioning
        # Multimodal image conditioning
        enable_image_conditioning=USE_PARALLAX,
        image_encoder_arch=IMAGE_ENCODER_ARCH,
        image_size=PARALLAX_IMAGE_SIZE,
        fusion_strategy=FUSION_STRATEGY,
    ).to(device)

    # Initialize Physics Loss Layer
    diff_physics_layer = DifferentiablePhysicsLoss()
    # (Optional: move to device if using advanced PyTorch-sim features)
    diff_physics_layer = diff_physics_layer.to(device)

    # ------------------------------------------------------------------
    # Optional: Initialize model weights from a pretrained checkpoint (SFT)
    # This loads ONLY model weights, not optimizer/scheduler/epoch state.
    # Precedence: CLI --init_from > INIT_FROM_CHECKPOINT env var
    # > training.init_from in the YAML config.
    # ------------------------------------------------------------------
    init_from_env = os.environ.get("INIT_FROM_CHECKPOINT")
    init_from_config = config.get("training.init_from", None)
    init_from = init_from_cli or init_from_env or init_from_config

    if init_from:
        init_path = os.path.expanduser(str(init_from))
        if not os.path.isfile(init_path):
            msg = (
                f"Requested init checkpoint not found: {init_path}. "
                "Set INIT_FROM_CHECKPOINT or --init_from to a valid .pth file."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Initializing model weights from checkpoint: {init_path}")
        print(f"\n  Loading pretrained weights from: {init_path}")
        checkpoint = torch.load(init_path, map_location=device)

        if "model_state_dict" not in checkpoint:
            msg = f"Checkpoint at {init_path} is missing 'model_state_dict'"
            logger.error(msg)
            raise KeyError(msg)

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Model weights initialized from checkpoint (optimizer/scheduler NOT restored)"
        )
        print("  ✓ Model weights loaded (fresh optimizer for SFT)")

    # ------------------------------------------------------------------
    # Optional: Inject LoRA adapters for efficient fine-tuning
    # ------------------------------------------------------------------
    if USE_LORA:
        print("\n" + "=" * 60)
        print("INJECTING LORA ADAPTERS")
        print("=" * 60)
        num_lora_layers = inject_lora_adapters(
            model,
            target_modules=LORA_TARGET_MODULES,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )
        print(f"  - LoRA rank: {LORA_RANK}")
        print(f"  - LoRA alpha: {LORA_ALPHA}")
        print(f"  - LoRA dropout: {LORA_DROPOUT}")
        print(f"  - Target modules: {LORA_TARGET_MODULES}")
        print(f"  - Layers modified: {num_lora_layers}")

        # Freeze base model parameters
        frozen_count = freeze_base_model(model)
        trainable, total, lora_params = count_lora_parameters(model)
        print(f"  - Base parameters frozen: {frozen_count:,}")
        print(f"  - LoRA parameters: {lora_params:,} ({lora_params/1e6:.2f}M)")
        print(f"  - Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")

    # Phase 3: Initialize diffusion refinement module (optional)
    diffusion_module = None
    diffusion_optimizer = None
    if USE_DIFFUSION:
        print("\n" + "=" * 60)
        print("INITIALIZING DIFFUSION REFINEMENT MODULE")
        print("=" * 60)
        unet = PoseRefinementUNet(
            pose_dim=INPUT_DIM, hidden_dims=[64, 128, 256], time_emb_dim=128
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        diffusion_module = DiffusionRefinementModule(
            unet, scheduler, device=str(device)
        )
        diffusion_optimizer = optim.Adam(unet.parameters(), lr=DIFFUSION_LR)

        from src.model.diffusion import count_parameters

        diffusion_params = count_parameters(unet)
        print(
            f"  - Diffusion UNet parameters: {diffusion_params:,} ({diffusion_params/1e6:.2f}M)"
        )
        print(f"  - Diffusion learning rate: {DIFFUSION_LR}")
        print(f"  - Diffusion loss weight: {DIFFUSION_LOSS_WEIGHT}")

    # Multi-task loss function
    def multi_task_loss(outputs, targets, action_targets=None, physics_targets=None):
        """
        Compute multi-task loss: pose + temporal + action (Phase 1) + physics (Phase 2)
        """
        logger.debug("ENTER multi_task_loss()")
        logger.debug(f"  outputs keys: {outputs.keys()}")
        logger.debug(f"  targets shape: {targets.shape}")
        logger.debug(
            f"  action_targets: {action_targets.shape if action_targets is not None else None}"
        )
        logger.debug(
            f"  physics_targets: {physics_targets.shape if physics_targets is not None else None}"
        )

        # Main pose reconstruction loss
        logger.debug("  Computing pose loss...")
        pose_loss = nn.MSELoss()(outputs["pose"], targets)
        logger.debug(f"  pose_loss computed: {pose_loss.item():.6f}")

        # Temporal consistency loss
        logger.debug("  Computing temporal loss...")
        temporal_loss = temporal_consistency_loss(outputs["pose"])
        logger.debug(f"  temporal_loss computed: {temporal_loss.item():.6f}")

        # Initialize loss components (store tensors, not .item() values)
        loss_components = {"pose_loss": pose_loss, "temporal_loss": temporal_loss}

        total_loss = pose_loss + TEMPORAL_LOSS_WEIGHT * temporal_loss
        logger.debug(f"  total_loss (pose + temporal): {total_loss.item():.6f}")

        # Phase 1: Action prediction loss
        if action_targets is not None and "action_logits" in outputs:
            logger.debug("  Computing action loss...")
            # action_logits: [seq_len, batch, num_actions]
            # action_targets: [batch, seq_len]
            action_logits = outputs["action_logits"].permute(
                1, 2, 0
            )  # [batch, num_actions, seq_len]
            action_loss = nn.CrossEntropyLoss()(action_logits, action_targets)
            total_loss += ACTION_LOSS_WEIGHT * action_loss
            loss_components["action_loss"] = action_loss
            logger.debug(f"  action_loss computed: {action_loss.item():.6f}")

        # Phase 2: Physics loss
        if physics_targets is not None and "physics" in outputs:
            logger.debug("  Computing physics loss...")
            # physics: [seq_len, batch, 6]
            # physics_targets: [batch, seq_len, 6] -> permute to [seq_len, batch, 6]
            physics_targets_permuted = physics_targets.permute(1, 0, 2)

            # Phase 2: Physics-aware loss function (Standard)
            phys_loss, phys_components = physics_loss(
                outputs["physics"], physics_targets_permuted
            )
            current_physics_loss = phys_loss

            # NEW: Differentiable Physics Layer (Brax)
            # Adds hard-constraint gradient signal to the existing physics supervision
            if diff_physics_layer is not None:
                logger.debug("  Computing differentiable physics loss...")
                # outputs['pose'] is [seq_len, batch, input_dim]
                # outputs['physics'] is [seq_len, batch, 6]
                diff_phys_loss = diff_physics_layer(outputs["pose"], outputs["physics"])
                current_physics_loss = (
                    current_physics_loss + DIFF_PHYSICS_LOSS_WEIGHT * diff_phys_loss
                )
                loss_components["diff_physics_loss"] = diff_phys_loss
                logger.debug(
                    f"  differentiable physics_loss computed: {diff_phys_loss.item():.6f}"
                )

            total_loss += PHYSICS_LOSS_WEIGHT * current_physics_loss
            loss_components["physics_loss"] = (
                phys_loss  # Keep original physics_loss for logging
            )
            loss_components.update(phys_components)
            logger.debug(f"  physics_loss computed: {phys_loss.item():.6f}")

        logger.debug(f"  FINAL total_loss: {total_loss.item():.6f}")
        logger.debug("EXIT multi_task_loss()")
        return total_loss, loss_components

    weight_decay = config.get("optimization.weight_decay", 0.01)

    # When using LoRA, only optimize LoRA parameters
    if USE_LORA:
        lora_params = list(get_lora_parameters(model))
        optimizer = optim.AdamW(
            lora_params, lr=LEARNING_RATE, weight_decay=weight_decay
        )
        logger.info(
            f"Optimizer configured for LoRA parameters only ({len(lora_params)} parameter groups)"
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
        )

    # Learning rate scheduler with longer warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            # Cosine decay after warmup
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Optional: resume training from a saved checkpoint
    # Precedence: CLI --resume_from > RESUME_FROM_CHECKPOINT env var
    # > training.resume_from in the YAML config.
    # ------------------------------------------------------------------
    resume_from_env = os.environ.get("RESUME_FROM_CHECKPOINT")
    resume_from_config = config.get("training.resume_from", None)
    resume_from = resume_from_cli or resume_from_env or resume_from_config

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_from:
        checkpoint_path = os.path.expanduser(str(resume_from))
        if not os.path.isfile(checkpoint_path):
            msg = (
                f"Requested resume checkpoint not found: {checkpoint_path}. "
                "Set RESUME_FROM_CHECKPOINT or --resume_from to a valid .pth file."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model and optimizer state
        if "model_state_dict" not in checkpoint:
            msg = f"Checkpoint at {checkpoint_path} is missing 'model_state_dict'"
            logger.error(msg)
            raise KeyError(msg)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Optional diffusion optimizer state
        if (
            diffusion_optimizer is not None
            and "diffusion_optimizer_state_dict" in checkpoint
        ):
            try:
                diffusion_optimizer.load_state_dict(
                    checkpoint["diffusion_optimizer_state_dict"]
                )
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(f"Failed to restore diffusion optimizer state: {e}")

        # Optional LR scheduler state
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(f"Failed to restore scheduler state: {e}")

        # Epoch and step counters
        ckpt_epoch = int(checkpoint.get("epoch", -1))
        if ckpt_epoch < -1:
            logger.warning(
                f"Checkpoint epoch {ckpt_epoch} is invalid; starting from epoch 0"
            )
            start_epoch = 0
        else:
            start_epoch = ckpt_epoch + 1

        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(
            checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss))
        )

        logger.info(
            f"Checkpoint loaded (epoch={ckpt_epoch}, next_epoch={start_epoch}, "
            f"global_step={global_step}, best_val_loss={best_val_loss:.6f})"
        )

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    print(f"  - Data path: {data_path}")

    # Choose dataset type based on multimodal configuration
    multimodal_dataset: MultimodalParallaxDataset | None = None
    if USE_PARALLAX:
        print("  - Mode: MULTIMODAL (2.5D parallax augmentation)")
        print(f"  - Parallax root: {PARALLAX_ROOT}")
        print(f"  - Image backend: {IMAGE_BACKEND}")
        multimodal_dataset = MultimodalParallaxDataset(
            parallax_root=PARALLAX_ROOT,
            motion_data_path=data_path,
            image_size=PARALLAX_IMAGE_SIZE,
            image_backend=IMAGE_BACKEND,
        )
        print(f"  - Multimodal samples (PNG frames): {len(multimodal_dataset)}")
        # For multimodal, we use the parallax dataset directly
        # (each item = one PNG frame with associated motion/camera/text)
        dataset = multimodal_dataset
    else:
        print("  - Mode: MOTION-ONLY")
        dataset = StickFigureDataset(data_path=data_path)
    print(f"  - Total samples loaded: {len(dataset)}")

    # Split into train/val/test (80/10/10) - improved from 90/10
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    num_workers = config.get("device.num_workers", 0)
    pin_memory = config.get("device.pin_memory", False)
    # Only pin memory when training on CUDA
    effective_pin_memory = bool(pin_memory and device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
    )
    DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
    )

    print(f"  - Training samples: {train_size} (80%)")
    print(f"  - Validation samples: {val_size} (10%)")
    print(f"  - Test samples: {test_size} (10%)")

    print("\n" + "=" * 60)
    print("STARTING TRAINING LOOP")
    print("=" * 60)

    # Track metrics
    for epoch in range(start_epoch, EPOCHS):
        # Training
        model.train()
        if diffusion_module is not None:
            diffusion_module.unet.train()

        total_train_loss = 0
        total_pose_loss = 0
        total_temporal_loss = 0
        total_action_loss = 0
        total_action_accuracy = 0
        total_physics_loss = 0  # Phase 2
        total_diffusion_loss = 0  # Phase 3
        total_smoothness_error = 0
        total_position_error = 0

        optimizer.zero_grad()  # Zero gradients at start
        logger.info(f"Starting epoch {epoch+1}/{EPOCHS}")

        for batch_idx, batch_data in enumerate(train_loader):
            logger.debug(f"BATCH {batch_idx+1}/{len(train_loader)}")

            # Unpack batch based on dataset type
            # Multimodal: (image, motion_frame, camera_pose, text_prompt, action_label)
            # Motion-only: (data, embedding, target, [actions], [physics])
            logger.debug(f"  Unpacking batch_data (len={len(batch_data)})")

            image_tensor = None
            image_camera_pose = None

            if USE_PARALLAX and multimodal_dataset is not None:
                # Multimodal batch: (image, motion_frame, camera_pose, text_prompt, action)
                # Note: text_prompt is a string, we need to embed it or use cached embeddings
                image_tensor, motion_frame, camera_pose, text_prompts, actions = (
                    batch_data
                )

                # For multimodal training with parallax frames:
                # - motion_frame is [batch, 20] single frame
                # - We create a pseudo-sequence by repeating the frame
                # - Target is the same as input (reconstruction objective)
                batch_size_curr = motion_frame.shape[0]

                # Create single-frame sequences: [batch, 1, 20]
                data = motion_frame.unsqueeze(1)  # [batch, 1, dim]
                target = data.clone()  # Reconstruction target

                # Get embeddings from the motion data (pre-computed in the .pt file)
                # For multimodal, we retrieve embeddings via sample index
                embeddings_list = []
                for i in range(batch_size_curr):
                    # Get sample index from dataset
                    sample_idx = multimodal_dataset.index[
                        (
                            train_dataset.indices[batch_idx * BATCH_SIZE + i]
                            if hasattr(train_dataset, "indices")
                            else i
                        )
                    ]["sample_idx"]
                    sample = multimodal_dataset.samples.get(sample_idx, {})
                    emb = sample.get("embedding")
                    if emb is None:
                        # Fallback: zero embedding (should not happen with proper data)
                        emb = torch.zeros(1024)
                    embeddings_list.append(emb)
                embedding = torch.stack(embeddings_list)

                image_camera_pose = camera_pose
                physics = None

                # Reshape actions for single-frame
                if actions is not None and not isinstance(actions, type(None)):
                    # actions is per-frame action label [batch]
                    actions = actions.unsqueeze(1) if actions.dim() == 1 else actions
                else:
                    actions = None

            elif len(batch_data) == 5:
                data, embedding, target, actions, physics = batch_data
            elif len(batch_data) == 4:
                data, embedding, target, actions = batch_data
                physics = None
            else:
                data, embedding, target = batch_data
                actions = None
                physics = None

            # Move to device
            logger.debug(f"  Moving tensors to device: {device}")
            data = data.to(device)
            embedding = embedding.to(device)
            target = target.to(device)
            if actions is not None:
                actions = actions.to(device)
            if physics is not None:
                physics = physics.to(device)
            if image_tensor is not None:
                image_tensor = image_tensor.to(device)
            if image_camera_pose is not None:
                image_camera_pose = image_camera_pose.to(device)

            # data: [batch, seq, dim] -> [seq, batch, dim] for transformer
            logger.debug(f"  Permuting data: {data.shape} -> ", end="")
            data = data.permute(1, 0, 2)
            logger.debug(f"{data.shape}")
            target = target.permute(1, 0, 2)

            # actions: [batch, seq] -> [seq, batch] for transformer
            if actions is not None:
                if actions.dim() == 2:
                    actions_seq = actions.permute(1, 0)
                else:
                    actions_seq = actions.unsqueeze(0)  # [1, batch] for single frame
                logger.debug(f"  actions_seq shape: {actions_seq.shape}")
            else:
                actions_seq = None
                logger.debug("  actions_seq: None")

            # Forward pass with multi-task outputs, action conditioning, and optional image
            logger.debug("  Running forward pass...")
            outputs = model(
                data,
                embedding,
                return_all_outputs=True,
                action_sequence=actions_seq,
                image_tensor=image_tensor,
                image_camera_pose=image_camera_pose,
            )
            logger.debug(f"  Forward pass complete. Output keys: {outputs.keys()}")

            # Compute multi-task loss (Phase 2: includes physics)
            logger.debug("  Computing multi-task loss...")
            loss, loss_components = multi_task_loss(outputs, target, actions, physics)
            logger.debug(f"  Loss computed: {loss.item():.6f}")

            # Scale loss for gradient accumulation
            logger.debug(f"  Scaling loss by 1/{GRAD_ACCUM_STEPS}")
            loss = loss / GRAD_ACCUM_STEPS

            logger.debug("  Running backward pass...")
            loss.backward()
            logger.debug("  Backward pass complete")

            # Accumulate losses (convert tensors to scalars)
            logger.debug("  Accumulating losses...")
            total_train_loss += loss.item() * GRAD_ACCUM_STEPS
            total_pose_loss += loss_components["pose_loss"].item()
            total_temporal_loss += loss_components["temporal_loss"].item()

            if "action_loss" in loss_components:
                total_action_loss += loss_components["action_loss"].item()
                logger.debug(
                    f"    action_loss: {loss_components['action_loss'].item():.6f}"
                )

                # Compute action accuracy
                if "action_logits" in outputs and actions is not None:
                    action_preds = (
                        outputs["action_logits"].argmax(dim=-1).permute(1, 0)
                    )  # [batch, seq]
                    action_acc = (action_preds == actions).float().mean().item()
                    total_action_accuracy += action_acc
                    logger.debug(f"    action_accuracy: {action_acc:.4f}")

            # Phase 2: Accumulate physics loss
            if "physics_loss" in loss_components:
                total_physics_loss += loss_components["physics_loss"].item()
                logger.debug(
                    f"    physics_loss: {loss_components['physics_loss'].item():.6f}"
                )

            # Phase 3: Diffusion refinement training (optional)
            if diffusion_module is not None and diffusion_optimizer is not None:
                # Get transformer predictions (detach to avoid backprop through transformer)
                transformer_poses = (
                    outputs["pose"].permute(1, 0, 2).detach()
                )  # [batch, seq, dim]

                # Train diffusion model to denoise transformer outputs
                diffusion_result = diffusion_module.train_step(
                    transformer_poses, diffusion_optimizer
                )
                total_diffusion_loss += diffusion_result["loss"]

            # Compute evaluation metrics
            metrics = compute_evaluation_metrics(outputs["pose"], target)
            total_smoothness_error += metrics["smoothness_error"]
            total_position_error += metrics["position_error"]

            # Gradient accumulation: only step every GRAD_ACCUM_STEPS
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                logger.debug(
                    f"  Gradient accumulation step {(batch_idx + 1) // GRAD_ACCUM_STEPS}"
                )

                # Gradient clipping
                logger.debug(f"  Clipping gradients (max_norm={MAX_GRAD_NORM})")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=MAX_GRAD_NORM
                )

                logger.debug("  Optimizer step")
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                logger.debug("  Gradients zeroed")

                # Print progress every 10 gradient steps
                if ((batch_idx + 1) // GRAD_ACCUM_STEPS) % 10 == 0:
                    avg_loss = total_train_loss / (batch_idx + 1)
                    logger.info(
                        f"  Step {(batch_idx + 1) // GRAD_ACCUM_STEPS}, Batch {batch_idx+1}/{len(train_loader)}, Avg Loss: {avg_loss:.4f}"
                    )
                    print(
                        f"  Step {(batch_idx + 1) // GRAD_ACCUM_STEPS}, Batch {batch_idx+1}/{len(train_loader)}, Avg Loss: {avg_loss:.4f}"
                    )

        # Step optimizer if there are remaining gradients
        logger.debug("Checking for remaining gradients...")
        if (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
            logger.debug("  Final gradient step for remaining batches")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        avg_pose_loss = total_pose_loss / len(train_loader)
        avg_temporal_loss = total_temporal_loss / len(train_loader)
        avg_action_loss = (
            total_action_loss / len(train_loader) if total_action_loss > 0 else 0
        )
        avg_action_accuracy = (
            total_action_accuracy / len(train_loader)
            if total_action_accuracy > 0
            else 0
        )
        avg_physics_loss = (
            total_physics_loss / len(train_loader) if total_physics_loss > 0 else 0
        )  # Phase 2
        avg_diffusion_loss = (
            total_diffusion_loss / len(train_loader) if total_diffusion_loss > 0 else 0
        )  # Phase 3
        avg_smoothness_error = total_smoothness_error / len(train_loader)
        avg_position_error = total_position_error / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_smoothness = 0
        total_val_position = 0

        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack batch (Phase 2: includes physics)
                if len(batch_data) == 5:
                    data, embedding, target, actions, physics = batch_data
                elif len(batch_data) == 4:
                    data, embedding, target, actions = batch_data
                    physics = None
                else:
                    data, embedding, target = batch_data
                    actions = None
                    physics = None

                data = data.to(device)
                embedding = embedding.to(device)
                target = target.to(device)
                if actions is not None:
                    actions = actions.to(device)
                if physics is not None:
                    physics = physics.to(device)

                data = data.permute(1, 0, 2)
                target = target.permute(1, 0, 2)

                if actions is not None:
                    actions_seq = actions.permute(1, 0)
                else:
                    actions_seq = None

                outputs = model(
                    data,
                    embedding,
                    return_all_outputs=True,
                    action_sequence=actions_seq,
                )
                loss, _ = multi_task_loss(outputs, target, actions, physics)
                total_val_loss += loss.item()

                # Compute metrics
                metrics = compute_evaluation_metrics(outputs["pose"], target)
                total_val_smoothness += metrics["smoothness_error"]
                total_val_position += metrics["position_error"]

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_smoothness = total_val_smoothness / len(val_loader)
        avg_val_position = total_val_position / len(val_loader)

        # Update learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Print comprehensive metrics
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"  Pose: {avg_pose_loss:.6f} | Temporal: {avg_temporal_loss:.6f}")
        if avg_action_loss > 0:
            print(
                f"  Action Loss: {avg_action_loss:.6f} | Action Acc: {avg_action_accuracy:.2%}"
            )
        if avg_physics_loss > 0:  # Phase 2
            print(f"  Physics Loss: {avg_physics_loss:.6f}")
        if avg_diffusion_loss > 0:  # Phase 3
            print(f"  Diffusion Loss: {avg_diffusion_loss:.6f}")
        print(
            f"  Smoothness: {avg_smoothness_error:.6f} | Position: {avg_position_error:.6f}"
        )
        print(
            f"  Val Smoothness: {avg_val_smoothness:.6f} | Val Position: {avg_val_position:.6f}"
        )
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_smoothness": avg_val_smoothness,
                "val_position": avg_val_position,
                "training_stage": TRAINING_STAGE,
                "lora_enabled": USE_LORA,
            }
            # Save LoRA state separately for easy extraction
            if USE_LORA:
                checkpoint_data["lora_state_dict"] = get_lora_state_dict(model)
                checkpoint_data["lora_config"] = {
                    "rank": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "target_modules": LORA_TARGET_MODULES,
                }
            torch.save(
                checkpoint_data,
                os.path.join(checkpoint_dir, "model_checkpoint_best.pth"),
            )
            print(
                f"  ✓ Best model saved to {checkpoint_dir}/model_checkpoint_best.pth (val_loss: {best_val_loss:.6f})"
            )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "training_stage": TRAINING_STAGE,
                "lora_enabled": USE_LORA,
            }
            if USE_LORA:
                checkpoint_data["lora_state_dict"] = get_lora_state_dict(model)
            torch.save(
                checkpoint_data,
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )
            print(
                f"  ✓ Checkpoint saved: {checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_checkpoint.pth"))
    print(f"Final model saved to {checkpoint_dir}/model_checkpoint.pth")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("\nModel ready for inference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stick figure generation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override path to training data directory (for RunPod deployments)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override path to checkpoint output directory (for RunPod deployments)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional path to checkpoint file for continued pretraining/resume (loads model, optimizer, scheduler, epoch)",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Optional path to checkpoint file for SFT initialization (loads model weights only, fresh optimizer)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path_override=args.data_path,
        checkpoint_dir_override=args.checkpoint_dir,
        resume_from_cli=args.resume_from,
        init_from_cli=args.init_from,
        seed=args.seed,
    )
