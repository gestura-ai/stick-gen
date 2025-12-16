"""
Stick-Gen RunPod Serverless Handler
Gestura AI - Text-to-Animation Inference

This handler processes text prompts and generates stick figure animations
using the trained StickFigureTransformer model.
"""

import os

import torch

import runpod
from runpod import RunPodLogger

# Initialize logger
log = RunPodLogger()

# Configuration from environment
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "base")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/model_checkpoint.pth")
CONFIG_PATH = os.getenv("CONFIG_PATH", f"/workspace/configs/{MODEL_VARIANT}.yaml")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Global model and embedder (loaded once at startup)
model = None
text_embedder = None


def load_model():
    """Load the StickFigureTransformer model and text embedder."""
    global model, text_embedder

    log.info(f"Loading Stick-Gen model (variant: {MODEL_VARIANT})...")
    log.info(f"Model path: {MODEL_PATH}")
    log.info(f"Device: {DEVICE}")

    # Import model architecture
    import sys

    sys.path.insert(0, "/workspace")
    import yaml
    from sentence_transformers import SentenceTransformer

    from src.model.transformer import StickFigureTransformer

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]

    # Initialize model
    model = StickFigureTransformer(
        input_dim=model_config["input_dim"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        output_dim=model_config.get("output_dim", model_config["input_dim"]),
        embedding_dim=model_config["embedding_dim"],
        dropout=model_config.get("dropout", 0.1),
        num_actions=model_config.get("num_actions", 64),
    )

    # Load checkpoint
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        log.info("Model checkpoint loaded successfully")
    else:
        log.warn(f"No checkpoint found at {MODEL_PATH}, using random weights")

    model.to(DEVICE)
    model.eval()

    # Load text embedder
    log.info("Loading text embedder (BAAI/bge-large-en-v1.5)...")
    text_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    text_embedder.to(DEVICE)

    log.info("Model and embedder loaded successfully")
    return model, text_embedder


def generate_animation(prompt: str, num_frames: int = 60, camera_data: dict = None):
    """Generate stick figure animation from text prompt."""
    global model, text_embedder

    if model is None or text_embedder is None:
        load_model()

    # Encode text prompt
    with torch.no_grad():
        text_embedding = text_embedder.encode(prompt, convert_to_tensor=True)
        text_embedding = text_embedding.unsqueeze(0).to(DEVICE)  # [1, 1024]

        # Initialize motion sequence (start from neutral pose)
        # 10 joints × 2 coords = 20 dimensions
        motion = torch.zeros(num_frames, 1, 20, device=DEVICE)

        # Prepare camera data if provided
        camera_tensor = None
        if camera_data:
            camera_tensor = torch.tensor(
                [
                    [
                        camera_data.get("x", 0.0),
                        camera_data.get("y", 0.0),
                        camera_data.get("zoom", 1.0),
                    ]
                    for _ in range(num_frames)
                ],
                device=DEVICE,
            ).unsqueeze(
                1
            )  # [seq_len, 1, 3]

        # Generate animation
        outputs = model(motion, text_embedding, camera_data=camera_tensor)

        # Extract pose predictions
        if isinstance(outputs, dict):
            generated_motion = outputs.get("pose", outputs.get("output", motion))
        else:
            generated_motion = outputs

        # Convert to list for JSON serialization
        animation_data = generated_motion.squeeze(1).cpu().numpy().tolist()

    return {
        "frames": animation_data,
        "num_frames": num_frames,
        "joints": 10,
        "prompt": prompt,
    }


def handler(job):
    """RunPod serverless handler function."""
    job_input = job.get("input", {})

    # Validate input
    if "prompt" not in job_input:
        return {"error": "Missing required field: 'prompt'"}

    prompt = job_input["prompt"]
    num_frames = job_input.get("num_frames", 60)
    camera_data = job_input.get("camera", None)

    log.info(f"Generating animation for prompt: '{prompt[:50]}...'")
    log.info(f"Frames: {num_frames}, Camera: {camera_data is not None}")

    try:
        result = generate_animation(prompt, num_frames, camera_data)
        log.info("Animation generated successfully")
        return {"status": "success", "animation": result}
    except Exception as e:
        log.error(f"Generation failed: {str(e)}")
        return {"error": str(e), "status": "failed"}


# Load model at startup (outside handler for efficiency)
if os.getenv("RUNPOD_POD_ID"):
    log.info("Running on RunPod - loading model at startup...")
    load_model()

# Start serverless worker
runpod.serverless.start({"handler": handler})
