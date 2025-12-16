import logging
import os
import sys
from pathlib import Path

import gradio as gr

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.generator import InferenceGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model (lazy loading)
generator = None


def load_model(model_size):
    global generator
    # Map friendly names to config/checkpoint paths
    # In a real scenario, these would point to actual different checkpoints
    # For now, we'll use the default checkpoint if available, or warn user

    checkpoint_path = "model_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        return "⚠️ Model checkpoint not found under 'model_checkpoint.pth'. Using procedural mode."

    try:
        generator = InferenceGenerator(model_path=checkpoint_path)
        return f"✅ Loaded {model_size} model successfully!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def generate_animation(prompt, style, camera_mode, use_llm):
    global generator
    if generator is None:
        # Try to load default
        msg = load_model("Default")
        if "❌" in msg:
            return None, msg

    output_path = "output.mp4"

    try:
        generator.generate(
            prompt=prompt,
            output_path=output_path,
            style=style.lower(),
            camera_mode=camera_mode.lower(),
            use_llm=use_llm,
        )
        return output_path, "✨ Generation Complete!"
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return None, f"❌ Error: {str(e)}"


# Define UI
with gr.Blocks(title="Stick-Gen Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏃‍♂️ Stick-Gen Studio
        ### Text-to-Stick-Figure Animation
        Generates realistic 10-second stick figure animations from natural language.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the action (e.g. 'A person walking angrily')",
                lines=3,
            )

            with gr.Row():
                style_input = gr.Dropdown(
                    choices=["Normal", "Sketch", "Ink", "Neon"],
                    value="Normal",
                    label="Render Style",
                )
                camera_input = gr.Dropdown(
                    choices=["Static", "Dynamic", "Cinematic"],
                    value="Static",
                    label="Camera Mode",
                )

            use_llm_checkbox = gr.Checkbox(label="Use LLM Story Engine", value=False)
            generate_btn = gr.Button("🎬 Generate Animation", variant="primary")

            status_output = gr.Markdown("")

        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Animation")

    generate_btn.click(
        fn=generate_animation,
        inputs=[prompt_input, style_input, camera_input, use_llm_checkbox],
        outputs=[video_output, status_output],
    )

    gr.Examples(
        examples=[
            ["A person walking happily", "Normal", "Static", False],
            ["Ninja running on rooftops", "Sketch", "Dynamic", False],
            ["Robot dancing in the rain", "Neon", "Cinematic", False],
            ["Two people shaking hands", "Ink", "Static", True],
        ],
        inputs=[prompt_input, style_input, camera_input, use_llm_checkbox],
    )

if __name__ == "__main__":
    demo.launch(share=False)
