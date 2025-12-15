import gradio as gr
import torch
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock imports for demo if model not available locally
try:
    from src.inference.generator import InferenceGenerator
    from src.model.transformer import StickFigureTransformer
    GENERATOR_AVAILABLE = True
except ImportError:
    print("Warning: Stick-Gen modules not found. Using mock generator for UI demo.")
    GENERATOR_AVAILABLE = False
    InferenceGenerator = None


class DemoApp:
    def __init__(self):
        self.generator = None
        self.model_loaded = False
        
    def load_model(self, model_size="medium", device="cpu"):
        if not GENERATOR_AVAILABLE:
            return "❌ Project modules not found. Please run from repository root."

        try:
            # Load actual model
            checkpoint_path = f"checkpoints/stick-gen-{model_size}.pth"
            if not os.path.exists(checkpoint_path):
                # Try alternative path
                checkpoint_path = f"model_checkpoint.pth"
                if not os.path.exists(checkpoint_path):
                    return f"⚠️ Model checkpoint not found. Using demo mode.\nExpected: checkpoints/stick-gen-{model_size}.pth"

            self.generator = InferenceGenerator(
                model_path=checkpoint_path,
                use_diffusion=False
            )
            self.model_loaded = True
            return f"✅ Model stick-gen-{model_size} loaded from {checkpoint_path}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def generate(self, prompt, duration, randomness, style):
        if not GENERATOR_AVAILABLE:
            return None, "⚠️ UI Demo Only (Core modules missing)"

        if not self.model_loaded:
            return None, "⚠️ Please load a model first using the 'Load Model' button"

        try:
            # Generate video using actual model
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                output_path = tmp.name

            self.generator.generate(
                prompt=prompt,
                output_path=output_path,
                style=style.lower()
            )

            if os.path.exists(output_path):
                return output_path, f"✅ Generated: {prompt}\nDuration: {duration}s | Style: {style}"
            else:
                return None, "❌ Generation completed but output file not found"

        except Exception as e:
            return None, f"❌ Generation failed: {str(e)}"


def create_demo():
    app = DemoApp()
    
    with gr.Blocks(title="Gestura Stick-Gen Demo") as demo:
        gr.Markdown(
            """
            # 🏃‍♂️ Stick-Gen: Text-to-Stick-Figure Animation
            Generate realistic stick figure animations from text prompts using Gestura AI's transformer model.
            """
        )
        
        with gr.Row():
            with gr.Column():
                # Input Controls
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="A person doing a backflip...",
                    lines=2
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    duration = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                        label="Duration (seconds)"
                    )
                    randomness = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.7,
                        label="Creativity (Temperature)"
                    )
                    style = gr.Dropdown(
                        choices=["Normal", "Sketch", "Neon", "Ink"],
                        value="Normal",
                        label="Render Style"
                    )
                    model_size = gr.Radio(
                        choices=["small", "medium", "large"],
                        value="medium",
                        label="Model Size"
                    )
                
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="secondary")
                    gen_btn = gr.Button("Generate Animation", variant="primary")
                
                status_output = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column():
                # Output Display
                video_output = gr.Video(label="Generated Animation")
                
        # Event Handlers
        load_btn.click(
            fn=app.load_model,
            inputs=[model_size],
            outputs=[status_output]
        )
        
        gen_btn.click(
            fn=app.generate,
            inputs=[prompt, duration, randomness, style],
            outputs=[video_output, status_output]
        )
        
        gr.Markdown(
            """
            ### model details
            - **Architecture**: Transformer (Motion Planning) + Diffusion (Refinement)
            - **Training Data**: AMASS, HumanML3D, Synthetic
            - **License**: Apache 2.0
            """
        )

    return demo

if __name__ == "__main__":
    try:
        demo = create_demo()
        demo.launch()
    except ImportError:
        print("Please install gradio: pip install gradio")
