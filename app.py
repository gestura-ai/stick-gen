# Stick-Gen Demo - HuggingFace Spaces Entry Point
# Gestura AI - https://gestura.ai
#
# This file serves as the entry point for HuggingFace Spaces deployment.
# Deploy with: huggingface-cli repo create GesturaAI/stick-gen-demo --type space --space_sdk gradio

import sys
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from examples.video_generation_demo import create_demo

# Create and launch the Gradio demo
demo = create_demo()

if __name__ == "__main__":
    demo.launch()
