# Installation Guide

Complete installation guide for the Stick-Gen project.

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Operating System**: macOS, Linux, or Windows
- **Disk Space**: ~120GB for full dataset and training
- **RAM**: 16GB minimum, 32GB recommended for training

## 🚀 Quick Install

### 1. Clone the Repository
```bash
git clone https://github.com/gestura-ai/stick-gen.git
cd stick-gen
```

### 2. Create Virtual Environment (Recommended)
```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

The project requires the following packages:

### Core ML/AI
- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.30.0` - Hugging Face transformers
- `numpy<2` - Numerical computing (pinned to 1.x for compatibility)

### Text Embeddings
- `sentence-transformers>=2.2.0` - For BAAI/bge-large-en-v1.5 embeddings

### Visualization
- `matplotlib>=3.5.0` - Plotting and visualization
- `pycairo>=1.20.0` - Cairo graphics library
- `manim>=0.17.0` - Mathematical animation engine

### Utilities
- `tqdm>=4.65.0` - Progress bars

### SMPL/AMASS Processing
- `chumpy>=0.70` - For SMPL model loading
- `scipy>=1.9.0` - Scientific computing

## 🎯 Optional: AMASS Dataset Setup

If you want to train with AMASS motion capture data:

### 1. Download AMASS Dataset
Follow the [AMASS Download Guide](AMASS_DOWNLOAD_GUIDE.md) to download the dataset.

### 2. Download SMPL Models
Follow the [SMPL-X Download Guide](SMPLX_DOWNLOAD_GUIDE.md) to download SMPL body models.

### 3. Convert AMASS Data
```bash
python scripts/amass/batch_convert_amass.py
```

## ✅ Verify Installation

### Test Basic Functionality
```bash
python3.9 -c "import torch; import transformers; print('✅ Installation successful!')"
```

### Test SMPL Installation (if using AMASS)
```bash
python scripts/validation/verify_smpl_installation.py
```

### Run Quick Test
```bash
python tests/integration/test_pipeline.py
```

## 🐛 Troubleshooting

### NumPy Compatibility Issues
If you encounter NumPy 2.x compatibility issues:
```bash
pip install "numpy<2"
```

### Cairo Installation Issues (macOS)
```bash
brew install cairo pkg-config
pip install pycairo
```

### Cairo Installation Issues (Linux)
```bash
sudo apt-get install libcairo2-dev pkg-config python3-dev
pip install pycairo
```

### SMPL Model Loading Issues
Ensure you have downloaded the SMPL models and placed them in `data/smpl_models/`. See [SMPLX_DOWNLOAD_GUIDE.md](SMPLX_DOWNLOAD_GUIDE.md).

## 📚 Next Steps

After installation:
1. **Read the Architecture**: [docs/architecture/AGENT.md](../architecture/AGENT.md)
2. **Generate Sample**: `./stick-gen "A person walking"`
3. **Train Model**: Follow [docs/training/TRAINING_GUIDE.md](../training/TRAINING_GUIDE.md)

## 🔗 Related Documentation

- [AMASS Download Guide](AMASS_DOWNLOAD_GUIDE.md)
- [SMPL-X Download Guide](SMPLX_DOWNLOAD_GUIDE.md)
- [Training Guide](../training/TRAINING_GUIDE.md)
- [Architecture Documentation](../architecture/AGENT.md)

