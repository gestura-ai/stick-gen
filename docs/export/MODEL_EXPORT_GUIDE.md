# Modern Model Export Formats Guide

Complete guide to exporting stick-gen models using industry-standard formats for maximum compatibility, security, and deployment flexibility.

---

## 📋 **Overview**

**Current State**: PyTorch `.pth` files (pickle-based, security risks, PyTorch-only)  
**Target State**: Multi-format support (Hugging Face, ONNX, Safetensors)

**Model**: StickFigureTransformer (15.5M parameters)  
**Use Cases**: Sharing, deployment, model hubs, cross-platform inference

---

## 🎯 **Recommended Formats**

### **1. Hugging Face Format (PRIMARY - Industry Standard)** ⭐⭐⭐

**Used by**: GPT-2, BERT, Llama, Mistral, Stable Diffusion, all modern LLMs

**Components**:
- `model.safetensors` - Secure weight storage (no pickle, no code execution)
- `config.json` - Model architecture configuration
- `README.md` - Model card with documentation, usage, metrics
- `tokenizer_config.json` (optional) - For text models

**Why Better than .pth**:
- ✅ **Security**: Safetensors cannot execute arbitrary code (unlike pickle)
- ✅ **Speed**: 2-3x faster loading than pickle
- ✅ **Memory**: Zero-copy loading, lower memory footprint
- ✅ **Compatibility**: Works with Hugging Face Hub, Transformers library
- ✅ **Discoverability**: Model cards make models searchable and documented
- ✅ **Versioning**: Easy to track model versions and updates

**Industry Adoption**:
- Hugging Face Hub: 500k+ models use this format
- Stability AI: All Stable Diffusion models
- Meta: Llama 2, Llama 3
- Mistral AI: All Mistral models
- Anthropic: Claude model exports

**File Structure**:
```
stick-gen-model/
├── model.safetensors          # Model weights (secure)
├── config.json                # Architecture config
├── README.md                  # Model card
├── training_args.json         # Training hyperparameters
└── generation_config.json     # Generation settings (optional)
```

---

### **2. ONNX Format (SECONDARY - Cross-Platform Deployment)** ⭐⭐

**Used by**: Production deployments, edge devices, cross-platform inference

**Components**:
- `model.onnx` - ONNX graph representation
- `config.json` - Model metadata

**Why Better than .pth**:
- ✅ **Platform-agnostic**: Works with TensorRT, ONNX Runtime, CoreML, TensorFlow
- ✅ **Optimized**: Graph optimizations for faster inference
- ✅ **Hardware acceleration**: Easy integration with GPUs, TPUs, NPUs
- ✅ **Mobile/Edge**: Deploy to iOS, Android, embedded systems
- ✅ **Language-agnostic**: Use from C++, C#, Java, JavaScript

**Deployment Targets**:
- ONNX Runtime (CPU/GPU)
- TensorRT (NVIDIA GPUs)
- CoreML (Apple devices)
- OpenVINO (Intel hardware)
- DirectML (Windows)

**Use Cases**:
- Production web services
- Mobile apps (iOS/Android)
- Edge devices (Raspberry Pi, Jetson)
- Browser inference (ONNX.js)

---

### **3. TorchScript Format (OPTIONAL - PyTorch Production)** ⭐

**Used by**: PyTorch production deployments, C++ inference

**Components**:
- `model.pt` - TorchScript serialized model
- `config.json` - Model metadata

**Why Better than .pth**:
- ✅ **No Python dependency**: Run in C++ environments
- ✅ **Optimized**: JIT compilation for faster inference
- ✅ **Mobile**: PyTorch Mobile for iOS/Android
- ✅ **Deployment**: Easier production deployment

**Use Cases**:
- C++ inference servers
- PyTorch Mobile apps
- Environments without Python

---

### **4. Safetensors Only (ALTERNATIVE - Lightweight)** ⭐⭐

**Used by**: When you want security without full Hugging Face integration

**Components**:
- `model.safetensors` - Just the weights
- `config.json` - Architecture config

**Why Better than .pth**:
- ✅ **Security**: No pickle vulnerabilities
- ✅ **Speed**: Fast loading
- ✅ **Simple**: Minimal dependencies

**Use Cases**:
- Internal model sharing
- Security-conscious environments
- When you don't need Hugging Face Hub

---

## 📊 **Format Comparison**

| Format | Security | Speed | Compatibility | Deployment | Hub Integration |
|--------|----------|-------|---------------|------------|-----------------|
| **.pth (current)** | ❌ Low | ⚠️ Medium | PyTorch only | Limited | None |
| **Hugging Face** | ✅ High | ✅ Fast | Excellent | Good | ✅ Full |
| **ONNX** | ✅ High | ✅ Very Fast | Universal | ✅ Excellent | Partial |
| **TorchScript** | ✅ High | ✅ Fast | PyTorch | Good | None |
| **Safetensors** | ✅ High | ✅ Fast | Good | Good | Partial |

---

## 🚀 **Recommended Strategy for Stick-Gen**

### **Phase 1: Hugging Face Format (PRIMARY)**
Export all models to Hugging Face format for:
- Sharing on Hugging Face Hub
- Documentation and discoverability
- Security and speed improvements
- Community adoption

### **Phase 2: ONNX Export (DEPLOYMENT)**
Export to ONNX for:
- Production web services
- Cross-platform deployment
- Hardware acceleration
- Mobile/edge deployment

### **Phase 3: Keep .pth (BACKWARD COMPATIBILITY)**
Maintain .pth support during transition:
- Existing checkpoints still work
- Gradual migration
- No breaking changes

---

## 📦 **What Changes Are Needed**

### **1. Dependencies**
```bash
pip install safetensors huggingface-hub onnx onnxruntime
```

### **2. Export Scripts**
Create `export_model.py` to convert .pth → all formats

### **3. Inference Updates**
Update `src/inference/generator.py` to support multiple formats:
```python
# Auto-detect format and load accordingly
if path.endswith('.safetensors'):
    load_safetensors(path)
elif path.endswith('.onnx'):
    load_onnx(path)
elif path.endswith('.pth'):
    load_pth(path)  # Backward compatibility
```

### **4. Model Card**
Create `README.md` with:
- Model description
- Training details
- Usage examples
- Performance metrics
- Limitations

---

## 🔄 **Migration Path**

### **Step 1: Export Existing Checkpoints**
```bash
python export_model.py \
    --input checkpoint_epoch_50.pth \
    --output stick-gen-v1 \
    --formats safetensors onnx
```

### **Step 2: Update Inference Code**
Add multi-format loading support

### **Step 3: Test Compatibility**
Verify all formats produce identical outputs

### **Step 4: Publish to Hub**
```bash
huggingface-cli login
python export_model.py --push-to-hub gestura-ai/stick-gen-v1
```

### **Step 5: Deprecate .pth (Optional)**
After transition period, make Hugging Face format primary

---

**Next**: See `export_model.py` for implementation details.

