# Stick-Gen Documentation

Welcome to the Stick-Gen documentation! This directory contains comprehensive documentation for the stick figure animation generation system.

## 📚 Documentation Structure

### [Setup](setup/)
Installation and environment setup guides:
- **[INSTALLATION.md](setup/INSTALLATION.md)** - Complete installation guide
- **[AMASS_DOWNLOAD_GUIDE.md](setup/AMASS_DOWNLOAD_GUIDE.md)** - How to download AMASS dataset
- **[SMPLX_DOWNLOAD_GUIDE.md](setup/SMPLX_DOWNLOAD_GUIDE.md)** - SMPL-X model setup
- **[SMPLX_QUICK_REFERENCE.md](setup/SMPLX_QUICK_REFERENCE.md)** - Quick reference for SMPL-X

### [Architecture](architecture/)
System architecture and design documentation:
- **[SPATIAL_MOVEMENT_PLAN.md](architecture/SPATIAL_MOVEMENT_PLAN.md)** - Spatial movement system design
- **[ADVANCED_IMPROVEMENTS.md](architecture/ADVANCED_IMPROVEMENTS.md)** - All 15 improvements explained

### [Training](training/)
Training pipeline and optimization guides:
- **[CONFIGURATION.md](training/CONFIGURATION.md)** - Training configuration guide for different hardware setups
- **[TRAINING_GUIDE.md](training/TRAINING_GUIDE.md)** - Complete training guide
- **[CPU_TRAINING_PLAN.md](training/CPU_TRAINING_PLAN.md)** - CPU-optimized training plan
- **[TRAINING_ARCHITECTURE_CLARIFICATION.md](training/TRAINING_ARCHITECTURE_CLARIFICATION.md)** - Architecture decisions
- **[TRAINING_PERFORMANCE_BOTTLENECK_REPORT.md](training/TRAINING_PERFORMANCE_BOTTLENECK_REPORT.md)** - Performance analysis
- **Configuration Files**:
  - `../configs/base.yaml` - Default configuration (15.8M params, CPU)
  - `../configs/small.yaml` - Budget CPU configuration (5.6M params)
  - `../configs/large.yaml` - GPU configuration (28M params)

### [Features](features/)
Feature-specific documentation:
- **[FACIAL_EXPRESSIONS.md](features/FACIAL_EXPRESSIONS.md)** - Facial expression system
- **[SPEECH_ANIMATION.md](features/SPEECH_ANIMATION.md)** - Speech animation system
- **[PHYSICS_SYSTEM.md](features/PHYSICS_SYSTEM.md)** - Physics-aware motion
- **[ACTION_CONDITIONING.md](features/ACTION_CONDITIONING.md)** - Action conditioning system

### [AMASS](amass/)
AMASS dataset integration documentation:
- **[AMASS_INTEGRATION.md](amass/AMASS_INTEGRATION.md)** - AMASS integration overview
- **[AMASS_CONVERSION_GUIDE.md](amass/AMASS_CONVERSION_GUIDE.md)** - Dataset conversion guide
- **[AMASS_FORMAT_AUTO_DETECTION.md](amass/AMASS_FORMAT_AUTO_DETECTION.md)** - Format detection system
- **[AMASS_TROUBLESHOOTING.md](amass/AMASS_TROUBLESHOOTING.md)** - Common issues and solutions

### [Reports](reports/)
Historical completion and validation reports:
- **[PHASE_5_COMPLETION_REPORT.md](reports/PHASE_5_COMPLETION_REPORT.md)** - Facial expressions implementation
- **[PHASE_7_COMPLETION_REPORT.md](reports/PHASE_7_COMPLETION_REPORT.md)** - Speech animation implementation
- **[PHASE_9_FINAL_REPORT.md](reports/PHASE_9_FINAL_REPORT.md)** - Final integration and validation
- **[VALIDATION_REPORT.md](reports/VALIDATION_REPORT.md)** - System validation results
- **[FINAL_VALIDATION_SUMMARY.md](reports/FINAL_VALIDATION_SUMMARY.md)** - Final validation summary

### [RunPod](runpod/)
Cloud GPU training deployment guides:
- **[README.md](runpod/README.md)** - RunPod documentation index and quick start
- **[RUNPOD_DEPLOYMENT.md](runpod/RUNPOD_DEPLOYMENT.md)** - Comprehensive deployment guide
- **[QUICK_START.md](runpod/QUICK_START.md)** - Get started in 5 minutes
- **[RUNPOD_PIPELINE_IMPROVEMENTS.md](runpod/RUNPOD_PIPELINE_IMPROVEMENTS.md)** - Latest enhancements
- **[ENABLE_GROK_API.md](runpod/ENABLE_GROK_API.md)** - LLM-enhanced dataset generation
- **[DEPLOYMENT_FIX_SUMMARY.md](runpod/DEPLOYMENT_FIX_SUMMARY.md)** - Deployment script fixes

### [Export](export/)
Model export and deployment guides:
- **[HUGGINGFACE_UPLOAD.md](export/HUGGINGFACE_UPLOAD.md)** - Upload trained model to Hugging Face Hub
- **[MODEL_RELEASE_GUIDE.md](export/MODEL_RELEASE_GUIDE.md)** - Complete model versioning and release strategy
- **[MODEL_EXPORT_GUIDE.md](export/MODEL_EXPORT_GUIDE.md)** - How to export trained models
- **[MODEL_CARD_TEMPLATE.md](export/MODEL_CARD_TEMPLATE.md)** - Model card template
- **Model Cards**:
  - `../model_cards/small.md` - Small variant (5.6M params)
  - `../model_cards/base.md` - Base variant (15.8M params)
  - `../model_cards/large.md` - Large variant (28M params)

## 🚀 Quick Start

1. **Installation**: Start with [setup/INSTALLATION.md](setup/INSTALLATION.md)
2. **Architecture**: Read [architecture/AGENT.md](architecture/AGENT.md) for system overview
3. **Cloud Training**: Use [runpod/QUICK_START.md](runpod/QUICK_START.md) for RunPod deployment
4. **Local Training**: Follow [training/TRAINING_GUIDE.md](training/TRAINING_GUIDE.md) to train models
5. **Features**: Explore [features/](features/) for specific feature documentation
6. **Examples**: See [../examples/README.md](../examples/README.md) for usage examples
7. **Release**: Use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before publishing

## 📖 Additional Resources

- **[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)** - Pre-release verification checklist
- **[CHANGELOG.md](../CHANGELOG.md)** - Project changelog
- **[CITATIONS.md](../CITATIONS.md)** - Dataset and model citations
- **[README.md](../README.md)** - Main project readme
- **[tests/README.md](../tests/README.md)** - Testing documentation
- **[examples/README.md](../examples/README.md)** - Example scripts and tutorials

## 🤝 Contributing

For contribution guidelines, see the main [README.md](../README.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

