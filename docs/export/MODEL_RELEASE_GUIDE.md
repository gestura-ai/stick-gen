# Model Release Guide

This guide documents the model versioning and release strategy for Stick-Gen, following Hugging Face and open-source ML best practices.

## Table of Contents

- [Versioning Strategy](#versioning-strategy)
- [Model Variants](#model-variants)
- [Release Process](#release-process)
- [Repository Structure](#repository-structure)
- [Best Practices](#best-practices)
- [Examples from Popular Models](#examples-from-popular-models)

## Versioning Strategy

### Semantic Versioning

Stick-Gen follows [Semantic Versioning 2.0.0](https://semver.org/) for model releases:

```
MAJOR.MINOR.PATCH (e.g., 1.2.3)
```

- **MAJOR** (1.x.x): Breaking changes, major architecture changes, incompatible API changes
- **MINOR** (x.1.x): New features, improved performance, backward-compatible changes
- **PATCH** (x.x.1): Bug fixes, documentation updates, minor improvements

### Version Examples

| Version | Type | Description |
|---------|------|-------------|
| 1.0.0 | MAJOR | Initial release |
| 1.1.0 | MINOR | Added diffusion refinement |
| 1.1.1 | PATCH | Fixed temporal consistency bug |
| 2.0.0 | MAJOR | Changed to 3D output (breaking change) |

### When to Increment

**MAJOR version** when you:
- Change model architecture significantly (e.g., 2D → 3D)
- Change input/output format (breaking API change)
- Remove or rename model outputs
- Change embedding model (incompatible with old embeddings)

**MINOR version** when you:
- Add new features (e.g., new expression types)
- Improve model performance (better training, more data)
- Add new decoder heads (backward compatible)
- Increase model capacity (more layers/parameters)

**PATCH version** when you:
- Fix bugs in model or training code
- Update documentation
- Improve training stability
- Fix numerical issues

## Model Variants

### Variant Naming Convention

Stick-Gen uses a **separate repository per variant** approach, following the pattern used by BERT, GPT-2, and other popular models:

```
GesturaAI/stick-gen-{variant}
```

### Available Variants

| Variant | Repository | Parameters | Use Case |
|---------|-----------|------------|----------|
| Small | `GesturaAI/stick-gen-small` | 5.6M | Budget CPU, edge devices |
| Base | `GesturaAI/stick-gen-base` | 15.8M | Standard deployment |
| Large | `GesturaAI/stick-gen-large` | 28M | High-quality, GPU |

### When to Create New Variant vs. Update Existing

**Create NEW variant repository** when:
- ✅ Different model size (parameters, layers, dimensions)
- ✅ Different hardware target (CPU vs. GPU)
- ✅ Different quality/speed tradeoff
- ✅ Different training configuration (epochs, data)

**Update EXISTING repository** when:
- ✅ Bug fixes to existing variant
- ✅ Improved training of same architecture
- ✅ Documentation updates
- ✅ Minor performance improvements

### Variant Configuration

Each variant has its own configuration file:

- `configs/small.yaml` → stick-gen-small
- `configs/base.yaml` → stick-gen-base
- `configs/large.yaml` → stick-gen-large

## Release Process

### Pre-Release Checklist

Before releasing a model, complete the following:

#### 1. Training Validation
- [ ] Model trained to convergence
- [ ] Validation loss plateaued
- [ ] No signs of overfitting
- [ ] Training logs saved and reviewed

#### 2. Evaluation
- [ ] Quantitative metrics computed on test set
- [ ] Qualitative assessment completed
- [ ] Benchmark prompts tested
- [ ] Failure cases documented

#### 3. Model Artifacts
- [ ] Best checkpoint saved
- [ ] Model size verified (matches expected parameters)
- [ ] Configuration file included
- [ ] Training logs archived

#### 4. Documentation
- [ ] Model card completed with all sections
- [ ] Metrics updated (no "TBD" values)
- [ ] Example outputs generated
- [ ] Limitations documented
- [ ] Known issues listed

#### 5. Code Quality
- [ ] All tests passing
- [ ] Code linted and formatted
- [ ] Dependencies updated in requirements.txt
- [ ] README updated

#### 6. Legal & Ethical
- [ ] License file included (MIT)
- [ ] Citations complete (AMASS, BGE, etc.)
- [ ] Ethical considerations documented
- [ ] Bias analysis completed

### Release Steps

#### Step 1: Prepare Model Files

```bash
# Validate checkpoint
python scripts/validation/validate_checkpoint.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml

# Generate example outputs
python scripts/validation/generate_examples.py \
  --checkpoint checkpoints/best_model.pt \
  --output examples/
```

#### Step 2: Update Model Card

```bash
# Update metrics in model_cards/base.md
# Replace all "TBD" with actual values
# Add example outputs
# Update changelog
```

#### Step 3: Upload to Hugging Face

```bash
# Upload model (choose variant)
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model.pt \
  --variant base \
  --version 1.0.0 \
  --repo-name GesturaAI/stick-gen-base
```

#### Step 4: Create GitHub Release

```bash
# Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Create GitHub release with:
# - Release notes
# - Changelog
# - Link to HF model
# - Example outputs
```

#### Step 5: Update Documentation

```bash
# Update main README.md
# Update docs/README.md
# Add release notes to CHANGELOG.md
```

### Post-Release

After releasing:

1. **Monitor Issues**: Watch for bug reports and user feedback
2. **Update Metrics**: Add community-reported metrics to model card
3. **Collect Examples**: Gather user-generated examples
4. **Plan Next Release**: Document improvements for next version

## Repository Structure

### Hugging Face Hub Structure

Each variant repository should contain:

```
GesturaAI/stick-gen-{variant}/
├── README.md                 # Model card (auto-generated from model_cards/{variant}.md)
├── pytorch_model.bin         # Model checkpoint
├── config.yaml               # Training configuration (from configs/{variant}.yaml)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── CITATIONS.md              # Complete citations
├── src/                      # Source code
│   ├── model/
│   ├── inference/
│   └── data_gen/
└── examples/                 # Example outputs
    ├── walk_wave.mp4
    ├── run_jump.mp4
    └── ...
```

### GitHub Repository Structure

Main repository at `gestura-ai/stick-gen`:

```
stick-gen/
├── README.md
├── LICENSE
├── CITATIONS.md
├── CHANGELOG.md
├── requirements.txt
├── configs/                 # Configuration files
│   ├── small.yaml          # Small variant
│   ├── base.yaml           # Base variant
│   └── large.yaml          # Large variant
├── model_cards/            # Model card templates
│   ├── small.md            # Small variant card
│   ├── base.md             # Base variant card
│   └── large.md            # Large variant card
├── examples/               # Example scripts
├── src/
├── docs/
├── tests/
└── scripts/
    └── push_to_huggingface.py   # Upload script
```

## Best Practices

### 1. Separate Repositories per Variant

**Why**: Follows industry standard (BERT, GPT-2, Stable Diffusion)

**Benefits**:
- Clear separation of different model sizes
- Independent versioning per variant
- Easier for users to find the right model
- Better discoverability on Hugging Face Hub

**Example**:
```
huggingface.co/GesturaAI/stick-gen-small
huggingface.co/GesturaAI/stick-gen-base
huggingface.co/GesturaAI/stick-gen-large
```

### 2. Consistent Naming Convention

**Pattern**: `{organization}/{model-name}-{variant}`

**Examples**:
- ✅ `GesturaAI/stick-gen-small`
- ✅ `GesturaAI/stick-gen-base`
- ✅ `GesturaAI/stick-gen-large`
- ❌ `GesturaAI/stick-gen-5.6m` (avoid parameter count in name)
- ❌ `GesturaAI/stick-gen-cpu` (avoid hardware in name)

### 3. Version Tags

Use Git tags for versioning:

```bash
# Tag format: v{MAJOR}.{MINOR}.{PATCH}
git tag -a v1.0.0 -m "Initial release"
git tag -a v1.1.0 -m "Added diffusion refinement"
git tag -a v1.1.1 -m "Fixed temporal consistency bug"
```

### 4. Comprehensive Model Cards

Include all sections:
- Model Details
- Intended Uses & Limitations
- Training Details
- Evaluation
- Technical Specifications
- Environmental Impact
- Ethical Considerations
- Citation

### 5. Reproducibility

Provide everything needed to reproduce:
- Complete training code
- Configuration files
- Dataset generation scripts
- Exact dependency versions
- Training logs

### 6. Example Outputs

Include diverse examples:
- Simple actions
- Complex actions
- Multi-step sequences
- Failure cases
- Edge cases

## Examples from Popular Models

### BERT Family

**Repository Structure**:
```
huggingface.co/bert-base-uncased
huggingface.co/bert-large-uncased
huggingface.co/bert-base-cased
huggingface.co/bert-large-cased
```

**Naming Pattern**: `{model}-{size}-{variant}`

**Lessons**:
- Separate repository per variant
- Clear size indicators (base, large)
- Consistent naming across variants

### GPT-2 Family

**Repository Structure**:
```
huggingface.co/gpt2              # 124M (base)
huggingface.co/gpt2-medium       # 355M
huggingface.co/gpt2-large        # 774M
huggingface.co/gpt2-xl           # 1.5B
```

**Naming Pattern**: `{model}-{size}` (base has no suffix)

**Lessons**:
- Base model has no size suffix
- Clear progression: medium → large → xl
- Parameter counts in model card, not name

### Stable Diffusion Family

**Repository Structure**:
```
huggingface.co/stabilityai/stable-diffusion-v1-4
huggingface.co/stabilityai/stable-diffusion-v1-5
huggingface.co/stabilityai/stable-diffusion-2-1
```

**Naming Pattern**: `{model}-v{major}-{minor}`

**Lessons**:
- Version number in repository name
- Separate repos for major versions
- Clear upgrade path

### Stick-Gen Approach

**Repository Structure**:
```
huggingface.co/GesturaAI/stick-gen-small
huggingface.co/GesturaAI/stick-gen-base
huggingface.co/GesturaAI/stick-gen-large
```

**Naming Pattern**: `stick-gen-{size}`

**Rationale**:
- Follows BERT/GPT-2 pattern
- Clear size indicators
- Base variant is default recommendation
- Version in tags, not repository name

## Updating Existing Models

### Minor Updates (v1.0.0 → v1.1.0)

1. Train improved model
2. Update model card with new metrics
3. Upload new checkpoint
4. Create new Git tag
5. Update changelog

```bash
# Upload updated model
python scripts/push_to_huggingface.py \
  --checkpoint checkpoints/best_model_v1.1.0.pt \
  --variant base \
  --version 1.1.0 \
  --repo-name GesturaAI/stick-gen-base
```

### Major Updates (v1.x.x → v2.0.0)

For breaking changes, consider:

**Option 1**: Update existing repository with clear migration guide
```
huggingface.co/GesturaAI/stick-gen-base (v2.0.0)
```

**Option 2**: Create new repository for v2
```
huggingface.co/GesturaAI/stick-gen-v2-base
```

**Recommendation**: Option 1 for most cases, Option 2 only if v1 and v2 need to coexist

## Troubleshooting

### Issue: Model size doesn't match expected parameters

**Solution**: Verify configuration matches variant
```bash
python scripts/validation/count_parameters.py \
  --checkpoint checkpoints/best_model.pt \
  --expected 15800000
```

### Issue: Upload fails due to large file size

**Solution**: Use Git LFS
```bash
cd hf_upload
git lfs install
git lfs track "*.bin"
git add .gitattributes pytorch_model.bin
git commit -m "Add model checkpoint"
git push
```

### Issue: Model card has "TBD" values

**Solution**: Run evaluation script
```bash
python scripts/validation/evaluate_model.py \
  --checkpoint checkpoints/best_model.pt \
  --output metrics.json
```

## See Also

- [HUGGINGFACE_UPLOAD.md](HUGGINGFACE_UPLOAD.md) - Detailed upload instructions
- [MODEL_EXPORT_GUIDE.md](MODEL_EXPORT_GUIDE.md) - Model export guide
- [Hugging Face Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Semantic Versioning](https://semver.org/)
- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)

