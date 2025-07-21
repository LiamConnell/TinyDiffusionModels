---
date: 2025-07-18
categories:
  - Baseline
  - Image Diffusion
  - MNIST
---

# MNIST Diffusion Baseline

**Status**: Complete  
**Type**: Baseline

## Objective

Establish a working baseline for image diffusion using the standard DDPM approach on MNIST digits. This serves as a validation of our diffusion implementation before moving to text modalities.

<!-- more -->

## Configuration

- **Model**: SimpleUNet with residual blocks and temporal embeddings
- **Training**: Standard DDPM on MNIST dataset
- **Architecture**: 
  - UNet backbone with down/up sampling
  - Temporal embedding for timestep conditioning
  - Residual connections throughout
- **Dataset**: MNIST handwritten digits (28x28 grayscale)
- **Hardware**: NVIDIA T4
- **Git Commit**: 4422ce927fbf61e226157e4a3f2ac8de91b583bb

## Hypothesis

Standard DDPM should work well for MNIST generation, providing a solid foundation for understanding diffusion mechanics before tackling text generation challenges.

## Results

### Quantitative
- Training converged successfully

### Qualitative  
- Generated samples are recognizable MNIST digits
- Clear progression from noise to structured digits during reverse process

The training progression shows the model learning to generate increasingly coherent MNIST digits:

#### Early Training (Epochs 1-3)
<div style="display: flex; gap: 20px; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="/assets/epoch_001.png" alt="Epoch 1" style="max-width: 200px;">
    <p><strong>Epoch 1</strong><br>Initial noise, barely recognizable patterns</p>
  </div>
  <div style="text-align: center;">
    <img src="/assets/epoch_002.png" alt="Epoch 2" style="max-width: 200px;">
    <p><strong>Epoch 2</strong><br>Some digit-like shapes emerging</p>
  </div>
  <div style="text-align: center;">
    <img src="/assets/epoch_003.png" alt="Epoch 3" style="max-width: 200px;">
    <p><strong>Epoch 3</strong><br>More defined structures appearing</p>
  </div>
</div>

#### Training Progression (Epochs 100-1000)
<div style="display: flex; gap: 20px; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="/assets/epoch_100.png" alt="Epoch 100" style="max-width: 200px;">
    <p><strong>Epoch 100</strong><br>Clear digit shapes, some noise remaining</p>
  </div>
  <div style="text-align: center;">
    <img src="/assets/epoch_500.png" alt="Epoch 500" style="max-width: 200px;">
    <p><strong>Epoch 500</strong><br>Well-formed digits, improved clarity</p>
  </div>
  <div style="text-align: center;">
    <img src="/assets/epoch_1000.png" alt="Epoch 1000" style="max-width: 200px;">
    <p><strong>Epoch 1000</strong><br>High-quality, recognizable MNIST digits</p>
  </div>
</div>

**Key Observations:**
- **Epochs 1-3**: Model learns basic structure and shape concepts
- **Epoch 100**: Recognizable digits with some artifacts
- **Epochs 500-1000**: Converged to high-quality digit generation

## Next Steps

- Move to text diffusion experiments using similar architecture principles
- Investigate embedding space approaches for text generation

---

**Sample Generation**:
```bash
uv run python -m src.mnist --sample
```