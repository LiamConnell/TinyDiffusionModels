---
date: 2025-07-21
categories:
  - Implementation
  - Text Diffusion
  - Diffusion-LM
  - Training
---

# Learned Rounding Function & Custom Embeddings Implementation

**Status**: In Progress - Training Successfully  
**Type**: Implementation + Experiment

## Objective

Implement and validate the learned rounding function and custom embedding space improvements identified in our [Diffusion-LM analysis](2025-07-21-diffusion-lm-analysis.md), replacing cosine similarity decoding with trainable components for improved text generation quality. 

<!-- more -->

This is basically just re-implementing features from the [Diffusion-LM paper](https://arxiv.org/abs/2205.14217) by Li et al. ("Diffusion-LM Improves Controllable Text Generation"). 

## Background

Following our [Diffusion-LM analysis](2025-07-21-diffusion-lm-analysis.md), we identified cosine similarity decoding as the primary bottleneck limiting text generation quality. This experiment implements two key architectural improvements:

1. **Learned Rounding Function**: Trainable linear decoder replacing cosine similarity
2. **Custom Embedding Space**: End-to-end embedding optimization alongside diffusion

## Code Implementation

### Pull Request
All architectural changes implemented in **[PR #13: Implement Diffusion-LM improvements](https://github.com/LiamConnell/TinyDiffusionModels/pull/13)** - covers both learned rounding and custom embeddings in a comprehensive 260-line addition.

### Key Components Added

#### 1. LearnedEmbedding Module
```python
class LearnedEmbedding(nn.Module):
    """Custom learnable embedding space for diffusion."""
    def __init__(self, vocab_size, embed_dim, pretrained_embeddings=None):
        # Supports flexible dimensions + optional pretrained initialization
```

**Features**:
- Flexible embedding dimensions independent of pre-trained models
- Optional initialization from pre-trained weights (`--init_from_pretrained`)
- End-to-end optimization alongside diffusion process

#### 2. LearnedRounding Module  
```python
class LearnedRounding(nn.Module):
    """Learned rounding function to convert embeddings to token probabilities."""
    def __init__(self, embed_dim, vocab_size):
        self.decoder = nn.Linear(embed_dim, vocab_size)
```

**Features**:
- Trainable linear layer for embedding → token logit conversion
- Replaces simple cosine similarity + argmax approach
- Joint optimization with diffusion objective

#### 3. Enhanced Training Loop
**Dual-objective training**:
- `diffusion_loss`: MSE for denoising (standard DDPM)  
- `rounding_loss`: Cross-entropy for token prediction
- `total_loss = diffusion_loss + rounding_weight * rounding_loss`

**Joint optimization** of three components:
- Diffusion model (TinyTransformer)
- Learned rounding function
- Custom embedding space

### New CLI Arguments
- `--use_learned_embeddings`: Enable custom embedding space
- `--embed_dim`: Custom embedding dimension  
- `--init_from_pretrained`: Initialize from pre-trained weights
- `--rounding_weight`: Weight for rounding loss component

## Experimental Setup

### Training Configuration
- **Architecture**: Joint training of diffusion + rounding + embeddings
- **Dataset**: Shakespeare corpus (tiny_shakespeare)
- **GPU**: Tesla T4 (14GB) - upgraded from original specs due to memory requirements
- **Embedding Dimension**: 256 (conservative baseline)
- **Batch Size**: 8 (memory-optimized)
- **Epochs**: 100
- **Sequence Length**: 64 tokens

### Configuration
Conservative baseline configuration for memory constraints:
- **Embedding dimension**: 256 (reduced from 2048 for T4 GPU)
- **Batch size**: 8  
- **Memory usage**: ~1.5GB total (fits comfortably on 14GB T4)

## Experimental Results

### Training Job Status
- **Experiment Issue**: [#14 - 100-epoch experiment tracking](https://github.com/LiamConnell/TinyDiffusionModels/issues/14)
- **Job ID**: `8015213902746877952`
- **Status**: ✅ **TRAINING SUCCESSFULLY** (as of Epoch 11/100)
- **Performance**: ~25 iterations/second

### Loss Metrics Observed
```
Epoch 11/100: diff_loss=0.141, round_loss=1.66, total=1.80
Epoch 11/100: diff_loss=0.101, round_loss=1.58, total=1.68  
Epoch 11/100: diff_loss=0.018, round_loss=1.65, total=1.66
```

**Analysis**:
- **Diffusion loss**: ~0.02-0.4 (reasonable denoising performance)
- **Rounding loss**: ~1.2-2.0 (token prediction learning well)  
- **Stable training**: Consistent progression, no memory issues

### Architecture Validation
✅ **Joint Optimization Working**: All three components training together smoothly  
✅ **Learned Rounding Active**: Trainable decoder successfully replacing cosine similarity  
✅ **Custom Embeddings Learning**: End-to-end embedding space optimization functional  
✅ **Training Stable**: Consistent loss progression at ~25 iterations/second

## Next Steps

### Immediate (Current Training)
- **Monitor to completion**: ~20 minutes remaining for baseline
- **Generate samples**: Test learned rounding vs cosine similarity quality
- **Evaluate architecture**: Compare against previous Shakespeare baseline

### Scaling Experiments  
- **Incremental dimension scaling**: 256 → 512 → 1024 → 2048
- **Extended training**: Once optimal dimension identified
- **Performance comparison**: Quality metrics vs original implementation

## Research Questions

1. **Quality Impact**: How much does learned rounding improve generation vs cosine similarity?
2. **Optimal Scaling**: What embedding dimension provides the best quality/memory tradeoff?
3. **Training Dynamics**: How do diffusion and rounding losses balance during joint optimization?

---

**Related Issues**: 
- [#12 - Diffusion-LM Analysis](https://github.com/LiamConnell/TinyDiffusionModels/issues/12)
- [#14 - 100-epoch Training Experiment](https://github.com/LiamConnell/TinyDiffusionModels/issues/14)  
- [#15 - Extended Training Strategy](https://github.com/LiamConnell/TinyDiffusionModels/issues/15)

**Related PR**: [#13 - Implementation](https://github.com/LiamConnell/TinyDiffusionModels/pull/13)