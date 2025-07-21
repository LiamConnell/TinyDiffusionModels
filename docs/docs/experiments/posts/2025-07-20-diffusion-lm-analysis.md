---
date: 2025-07-20
categories:
  - Research
  - Text Diffusion
  - Literature Review
---

# Diffusion-LM vs Current Implementation Analysis

**Status**: Complete  
**Type**: Research

## Objective

Comprehensive comparison between our current text diffusion implementation and the Diffusion-LM paper approach to identify potential improvements and architectural differences that could enhance text generation quality.

<!-- more -->

## Background

Following poor text generation quality in our Shakespeare baseline experiments, this research investigates how our approach differs from established methods in the literature, specifically focusing on the [Diffusion-LM paper](https://arxiv.org/abs/2205.14217) by Li et al. ("Diffusion-LM Improves Controllable Text Generation").

## Key Findings

### Architectural Differences Identified

#### 1. Token Decoding Strategy
- **Current Implementation**: Simple cosine similarity + argmax for embedding-to-token conversion
- **Diffusion-LM**: Learned softmax rounding function trained end-to-end
- **Implication**: Our decoding bottleneck may be addressable through learned mappings

#### 2. Embedding Space / Training Targets
- **Current Implementation**: Pre-trained embeddings (Gemma-2b-it) as diffusion target
- **Diffusion-LM**: Custom embedding space learned jointly with diffusion process
- **Implication**: Trade-off between leveraging pre-trained knowledge vs. task-specific optimization

### Critical Insights

#### Decoding as Primary Bottleneck
Our hypothesis that embedding-to-token decoding is the main quality bottleneck aligns with Diffusion-LM's emphasis on learned rounding functions. The paper's approach suggests that:
- Simple nearest-neighbor decoding loses semantic information
- Learned mappings can preserve diffusion process benefits through to final tokens
- End-to-end training of decoding improves coherence

#### Embedding Space Considerations
- **Advantage of Pre-trained Embeddings**: Rich semantic representations, faster convergence
- **Advantage of Custom Space**: Optimized for diffusion process, potentially better quality
- **Research Question**: Can we get best of both worlds through fine-tuning approaches?

## Potential Improvements for Our Implementation

### High-Priority Enhancements
1. **Learned Rounding Function**: Replace cosine similarity with trainable softmax mapping
2. **Custom Embedding Space**

### Other Enhancements
2. **Fluency Regularization**: Add explicit regularization terms for linguistic coherence
3. **Gradient-based Control**: Implement controllable generation during diffusion

### Implementation Complexity Analysis
- **Learned Rounding**: Medium complexity, high potential impact
- **Gradient Control**: High complexity, medium potential impact  
- **Custom Embedding Space**: High complexity, uncertain impact given our pre-trained approach

## Research Questions Raised

1. **Pre-trained vs Custom Embeddings**: Should we abandon Gemma embeddings for task-specific space?


## Next Steps

1. Begin Phase 1 experiments with learned rounding function implementation
2. Establish better evaluation metrics for text diffusion quality
3. Create systematic comparison framework for different approaches

---

**Related GitHub Issue**: [#12](https://github.com/LiamConnell/TinyDiffusionModels/issues/12) - Comparison with Diffusion-LM paper approach  
