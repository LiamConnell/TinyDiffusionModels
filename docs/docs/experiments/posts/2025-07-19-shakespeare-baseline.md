---
date: 2025-07-19
categories:
  - Baseline
  - Text Diffusion
  - Shakespeare
---

# Shakespeare Text Diffusion Baseline

**Status**: Complete  
**Type**: Baseline

## Objective

Implement initial text diffusion in embedding space using Shakespeare corpus. Explore whether standard diffusion approaches can work for text generation through continuous embeddings.

<!-- more -->

## Configuration

- **Model**: TinyTransformer for embedding space diffusion
- **Training**: Pure diffusion in embedding space
- **Architecture**: Transformer encoder adapted for diffusion
- **Dataset**: Shakespeare corpus, tokenized and embedded
- **Decoding**: Cosine similarity between generated embeddings and token embeddings
- **Hardware**: T4
- **Git Commit**: 4422ce927fbf61e226157e4a3f2ac8de91b583bb

## Hypothesis

Text diffusion in embedding space should be possible, though the continuous-to-discrete mapping (embeddings to tokens) may present challenges for generation quality.

## Results

### Quantitative
- Training converges and loss decreases as expected
- Model learns to denoise embeddings progressively

### Qualitative  
- Generated text quality is poor (samples stored in `samples/bad_text/`)
- Text lacks coherence and often produces nonsensical sequences. Seems like random characters. 
- Clear disconnect between continuous embedding space and discrete token outputs

## Key Learnings

- **Embedding space diffusion is technically feasible**: The mathematical framework works
- **Decoding is the major bottleneck**: Cosine similarity approach has significant limitations
- **Continuous-discrete gap is challenging**: Moving from smooth embeddings to sharp token decisions loses information
- **Need better bridging strategy**: Simple nearest-neighbor decoding insufficient for quality text

## Next Steps

- Experiment with guided generation combining autoregressive and diffusion approaches
- Investigate better decoding strategies beyond cosine similarity
- Consider hybrid approaches that maintain some discrete structure
- Explore different pre-trained embedding models

---

**Sample Generation**:
```bash
uv run python -m src.shakespeare --sample
```

**Training**:
```bash
uv run python -m src.shakespeare --train
```