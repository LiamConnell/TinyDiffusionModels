# TextDiffusion

TextDiffusion implements both image and text diffusion models for generative AI research.

## Overview

This project explores diffusion models across two modalities:

- **Image Diffusion** (`src/mnist.py`): Standard DDPM implementation for MNIST digit generation
- **Text Diffusion** (`src/shakespeare.py`): Experimental embedding-space diffusion for text generation

## Key Features

### Image Diffusion
- UNet architecture with residual blocks and temporal embeddings
- Standard DDPM forward/reverse diffusion process
- Trained on MNIST dataset for digit generation

### Text Diffusion
- Transformer-based architecture for embedding space diffusion
- Two generation modes:
  - Pure diffusion: Generate embeddings then decode to tokens
  - Guided generation: Mix autoregressive and diffusion logits
- Trained on Shakespeare corpus

## Architecture

The project follows a modular design with separate implementations for each modality:

```
src/
├── mnist.py          # Image diffusion implementation
└── shakespeare.py    # Text diffusion implementation

deployment/
├── configs/          # Vertex AI job configurations
├── deploy.py         # Job submission script
└── monitor.py        # Job monitoring script
```

## Quick Start

### Local Training

```bash
# Setup environment
uv venv
uv pip install -r requirements.txt

# Train image diffusion
uv run python -m src.mnist --train

# Train text diffusion
uv run python -m src.shakespeare --train

# Generate samples
uv run python -m src.mnist --sample
uv run python -m src.shakespeare --sample
```

### Cloud Deployment

Deploy training jobs to Google Cloud Vertex AI:

```bash
# Deploy text diffusion training
uv run python deployment/deploy.py shakespeare-training

# Monitor job progress
uv run python deployment/monitor.py JOB_ID
```

## Research Focus

The text diffusion component represents experimental research into:

- Embedding-space diffusion vs. discrete token diffusion
- Hybrid autoregressive-diffusion generation
- Quality comparison between pure and guided generation modes

Results and findings are documented in the [Experiment Journal](experiments.md).
