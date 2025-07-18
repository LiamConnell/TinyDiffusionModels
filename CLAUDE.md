# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TextDiffusion implements both image and text diffusion models. The image diffusion (`mnist.py`) generates MNIST digits using standard DDPM. The text diffusion (`shakespeare.py`) is experimental, operating in embedding space rather than discrete tokens, with both pure diffusion and guided generation modes.

## Common Commands

### Setup
```bash
uv venv
uv pip install -r requirements.txt
```

### Local Training

#### Image Diffusion
```bash
# Train the model
uv run python mnist.py --train

# Generate samples
uv run python mnist.py --sample
```

#### Text Diffusion
```bash
# Train the model
uv run python shakespeare.py --train

# Generate via pure diffusion
uv run python shakespeare.py --sample

# Generate via guided AR+diffusion (experimental)
uv run python shakespeare.py --guided_sample
```

### Vertex AI Training (Robust Cloud Storage)

#### Setup
```bash
# One-time setup
./deployment/setup_vertex_training.sh
```

#### Submit Training Jobs
```bash
# Text diffusion training
python deployment/submit_vertex_training.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py

# Image diffusion training
python deployment/submit_vertex_training.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script mnist.py

# With custom parameters
python deployment/submit_vertex_training.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py --epochs 10 --batch-size 32
```

#### Submit Sampling Jobs
```bash
# Text diffusion sampling (pure diffusion)
python deployment/submit_vertex_sampling.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py

# Text diffusion sampling (guided generation)
python deployment/submit_vertex_sampling.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py --sample-mode guided_sample

# Image diffusion sampling
python deployment/submit_vertex_sampling.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script mnist.py

# With custom parameters
python deployment/submit_vertex_sampling.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py --num-samples 10 --checkpoint gs://YOUR_BUCKET/diffusion/checkpoints/custom-model.pth
```

#### Download Checkpoints
```bash
# Download trained models from Cloud Storage
gsutil cp gs://your-bucket/diffusion/checkpoints/text-model.pth ./
gsutil cp gs://your-bucket/diffusion/checkpoints/image-model.pth ./
```

## Architecture

### Image Diffusion (`mnist.py`)
- **SimpleUNet**: UNet with residual blocks and temporal embeddings
- **Standard DDPM**: Forward/reverse diffusion on pixel space
- **Training**: Learns to predict noise at each timestep
- **Sampling**: Reverse diffusion from pure noise

### Text Diffusion (`shakespeare.py`)
- **TinyTransformer**: Transformer encoder for embedding space diffusion
- **Embedding Space**: Uses pre-trained model embeddings as target space
- **Two Generation Modes**:
  - Pure diffusion: Generate embeddings then decode to tokens
  - Guided generation: Mix autoregressive and diffusion logits using alpha parameter

### Key Components
- `q_sample()`: Forward diffusion (add noise)
- `p_sample()`: Reverse diffusion step (denoise)
- `guided_generate()`: Hybrid AR+diffusion generation
- `load_text_dataset()`: Shakespeare corpus loading
- `tokenize_corpus()`: Text to token conversion

## Data Flow

**Image**: MNIST → Add noise → Train denoiser → Generate via reverse diffusion

**Text**: Shakespeare corpus → Tokenize → Convert to embeddings → Train diffusion in embedding space → Generate via diffusion or guided mode

## Key Files

### Core Implementation
- `mnist.py`: Image diffusion implementation
- `shakespeare.py`: Text diffusion implementation  
- `Scratch.ipynb`: Experimental notebook

### Vertex AI Training
- `deployment/vertex_ai_config.yaml`: Vertex AI job configuration with Cloud Storage
- `deployment/submit_vertex_training.py`: Training job submission script
- `deployment/vertex_ai_sampling_config.yaml`: Vertex AI sampling job configuration
- `deployment/submit_vertex_sampling.py`: Sampling job submission script
- `deployment/setup_vertex_training.sh`: One-time Vertex AI setup
- `deployment/Dockerfile`: Container configuration for training jobs

### Outputs
- `samples/`: Generated outputs (images and text)
- `*.pth`: Local model checkpoints
- `gs://bucket/diffusion/checkpoints/`: Cloud Storage checkpoints (Vertex AI)
- `gs://bucket/diffusion/outputs/`: Cloud Storage training outputs

## Technical Notes

### Local Training
- Text diffusion quality is experimental (see `samples/bad_text/`)
- Uses cosine similarity for embedding-to-token decoding
- Guided generation alpha parameter controls AR vs diffusion mixture
- Both implementations follow similar diffusion mathematics but operate in different spaces

### Vertex AI Training
- **Automatic Checkpoint Saving**: Checkpoints automatically saved to Cloud Storage
- **Robust Storage**: Uses Google Cloud Storage for durability and accessibility
- **GPU Acceleration**: Configurable machine types and accelerators
- **Multi-Model Support**: Supports both text and image diffusion training
- **Container-Based**: Uses Docker for consistent training environments

### Checkpoint Management
- **Local**: Saved as `.pth` files in project directory
- **Cloud**: Automatically saved to `gs://bucket/diffusion/checkpoints/` during Vertex AI training
- **Access**: Download with `gsutil cp` command for local inference