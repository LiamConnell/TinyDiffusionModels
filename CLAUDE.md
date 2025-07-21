# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TextDiffusion implements both image and text diffusion models. The image diffusion (`src/mnist.py`) generates MNIST digits using standard DDPM. The text diffusion (`src/shakespeare.py`) is experimental, operating in embedding space rather than discrete tokens, with both pure diffusion and guided generation modes.

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
uv run python -m src.mnist.py --train

# Generate samples
uv run python -m src.mnist.py --sample
```

#### Text Diffusion
```bash
# Train the model
uv run python -m src.shakespeare.py --train

# Generate via pure diffusion
uv run python -m src.shakespeare.py --sample

# Generate via guided AR+diffusion (experimental)
uv run python -m src.shakespeare.py --guided_sample
```

### Vertex AI Deployment

**IMPORTANT**: By default, deployment builds a fresh Docker container with the latest code to ensure deployments use current fixes. Use `--no-build` only when you're certain the existing container is up to date.

#### Deploy Jobs (Automatic Container Build)
```bash
# Train text diffusion (builds container with latest code)
uv run python deployment/deploy.py shakespeare-training

# Sample text diffusion (builds container with latest code)
uv run python deployment/deploy.py shakespeare-sampling

# Train image diffusion (builds container with latest code)
uv run python deployment/deploy.py mnist-training

# Sample image diffusion (builds container with latest code)
uv run python deployment/deploy.py mnist-sampling

# Skip container build (use existing container - faster but may have stale code)
uv run python deployment/deploy.py shakespeare-sampling --no-build
```

#### Monitor Jobs
```bash
# Check job status
uv run python deployment/monitor.py JOB_ID

# Read job logs
uv run python deployment/monitor.py JOB_ID --logs

# Show full job details
uv run python deployment/monitor.py JOB_ID --full
```

#### Download Checkpoints
```bash
# Download trained models from Cloud Storage
gsutil cp gs://text-diffusion/diffusion/checkpoints/text-model.pth ./
gsutil cp gs://text-diffusion/diffusion/checkpoints/image-model.pth ./
```

## Architecture

### Image Diffusion (`src/mnist.py`)
- **SimpleUNet**: UNet with residual blocks and temporal embeddings
- **Standard DDPM**: Forward/reverse diffusion on pixel space
- **Training**: Learns to predict noise at each timestep
- **Sampling**: Reverse diffusion from pure noise

### Text Diffusion (`src/shakespeare.py`)
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
- `src/mnist.py`: Image diffusion implementation
- `src/shakespeare.py`: Text diffusion implementation  
- `Scratch.ipynb`: Experimental notebook

### Vertex AI Deployment
- `deployment/configs/shakespeare-training.yaml`: Text diffusion training config
- `deployment/configs/shakespeare-sampling.yaml`: Text diffusion sampling config  
- `deployment/configs/mnist-training.yaml`: Image diffusion training config
- `deployment/configs/mnist-sampling.yaml`: Image diffusion sampling config
- `deployment/deploy.py`: Simple job submission script
- `deployment/monitor.py`: Job monitoring and logs script
- `deployment/Dockerfile`: Container configuration for training jobs

### Outputs
- `samples/`: Generated outputs (images and text)
- `*.pth`: Local model checkpoints
- `gs://text-diffusion/diffusion/checkpoints/`: Cloud Storage checkpoints (Vertex AI)
- `gs://text-diffusion/diffusion/outputs/`: Cloud Storage training outputs

## Technical Notes

### Local Training
- Text diffusion quality is experimental (see `samples/bad_text/`)
- Uses cosine similarity for embedding-to-token decoding
- Guided generation alpha parameter controls AR vs diffusion mixture
- Both implementations follow similar diffusion mathematics but operate in different spaces

### Vertex AI Deployment
- **Zero Configuration**: All values hardcoded in config files - no environment variables needed
- **Predefined Configs**: 4 ready-to-submit job configurations for all use cases
- **Natural Language Interface**: Claude Code can modify configs based on user requests
- **Built-in Monitoring**: Simple status checking and log viewing with automatic job tracking
- **GPU Acceleration**: T4 GPUs with configurable machine types

### Checkpoint Management
- **Local**: Saved as `.pth` files in project directory
- **Cloud**: Automatically saved to `gs://text-diffusion/diffusion/checkpoints/` during Vertex AI training
- **Access**: Download with `gsutil cp` command for local inference

## Claude Code Deployment Operations

**IMPORTANT**: When the user submits a job, Claude Code should ALWAYS monitor the job status by default. Check the status periodically and provide updates to the user.

### Job Submission and Monitoring Workflow
1. **Submit Job**: Use `uv run python deployment/deploy.py JOB_TYPE`
2. **Monitor Job**: Immediately start monitoring with `uv run python deployment/monitor.py JOB_ID`
3. **Provide Updates**: Check status every few minutes and report to user
4. **Handle Completion**: When job completes (success/failure), inform user of final status

### Config Modifications
Edit files in `deployment/configs/` directly based on natural language requests:
- `shakespeare-training.yaml` - text diffusion training config
- `shakespeare-sampling.yaml` - text diffusion sampling config  
- `mnist-training.yaml` - image diffusion training config
- `mnist-sampling.yaml` - image diffusion sampling config

### Job Operations
- **Submit**: `uv run python deployment/deploy.py JOB_TYPE` (builds fresh container)
- **Submit (no build)**: `uv run python deployment/deploy.py JOB_TYPE --no-build` (uses existing container)
- **Monitor**: `uv run python deployment/monitor.py JOB_ID`
- **Get Logs**: `uv run python deployment/monitor.py JOB_ID --logs`
- **Full Details**: `uv run python deployment/monitor.py JOB_ID --full`

### Container Build Process
- **Default Behavior**: Every deployment builds and pushes a fresh Docker container with the latest source code
- **Why This Matters**: Ensures deployments always use current bug fixes and code changes
- **Build Steps**: 
  1. `docker build -t gcr.io/learnagentspace/text-diffusion:latest .`
  2. `docker push gcr.io/learnagentspace/text-diffusion:latest`
- **Skip Building**: Use `--no-build` flag for faster deployment when container is already up to date

### Common Config Changes
- **Machine type**: Edit `machineType` field (e.g., `n1-standard-4`, `n1-highmem-8`)
- **GPU type**: Edit `acceleratorType` field (e.g., `NVIDIA_TESLA_T4`, `NVIDIA_TESLA_V100`)  
- **Training epochs**: Edit `--epochs` in args array
- **Batch size**: Edit `--batch_size` in args array
- **Sample count**: Edit `--n` in args array (sampling jobs only)

### Troubleshooting and Testing
When user asks for testing, troubleshooting, or if a job fails:

1. **Get Job Logs**: `uv run python deployment/monitor.py JOB_ID --logs`
2. **Analyze Errors**: Look for error messages, stack traces, or failure patterns
3. **Common Issues to Check**:
   - Out of memory errors (suggest smaller batch size or different machine type)
   - GPU compatibility issues (suggest different accelerator type)
   - File/checkpoint not found (check paths in config)
   - Container/dependency issues (check Dockerfile and requirements)
   - Timeout issues (suggest longer-running machine type)
4. **Suggest Fixes**: Modify configs based on error analysis and resubmit job
5. **Verify Fix**: Monitor the new job to ensure the issue is resolved

### Example Troubleshooting Flow
```bash
# Job fails - get logs to diagnose
uv run python deployment/monitor.py 12345 --logs

# Common fixes based on error type:
# - OOM error: Reduce batch_size in config file
# - GPU error: Change acceleratorType in config file  
# - File error: Check checkpoint paths in config file
# - Then resubmit the job
uv run python deployment/deploy.py shakespeare-training
```

## Experiment Issues

Issues labeled `experiment` document training runs and experimental configurations. They should include:
- Job ID and monitoring commands
- Complete deployment configuration (epochs, batch size, machine type)  
- Git commit hash for reproducibility
- Expected outcome and current status

## GitHub Issue Management

- Usually when I refer to an issue, I mean a github issue. Creating an issue, investigating/fixing an issue, etc. 
- Be sure to always read the comments of the issue because a lot of important context can be there.

## Documentation and Experiment Tracking

- I use mkdocs to document the project and research in the docs/ directory. 
- I have an experiment journal where I write entries in a blog style. An example entry is `docs/docs/experiments/posts/2025-07-18-mnist-baseline.md`.