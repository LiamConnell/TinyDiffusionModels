# TextDiffusion

[Project Writeup and Experiment Log](https://liamconnell.github.io/TinyDiffusionModels/)

Implementation of diffusion models for both images and text. Image diffusion uses standard DDPM on MNIST digits. Text diffusion is experimental, operating in embedding space with both pure diffusion and guided generation modes.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

## Local Training & Sampling

### Image Diffusion

```bash
# Train the model
uv run python -m src.mnist --train

# Generate samples
uv run python -m src.mnist --sample
```

### Text Diffusion 

```bash
# Train the model
uv run python -m src.shakespeare --train

# Generate via pure diffusion
uv run python -m src.shakespeare --sample

# Generate via guided AR+diffusion (experimental)
uv run python -m src.shakespeare --guided_sample
```

## Vertex AI Deployment

### Deploy Jobs (Zero Configuration Required)

All configuration is predefined - simply choose your job type:

```bash
# Train text diffusion
uv run python deployment/deploy.py shakespeare-training

# Sample text diffusion  
uv run python deployment/deploy.py shakespeare-sampling

# Train image diffusion
uv run python deployment/deploy.py mnist-training

# Sample image diffusion
uv run python deployment/deploy.py mnist-sampling
```

### Monitor Jobs

```bash
# Check job status
uv run python deployment/monitor.py JOB_ID

# Read job logs
uv run python deployment/monitor.py JOB_ID --logs

# Show full job details
uv run python deployment/monitor.py JOB_ID --full
```

### Download Checkpoints

```bash
# Download trained models from Cloud Storage
gsutil cp gs://text-diffusion/diffusion/checkpoints/text-model.pth ./
gsutil cp gs://text-diffusion/diffusion/checkpoints/image-model.pth ./
```

**Features:**
- **Zero Configuration**: All values hardcoded in config files - no environment variables needed
- **Predefined Configs**: 4 ready-to-submit job configurations for all use cases
- **GPU Acceleration**: T4 GPUs with configurable machine types
- **Automatic Checkpoints**: Models saved to `gs://text-diffusion/diffusion/checkpoints/`
- **Output Storage**: Training outputs saved to `gs://text-diffusion/diffusion/outputs/`

## Project Structure

```
src/
├── mnist.py          # Image diffusion on MNIST
├── shakespeare.py    # Text diffusion in embedding space
└── utils.py         # Shared utilities

deployment/
├── configs/         # Predefined Vertex AI job configs
├── deploy.py        # Simple job submission
└── monitor.py       # Job monitoring and logs

samples/             # Generated outputs
data/               # Training data (MNIST)
```

## Architecture

**Image Diffusion**: Standard DDPM with UNet, trains to predict noise at each timestep  
**Text Diffusion**: Experimental approach using transformer in embedding space, supports both pure diffusion and guided generation mixing autoregressive and diffusion logits