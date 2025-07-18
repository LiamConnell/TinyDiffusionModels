### Setup

```
uv venv
uv pip install -r requirements.txt
```

### Image Diffusion

```
uv run python mnist.py --train
uv run python mnist.py --sample
```

### Text Diffusion 

```
uv run python shakespeare.py --train
uv run python shakespeare.py --sample
```

#### (Experimental) Guided Generation

```
uv run python shakespeare.py --guided_sample
```

### Deployment

#### Google Cloud Vertex AI Training

Train models on Vertex AI with GPU acceleration and automatic checkpoint saving to Cloud Storage:

```bash
# Setup (one-time)
./deployment/setup_vertex_training.sh

# Submit text diffusion training job
python deployment/submit_vertex_training.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py

# Submit image diffusion training job  
python deployment/submit_vertex_training.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script mnist.py

# With custom parameters
python deployment/submit_vertex_training.py --project-id YOUR_PROJECT --bucket YOUR_BUCKET --script shakespeare.py --epochs 10 --batch-size 32
```

**Features:**
- **Automatic Checkpoint Saving**: Model checkpoints are automatically saved to `gs://your-bucket/diffusion/checkpoints/`
- **Output Storage**: Training outputs saved to `gs://your-bucket/diffusion/outputs/`
- **Script Selection**: Choose between `shakespeare.py` (text) or `mnist.py` (image) training
- **Robust Storage**: Uses Cloud Storage for durability and easy access

**Checkpoint Access:**
```bash
# Download trained checkpoints
gsutil cp gs://your-bucket/diffusion/checkpoints/text-model.pth ./
gsutil cp gs://your-bucket/diffusion/checkpoints/image-model.pth ./
```

Configuration is managed through `.env` file (created by setup script). Monitor jobs at the [Vertex AI Console](https://console.cloud.google.com/ai/training/jobs).