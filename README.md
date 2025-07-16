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

Train models on Vertex AI with GPU acceleration:

```bash
# Setup (one-time)
./setup_vertex_training.sh

# Submit training job
python submit_vertex_training.py

# With custom parameters
python submit_vertex_training.py --epochs 10 --batch-size 32
```

Configuration is managed through `.env` file (created by setup script). Monitor jobs at the [Vertex AI Console](https://console.cloud.google.com/ai/training/jobs).