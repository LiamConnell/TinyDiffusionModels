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