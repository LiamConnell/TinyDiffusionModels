FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy source code
COPY src/ src/
COPY .env .

# Create samples directory
RUN mkdir -p samples

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Run training by default
CMD ["python", "-m", "src.shakespeare", "--train"]