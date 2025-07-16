#!/usr/bin/env python3
"""
Submit text diffusion training job to Vertex AI
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
import string
import re

def run_command(cmd, check=True):
    """Run shell command and handle errors"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    if result.stdout:
        print(result.stdout)
    
    return result

def build_and_push_image(project_id, image_name):
    """Build and push Docker image to Google Container Registry"""
    image_uri = f"gcr.io/{project_id}/{image_name}:latest"
    
    # Build image
    run_command(["docker", "build", "-t", image_uri, "."])
    
    # Configure Docker for GCR
    run_command(["gcloud", "auth", "configure-docker", "--quiet"])
    
    # Push image
    run_command(["docker", "push", image_uri])
    
    return image_uri

def substitute_env_vars(template_content):
    """Substitute environment variables in template content with default support"""
    
    def replace_with_default(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return os.getenv(var_name, default_value)
    
    # Handle ${VAR:-default} syntax
    result = re.sub(r'\$\{([^}:]+)(?::[-]([^}]*))?\}', replace_with_default, template_content)
    
    # Handle remaining ${VAR} syntax
    template = string.Template(result)
    return template.safe_substitute(os.environ)

def submit_training_job(project_id, region, image_uri, bucket_name, epochs=3, batch_size=16):
    """Submit training job to Vertex AI"""
    
    # Set required environment variables (only if not already set)
    env_defaults = {
        'VERTEX_JOB_NAME': f"text-diffusion-training-{os.getenv('USER', 'user')}",
        'VERTEX_IMAGE_URI': image_uri,
        'VERTEX_OUTPUT_URI': f"gs://{bucket_name}/text-diffusion/outputs",
        'TRAIN_EPOCHS': str(epochs),
        'TRAIN_BATCH_SIZE': str(batch_size),
        'VERTEX_MACHINE_TYPE': 'n1-standard-4',
        'VERTEX_ACCELERATOR_TYPE': 'NVIDIA_TESLA_T4',
        'VERTEX_ACCELERATOR_COUNT': '1',
        'VERTEX_REPLICA_COUNT': '1'
    }
    
    # Only set defaults for missing environment variables
    for key, default_value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    # Read and substitute template
    template_path = Path("vertex_ai_config.yaml")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file {template_path} not found")
    
    template_content = template_path.read_text()
    job_config = substitute_env_vars(template_content)
    
    # Write config to temp file
    config_path = Path("temp_job_config.yaml")
    config_path.write_text(job_config)
    
    try:
        # Submit job
        cmd = [
            "gcloud", "ai", "custom-jobs", "create",
            "--display-name", os.environ.get('VERTEX_JOB_NAME', f"text-diffusion-training-{os.getenv('USER', 'user')}"),
            "--region", region,
            "--project", project_id,
            "--config", str(config_path),
            "--format", "json"
        ]
        
        result = run_command(cmd)
        print("✅ Training job submitted successfully!")
        return result
        
    finally:
        # Clean up temp file
        if config_path.exists():
            config_path.unlink()

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Submit text diffusion training to Vertex AI")
    parser.add_argument("--project-id", help="Google Cloud project ID (overrides env var)")
    parser.add_argument("--region", help="Training region (overrides env var)")
    parser.add_argument("--bucket", help="GCS bucket name for outputs (overrides env var)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides env var)")
    parser.add_argument("--batch-size", type=int, help="Training batch size (overrides env var)")
    parser.add_argument("--image-name", help="Docker image name (overrides env var)")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build/push")
    
    args = parser.parse_args()
    
    print("🚀 Submitting text diffusion training to Vertex AI...")
    
    # Get configuration from environment or command line args
    project_id = args.project_id or os.getenv("GCP_PROJECT_ID")
    region = args.region or os.getenv("VERTEX_REGION", "us-central1")
    bucket_name = args.bucket or os.getenv("GCS_BUCKET")
    epochs = args.epochs or int(os.getenv("TRAIN_EPOCHS", "3"))
    batch_size = args.batch_size or int(os.getenv("TRAIN_BATCH_SIZE", "16"))
    image_name = args.image_name or os.getenv("VERTEX_IMAGE_NAME", "text-diffusion")
    
    # Validate required parameters
    if not project_id:
        print("❌ Error: --project-id required or set GCP_PROJECT_ID in .env")
        sys.exit(1)
    if not bucket_name:
        print("❌ Error: --bucket required or set GCS_BUCKET in .env")
        sys.exit(1)
    
    # Check required environment variables
    if not os.getenv("HF_TOKEN"):
        print("⚠️  Warning: HF_TOKEN not set. You may need to set it for Hugging Face access.")
    
    # Build and push image (unless skipped)
    if not args.skip_build:
        print("🔨 Building and pushing Docker image...")
        image_uri = build_and_push_image(project_id, image_name)
    else:
        image_uri = f"gcr.io/{project_id}/{image_name}:latest"
        print(f"📦 Using existing image: {image_uri}")
    
    # Submit training job
    print("📋 Submitting training job...")
    submit_training_job(
        project_id=project_id,
        region=region,
        image_uri=image_uri,
        bucket_name=bucket_name,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print(f"""
📊 You can monitor your job at:
https://console.cloud.google.com/vertex-ai/training/custom-jobs?inv=1&invt=Ab23xg&project={project_id}

📁 Outputs will be saved to:
gs://{bucket_name}/text-diffusion/outputs/
""")

if __name__ == "__main__":
    main()