#!/usr/bin/env python3
"""
Submit text diffusion sampling job to Vertex AI
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

def submit_sampling_job(project_id, region, image_uri, bucket_name, checkpoint_path=None, num_samples=5, script="shakespeare.py", sample_mode="sample"):
    """Submit sampling job to Vertex AI"""
    
    # Set required environment variables (only if not already set)
    model_name = "text-model" if script == "shakespeare.py" else "image-model"
    
    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = f"gs://{bucket_name}/diffusion/checkpoints/{model_name}.pth"
    
    env_defaults = {
        'VERTEX_JOB_NAME': f"diffusion-sampling-{model_name}-{os.getenv('USER', 'user')}",
        'VERTEX_IMAGE_URI': image_uri,
        'BUCKET_NAME': bucket_name,
        'SAMPLE_SCRIPT': script,
        'MODEL_NAME': model_name,
        'CHECKPOINT_PATH': checkpoint_path,
        'NUM_SAMPLES': str(num_samples),
        'SAMPLE_MODE': sample_mode,
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
    template_path = Path(__file__).parent / "vertex_ai_sampling_config.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file {template_path} not found")
    
    template_content = template_path.read_text()
    job_config = substitute_env_vars(template_content)
    
    # Write config to temp file
    config_path = Path("temp_sampling_config.yaml")
    config_path.write_text(job_config)
    
    try:
        # Submit job
        cmd = [
            "gcloud", "ai", "custom-jobs", "create",
            "--display-name", os.environ.get('VERTEX_JOB_NAME', f"text-diffusion-sampling-{os.getenv('USER', 'user')}"),
            "--region", region,
            "--project", project_id,
            "--config", str(config_path),
            "--format", "json"
        ]
        
        result = run_command(cmd)
        print("✅ Sampling job submitted successfully!")
        return result
        
    finally:
        # Clean up temp file
        if config_path.exists():
            config_path.unlink()

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Submit text diffusion sampling to Vertex AI")
    parser.add_argument("--project-id", help="Google Cloud project ID (overrides env var)")
    parser.add_argument("--region", help="Sampling region (overrides env var)")
    parser.add_argument("--bucket", help="GCS bucket name for checkpoints and outputs (overrides env var)")
    parser.add_argument("--checkpoint", help="Path to checkpoint file (local or GCS)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--script", choices=["shakespeare.py", "mnist.py"], default="shakespeare.py", help="Sampling script to run")
    parser.add_argument("--sample-mode", choices=["sample", "guided_sample"], default="sample", help="Sampling mode (for text diffusion)")
    parser.add_argument("--image-name", help="Docker image name (overrides env var)")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build/push")
    
    args = parser.parse_args()
    
    print("🚀 Submitting diffusion sampling to Vertex AI...")
    
    # Get configuration from environment or command line args
    project_id = args.project_id or os.getenv("GCP_PROJECT_ID")
    region = args.region or os.getenv("VERTEX_REGION", "us-central1")
    bucket_name = args.bucket or os.getenv("GCS_BUCKET")
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
    
    # Submit sampling job
    print("📋 Submitting sampling job...")
    submit_sampling_job(
        project_id=project_id,
        region=region,
        image_uri=image_uri,
        bucket_name=bucket_name,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        script=args.script,
        sample_mode=args.sample_mode
    )
    
    print(f"""
📊 You can monitor your job at:
https://console.cloud.google.com/vertex-ai/training/custom-jobs?inv=1&invt=Ab23xg&project={project_id}

📁 Outputs will be saved to:
gs://{bucket_name}/diffusion/outputs/

💾 Using checkpoint from:
{args.checkpoint or f"gs://{bucket_name}/diffusion/checkpoints/{('text-model' if args.script == 'shakespeare.py' else 'image-model')}.pth"}
""")

if __name__ == "__main__":
    main()