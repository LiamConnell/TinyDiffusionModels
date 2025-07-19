#!/usr/bin/env python3
"""
Ultra-simple deployment script for TextDiffusion jobs.
All values are hardcoded - no environment variables or templating needed.
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path

def build_container():
    """Build and push the Docker container with latest code"""
    
    print("🔨 Building Docker container...")
    
    # Build container
    build_cmd = [
        "docker", "build", 
        "-t", "gcr.io/learnagentspace/text-diffusion:latest",
        "."
    ]
    
    try:
        print(f"Running: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, check=True)
        print("✅ Container built successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building container: {e}")
        sys.exit(1)
    
    # Push container
    print("📤 Pushing container to registry...")
    push_cmd = [
        "docker", "push", 
        "gcr.io/learnagentspace/text-diffusion:latest"
    ]
    
    try:
        print(f"Running: {' '.join(push_cmd)}")
        result = subprocess.run(push_cmd, check=True)
        print("✅ Container pushed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error pushing container: {e}")
        sys.exit(1)

def deploy_job(job_type: str, build: bool = True):
    """Deploy a predefined job configuration"""
    
    # Build container with latest code if requested
    if build:
        build_container()
    
    # Use config directly (no substitutions needed)
    config_path = Path(__file__).parent / "configs" / f"{job_type}.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"🚀 Deploying {job_type} job...")
    
    # Submit job
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        "--display-name", f"{job_type}-job",
        "--region", "us-central1", 
        "--project", "learnagentspace",
        "--config", str(config_path),
        "--format", "json"
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_data = json.loads(result.stdout)
        job_id = job_data["name"].split("/")[-1]
        
        print(f"✅ Job submitted successfully!")
        print(f"📋 Job ID: {job_id}")
        print(f"🔗 Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=learnagentspace")
        
        return job_id
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error submitting job: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing job response: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Deploy TextDiffusion jobs to Vertex AI")
    parser.add_argument("job_type", choices=[
        "shakespeare-training", "shakespeare-sampling", 
        "mnist-training", "mnist-sampling"
    ], help="Type of job to deploy")
    parser.add_argument("--no-build", action="store_true", 
                       help="Skip building and pushing the Docker container")
    
    args = parser.parse_args()
    
    job_id = deploy_job(args.job_type, build=not args.no_build)
    print(f"\n💡 To monitor this job, run:")
    print(f"   uv run python deployment/monitor.py {job_id}")
    print(f"   uv run python deployment/monitor.py {job_id} --logs")

if __name__ == "__main__":
    main()