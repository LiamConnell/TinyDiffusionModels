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

def deploy_job(job_type: str):
    """Deploy a predefined job configuration"""
    
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
    
    args = parser.parse_args()
    
    job_id = deploy_job(args.job_type)
    print(f"\n💡 To monitor this job, run:")
    print(f"   uv run python deployment/monitor.py {job_id}")
    print(f"   uv run python deployment/monitor.py {job_id} --logs")

if __name__ == "__main__":
    main()