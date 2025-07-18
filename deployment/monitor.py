#!/usr/bin/env python3
"""
Simple job monitoring script for TextDiffusion Vertex AI jobs.
All values are hardcoded - no environment variables needed.
"""

import argparse
import subprocess
import json
import sys

def get_job_status(job_id: str):
    """Get job status"""
    cmd = [
        "gcloud", "ai", "custom-jobs", "describe", job_id,
        "--project", "learnagentspace",
        "--region", "us-central1",
        "--format", "json"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_data = json.loads(result.stdout)
        return job_data
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting job status: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing job data: {e}")
        sys.exit(1)

def get_job_logs(job_id: str):
    """Get job logs"""
    # Get job details to find creation time for log filtering
    job_data = get_job_status(job_id)
    create_time = job_data.get("createTime", "")
    
    # Build query with job_id and optional timestamp filter
    query = f'resource.labels.job_id="{job_id}"'
    if create_time:
        query += f' timestamp>="{create_time}"'
    
    cmd = [
        "gcloud", "logging", "read", 
        query,
        "--project", "learnagentspace",
        "--limit", "100",
        "--format", "value(timestamp,textPayload,jsonPayload.message)",
        "--freshness", "7d"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout if result.stdout.strip() else "No logs found. The job may still be starting or logs may not be available yet."
    except subprocess.CalledProcessError as e:
        return f"❌ Error getting logs: {e.stderr}"

def format_job_status(job_data):
    """Format job status for display"""
    state = job_data.get("state", "UNKNOWN")
    create_time = job_data.get("createTime", "Unknown")
    update_time = job_data.get("updateTime", "Unknown")
    
    # State emoji mapping
    state_emoji = {
        "JOB_STATE_QUEUED": "⏳",
        "JOB_STATE_PENDING": "⏳", 
        "JOB_STATE_RUNNING": "🏃",
        "JOB_STATE_SUCCEEDED": "✅",
        "JOB_STATE_FAILED": "❌",
        "JOB_STATE_CANCELLING": "🛑",
        "JOB_STATE_CANCELLED": "🛑",
        "JOB_STATE_PAUSED": "⏸️"
    }
    
    emoji = state_emoji.get(state, "❓")
    
    print(f"{emoji} Job State: {state}")
    print(f"📅 Created: {create_time}")
    print(f"🔄 Updated: {update_time}")
    
    # Show error if failed
    if state == "JOB_STATE_FAILED" and "error" in job_data:
        error = job_data["error"]
        print(f"💥 Error: {error.get('message', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Monitor TextDiffusion Vertex AI jobs")
    parser.add_argument("job_id", help="Job ID to monitor")
    parser.add_argument("--logs", action="store_true", help="Show job logs")
    parser.add_argument("--full", action="store_true", help="Show full job details")
    
    args = parser.parse_args()
    
    print(f"📋 Monitoring job: {args.job_id}")
    print()
    
    if args.logs:
        print("📄 Job Logs:")
        print("=" * 50)
        logs = get_job_logs(args.job_id)
        if logs.strip():
            print(logs)
        else:
            print("No logs available yet. Job may still be starting up.")
    
    elif args.full:
        print("📊 Full Job Details:")
        print("=" * 50)
        job_data = get_job_status(args.job_id)
        print(json.dumps(job_data, indent=2))
    
    else:
        print("📊 Job Status:")
        print("=" * 20)
        job_data = get_job_status(args.job_id)
        format_job_status(job_data)
        
        print()
        print("💡 To see logs: uv run python deployment/monitor.py", args.job_id, "--logs")
        print("💡 To see full details: uv run python deployment/monitor.py", args.job_id, "--full")

if __name__ == "__main__":
    main()