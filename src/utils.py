import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import subprocess

import torch
from google.cloud import storage


def is_gcs_path(path: Union[str, Path]) -> bool:
    """Check if path is a Google Cloud Storage path."""
    return str(path).startswith("gs://")


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse GCS path into bucket and blob names."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Not a GCS path: {gcs_path}")
    
    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    return bucket_name, blob_name


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download file from GCS to local path."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload local file to GCS."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)




def load_checkpoint(ckpt_path: Union[str, Path], device: str) -> dict:
    """Load checkpoint from local path or GCS."""
    ckpt_path = str(ckpt_path)
    
    if is_gcs_path(ckpt_path):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            try:
                # Download from GCS to temp file
                print(f"Downloading checkpoint from GCS: {ckpt_path}")
                download_from_gcs(ckpt_path, tmp.name)
                return torch.load(tmp.name, map_location=device)
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint from {ckpt_path}: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        return torch.load(ckpt_path, map_location=device)


def save_checkpoint(model_state: dict, ckpt_path: Union[str, Path]) -> None:
    """Save checkpoint to local path or GCS."""
    ckpt_path = str(ckpt_path)
    
    if is_gcs_path(ckpt_path):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            try:
                torch.save(model_state, tmp.name)
                print(f"Uploading checkpoint to GCS: {ckpt_path}")
                upload_to_gcs(tmp.name, ckpt_path)
                print(f"✔ Uploaded checkpoint to {ckpt_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to upload checkpoint to {ckpt_path}: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        torch.save(model_state, ckpt_path)
        print(f"✔ Saved checkpoint to {ckpt_path}")


def save_samples(content: Union[str, bytes], sample_path: Union[str, Path], 
                 mode: str = "w") -> None:
    """Save samples to local path or GCS."""
    sample_path = str(sample_path)
    
    if is_gcs_path(sample_path):
        # Get file suffix
        suffix = Path(sample_path).suffix
        with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False) as tmp:
            try:
                if isinstance(content, str):
                    tmp.write(content)
                else:
                    tmp.write(content)
                tmp.flush()
                tmp.close()  # Explicitly close file before upload
                print(f"Uploading sample to GCS: {sample_path}")
                upload_to_gcs(tmp.name, sample_path)
                print(f"✔ Uploaded sample to {sample_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to upload sample to {sample_path}: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        # Ensure parent directory exists
        Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, str):
            Path(sample_path).write_text(content)
        else:
            Path(sample_path).write_bytes(content)
        print(f"✔ Saved sample to {sample_path}")


def get_vertex_checkpoint_path(base_name: str) -> str:
    """Get appropriate checkpoint path for Vertex AI or local training."""
    if "AIP_MODEL_DIR" in os.environ:
        return os.path.join(os.environ["AIP_MODEL_DIR"], base_name)
    return base_name


def get_samples_dir(base_dir: str = "samples") -> Union[str, Path]:
    """Get samples directory path, supporting both local and cloud storage."""
    if "AIP_MODEL_DIR" in os.environ:
        # In Vertex AI, save samples to cloud storage
        model_dir = os.environ["AIP_MODEL_DIR"]
        # AIP_MODEL_DIR is already the outputs directory, so just append base_dir
        if model_dir.startswith("gs://"):
            # Return string for GCS paths to avoid Path normalization issues
            return f"{model_dir}/{base_dir}"
        else:
            return Path(model_dir) / base_dir
    return Path(base_dir)