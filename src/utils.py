import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import subprocess

import torch
from anypath import AnyPath


def is_gcs_path(path: Union[str, Path]) -> bool:
    """Check if path is a Google Cloud Storage path."""
    return str(path).startswith("gs://")


def download_checkpoint_from_gcs(gcs_path: str, local_path: str) -> str:
    """Download checkpoint from GCS if it doesn't exist locally."""
    if os.path.exists(local_path):
        print(f"Checkpoint already exists locally: {local_path}")
        return local_path
    
    if gcs_path.startswith("gs://"):
        print(f"Downloading checkpoint from GCS: {gcs_path}")
        try:
            subprocess.run(["gsutil", "cp", gcs_path, local_path], check=True)
            print(f"Downloaded checkpoint to: {local_path}")
            return local_path
        except subprocess.CalledProcessError as e:
            print(f"Failed to download from GCS: {e}")
            raise
    else:
        return gcs_path


def load_checkpoint(ckpt_path: Union[str, Path], device: str) -> dict:
    """Load checkpoint from local path or GCS using AnyPath."""
    path = AnyPath(ckpt_path)
    
    if path.is_cloud():
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            try:
                # Download to temp file
                path.download_to(tmp.name)
                return torch.load(tmp.name, map_location=device)
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint from {ckpt_path}: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        return torch.load(str(path), map_location=device)


def save_checkpoint(model_state: dict, ckpt_path: Union[str, Path]) -> None:
    """Save checkpoint to local path or GCS using AnyPath."""
    path = AnyPath(ckpt_path)
    
    if path.is_cloud():
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            try:
                torch.save(model_state, tmp.name)
                path.upload_from(tmp.name)
                print(f"✔ Uploaded checkpoint to {ckpt_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to upload checkpoint to {ckpt_path}: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        torch.save(model_state, str(path))
        print(f"✔ Saved checkpoint to {ckpt_path}")


def save_samples(content: Union[str, bytes], sample_path: Union[str, Path], 
                 mode: str = "w") -> None:
    """Save samples to local path or GCS using AnyPath."""
    path = AnyPath(sample_path)
    
    if path.is_cloud():
        with tempfile.NamedTemporaryFile(mode=mode, suffix=path.suffix, delete=False) as tmp:
            try:
                if isinstance(content, str):
                    tmp.write(content)
                else:
                    tmp.write(content)
                tmp.flush()
                path.upload_from(tmp.name)
                print(f"✔ Uploaded sample to {sample_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to upload sample to {sample_path}: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)
        print(f"✔ Saved sample to {sample_path}")


def get_vertex_checkpoint_path(base_name: str) -> str:
    """Get appropriate checkpoint path for Vertex AI or local training."""
    if "AIP_MODEL_DIR" in os.environ:
        return os.path.join(os.environ["AIP_MODEL_DIR"], base_name)
    return base_name


def get_samples_dir(base_dir: str = "samples") -> AnyPath:
    """Get samples directory path, supporting both local and cloud storage."""
    if "AIP_MODEL_DIR" in os.environ:
        # In Vertex AI, save samples to cloud storage
        model_dir = AnyPath(os.environ["AIP_MODEL_DIR"])
        return model_dir.parent / "outputs" / base_dir
    return AnyPath(base_dir)