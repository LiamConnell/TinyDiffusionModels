import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest
import torch

from src.utils import (
    is_gcs_path,
    parse_gcs_path,
    download_from_gcs,
    upload_to_gcs,
    load_checkpoint,
    save_checkpoint,
    save_samples,
    get_vertex_checkpoint_path,
    get_samples_dir
)


class TestIsGcsPath:
    def test_is_gcs_path_true(self):
        assert is_gcs_path("gs://bucket/path") is True
        assert is_gcs_path("gs://bucket") is True

    def test_is_gcs_path_false(self):
        assert is_gcs_path("/local/path") is False
        assert is_gcs_path("bucket/path") is False
        assert is_gcs_path("s3://bucket/path") is False
        assert is_gcs_path(Path("/local/path")) is False


class TestParseGcsPath:
    def test_parse_gcs_path_with_blob(self):
        bucket, blob = parse_gcs_path("gs://my-bucket/path/to/file.txt")
        assert bucket == "my-bucket"
        assert blob == "path/to/file.txt"

    def test_parse_gcs_path_bucket_only(self):
        bucket, blob = parse_gcs_path("gs://my-bucket")
        assert bucket == "my-bucket"
        assert blob == ""

    def test_parse_gcs_path_with_slash(self):
        bucket, blob = parse_gcs_path("gs://my-bucket/")
        assert bucket == "my-bucket"
        assert blob == ""

    def test_parse_gcs_path_invalid(self):
        with pytest.raises(ValueError, match="Not a GCS path"):
            parse_gcs_path("/local/path")

    def test_parse_gcs_path_s3(self):
        with pytest.raises(ValueError, match="Not a GCS path"):
            parse_gcs_path("s3://bucket/path")


class TestDownloadFromGcs:
    @patch('src.utils.storage.Client')
    def test_download_from_gcs_success(self, mock_client_class):
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        download_from_gcs("gs://my-bucket/file.txt", "/local/file.txt")
        
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("file.txt")
        mock_blob.download_to_filename.assert_called_once_with("/local/file.txt")


class TestUploadToGcs:
    @patch('src.utils.storage.Client')
    def test_upload_to_gcs_success(self, mock_client_class):
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        upload_to_gcs("/local/file.txt", "gs://my-bucket/file.txt")
        
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("file.txt")
        mock_blob.upload_from_filename.assert_called_once_with("/local/file.txt")


class TestLoadCheckpoint:
    @patch('src.utils.torch.load')
    def test_load_checkpoint_local(self, mock_torch_load):
        mock_checkpoint = {"model": "state"}
        mock_torch_load.return_value = mock_checkpoint
        
        result = load_checkpoint("/local/checkpoint.pth", "cpu")
        
        mock_torch_load.assert_called_once_with("/local/checkpoint.pth", map_location="cpu")
        assert result == mock_checkpoint

    @patch('src.utils.torch.load')
    @patch('src.utils.download_from_gcs')
    @patch('src.utils.tempfile.NamedTemporaryFile')
    @patch('src.utils.os.unlink')
    def test_load_checkpoint_gcs(self, mock_unlink, mock_temp_file, mock_download, mock_torch_load):
        mock_checkpoint = {"model": "state"}
        mock_torch_load.return_value = mock_checkpoint
        
        mock_temp = Mock()
        mock_temp.name = "/tmp/tempfile.pth"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        result = load_checkpoint("gs://bucket/checkpoint.pth", "cuda")
        
        mock_download.assert_called_once_with("gs://bucket/checkpoint.pth", "/tmp/tempfile.pth")
        mock_torch_load.assert_called_once_with("/tmp/tempfile.pth", map_location="cuda")
        mock_unlink.assert_called_once_with("/tmp/tempfile.pth")
        assert result == mock_checkpoint

    @patch('src.utils.download_from_gcs')
    @patch('src.utils.tempfile.NamedTemporaryFile')
    @patch('src.utils.os.unlink')
    def test_load_checkpoint_gcs_error(self, mock_unlink, mock_temp_file, mock_download):
        mock_download.side_effect = Exception("Download failed")
        
        mock_temp = Mock()
        mock_temp.name = "/tmp/tempfile.pth"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        with pytest.raises(RuntimeError, match="Failed to download checkpoint"):
            load_checkpoint("gs://bucket/checkpoint.pth", "cpu")
        
        mock_unlink.assert_called_once_with("/tmp/tempfile.pth")


class TestSaveCheckpoint:
    @patch('src.utils.torch.save')
    def test_save_checkpoint_local(self, mock_torch_save):
        model_state = {"model": "state"}
        
        save_checkpoint(model_state, "/local/checkpoint.pth")
        
        mock_torch_save.assert_called_once_with(model_state, "/local/checkpoint.pth")

    @patch('src.utils.torch.save')
    @patch('src.utils.upload_to_gcs')
    @patch('src.utils.tempfile.NamedTemporaryFile')
    @patch('src.utils.os.unlink')
    def test_save_checkpoint_gcs(self, mock_unlink, mock_temp_file, mock_upload, mock_torch_save):
        model_state = {"model": "state"}
        
        mock_temp = Mock()
        mock_temp.name = "/tmp/tempfile.pth"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        save_checkpoint(model_state, "gs://bucket/checkpoint.pth")
        
        mock_torch_save.assert_called_once_with(model_state, "/tmp/tempfile.pth")
        mock_upload.assert_called_once_with("/tmp/tempfile.pth", "gs://bucket/checkpoint.pth")
        mock_unlink.assert_called_once_with("/tmp/tempfile.pth")

    @patch('src.utils.torch.save')
    @patch('src.utils.upload_to_gcs')
    @patch('src.utils.tempfile.NamedTemporaryFile')
    @patch('src.utils.os.unlink')
    def test_save_checkpoint_gcs_error(self, mock_unlink, mock_temp_file, mock_upload, mock_torch_save):
        mock_upload.side_effect = Exception("Upload failed")
        
        mock_temp = Mock()
        mock_temp.name = "/tmp/tempfile.pth"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        with pytest.raises(RuntimeError, match="Failed to upload checkpoint"):
            save_checkpoint({"model": "state"}, "gs://bucket/checkpoint.pth")
        
        mock_unlink.assert_called_once_with("/tmp/tempfile.pth")


class TestSaveSamples:
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.utils.Path')
    def test_save_samples_local_text(self, mock_path_class, mock_file):
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.parent = Mock()
        
        save_samples("sample text", "/local/sample.txt")
        
        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_path.write_text.assert_called_once_with("sample text")

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.utils.Path')
    def test_save_samples_local_bytes(self, mock_path_class, mock_file):
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.parent = Mock()
        
        save_samples(b"sample bytes", "/local/sample.bin")
        
        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_path.write_bytes.assert_called_once_with(b"sample bytes")

    @patch('src.utils.upload_to_gcs')
    @patch('src.utils.tempfile.NamedTemporaryFile')
    @patch('src.utils.os.unlink')
    def test_save_samples_gcs_text(self, mock_unlink, mock_temp_file, mock_upload):
        mock_temp = Mock()
        mock_temp.name = "/tmp/tempfile.txt"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        save_samples("sample text", "gs://bucket/sample.txt")
        
        mock_temp.write.assert_called_once_with("sample text")
        mock_temp.flush.assert_called_once()
        mock_temp.close.assert_called_once()
        mock_upload.assert_called_once_with("/tmp/tempfile.txt", "gs://bucket/sample.txt")
        mock_unlink.assert_called_once_with("/tmp/tempfile.txt")


class TestGetVertexCheckpointPath:
    def test_get_vertex_checkpoint_path_local(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_vertex_checkpoint_path("model.pth")
            assert result == "model.pth"

    def test_get_vertex_checkpoint_path_vertex(self):
        with patch.dict(os.environ, {"AIP_MODEL_DIR": "gs://bucket/models"}, clear=True):
            result = get_vertex_checkpoint_path("model.pth")
            assert result == "gs://bucket/models/model.pth"


class TestGetSamplesDir:
    def test_get_samples_dir_local_default(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_samples_dir()
            assert result == Path("samples")

    def test_get_samples_dir_local_custom(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_samples_dir("outputs")
            assert result == Path("outputs")

    def test_get_samples_dir_vertex_gcs(self):
        with patch.dict(os.environ, {"AIP_MODEL_DIR": "gs://bucket/models"}, clear=True):
            result = get_samples_dir()
            assert result == "gs://bucket/models/samples"

    def test_get_samples_dir_vertex_gcs_custom(self):
        with patch.dict(os.environ, {"AIP_MODEL_DIR": "gs://bucket/models"}, clear=True):
            result = get_samples_dir("outputs")
            assert result == "gs://bucket/models/outputs"

    def test_get_samples_dir_vertex_gcs_trailing_slash(self):
        with patch.dict(os.environ, {"AIP_MODEL_DIR": "gs://bucket/models/"}, clear=True):
            result = get_samples_dir()
            assert result == "gs://bucket/models/samples"

    def test_get_samples_dir_vertex_local(self):
        with patch.dict(os.environ, {"AIP_MODEL_DIR": "/local/models"}, clear=True):
            result = get_samples_dir()
            assert result == Path("/local/models") / "samples"