import os
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.utils import (
    load_checkpoint,
    save_checkpoint,
    save_samples,
    get_vertex_checkpoint_path,
    get_samples_dir
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint operations."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.linear2(self.linear(x))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model():
    """Create a sample model with some trained parameters."""
    model = SimpleModel()
    # Set some specific weights to verify they persist
    with torch.no_grad():
        model.linear.weight.fill_(0.5)
        model.linear.bias.fill_(0.1)
        model.linear2.weight.fill_(0.3)
        model.linear2.bias.fill_(0.2)
    return model


class TestCheckpointIntegration:
    def test_save_and_load_checkpoint_local(self, temp_dir, sample_model):
        """Test saving and loading a checkpoint to/from local filesystem."""
        checkpoint_path = Path(temp_dir) / "test_model.pth"
        
        # Create checkpoint data
        checkpoint_data = {
            'model_state_dict': sample_model.state_dict(),
            'epoch': 42,
            'loss': 0.123,
            'optimizer_state_dict': {'lr': 0.001}
        }
        
        # Save checkpoint
        save_checkpoint(checkpoint_data, str(checkpoint_path))
        
        # Verify file was created
        assert checkpoint_path.exists()
        assert checkpoint_path.is_file()
        
        # Load checkpoint
        loaded_data = load_checkpoint(str(checkpoint_path), 'cpu')
        
        # Verify data integrity
        assert loaded_data['epoch'] == 42
        assert loaded_data['loss'] == 0.123
        assert loaded_data['optimizer_state_dict']['lr'] == 0.001
        
        # Verify model state dict
        original_state = checkpoint_data['model_state_dict']
        loaded_state = loaded_data['model_state_dict']
        
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])
    
    def test_load_checkpoint_into_model(self, temp_dir, sample_model):
        """Test loading a checkpoint into a new model instance."""
        checkpoint_path = Path(temp_dir) / "model_for_loading.pth"
        
        # Save original model
        checkpoint_data = {
            'model_state_dict': sample_model.state_dict(),
            'epoch': 10
        }
        save_checkpoint(checkpoint_data, str(checkpoint_path))
        
        # Create new model with different weights
        new_model = SimpleModel()
        with torch.no_grad():
            new_model.linear.weight.fill_(0.999)
            new_model.linear.bias.fill_(0.888)
        
        # Load checkpoint into new model
        loaded_data = load_checkpoint(str(checkpoint_path), 'cpu')
        new_model.load_state_dict(loaded_data['model_state_dict'])
        
        # Verify weights match original model
        for (name1, param1), (name2, param2) in zip(
            sample_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    def test_checkpoint_with_different_devices(self, temp_dir, sample_model):
        """Test saving on one device and loading on another."""
        checkpoint_path = Path(temp_dir) / "device_test.pth"
        
        # Save checkpoint
        checkpoint_data = {'model_state_dict': sample_model.state_dict()}
        save_checkpoint(checkpoint_data, str(checkpoint_path))
        
        # Load with explicit device mapping
        loaded_cpu = load_checkpoint(str(checkpoint_path), 'cpu')
        
        # Verify all tensors are on CPU
        for tensor in loaded_cpu['model_state_dict'].values():
            assert tensor.device.type == 'cpu'


class TestSaveSamplesIntegration:
    def test_save_text_samples_local(self, temp_dir):
        """Test saving text samples to local filesystem."""
        sample_path = Path(temp_dir) / "text_sample.txt"
        sample_text = "Generated text sample\nLine 2\nLine 3"
        
        save_samples(sample_text, str(sample_path))
        
        # Verify file was created
        assert sample_path.exists()
        assert sample_path.is_file()
        
        # Verify content
        loaded_text = sample_path.read_text()
        assert loaded_text == sample_text
    
    def test_save_binary_samples_local(self, temp_dir):
        """Test saving binary samples to local filesystem."""
        sample_path = Path(temp_dir) / "binary_sample.bin"
        sample_data = b"\x00\x01\x02\x03\xFF\xFE"
        
        save_samples(sample_data, str(sample_path))
        
        # Verify file was created
        assert sample_path.exists()
        assert sample_path.is_file()
        
        # Verify content
        loaded_data = sample_path.read_bytes()
        assert loaded_data == sample_data
    
    def test_save_samples_creates_directories(self, temp_dir):
        """Test that save_samples creates parent directories."""
        nested_path = Path(temp_dir) / "nested" / "deep" / "sample.txt"
        sample_text = "Test content"
        
        # Verify parent directories don't exist
        assert not nested_path.parent.exists()
        
        save_samples(sample_text, str(nested_path))
        
        # Verify file and directories were created
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert nested_path.read_text() == sample_text
    
    def test_save_samples_different_modes(self, temp_dir):
        """Test saving samples with different file modes."""
        # Test text mode (default)
        text_path = Path(temp_dir) / "text_mode.txt"
        save_samples("Text content", str(text_path), mode="w")
        assert text_path.read_text() == "Text content"
        
        # Test binary mode
        binary_path = Path(temp_dir) / "binary_mode.bin"
        save_samples(b"Binary content", str(binary_path), mode="wb")
        assert binary_path.read_bytes() == b"Binary content"


class TestVertexAIIntegration:
    def test_get_vertex_checkpoint_path_local_environment(self):
        """Test get_vertex_checkpoint_path in local environment."""
        # Ensure AIP_MODEL_DIR is not set
        original_env = os.environ.get("AIP_MODEL_DIR")
        if "AIP_MODEL_DIR" in os.environ:
            del os.environ["AIP_MODEL_DIR"]
        
        try:
            result = get_vertex_checkpoint_path("my_model.pth")
            assert result == "my_model.pth"
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["AIP_MODEL_DIR"] = original_env
    
    def test_get_vertex_checkpoint_path_vertex_environment(self):
        """Test get_vertex_checkpoint_path in Vertex AI environment."""
        original_env = os.environ.get("AIP_MODEL_DIR")
        os.environ["AIP_MODEL_DIR"] = "gs://my-bucket/models"
        
        try:
            result = get_vertex_checkpoint_path("my_model.pth")
            assert result == "gs://my-bucket/models/my_model.pth"
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["AIP_MODEL_DIR"] = original_env
            else:
                del os.environ["AIP_MODEL_DIR"]
    
    def test_get_samples_dir_local_environment(self):
        """Test get_samples_dir in local environment."""
        original_env = os.environ.get("AIP_MODEL_DIR")
        if "AIP_MODEL_DIR" in os.environ:
            del os.environ["AIP_MODEL_DIR"]
        
        try:
            # Test default
            result = get_samples_dir()
            assert result == Path("samples")
            
            # Test custom
            result = get_samples_dir("outputs")
            assert result == Path("outputs")
        finally:
            if original_env is not None:
                os.environ["AIP_MODEL_DIR"] = original_env
    
    def test_get_samples_dir_vertex_gcs_environment(self):
        """Test get_samples_dir in Vertex AI GCS environment."""
        original_env = os.environ.get("AIP_MODEL_DIR")
        os.environ["AIP_MODEL_DIR"] = "gs://my-bucket/outputs"
        
        try:
            # Test default
            result = get_samples_dir()
            assert result == "gs://my-bucket/outputs/samples"
            
            # Test custom
            result = get_samples_dir("generated")
            assert result == "gs://my-bucket/outputs/generated"
            
            # Test with trailing slash
            os.environ["AIP_MODEL_DIR"] = "gs://my-bucket/outputs/"
            result = get_samples_dir()
            assert result == "gs://my-bucket/outputs/samples"
        finally:
            if original_env is not None:
                os.environ["AIP_MODEL_DIR"] = original_env
            else:
                del os.environ["AIP_MODEL_DIR"]
    
    def test_get_samples_dir_vertex_local_environment(self):
        """Test get_samples_dir in Vertex AI local environment."""
        original_env = os.environ.get("AIP_MODEL_DIR")
        os.environ["AIP_MODEL_DIR"] = "/tmp/vertex_outputs"
        
        try:
            result = get_samples_dir()
            assert result == Path("/tmp/vertex_outputs") / "samples"
            
            result = get_samples_dir("custom")
            assert result == Path("/tmp/vertex_outputs") / "custom"
        finally:
            if original_env is not None:
                os.environ["AIP_MODEL_DIR"] = original_env
            else:
                del os.environ["AIP_MODEL_DIR"]


class TestFullWorkflowIntegration:
    def test_training_workflow_simulation(self, temp_dir, sample_model):
        """Test a complete training workflow with checkpoints and samples."""
        # Simulate training setup
        samples_dir = Path(temp_dir) / "samples"
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        
        # Create directories
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate training loop
        for epoch in range(3):
            # Simulate some training...
            loss = 1.0 / (epoch + 1)  # Decreasing loss
            
            # Save checkpoint
            checkpoint_data = {
                'model_state_dict': sample_model.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'learning_rate': 0.001 * (0.9 ** epoch)
            }
            
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
            save_checkpoint(checkpoint_data, str(checkpoint_path))
            
            # Save sample output
            sample_text = f"Epoch {epoch}: Generated sample text with loss {loss:.3f}"
            sample_path = samples_dir / f"epoch_{epoch}_sample.txt"
            save_samples(sample_text, str(sample_path))
        
        # Verify all checkpoints were saved
        assert len(list(checkpoint_dir.glob("*.pth"))) == 3
        
        # Verify all samples were saved
        assert len(list(samples_dir.glob("*.txt"))) == 3
        
        # Load and verify final checkpoint
        final_checkpoint = load_checkpoint(str(checkpoint_dir / "epoch_2.pth"), 'cpu')
        assert final_checkpoint['epoch'] == 2
        assert final_checkpoint['loss'] == 1.0 / 3
        
        # Verify sample content
        final_sample = (samples_dir / "epoch_2_sample.txt").read_text()
        assert "Epoch 2" in final_sample
        assert "0.333" in final_sample
    
    def test_checkpoint_resuming_workflow(self, temp_dir, sample_model):
        """Test resuming training from a checkpoint."""
        checkpoint_path = Path(temp_dir) / "resume_test.pth"
        
        # Save initial checkpoint
        initial_data = {
            'model_state_dict': sample_model.state_dict(),
            'epoch': 5,
            'loss': 0.456,
            'optimizer_state': {'momentum_buffer': [1, 2, 3]}
        }
        save_checkpoint(initial_data, str(checkpoint_path))
        
        # Simulate resuming training
        loaded_data = load_checkpoint(str(checkpoint_path), 'cpu')
        
        # Create new model and load state
        resumed_model = SimpleModel()
        resumed_model.load_state_dict(loaded_data['model_state_dict'])
        
        # Verify resuming works correctly
        assert loaded_data['epoch'] == 5
        assert loaded_data['loss'] == 0.456
        assert loaded_data['optimizer_state']['momentum_buffer'] == [1, 2, 3]
        
        # Verify model weights match
        for (name1, param1), (name2, param2) in zip(
            sample_model.named_parameters(), resumed_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)