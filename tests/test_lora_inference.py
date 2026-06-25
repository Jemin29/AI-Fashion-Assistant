"""
week4/tests/test_lora_inference.py
==================================
Unit tests for LoraInferenceSystem.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.lora.style_manager.lora_registry import LoraRegistry
from src.lora.inference.lora_inference import LoraInferenceSystem
from src.utils.config_manager import get_default_config


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_registry(temp_workspace):
    """Instantiate a registry with dummy registered files for all 4 brand styles."""
    db_file = temp_workspace / "lora_registry.json"
    registry = LoraRegistry(registry_path=db_file)
    
    # Create dummy weights files for all brands to avoid KeyError during switching
    for brand in ["nike", "gucci", "zara", "h&m"]:
        weight_file = temp_workspace / f"{brand}.safetensors"
        with open(weight_file, "wb") as f:
            f.write(brand.upper().encode())
        registry.register_model(brand=brand, model_path=weight_file)
        
    return registry


class TestLoraInferenceSystem:
    """Verify initialization, pipeline loading, generation, batching, and metadata sidecar generation."""

    def test_initialization(self, temp_workspace, mock_registry):
        """Test initializing LoraInferenceSystem with different parameters."""
        config = get_default_config()
        output_dir = temp_workspace / "inference_out"
        
        system = LoraInferenceSystem(
            config=config,
            registry=mock_registry,
            output_dir=output_dir,
            device="cpu",
            dry_run=True
        )
        
        assert system.config == config
        assert system.registry == mock_registry
        assert system.device == "cpu"
        assert system.dry_run is True
        assert system.output_dir == output_dir.resolve()
        assert system.pipeline is None
        assert system.switcher is None

    def test_load_pipeline_dry_run(self, mock_registry, temp_workspace):
        """Test pipeline loading in dry-run mode."""
        output_dir = temp_workspace / "inference_out"
        system = LoraInferenceSystem(
            registry=mock_registry,
            output_dir=output_dir,
            dry_run=True
        )
        
        system.load_pipeline()
        
        assert system.pipeline is not None
        assert system.switcher is not None
        
        # Test mock pipeline calls
        system.pipeline.load_lora_weights("dummy_path", "nike")
        assert system.pipeline.loaded_adapters["nike"] == "dummy_path"
        
        system.pipeline.set_adapters(["nike"], [0.8])
        assert system.pipeline.active_adapters == ["nike"]
        assert system.pipeline.active_weights == [0.8]
        
        system.pipeline.unload_lora_weights()
        assert not system.pipeline.loaded_adapters
        assert not system.pipeline.active_adapters

    def test_generate_single_dry_run(self, mock_registry, temp_workspace):
        """Verify generating a single style design image with metadata sidecar."""
        output_dir = temp_workspace / "inference_out"
        system = LoraInferenceSystem(
            registry=mock_registry,
            output_dir=output_dir,
            dry_run=True
        )
        
        prompt = "Red streetwear hoodie"
        brand = "nike"
        scale = 1.2
        seed = 42
        
        res = system.generate(prompt=prompt, brand=brand, scale=scale, seed=seed)
        
        assert res["success"] is True
        assert res["brand"] == "nike"
        assert res["scale"] == 1.2
        assert "sportswear" in res["prompt"]
        assert res["dry_run"] is True
        
        # Verify output files
        image_path = Path(res["image_path"])
        metadata_path = Path(res["metadata_path"])
        
        assert image_path.exists()
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        assert metadata["prompt"] == prompt
        assert metadata["enriched_prompt"] == res["prompt"]
        assert metadata["brand"] == brand
        assert metadata["scale"] == scale
        assert metadata["seed"] == seed
        assert metadata["device"] == "cpu"
        assert metadata["resolution"] == "512x512"
        assert "timestamp" in metadata

    def test_generate_batch_dry_run(self, mock_registry, temp_workspace):
        """Verify batch generation with mismatched/padding parameter arrays."""
        output_dir = temp_workspace / "inference_out"
        system = LoraInferenceSystem(
            registry=mock_registry,
            output_dir=output_dir,
            dry_run=True
        )
        
        prompts = ["Red streetwear hoodie", "Casual linen shirt"]
        brands = ["nike"]  # Should be padded
        scales = [1.2, 0.8]
        seeds = [10, 20]
        
        results = system.generate_batch(
            prompts=prompts,
            brands=brands,
            scales=scales,
            seeds=seeds
        )
        
        assert len(results) == 2
        
        # First item (nike)
        assert results[0]["success"] is True
        assert results[0]["brand"] == "nike"
        assert results[0]["scale"] == 1.2
        
        # Second item (should pad brand nike, scale 0.8)
        assert results[1]["success"] is True
        assert results[1]["brand"] == "nike"  # padded from first element
        assert results[1]["scale"] == 0.8

    def test_generate_batch_invalid_input(self, mock_registry):
        """Verify ValueError raised when prompts, brands, or scales are empty."""
        system = LoraInferenceSystem(registry=mock_registry, dry_run=True)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            system.generate_batch(prompts=[], brands=["nike"], scales=[1.0])
            
        with pytest.raises(ValueError, match="cannot be empty"):
            system.generate_batch(prompts=["Red hoodie"], brands=[], scales=[1.0])

        with pytest.raises(ValueError, match="cannot be empty"):
            system.generate_batch(prompts=["Red hoodie"], brands=["nike"], scales=[])

    @patch("diffusers.StableDiffusionXLPipeline.from_pretrained")
    def test_real_pipeline_loading_and_generation(self, mock_from_pretrained, mock_registry, temp_workspace):
        """Test pipeline loading and generation flow with a mocked StableDiffusionXLPipeline (non-dry_run)."""
        mock_pipe_instance = MagicMock()
        
        class MockPipelineOutput:
            def __init__(self):
                from PIL import Image
                self.images = [Image.new("RGB", (100, 100), color=(0, 255, 0))]
                
        mock_pipe_instance.return_value = MockPipelineOutput()
        mock_from_pretrained.return_value = mock_pipe_instance
        
        output_dir = temp_workspace / "inference_out"
        system = LoraInferenceSystem(
            registry=mock_registry,
            output_dir=output_dir,
            dry_run=False,
            device="cpu"
        )
        
        system.load_pipeline()
        
        assert system.pipeline == mock_pipe_instance
        mock_from_pretrained.assert_called_once()
        
        # Generate in non-dry-run mode
        prompt = "Luxe gold silk gown"
        brand = "gucci"
        scale = 1.5
        seed = 123
        
        res = system.generate(prompt=prompt, brand=brand, scale=scale, seed=seed)
        
        assert res["success"] is True
        assert res["dry_run"] is False
        
        # Check that pipeline call was invoked
        mock_pipe_instance.assert_called_once()
        
        # Verify metadata resolution is 1024x1024
        metadata_path = Path(res["metadata_path"])
        assert metadata_path.exists()
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["resolution"] == "1024x1024"
        assert metadata["seed"] == 123
