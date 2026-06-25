"""
week4/tests/test_style_mixer.py
==============================
Unit tests for the StyleMixer blending engine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.lora.style_manager.lora_registry import LoraRegistry
from src.lora.style_manager.style_mixer import StyleMixer


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_registry(temp_workspace):
    """Instantiate a registry with dummy registered files for Nike, Gucci, and Zara."""
    db_file = temp_workspace / "lora_registry.json"
    registry = LoraRegistry(registry_path=db_file)
    
    # Create dummy weights files
    nike_file = temp_workspace / "nike.safetensors"
    with open(nike_file, "wb") as f:
        f.write(b"NIKE")

    gucci_file = temp_workspace / "gucci.safetensors"
    with open(gucci_file, "wb") as f:
        f.write(b"GUCCI")

    zara_file = temp_workspace / "zara.safetensors"
    with open(zara_file, "wb") as f:
        f.write(b"ZARA")

    registry.register_model(brand="nike", model_path=nike_file)
    registry.register_model(brand="gucci", model_path=gucci_file)
    registry.register_model(brand="zara", model_path=zara_file)
    return registry


class TestStyleMixer:
    """Verify prompt blending formulas, weights normalization rules, and dynamic generation."""

    def test_generate_blended_prompt(self, mock_registry):
        """Verify prompt triggers are blended proportionally according to weights."""
        mixer = StyleMixer(registry=mock_registry)
        
        base_prompt = "Oversized hoodie"
        brand_weights = {"nike": 0.7, "gucci": 0.3}
        
        blended = mixer.generate_blended_prompt(base_prompt, brand_weights)
        
        assert base_prompt in blended
        assert "70% sportswear" in blended
        assert "30% luxury" in blended

        # Try a different blend ratio
        brand_weights_2 = {"nike": 0.5, "zara": 0.5}
        blended_2 = mixer.generate_blended_prompt(base_prompt, brand_weights_2)
        assert "50% sportswear" in blended_2
        assert "50% casual" in blended_2

    def test_mix_styles_normalization(self, mock_registry):
        """Verify input weights are normalized and toggled correctly in registry and pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.unload_lora_weights = MagicMock()
        mock_pipeline.load_lora_weights = MagicMock()
        mock_pipeline.set_adapters = MagicMock()

        mixer = StyleMixer(registry=mock_registry, inference_pipeline=mock_pipeline)

        # Mix 70% Nike and 30% Gucci (normalized automatically)
        brand_weights = {"nike": 7.0, "gucci": 3.0}
        norm_weights = mixer.mix_styles(brand_weights)

        # Check normalization
        assert norm_weights["nike"] == 0.7
        assert norm_weights["gucci"] == 0.3
        
        # Verify active and loaded states in registry
        assert mock_registry.models["nike"]["active"] is True
        assert mock_registry.models["nike"]["scale"] == 0.7
        assert mock_registry.models["nike"]["loaded"] is True

        assert mock_registry.models["gucci"]["active"] is True
        assert mock_registry.models["gucci"]["scale"] == 0.3
        assert mock_registry.models["gucci"]["loaded"] is True

        # Zara should be inactive
        assert mock_registry.models["zara"]["active"] is False

        # Verify pipeline method calls
        assert mock_pipeline.unload_lora_weights.call_count == 1
        assert mock_pipeline.load_lora_weights.call_count == 2
        mock_pipeline.set_adapters.assert_called_once_with(
            ["nike", "gucci"],
            adapter_weights=[0.7, 0.3]
        )

    def test_mix_styles_validation_errors(self, mock_registry):
        """Verify invalid weights and unsupported brands trigger ValueError."""
        mixer = StyleMixer(registry=mock_registry)

        # Unsupported brand
        with pytest.raises(ValueError):
            mixer.mix_styles({"nike": 0.5, "adidas": 0.5})

        # Negative weight
        with pytest.raises(ValueError):
            mixer.mix_styles({"nike": -0.2, "gucci": 1.2})

        # Sum of weights <= 0
        with pytest.raises(ValueError):
            mixer.mix_styles({"nike": 0.0, "gucci": 0.0})

        # Empty dict
        with pytest.raises(ValueError):
            mixer.mix_styles({})

    def test_generate_mixed_design_dry_run(self, mock_registry, temp_workspace):
        """Verify dry-run mode switches adapters, blends prompts, and draws mixed color canvas."""
        mixer = StyleMixer(
            registry=mock_registry,
            output_dir=temp_workspace / "mixer_out"
        )

        res = mixer.generate_mixed_design(
            prompt="Winter parka jacket",
            brand_weights={"nike": 0.6, "gucci": 0.4},
            dry_run=True
        )

        assert res["success"] is True
        assert res["dry_run"] is True
        assert "blended style of 60%" in res["prompt"]
        assert "40% luxury" in res["prompt"]
        assert res["weights"]["nike"] == 0.6
        assert res["weights"]["gucci"] == 0.4

        # Verify image exists
        image_path = Path(res["image_path"])
        assert image_path.exists()
        assert image_path.suffix == ".png"
