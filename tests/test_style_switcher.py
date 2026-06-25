"""
week4/tests/test_style_switcher.py
==================================
Unit tests for the StyleSwitcher dynamic switching engine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.lora.style_manager.lora_registry import LoraRegistry
from src.lora.style_manager.style_switcher import StyleSwitcher


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_registry(temp_workspace):
    """Instantiate a registry with dummy registered files for Nike and Gucci."""
    db_file = temp_workspace / "lora_registry.json"
    registry = LoraRegistry(registry_path=db_file)
    
    # Create dummy weights files
    nike_file = temp_workspace / "nike.safetensors"
    with open(nike_file, "wb") as f:
        f.write(b"NIKE")

    gucci_file = temp_workspace / "gucci.safetensors"
    with open(gucci_file, "wb") as f:
        f.write(b"GUCCI")

    registry.register_model(brand="nike", model_path=nike_file)
    registry.register_model(brand="gucci", model_path=gucci_file)
    return registry


class TestStyleSwitcher:
    """Verify prompt preprocess expansions, weights switching commands, and dynamic generation."""

    def test_preprocess_prompt(self, mock_registry):
        """Verify brand style presets are correctly appended to base prompts."""
        switcher = StyleSwitcher(registry=mock_registry)
        
        base_prompt = "Black oversized hoodie"
        
        # Nike prompt expansion
        nike_prompt = switcher.preprocess_prompt(base_prompt, brand="nike")
        assert base_prompt in nike_prompt
        assert "sportswear" in nike_prompt
        assert "athletic fit" in nike_prompt

        # Gucci prompt expansion
        gucci_prompt = switcher.preprocess_prompt(base_prompt, brand="gucci")
        assert base_prompt in gucci_prompt
        assert "luxury" in gucci_prompt
        assert "haute-couture" in gucci_prompt

        # Zara prompt expansion
        zara_prompt = switcher.preprocess_prompt(base_prompt, brand="zara")
        assert base_prompt in zara_prompt
        assert "casual" in zara_prompt
        assert "minimalist look" in zara_prompt

        # H&M prompt expansion
        hm_prompt = switcher.preprocess_prompt(base_prompt, brand="h&m")
        assert base_prompt in hm_prompt
        assert "basic" in hm_prompt
        assert "organic textures" in hm_prompt

        # Double preprocessing check (should not duplicate)
        double_preprocess = switcher.preprocess_prompt(nike_prompt, brand="nike")
        assert double_preprocess == nike_prompt

    def test_switch_style_toggles(self, mock_registry):
        """Verify registry toggles and pipeline weights loading methods are invoked."""
        mock_pipeline = MagicMock()
        mock_pipeline.unload_lora_weights = MagicMock()
        mock_pipeline.load_lora_weights = MagicMock()
        mock_pipeline.set_adapters = MagicMock()

        switcher = StyleSwitcher(registry=mock_registry, inference_pipeline=mock_pipeline)

        # Initially inactive
        assert not mock_registry.list_models(filter_active=True)

        # Switch to Nike
        switcher.switch_style(brand="nike", scale=0.8)
        
        # Registry asserts
        assert mock_registry.models["nike"]["active"] is True
        assert mock_registry.models["nike"]["scale"] == 0.8
        assert mock_registry.models["nike"]["loaded"] is True
        assert mock_registry.models["gucci"]["active"] is False

        # Pipeline asserts
        mock_pipeline.unload_lora_weights.assert_called_once()
        mock_pipeline.load_lora_weights.assert_called_once_with(
            mock_registry.models["nike"]["model_path"],
            adapter_name="nike"
        )
        mock_pipeline.set_adapters.assert_called_once_with(["nike"], adapter_weights=[0.8])

        # Switch to Gucci
        switcher.switch_style(brand="gucci", scale=1.5)
        
        # Nike should deactivate, Gucci should activate
        assert mock_registry.models["nike"]["active"] is False
        assert mock_registry.models["gucci"]["active"] is True
        assert mock_registry.models["gucci"]["scale"] == 1.5

    def test_generate_styled_design_dry_run(self, mock_registry, temp_workspace):
        """Verify dry-run mode switches adapters, builds prompt, and draws mock design canvas."""
        switcher = StyleSwitcher(
            registry=mock_registry,
            output_dir=temp_workspace / "switcher_out"
        )

        res = switcher.generate_styled_design(
            prompt="Red streetwear jacket",
            brand="nike",
            scale=1.2,
            dry_run=True
        )

        assert res["success"] is True
        assert res["dry_run"] is True
        assert "sportswear" in res["prompt"]
        assert "techwear style" in res["prompt"]
        assert res["scale"] == 1.2

        # Verify generated mock image is present
        image_path = Path(res["image_path"])
        assert image_path.exists()
        assert image_path.suffix == ".png"

    def test_unregistered_style_raises_error(self, mock_registry):
        """Verify switching to an unregistered brand style raises a KeyError."""
        switcher = StyleSwitcher(registry=mock_registry)
        
        with pytest.raises(KeyError):
            switcher.switch_style(brand="zara")
