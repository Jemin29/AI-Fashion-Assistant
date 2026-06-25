"""
week4/tests/test_personalized_generator.py
=========================================
Unit tests for the PersonalizedFashionGenerator system.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.lora.style_manager.lora_registry import LoraRegistry
from src.lora.inference.lora_inference import LoraInferenceSystem
from src.lora.inference.personalized_generator import PersonalizedFashionGenerator


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


@pytest.fixture
def mock_inference_system(mock_registry, temp_workspace):
    """Instantiate a LoraInferenceSystem in dry-run mode."""
    output_dir = temp_workspace / "inference_out"
    return LoraInferenceSystem(
        registry=mock_registry,
        output_dir=output_dir,
        dry_run=True
    )


class TestPersonalizedFashionGenerator:
    """Verify prompt formatting, personalized generation execution, and sidecar metadata formatting."""

    def test_build_personalized_prompt(self, mock_inference_system):
        """Verify build_personalized_prompt merges preferences correctly."""
        generator = PersonalizedFashionGenerator(inference_system=mock_inference_system)
        
        prompt = generator.build_personalized_prompt(
            base_item="hoodie",
            color="black",
            style="streetwear"
        )
        
        assert "black" in prompt
        assert "streetwear" in prompt
        assert "hoodie" in prompt

    def test_generate_personalized_design_success(self, mock_inference_system):
        """Verify successful design generation and preferences metadata serialization."""
        generator = PersonalizedFashionGenerator(inference_system=mock_inference_system)
        
        preferences = {
            "favorite_brand": "nike",
            "favorite_color": "black",
            "preferred_style": "streetwear"
        }
        
        res = generator.generate_personalized_design(
            base_item="hoodie",
            preferences=preferences,
            scale=1.2,
            seed=42
        )
        
        assert res["success"] is True
        assert res["preferences"] == preferences
        assert "nike" in res["brand"]
        assert "black" in res["prompt"]
        assert "streetwear" in res["prompt"]
        assert "sportswear" in res["prompt"] # brand token appended by style switcher
        
        # Verify output metadata file contains preferences key
        metadata_path = Path(res["metadata_path"])
        assert metadata_path.exists()
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        assert "preferences" in metadata
        assert metadata["preferences"]["favorite_brand"] == "nike"
        assert metadata["preferences"]["favorite_color"] == "black"
        assert metadata["preferences"]["preferred_style"] == "streetwear"

    def test_generate_personalized_design_default_preferences(self, mock_inference_system):
        """Verify default preferences are applied when empty preferences dict is passed."""
        generator = PersonalizedFashionGenerator(inference_system=mock_inference_system)
        
        res = generator.generate_personalized_design(
            base_item="hoodie",
            preferences={}
        )
        
        assert res["success"] is True
        assert res["preferences"]["favorite_brand"] == "nike"
        assert res["preferences"]["favorite_color"] == "black"
        assert res["preferences"]["preferred_style"] == "streetwear"
