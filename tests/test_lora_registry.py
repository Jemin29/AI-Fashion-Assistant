"""
week4/tests/test_lora_registry.py
=================================
Unit tests for the LoraRegistry style adapter manager.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.lora.style_manager.lora_registry import LoraRegistry


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_weights_file(temp_workspace):
    """Create a dummy weights file representing an adapter safetensors file."""
    path = temp_workspace / "nike_style_adapter.safetensors"
    with open(path, "wb") as f:
        f.write(b"DUMMY_LORA_WEIGHTS_DATA_SAFETENSORS")
    return path


class TestLoraRegistry:
    """Verify registry database serialization, validation rules, load triggers, and state queries."""

    def test_registry_initialization(self, temp_workspace):
        """Verify DB file resolved and initialized empty."""
        db_file = temp_workspace / "lora_registry.json"
        registry = LoraRegistry(registry_path=db_file)
        assert registry.registry_path == db_file.resolve()
        assert not registry.models

    def test_register_model_success(self, temp_workspace, dummy_weights_file):
        """Verify successful model registration and file metadata compilation."""
        db_file = temp_workspace / "lora_registry.json"
        registry = LoraRegistry(registry_path=db_file)

        extra_meta = {"rank": 8, "alpha": 16, "base_model": "sdxl-1.0"}
        entry = registry.register_model(
            brand="nike",
            model_path=dummy_weights_file,
            metadata=extra_meta
        )

        assert entry["brand"] == "nike"
        assert entry["model_path"] == str(dummy_weights_file.resolve().as_posix())
        assert entry["size_bytes"] == dummy_weights_file.stat().st_size
        assert entry["loaded"] is False
        assert entry["active"] is False
        assert entry["scale"] == 1.0
        assert entry["metadata"] == extra_meta

        # Verify DB file persisted on disk
        assert db_file.exists()
        with open(db_file, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        assert "nike" in persisted
        assert persisted["nike"]["metadata"]["rank"] == 8

    def test_register_model_validation_failures(self, temp_workspace, dummy_weights_file):
        """Verify brand checks and missing files raise errors."""
        registry = LoraRegistry(registry_path=temp_workspace / "db.json")

        # Unsupported brand
        with pytest.raises(ValueError):
            registry.register_model(brand="adidas", model_path=dummy_weights_file)

        # Missing weight file
        with pytest.raises(FileNotFoundError):
            registry.register_model(brand="nike", model_path=temp_workspace / "missing.safetensors")

    def test_load_model_state(self, temp_workspace, dummy_weights_file):
        """Verify toggle loading changes state status successfully."""
        registry = LoraRegistry(registry_path=temp_workspace / "db.json")
        registry.register_model(brand="nike", model_path=dummy_weights_file)

        assert registry.models["nike"]["loaded"] is False
        
        # Load model
        entry = registry.load_model(brand="nike")
        assert entry["loaded"] is True
        assert registry.models["nike"]["loaded"] is True

        # Error if brand not registered
        with pytest.raises(KeyError):
            registry.load_model(brand="gucci")

    def test_activation_and_blend_weights(self, temp_workspace, dummy_weights_file):
        """Verify activation and blend scales limits."""
        registry = LoraRegistry(registry_path=temp_workspace / "db.json")
        registry.register_model(brand="nike", model_path=dummy_weights_file)

        assert registry.models["nike"]["active"] is False
        
        # Activate model
        registry.activate_model(brand="nike", scale=0.8)
        assert registry.models["nike"]["active"] is True
        assert registry.models["nike"]["scale"] == 0.8

        # Try invalid scales
        with pytest.raises(ValueError):
            registry.activate_model(brand="nike", scale=-0.5)

        with pytest.raises(ValueError):
            registry.activate_model(brand="nike", scale=2.5)

        # Deactivate model
        registry.deactivate_model(brand="nike")
        assert registry.models["nike"]["active"] is False
        assert registry.models["nike"]["scale"] == 1.0

    def test_list_models_filtering(self, temp_workspace, dummy_weights_file):
        """Verify list queries and active filtering options."""
        registry = LoraRegistry(registry_path=temp_workspace / "db.json")
        
        # Create a second dummy weights file
        gucci_weights = temp_workspace / "gucci.safetensors"
        with open(gucci_weights, "wb") as f:
            f.write(b"GUCCI")

        registry.register_model(brand="nike", model_path=dummy_weights_file)
        registry.register_model(brand="gucci", model_path=gucci_weights)

        # List all
        all_models = registry.list_models()
        assert len(all_models) == 2
        assert "nike" in all_models
        assert "gucci" in all_models

        # Active filter (currently none are active)
        active_models = registry.list_models(filter_active=True)
        assert len(active_models) == 0

        # Activate Nike
        registry.activate_model(brand="nike", scale=1.2)
        active_models = registry.list_models(filter_active=True)
        assert len(active_models) == 1
        assert "nike" in active_models
        assert "gucci" not in active_models
