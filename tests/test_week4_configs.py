"""
week4/tests/test_configs.py
===========================
Unit tests verifying the Week 4 Pydantic configuration validation system.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config_manager import (
    Week4Config,
    LoraConfig,
    DatasetConfig,
    TrainerConfig,
    InferenceConfig,
    get_default_config
)


class TestWeek4Configurations:
    """Validate Pydantic configurations boundaries, yaml serializations, and defaults."""

    def test_default_config_loading(self):
        """Verify defaults match architectural specs."""
        cfg = get_default_config()
        assert isinstance(cfg, Week4Config)
        assert cfg.lora.r == 8
        assert cfg.lora.alpha == 16
        assert "to_q" in cfg.lora.target_modules
        assert cfg.dataset.resolution == 1024
        assert cfg.trainer.mixed_precision == "fp16"
        assert cfg.inference.lora_scale == 0.8

    def test_lora_rank_boundaries(self):
        """Verify boundaries constraints trigger validation errors."""
        # Rank too small
        with pytest.raises(ValidationError):
            LoraConfig(r=0)
            
        # Rank too large
        with pytest.raises(ValidationError):
            LoraConfig(r=512)

        # Valid rank
        cfg = LoraConfig(r=32)
        assert cfg.r == 32

    def test_dataset_resolution_boundaries(self):
        """Verify resolution constraints."""
        with pytest.raises(ValidationError):
            DatasetConfig(resolution=100)

        with pytest.raises(ValidationError):
            DatasetConfig(resolution=4000)

        cfg = DatasetConfig(resolution=512)
        assert cfg.resolution == 512

    def test_trainer_precision_regex(self):
        """Verify precision settings only accept fp16, bf16, or no."""
        with pytest.raises(ValidationError):
            TrainerConfig(mixed_precision="fp64")

        cfg = TrainerConfig(mixed_precision="bf16")
        assert cfg.mixed_precision == "bf16"

    def test_config_sync_output_path(self):
        """Verify model validators synchronize paths dynamically."""
        cfg = Week4Config(output_root="custom_outputs")
        assert cfg.trainer.output_dir == str(Path("custom_outputs/trainer"))

    def test_yaml_serialization_cycle(self):
        """Verify configurations can save to YAML and parse back accurately."""
        cfg = Week4Config()
        cfg.lora.r = 64
        cfg.trainer.learning_rate = 5e-5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_yaml = Path(tmpdir) / "config.yaml"
            cfg.save(temp_yaml)
            
            assert temp_yaml.exists()
            
            # Load back
            loaded = Week4Config.from_yaml(temp_yaml)
            assert loaded.lora.r == 64
            assert loaded.trainer.learning_rate == 5e-5
            assert loaded.dataset.resolution == 1024
