"""
tests/test_model_config.py
==========================
Unit tests for the centralized Pydantic-based configuration manager.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.model_config import (
    CentralizedConfig,
    SDXLSettings,
    DeviceSettings,
    ImageResolution,
    BatchSettings,
    SeedSettings,
    OutputPaths,
    SIZE_PRESETS,
)


def test_sdxl_settings_validation():
    """Verify validations on SDXL hyper-parameters."""
    # Defaults
    settings = SDXLSettings()
    assert settings.guidance_scale == 7.5
    assert settings.num_inference_steps == 30
    assert settings.scheduler == "euler"
    assert settings.use_refiner is False

    # Valid values
    s2 = SDXLSettings(refiner_strength=0.5, guidance_scale=12.0, num_inference_steps=50)
    assert s2.refiner_strength == 0.5
    assert s2.guidance_scale == 12.0
    assert s2.num_inference_steps == 50

    # Invalid refiner strength
    with pytest.raises(ValidationError):
        SDXLSettings(refiner_strength=1.5)
    with pytest.raises(ValidationError):
        SDXLSettings(refiner_strength=-0.1)

    # Invalid guidance scale
    with pytest.raises(ValidationError):
        SDXLSettings(guidance_scale=-1.0)
    with pytest.raises(ValidationError):
        SDXLSettings(guidance_scale=0.0)

    # Invalid inference steps
    with pytest.raises(ValidationError):
        SDXLSettings(num_inference_steps=0)
    with pytest.raises(ValidationError):
        SDXLSettings(num_inference_steps=-10)


def test_device_settings_validation():
    """Verify hardware and device options validations."""
    settings = DeviceSettings()
    assert settings.device == "auto"
    assert settings.torch_dtype == "float16"

    # Valid settings
    s2 = DeviceSettings(device="cuda", torch_dtype="bfloat16")
    assert s2.device == "cuda"
    assert s2.torch_dtype == "bfloat16"

    # Invalid device
    with pytest.raises(ValidationError):
        DeviceSettings(device="invalid-device")

    # Invalid dtype
    with pytest.raises(ValidationError):
        DeviceSettings(torch_dtype="int16")


def test_image_resolution_presets_and_validation():
    """Verify presets loading and divisibility checks on image resolution."""
    res = ImageResolution(width=1024, height=1024)
    assert res.width == 1024
    assert res.height == 1024

    # Size preset square_512
    res_preset = ImageResolution(size_preset="square_512")
    assert res_preset.width == 512
    assert res_preset.height == 512

    # Size preset portrait_1024
    res_portrait = ImageResolution(size_preset="portrait_1024")
    assert res_portrait.width == 832
    assert res_portrait.height == 1216

    # Invalid size preset
    with pytest.raises(ValidationError):
        ImageResolution(size_preset="unknown_preset")

    # Not divisible by 8
    with pytest.raises(ValidationError):
        ImageResolution(width=1021, height=1024)
    with pytest.raises(ValidationError):
        ImageResolution(width=1024, height=1027)


def test_batch_settings_validation():
    """Verify batch configurations validations."""
    b = BatchSettings()
    assert b.batch_size == 1
    assert b.max_workers == 2

    # Invalid values
    with pytest.raises(ValidationError):
        BatchSettings(batch_size=0)
    with pytest.raises(ValidationError):
        BatchSettings(max_workers=-5)


def test_seed_settings():
    """Verify seed configurations."""
    s = SeedSettings()
    assert s.seed == -1
    assert s.use_random_seed is True


def test_output_paths_directories_creation(tmp_path):
    """Verify OutputPaths initialization and directory creation."""
    paths = OutputPaths(
        output_dir=tmp_path / "out",
        metadata_dir=tmp_path / "meta",
        batch_dir=tmp_path / "batch",
        experiments_dir=tmp_path / "exp",
        log_dir=tmp_path / "logs",
    )

    assert not paths.output_dir.exists()
    
    paths.create_directories()
    
    assert paths.output_dir.exists()
    assert paths.metadata_dir.exists()
    assert paths.batch_dir.exists()
    assert paths.experiments_dir.exists()
    assert paths.log_dir.exists()


def test_yaml_serialization_deserialization(tmp_path):
    """Verify loading config from YAML and writing config to YAML."""
    yaml_file = tmp_path / "config.yaml"
    
    config = CentralizedConfig()
    config.sdxl.guidance_scale = 9.5
    config.resolution.size_preset = "portrait_1024"
    config.batch.batch_size = 4
    config.paths.output_dir = tmp_path / "custom_out"
    
    config.save_to_yaml(yaml_file)
    assert yaml_file.exists()
    
    # Reload from YAML
    reloaded = CentralizedConfig.load_from_yaml(yaml_file)
    assert reloaded.sdxl.guidance_scale == 9.5
    assert reloaded.resolution.width == 832
    assert reloaded.resolution.height == 1216
    assert reloaded.batch.batch_size == 4
    assert reloaded.paths.output_dir == tmp_path / "custom_out"


def test_environment_variable_overrides():
    """Verify env variables can override settings nested under prefixes."""
    os.environ["FASHION_CONFIG_SDXL__GUIDANCE_SCALE"] = "15.5"
    os.environ["FASHION_CONFIG_DEVICE__DEVICE"] = "mps"
    os.environ["FASHION_CONFIG_RESOLUTION__SIZE_PRESET"] = "square_512"
    os.environ["FASHION_CONFIG_BATCH__BATCH_SIZE"] = "8"
    
    try:
        config = CentralizedConfig()
        assert config.sdxl.guidance_scale == 15.5
        assert config.device.device == "mps"
        assert config.resolution.width == 512
        assert config.resolution.height == 512
        assert config.batch.batch_size == 8
    finally:
        # Clean up env
        os.environ.pop("FASHION_CONFIG_SDXL__GUIDANCE_SCALE", None)
        os.environ.pop("FASHION_CONFIG_DEVICE__DEVICE", None)
        os.environ.pop("FASHION_CONFIG_RESOLUTION__SIZE_PRESET", None)
        os.environ.pop("FASHION_CONFIG_BATCH__BATCH_SIZE", None)
