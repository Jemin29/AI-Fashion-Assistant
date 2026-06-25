"""
week3/config_manager.py
=======================
Centralized Configuration Manager for Week 3 — ControlNet & Style Control.
Loads configurations from YAML overlays and validates them using Pydantic v2.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Resolve Paths ─────────────────────────────────────────────────────────────
_WEEK3_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _WEEK3_DIR / "configs"
_PROJECT_ROOT = _WEEK3_DIR.parent


# =============================================================================
# ── Sub-models: Model Specs
# =============================================================================

class ModelSpec(BaseModel):
    """Configuration specification for a single model (Base or Refiner)."""
    repo_id: str = ""
    variant: Optional[str] = "fp16"
    use_safetensors: bool = True
    local_path: Optional[str] = None


class RuntimeConfig(BaseModel):
    """Memory optimization and hardware device configs."""
    device: str = "auto"
    torch_dtype: str = "float16"
    enable_xformers: bool = True
    enable_cpu_offload: bool = False
    enable_sequential_offload: bool = False
    enable_attention_slicing: bool = False


class ModelConfig(BaseModel):
    """SDXL model parameters."""
    base: ModelSpec = Field(default_factory=ModelSpec)
    refiner: ModelSpec = Field(default_factory=ModelSpec)
    vae: ModelSpec = Field(default_factory=ModelSpec)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


# =============================================================================
# ── Sub-models: ControlNet Preprocessors
# =============================================================================

class CannyConfig(BaseModel):
    """Thresholds and size settings for the Canny edge detector."""
    low_threshold: int = 100
    high_threshold: int = 200
    resolution: int = 1024


class OpenposeConfig(BaseModel):
    """Body pose detection settings."""
    detect_hand_and_face: bool = True
    resolution: int = 1024


class DepthConfig(BaseModel):
    """Depth estimation preprocessor settings."""
    model_type: str = "dpt_hybrid"
    resolution: int = 1024


class PreprocessorsConfig(BaseModel):
    """Nested configuration for all preprocessors."""
    canny: CannyConfig = Field(default_factory=CannyConfig)
    openpose: OpenposeConfig = Field(default_factory=OpenposeConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)


class ControlNetSpec(BaseModel):
    """Parameters mapping the ControlNet model configuration."""
    enabled: bool = True
    model_id: str = "diffusers/controlnet-canny-sdxl-1.0"
    conditioning_scale: float = 0.8
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0

    @field_validator("conditioning_scale")
    @classmethod
    def _validate_scale(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"conditioning_scale must be between 0.0 and 1.0, got {v}")
        return v


# =============================================================================
# ── Main Settings Configuration (Pydantic settings)
# =============================================================================

class Week3Config(BaseSettings):
    """
    Main configurations class for Week 3.
    Binds properties to environment variables prefixed with `FASHION_CONFIG_`.
    """
    model_config = SettingsConfigDict(
        env_prefix="FASHION_CONFIG_",
        env_nested_delimiter="__",
        extra="ignore"
    )

    # Core models and layers
    model: ModelConfig = Field(default_factory=ModelConfig)
    controlnet: ControlNetSpec = Field(default_factory=ControlNetSpec)
    preprocessors: PreprocessorsConfig = Field(default_factory=PreprocessorsConfig)

    # General Paths
    output_dir: Path = _WEEK3_DIR / "outputs"
    log_dir: Path = _WEEK3_DIR / "logs"

    @field_validator("output_dir", "log_dir", mode="before")
    @classmethod
    def _resolve_paths(cls, v: Any) -> Path:
        return Path(v).resolve()


# =============================================================================
# ── Config Loader functions
# =============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Helper to safely read a YAML configuration file."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as err:
        print(f"Warning: Failed to load YAML at {path}: {err}", file=sys.stderr)
        return {}


def load_raw_configs() -> Dict[str, Any]:
    """Merges all default configs/ YAML overlays."""
    merged: Dict[str, Any] = {}
    
    # 1. Load model configuration overlay
    model_path = _CONFIGS_DIR / "model_config.yaml"
    merged.update(_load_yaml(model_path))
    
    # 2. Load ControlNet configurations overlay
    cnet_path = _CONFIGS_DIR / "controlnet_config.yaml"
    cnet_data = _load_yaml(cnet_path)
    
    # Safely merge nested dictionaries
    for key, val in cnet_data.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key].update(val)
        else:
            merged[key] = val
            
    return merged


@lru_cache(maxsize=1)
def get_config() -> Week3Config:
    """
    Load settings singleton with LRU caching.
    Precedence: Env variables overrides > configs/*.yaml > Pydantic defaults.
    """
    raw_data = load_raw_configs()
    return Week3Config(**raw_data)
