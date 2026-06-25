"""
configs/model_config.py
======================
Centralized configuration manager using Pydantic v2.
Supports hyperparameter settings, hardware overrides, and directory structures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default directories
DEFAULT_OUTPUT_DIR = Path("week2/outputs/generated")
DEFAULT_METADATA_DIR = Path("week2/outputs/metadata")
DEFAULT_BATCH_DIR = Path("week2/outputs/batch_runs")
DEFAULT_EXPERIMENTS_DIR = Path("week2/outputs/experiments")
DEFAULT_LOG_DIR = Path("logs")

# SDXL presets
SIZE_PRESETS: Dict[str, tuple[int, int]] = {
    "square_1024": (1024, 1024),
    "portrait_1024": (832, 1216),
    "landscape_1024": (1216, 832),
    "square_512": (512, 512),
}


class SDXLSettings(BaseModel):
    """SDXL pipeline settings and hyper-parameters."""
    model_id: str = Field(
        default="stabilityai/stable-diffusion-xl-base-1.0",
        description="HuggingFace repository ID for SDXL base model."
    )
    refiner_id: Optional[str] = Field(
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
        description="HuggingFace repository ID for SDXL refiner model."
    )
    use_refiner: bool = Field(
        default=False,
        description="Whether to run the SDXL refiner pass."
    )
    refiner_strength: float = Field(
        default=0.3,
        description="Denoising strength for the refiner pass (0.0 to 1.0)."
    )
    guidance_scale: float = Field(
        default=7.5,
        description="Classifier-Free Guidance (CFG) scale."
    )
    num_inference_steps: int = Field(
        default=30,
        description="Number of denoising/inference steps."
    )
    scheduler: str = Field(
        default="euler",
        description="Inference scheduler name (e.g. euler, dpm++, heun, etc.)."
    )
    scheduler_karras_sigmas: bool = Field(
        default=False,
        description="Whether to use Karras sigmas with the scheduler."
    )

    @field_validator("refiner_strength")
    @classmethod
    def _validate_refiner_strength(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"refiner_strength must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("guidance_scale")
    @classmethod
    def _validate_guidance_scale(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"guidance_scale must be positive, got {v}")
        return v

    @field_validator("num_inference_steps")
    @classmethod
    def _validate_inference_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {v}")
        return v


class DeviceSettings(BaseModel):
    """Execution hardware and memory optimization settings."""
    device: str = Field(
        default="auto",
        description="Target compute device (auto, cuda, cpu, mps)."
    )
    torch_dtype: str = Field(
        default="float16",
        description="Tensor precision format (float16, bfloat16, float32)."
    )
    low_vram_mode: bool = Field(
        default=False,
        description="Enable model CPU offload for low-VRAM memory optimizations."
    )
    sequential_offload: bool = Field(
        default=False,
        description="Enable sequential CPU offload for minimum VRAM usage."
    )
    enable_xformers: bool = Field(
        default=False,
        description="Enable xFormers memory-efficient attention."
    )
    attention_slicing: bool = Field(
        default=False,
        description="Enable VAE/attention slicing."
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        v_lower = v.lower().strip()
        valid = ("auto", "cuda", "cpu", "mps")
        if v_lower not in valid:
            raise ValueError(f"device must be one of {valid}, got {v}")
        return v_lower

    @field_validator("torch_dtype")
    @classmethod
    def _validate_dtype(cls, v: str) -> str:
        v_lower = v.lower().strip()
        valid = ("float16", "bfloat16", "float32")
        if v_lower not in valid:
            raise ValueError(f"torch_dtype must be one of {valid}, got {v}")
        return v_lower


class ImageResolution(BaseModel):
    """Output image dimensions and preset resolution configuration."""
    width: int = Field(
        default=1024,
        description="Output image width in pixels."
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels."
    )
    size_preset: Optional[str] = Field(
        default=None,
        description="Size preset name (e.g. square_1024, portrait_1024). Overrides width/height."
    )

    @model_validator(mode="after")
    def _resolve_presets(self) -> ImageResolution:
        if self.size_preset:
            if self.size_preset in SIZE_PRESETS:
                self.width, self.height = SIZE_PRESETS[self.size_preset]
            else:
                valid_presets = list(SIZE_PRESETS.keys())
                raise ValueError(f"Invalid size_preset {self.size_preset!r}. Must be one of {valid_presets}")
        
        # Check divisibility by 8
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError(f"Width ({self.width}) and Height ({self.height}) must be divisible by 8.")
        
        return self


class BatchSettings(BaseModel):
    """Batch size and workers count settings."""
    batch_size: int = Field(
        default=1,
        description="Number of images generated per prompt request."
    )
    max_workers: int = Field(
        default=2,
        description="Parallel workers count for batch generators."
    )

    @field_validator("batch_size", "max_workers")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class SeedSettings(BaseModel):
    """Seed configuration for reproducible image generation."""
    seed: int = Field(
        default=-1,
        description="RNG seed value (-1 for random seed)."
    )
    use_random_seed: bool = Field(
        default=True,
        description="Whether to generate random seeds for negative seed values."
    )


class OutputPaths(BaseModel):
    """Global file output directories settings."""
    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR,
        description="Root output folder for generated fashion images."
    )
    metadata_dir: Path = Field(
        default=DEFAULT_METADATA_DIR,
        description="Output folder for saving sidecar metadata JSONs."
    )
    batch_dir: Path = Field(
        default=DEFAULT_BATCH_DIR,
        description="Manifests and logs folder for batch runs."
    )
    experiments_dir: Path = Field(
        default=DEFAULT_EXPERIMENTS_DIR,
        description="SQLite database folder for tracking experiments."
    )
    log_dir: Path = Field(
        default=DEFAULT_LOG_DIR,
        description="Output logs directory."
    )

    def create_directories(self) -> None:
        """Utility to ensure all output paths exist on disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


class CentralizedConfig(BaseSettings):
    """Centralized System Configuration combining all settings groups."""
    sdxl: SDXLSettings = Field(default_factory=SDXLSettings)
    device: DeviceSettings = Field(default_factory=DeviceSettings)
    resolution: ImageResolution = Field(default_factory=ImageResolution)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    seed: SeedSettings = Field(default_factory=SeedSettings)
    paths: OutputPaths = Field(default_factory=OutputPaths)

    # Use SettingsConfigDict for loading env variables
    model_config = SettingsConfigDict(
        env_prefix="FASHION_CONFIG_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path]) -> CentralizedConfig:
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        return cls.model_validate(data)

    def save_to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save active configuration to a YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump(mode="json")
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
