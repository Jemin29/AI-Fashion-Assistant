"""
src/utils/config_manager.py
===========================
Unified configuration manager for Weeks 2, 3, 4, and 5.
Consolidates config models and loader utilities across generation, LoRA, and RAG pipelines.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Import Week 2 configs and helpers ─────────────────────────────────────────
from week2.config_manager import (
    Week2Config,
    Week2EnvSettings,
    GenerationPreset,
    GenerationConfig,
    ModelSpec,
    RuntimeConfig,
    CacheConfig,
    StylePreset,
    PromptStructureConfig,
    PromptConfig,
    ClipEvalConfig,
    AestheticEvalConfig,
    QualityEvalConfig,
    FIDConfig,
    EvalReportConfig,
    AutoCurationConfig,
    EvaluationConfig,
    _deep_merge,
    _load_yaml,
    get_config as get_week2_config,
    reload_config as reload_week2_config,
    get_env_settings as get_week2_env_settings,
)


# =============================================================================
# ── Week 4 Configuration Schemes (LoRA Fine-tuning)
# =============================================================================

class LoraConfig(BaseModel):
    """Configuration for LoRA adapter injection."""
    r: int = Field(default=8, ge=1, le=256)
    alpha: int = Field(default=16, ge=1)
    target_modules: List[str] = Field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    bias: str = Field(default="none")


class DatasetConfig(BaseModel):
    """Configuration for training dataset preprocessing."""
    resolution: int = Field(default=1024, ge=256, le=2048)
    center_crop: bool = True
    random_flip: bool = True


class TrainerConfig(BaseModel):
    """Configuration for LoRA model training loops."""
    mixed_precision: str = Field(default="fp16", pattern="^(fp16|bf16|no)$")
    learning_rate: float = Field(default=1e-4, ge=0.0)
    output_dir: str = Field(default="outputs/trainer")
    num_epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=1, ge=1)



class InferenceConfig(BaseModel):
    """Configuration for model evaluation and testing inference."""
    lora_scale: float = Field(default=0.8, ge=0.0, le=1.0)
    num_inference_steps: int = Field(default=30, ge=1)
    base_model_id: str = Field(default="stabilityai/stable-diffusion-xl-base-1.0", description="Base SDXL model identifier.")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale.")


class Week4Config(BaseModel):
    """Week 4 Configuration bundling LoRA specifications."""
    lora: LoraConfig = Field(default_factory=LoraConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output_root: str = Field(default="outputs")

    @model_validator(mode="after")
    def sync_paths(self) -> Week4Config:
        """Sync training outputs with output root if default."""
        if self.output_root != "outputs":
            self.trainer.output_dir = str(Path(self.output_root) / "trainer")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to flat dictionary."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Serialize configuration model to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration model to target YAML file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> Week4Config:
        """Instantiate configuration model from target YAML file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# =============================================================================
# ── Week 5 Configuration Schemes (Fashion Intelligence & RAG)
# =============================================================================

class EmbeddingConfig(BaseModel):
    """Configuration for text embedding models."""
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Pre-trained transformer model ID."
    )
    dimension: int = Field(
        default=384,
        ge=1,
        description="Dimensionality of generated dense vectors."
    )
    device: str = Field(
        default="cpu",
        pattern="^(cpu|cuda|mps)$",
        description="Target execution device."
    )
    cache_folder: str = Field(
        default="outputs/embeddings/cache",
        description="Folder path where embedding models are cached."
    )


class VectorDbConfig(BaseModel):
    """Configuration for vector database (FAISS) settings."""
    index_type: str = Field(
        default="FlatL2",
        pattern="^(FlatL2|InnerProduct)$",
        description="Type of metric index to compile."
    )
    storage_path: str = Field(
        default="outputs/vector_db/faiss_index",
        description="Path on disk to load/save index partitions."
    )
    auto_save: bool = Field(
        default=True,
        description="Persist vector index to disk after mutations."
    )


class RetrievalConfig(BaseModel):
    """Configuration for RAG search and retrieval routing."""
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of matches to return."
    )
    hybrid_search: bool = Field(
        default=True,
        description="Combine keyword searching with vector queries."
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Score weight coefficient for BM25/keyword components."
    )
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Score weight coefficient for dense vector search components."
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> RetrievalConfig:
        """Ensure hybrid search weights sum to exactly 1.0."""
        total = self.keyword_weight + self.vector_weight
        if abs(total - 1.0) > 1e-6:
            self.keyword_weight = round(self.keyword_weight / total, 4)
            self.vector_weight = round(self.vector_weight / total, 4)
        return self


class RecommendationConfig(BaseModel):
    """Configuration for fashion design recommendation algorithms."""
    max_recommendations: int = Field(
        default=10,
        ge=1,
        description="Max recommendations count to return."
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold limit to permit matches."
    )
    diversity_bias: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Scale penalty score mapping similarity vs product catalog diversity."
    )


class TrendConfig(BaseModel):
    """Configuration for fashion trend analysis and forecasting."""
    time_window_days: int = Field(
        default=30,
        ge=1,
        description="Lookback window in days to ingest mentions."
    )
    min_mention_count: int = Field(
        default=3,
        ge=1,
        description="Minimum frequency criteria to classify a design element as a trend."
    )
    growth_threshold: float = Field(
        default=0.1,
        description="Mentions velocity criteria to classify element as an active trend."
    )


class Week5Config(BaseModel):
    """Centralized Week 5 Configuration bundling sub-configs."""
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDbConfig = Field(default_factory=VectorDbConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    recommendations: RecommendationConfig = Field(default_factory=RecommendationConfig)
    trends: TrendConfig = Field(default_factory=TrendConfig)
    log_dir: str = Field(default="logs", description="Base logging outputs path.")
    output_root: str = Field(default="outputs", description="Root output directory.")

    @model_validator(mode="after")
    def sync_output_paths(self) -> Week5Config:
        """Sync output subdirs with output root if default."""
        root = Path(self.output_root)
        
        cache_path = Path(self.embeddings.cache_folder)
        if not cache_path.is_absolute() and not self.embeddings.cache_folder.startswith(self.output_root):
            self.embeddings.cache_folder = str((root / "embeddings" / "cache").as_posix())

        db_path = Path(self.vector_db.storage_path)
        if not db_path.is_absolute() and not self.vector_db.storage_path.startswith(self.output_root):
            self.vector_db.storage_path = str((root / "vector_db" / "faiss_index").as_posix())

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert configurations to flat dict."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Serialize configuration model to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration model to target YAML file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> Week5Config:
        """Instantiate config model from a YAML configuration file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# =============================================================================
# ── Convenience Routing API
# =============================================================================

def get_default_config() -> Union[Week5Config, Week4Config]:
    """
    Return default configuration instance dynamically based on calling module.
    If called from LoRA / training components, returns Week4Config.
    If called from RAG / intelligence components, returns Week5Config.
    """
    for frame_info in inspect.stack():
        module_name = frame_info.frame.f_globals.get('__name__', '')
        if any(k in module_name for k in ("lora", "week4", "kohya")):
            return Week4Config()
    return Week5Config()


def get_config(*args, **kwargs) -> Any:
    """
    Return the singleton config instance dynamically.
    If called from Week 2, 3, 4, or 5 contexts, routes appropriately.
    """
    for frame_info in inspect.stack():
        module_name = frame_info.frame.f_globals.get('__name__', '')
        if any(k in module_name for k in ("controlnet", "week3", "pose", "depth", "sketch")):
            try:
                from week3.config_manager import get_config as get_week3_config
                return get_week3_config()
            except ImportError:
                pass
        if any(k in module_name for k in ("lora", "week4", "kohya")):
            return get_default_config()
        if any(k in module_name for k in ("week5", "rag", "recommendations", "trends", "evaluation", "search", "assistant")):
            return get_default_config()
        if "week2" in module_name:
            return get_week2_config(*args, **kwargs)
    return get_week2_config(*args, **kwargs)


def reload_config(*args, **kwargs) -> Any:
    """Force config reload."""
    for frame_info in inspect.stack():
        module_name = frame_info.frame.f_globals.get('__name__', '')
        if any(k in module_name for k in ("controlnet", "week3", "pose", "depth", "sketch")):
            try:
                from week3.config_manager import get_config as get_week3_config
                get_week3_config.cache_clear()
                return get_week3_config()
            except ImportError:
                pass
        if "week2" in module_name:
            return reload_week2_config(*args, **kwargs)
    return reload_week2_config(*args, **kwargs)

