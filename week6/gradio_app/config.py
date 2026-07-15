"""
Week 6 — AI Fashion Creative Studio
Configuration System

Provides a unified, Pydantic-validated configuration loader that reads from:
  1. week6/configs/app_config.yaml
  2. week6/configs/services_config.yaml
  3. Environment variables (override YAML)
  4. week6/.env file (loaded via python-dotenv)

Usage:
    from week6.gradio_app.config import get_config, AppConfig

    cfg = get_config()
    print(cfg.server.port)          # 7860
    print(cfg.mock.global_mock)     # True
    print(cfg.services.rag.embedding_model)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_WEEK6_ROOT = Path(__file__).resolve().parent.parent
_APP_CONFIG = _WEEK6_ROOT / "configs" / "app_config.yaml"
_SVC_CONFIG = _WEEK6_ROOT / "configs" / "services_config.yaml"
_ENV_FILE = _WEEK6_ROOT / ".env"


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic sub-models for each config section
# ══════════════════════════════════════════════════════════════════════════════

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    debug: bool = False
    max_threads: int = 40
    show_error: bool = True
    quiet: bool = False


class MockConfig(BaseModel):
    global_mock: bool = True
    generation: bool = True
    controlnet: bool = True
    lora: bool = True
    rag: bool = False
    recommendations: bool = False
    trends: bool = False


class FeatureConfig(BaseModel):
    home_page: bool = True
    style_studio: bool = True
    controlnet_studio: bool = True
    brand_studio: bool = True
    fashion_qa: bool = True
    trend_explorer: bool = True
    recommend_hub: bool = True
    eval_dashboard: bool = True
    wardrobe_gen: bool = True


class PathsConfig(BaseModel):
    project_root: str = ".."
    outputs_dir: str = "week6/outputs"
    logs_dir: str = "week6/logs"
    assets_dir: str = str(_WEEK6_ROOT / "assets")
    generated_images: str = "week6/outputs/generated"
    sketch_outputs: str = "week6/outputs/sketches"
    reports: str = "week6/outputs/reports"

    @field_validator("assets_dir", mode="before")
    @classmethod
    def resolve_assets_dir(cls, v: Any) -> str:
        if isinstance(v, str):
            path = Path(v)
            if not path.is_absolute():
                parts = path.parts
                if parts and parts[0] == "week6":
                    return str((_WEEK6_ROOT.parent / path).resolve())
                else:
                    return str((_WEEK6_ROOT / path).resolve())
        return v


class UIConfig(BaseModel):
    theme: str = "dark"
    primary_color: str = "hsl(245, 70%, 60%)"
    accent_color: str = "hsl(15, 90%, 65%)"
    font_family: str = "Outfit, Inter, sans-serif"
    max_gallery_images: int = 8
    chat_history_limit: int = 20
    enable_animations: bool = True


class SessionConfig(BaseModel):
    max_session_history: int = 50
    auto_save_outputs: bool = True
    output_format: str = "png"
    output_quality: int = 95


class AnalyticsConfig(BaseModel):
    log_interactions: bool = True
    log_generation_params: bool = True
    log_rag_queries: bool = True
    log_latency: bool = True


class AppConfig(BaseModel):
    """Root application configuration model."""
    name: str = "AI Fashion Creative Studio"
    version: str = "1.0.0"
    description: str = "Week 6 Gradio Creative Studio"
    week: int = 6
    server: ServerConfig = Field(default_factory=ServerConfig)
    mock: MockConfig = Field(default_factory=MockConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)


# ── Services sub-models ───────────────────────────────────────────────────────

class GenerationDefaultParams(BaseModel):
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    negative_prompt: str = "low quality, blurry, watermark, text, distorted"


class GenerationServiceConfig(BaseModel):
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    torch_dtype: str = "float16"
    device: str = "auto"
    enable_xformers: bool = True
    default_params: GenerationDefaultParams = Field(default_factory=GenerationDefaultParams)


class LoRABrandConfig(BaseModel):
    adapter_path: str
    trigger_word: str
    description: str


class LoRAServiceConfig(BaseModel):
    adapters_dir: str = "week4/lora_outputs"
    brands: Dict[str, LoRABrandConfig] = Field(default_factory=dict)
    default_lora_scale: float = 0.85
    max_mix_adapters: int = 3


class ChromaDBConfig(BaseModel):
    persist_directory: str = "week5/chromadb_store"
    in_memory: bool = True
    collections: List[str] = Field(
        default_factory=lambda: ["fashion_knowledge", "trend_items", "style_profiles", "brand_profiles"]
    )


class RAGServiceConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    force_mock_embeddings: bool = True
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)


class RecommendationServiceConfig(BaseModel):
    n_results: int = 5
    similarity_weight: float = 0.7
    rule_weight: float = 0.3


class TrendServiceConfig(BaseModel):
    forecast_horizon: int = 2
    velocity_window: int = 4
    confidence_threshold: float = 0.6


class ServicesConfig(BaseModel):
    """Root services configuration model."""
    generation: GenerationServiceConfig = Field(default_factory=GenerationServiceConfig)
    lora: LoRAServiceConfig = Field(default_factory=LoRAServiceConfig)
    rag: RAGServiceConfig = Field(default_factory=RAGServiceConfig)
    recommendations: RecommendationServiceConfig = Field(default_factory=RecommendationServiceConfig)
    trends: TrendServiceConfig = Field(default_factory=TrendServiceConfig)


# ══════════════════════════════════════════════════════════════════════════════
# Loader functions
# ══════════════════════════════════════════════════════════════════════════════

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning empty dict if missing."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _apply_env_overrides(cfg_dict: dict) -> dict:
    """Apply environment variable overrides to config dict."""
    env_map = {
        "FASHION_STUDIO_HOST": ("server", "host"),
        "FASHION_STUDIO_PORT": ("server", "port"),
        "FASHION_STUDIO_SHARE": ("server", "share"),
        "FASHION_STUDIO_MOCK_MODE": ("mock", "global_mock"),
    }
    for env_key, (section, key) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            section_dict = cfg_dict.setdefault(section, {})
            # Type coercion for booleans and ints
            if val.lower() in ("true", "false"):
                section_dict[key] = val.lower() == "true"
            elif val.isdigit():
                section_dict[key] = int(val)
            else:
                section_dict[key] = val
    return cfg_dict


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Load and return the validated AppConfig (cached after first call).

    Priority: env vars > YAML file > Pydantic defaults.

    Returns:
        AppConfig: Fully validated application configuration.
    """
    if _DOTENV_AVAILABLE and _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)

    raw = _load_yaml(_APP_CONFIG)
    # Flatten 'app' top-level key if present
    if "app" in raw:
        app_meta = raw.pop("app")
        raw.update(app_meta)
    raw = _apply_env_overrides(raw)
    cfg = AppConfig.model_validate(raw)
    if cfg.mock.global_mock:
        from loguru import logger
        banner = (
            "\n" + "═" * 60 + "\n"
            "⚠️  RUNNING IN MOCK MODE (GLOBAL_MOCK=True)  ⚠️\n"
            "This run is in Mock Mode. No GPU or real weights will be loaded.\n"
            "To run in Real GPU Mode, use --no-mock or configure .env file.\n"
            + "═" * 60
        )
        logger.warning(banner)
    return cfg


@lru_cache(maxsize=1)
def get_services_config() -> ServicesConfig:
    """
    Load and return the validated ServicesConfig (cached after first call).

    Returns:
        ServicesConfig: Fully validated services configuration.
    """
    raw = _load_yaml(_SVC_CONFIG)
    return ServicesConfig.model_validate(raw)


def reload_config() -> AppConfig:
    """
    Force-reload the AppConfig by clearing the LRU cache.

    Useful during development when YAML files change.

    Returns:
        AppConfig: Freshly loaded configuration.
    """
    get_config.cache_clear()
    get_services_config.cache_clear()
    return get_config()
