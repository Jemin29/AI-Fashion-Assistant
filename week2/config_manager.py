"""
week2/config_manager.py
=======================
Typed configuration loader for Week 2 — Text-to-Image Fashion Generation.

Architecture
------------
- Pydantic v2 BaseSettings for env-variable binding
- YAML overlay for file-based config (generation, model, prompt, eval)
- Singleton pattern so configs are loaded once per process
- Full type-safety and validation at load time

Usage
-----
    from week2.config_manager import get_config

    cfg = get_config()
    print(cfg.generation.width)          # 1024
    print(cfg.model.runtime.device)      # "auto"
    print(cfg.prompts.style_presets)     # dict[str, StylePreset]
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_WEEK2_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _WEEK2_DIR.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"



# =============================================================================
# ── Sub-models: Generation Config
# =============================================================================

class GenerationPreset(BaseModel):
    """A named generation preset (draft / standard / high_quality / portrait)."""
    width: int                          = 1024
    height: int                         = 1024
    num_inference_steps: int            = 30
    guidance_scale: float               = 7.5
    use_refiner: bool                   = False
    refiner_strength: float             = 0.3


class GenerationConfig(BaseModel):
    """All SDXL sampling hyperparameters."""
    width: int                          = 1024
    height: int                         = 1024
    num_inference_steps: int            = 30
    guidance_scale: float               = 7.5
    guidance_rescale: float             = 0.0
    use_refiner: bool                   = False
    refiner_strength: float             = 0.3
    refiner_aesthetic_score: float      = 6.0
    refiner_negative_aesthetic_score: float = 2.5
    seed: int                           = -1
    num_images_per_prompt: int          = 1
    scheduler: str                      = "euler"
    scheduler_karras_sigmas: bool       = False
    output_format: str                  = "png"
    output_quality: int                 = 95
    save_metadata_sidecar: bool         = True
    clip_skip: int                      = 0
    eta: float                          = 0.0
    denoising_end: float                = 1.0
    presets: Dict[str, GenerationPreset] = Field(default_factory=dict)

    @field_validator("output_format")
    @classmethod
    def _valid_format(cls, v: str) -> str:
        if v not in ("png", "jpg", "jpeg", "webp"):
            raise ValueError(f"output_format must be one of png/jpg/webp, got {v!r}")
        return v


# =============================================================================
# ── Sub-models: Model Config
# =============================================================================

class ModelSpec(BaseModel):
    """Spec for a single model (base / refiner / vae)."""
    repo_id: str                        = ""
    variant: Optional[str]              = "fp16"
    use_safetensors: bool               = True
    local_path: Optional[str]           = None


class RuntimeConfig(BaseModel):
    """Compute / memory-optimisation settings."""
    device: str                         = "auto"
    torch_dtype: str                    = "float16"
    enable_xformers: bool               = True
    enable_attention_slicing: bool      = False
    enable_vae_slicing: bool            = False
    enable_vae_tiling: bool             = False
    enable_cpu_offload: bool            = False
    enable_sequential_offload: bool     = False
    keep_models_in_memory: bool         = True
    max_memory_gb: Optional[float]      = None

    @field_validator("torch_dtype")
    @classmethod
    def _valid_dtype(cls, v: str) -> str:
        if v not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"torch_dtype must be float16/bfloat16/float32, got {v!r}")
        return v


class CacheConfig(BaseModel):
    huggingface_home: str               = ".cache/huggingface"
    model_cache: str                    = ".cache/models"
    clip_cache: str                     = ".cache/clip"


class ModelConfig(BaseModel):
    base: ModelSpec                     = Field(default_factory=ModelSpec)
    refiner: ModelSpec                  = Field(default_factory=ModelSpec)
    vae: ModelSpec                      = Field(default_factory=ModelSpec)
    runtime: RuntimeConfig              = Field(default_factory=RuntimeConfig)
    cache: CacheConfig                  = Field(default_factory=CacheConfig)


# =============================================================================
# ── Sub-models: Prompt Config
# =============================================================================

class StylePreset(BaseModel):
    positive: List[str]                 = Field(default_factory=list)
    negative: List[str]                 = Field(default_factory=list)


class PromptStructureConfig(BaseModel):
    separator: str                      = ", "
    max_tokens: int                     = 77
    truncation: str                     = "ellipsis"
    deduplicate_tags: bool              = True


class PromptConfig(BaseModel):
    structure: PromptStructureConfig    = Field(default_factory=PromptStructureConfig)
    quality_boosters: List[str]         = Field(default_factory=list)
    fashion_technical: List[str]        = Field(default_factory=list)
    negative: Dict[str, List[str]]      = Field(default_factory=dict)
    style_presets: Dict[str, StylePreset] = Field(default_factory=dict)
    gender_modifiers: Dict[str, List[str]] = Field(default_factory=dict)
    season_modifiers: Dict[str, List[str]] = Field(default_factory=dict)
    category_templates: Dict[str, str]  = Field(default_factory=dict)

    @property
    def global_negative(self) -> List[str]:
        return (
            self.negative.get("global", []) +
            self.negative.get("fashion_specific", [])
        )


# =============================================================================
# ── Sub-models: Evaluation Config
# =============================================================================

class ClipEvalConfig(BaseModel):
    enabled: bool                       = True
    model_name: str                     = "ViT-L-14"
    pretrained: str                     = "openai"
    batch_size: int                     = 8
    min_similarity_threshold: float     = 0.20
    warn_threshold: float               = 0.25


class AestheticEvalConfig(BaseModel):
    enabled: bool                       = False
    model_name: str                     = ""
    min_acceptable_score: float         = 5.0
    warn_score: float                   = 6.0


class QualityEvalConfig(BaseModel):
    enabled: bool                       = True
    compute_ssim: bool                  = False
    compute_psnr: bool                  = False
    check_resolution: bool              = True
    min_width: int                      = 512
    min_height: int                     = 512
    check_artifacts: bool               = True
    check_black_image: bool             = True
    check_white_image: bool             = True
    black_threshold: int                = 5
    white_threshold: int                = 250


class FIDConfig(BaseModel):
    enabled: bool                       = False
    reference_dir: Optional[str]        = None
    batch_size: int                     = 32
    num_workers: int                    = 4


class EvalReportConfig(BaseModel):
    save_json: bool                     = True
    save_html: bool                     = False
    output_dir: str                     = "week2/outputs/evaluation_reports"
    filename_pattern: str               = "eval_{timestamp}.json"
    include_per_image_scores: bool      = True
    include_summary_stats: bool         = True


class AutoCurationConfig(BaseModel):
    enabled: bool                       = False
    reject_black_images: bool           = True
    reject_white_images: bool           = True
    reject_below_clip_threshold: bool   = False
    reject_below_aesthetic_score: bool  = False


class EvaluationConfig(BaseModel):
    clip: ClipEvalConfig                = Field(default_factory=ClipEvalConfig)
    aesthetic: AestheticEvalConfig      = Field(default_factory=AestheticEvalConfig)
    quality: QualityEvalConfig          = Field(default_factory=QualityEvalConfig)
    fid: FIDConfig                      = Field(default_factory=FIDConfig)
    report: EvalReportConfig            = Field(default_factory=EvalReportConfig)
    auto_curation: AutoCurationConfig   = Field(default_factory=AutoCurationConfig)


# =============================================================================
# ── Root Config (aggregates all sub-configs)
# =============================================================================

class Week2Config(BaseModel):
    """Root configuration object for Week 2."""
    generation: GenerationConfig        = Field(default_factory=GenerationConfig)
    model: ModelConfig                  = Field(default_factory=ModelConfig)
    prompts: PromptConfig               = Field(default_factory=PromptConfig)
    evaluation: EvaluationConfig        = Field(default_factory=EvaluationConfig)

    # Convenience paths (derived from env / defaults)
    output_dir: Path                    = _WEEK2_DIR / "outputs" / "images"
    metadata_dir: Path                  = _WEEK2_DIR / "outputs" / "metadata"
    log_dir: Path                       = _WEEK2_DIR / "logs"

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# ── Environment Settings (Pydantic-Settings)
# =============================================================================

class Week2EnvSettings(BaseSettings):
    """Reads Week 2 settings from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=[
            str(_WEEK2_DIR / ".env"),
            str(_PROJECT_ROOT / ".env"),
        ],
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Hugging Face
    hf_token:         Optional[str] = None
    hf_home:          str           = ".cache/huggingface"

    # Models
    sdxl_base_model:  str           = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_model: str         = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_vae_model:   str           = "madebyollin/sdxl-vae-fp16-fix"

    # Runtime
    device:           str           = "auto"
    torch_dtype:      str           = "float16"
    enable_xformers:  bool          = True
    enable_cpu_offload: bool        = False
    enable_sequential_offload: bool = False

    # Defaults
    default_width:    int           = 1024
    default_height:   int           = 1024
    default_steps:    int           = 30
    default_guidance_scale: float   = 7.5
    default_seed:     int           = -1
    default_num_images: int         = 1

    # Paths
    output_dir:       str           = "week2/outputs/images"
    metadata_dir:     str           = "week2/outputs/metadata"
    log_level:        str           = "INFO"
    log_dir:          str           = "week2/logs"
    log_rotation:     str           = "10 MB"
    log_retention:    str           = "7 days"
    log_colorize:     bool          = True

    # Evaluation
    clip_model_name:  str           = "ViT-L-14"
    clip_pretrained:  str           = "openai"
    aesthetic_threshold: float      = 5.0
    eval_on_generate: bool          = False

    # Dev
    debug_mode:       bool          = False
    profile_inference: bool         = False


# =============================================================================
# ── YAML Loader
# =============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning an empty dict if it doesn't exist."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge `override` into `base`."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# =============================================================================
# ── Config Builder
# =============================================================================

def _build_config(env: Week2EnvSettings) -> Week2Config:
    """
    Build the root Week2Config by:
    1. Loading each YAML file
    2. Applying environment overrides
    3. Validating via Pydantic
    """
    # ── Load YAML files ────────────────────────────────────────────────────
    gen_yaml   = _load_yaml(_CONFIGS_DIR / "generation_config.yaml")
    model_yaml = _load_yaml(_CONFIGS_DIR / "model_config.yaml")
    prompt_yaml= _load_yaml(_CONFIGS_DIR / "prompt_config.yaml")
    eval_yaml  = _load_yaml(_CONFIGS_DIR / "evaluation_config.yaml")

    # ── Apply env overrides to generation ─────────────────────────────────
    gen_section = gen_yaml.get("generation", {})
    gen_section.setdefault("width",              env.default_width)
    gen_section.setdefault("height",             env.default_height)
    gen_section.setdefault("num_inference_steps",env.default_steps)
    gen_section.setdefault("guidance_scale",     env.default_guidance_scale)
    gen_section.setdefault("seed",               env.default_seed)
    gen_section.setdefault("num_images_per_prompt", env.default_num_images)
    gen_section["presets"] = {
        k: GenerationPreset(**v)
        for k, v in gen_yaml.get("presets", {}).items()
    }

    # ── Apply env overrides to model/runtime ──────────────────────────────
    model_section = model_yaml.get("models", {})
    runtime_section = model_yaml.get("runtime", {})
    runtime_section["device"]     = env.device
    runtime_section["torch_dtype"]= env.torch_dtype
    runtime_section["enable_xformers"] = env.enable_xformers
    runtime_section["enable_cpu_offload"] = env.enable_cpu_offload
    runtime_section["enable_sequential_offload"] = env.enable_sequential_offload

    # Update model repo IDs from env
    if "base" not in model_section:
        model_section["base"] = {}
    model_section["base"]["repo_id"] = env.sdxl_base_model
    if "refiner" not in model_section:
        model_section["refiner"] = {}
    model_section["refiner"]["repo_id"] = env.sdxl_refiner_model
    if "vae" not in model_section:
        model_section["vae"] = {}
    model_section["vae"]["repo_id"] = env.sdxl_vae_model

    # ── Build Prompt Config ────────────────────────────────────────────────
    style_presets_raw = prompt_yaml.get("style_presets", {})
    style_presets = {
        k: StylePreset(**v) for k, v in style_presets_raw.items()
    }
    prompt_dict = dict(prompt_yaml)
    prompt_dict["style_presets"] = style_presets

    # ── Build Evaluation Config ────────────────────────────────────────────
    eval_section = eval_yaml.get("evaluation", {})
    if "clip" in eval_section:
        eval_section["clip"]["model_name"] = env.clip_model_name
        eval_section["clip"]["pretrained"] = env.clip_pretrained
    curation = eval_yaml.get("auto_curation", {})

    # ── Assemble root config ───────────────────────────────────────────────
    return Week2Config(
        generation  = GenerationConfig(**gen_section),
        model       = ModelConfig(
            base    = ModelSpec(**model_section.get("base", {})),
            refiner = ModelSpec(**model_section.get("refiner", {})),
            vae     = ModelSpec(**model_section.get("vae", {})),
            runtime = RuntimeConfig(**runtime_section),
            cache   = CacheConfig(**model_yaml.get("cache", {})),
        ),
        prompts     = PromptConfig(**prompt_dict),
        evaluation  = EvaluationConfig(
            clip         = ClipEvalConfig(**eval_section.get("clip", {})),
            aesthetic    = AestheticEvalConfig(**eval_section.get("aesthetic", {})),
            quality      = QualityEvalConfig(**eval_section.get("quality", {})),
            fid          = FIDConfig(**eval_section.get("fid", {})),
            report       = EvalReportConfig(**eval_section.get("report", {})),
            auto_curation= AutoCurationConfig(**curation),
        ),
        output_dir   = Path(env.output_dir),
        metadata_dir = Path(env.metadata_dir),
        log_dir      = Path(env.log_dir),
    )


# =============================================================================
# ── Public API
# =============================================================================

@lru_cache(maxsize=1)
def get_config() -> Week2Config:
    """
    Return the singleton Week2Config instance.

    Config is loaded once on first call; subsequent calls return the cached
    instance. Call `reload_config()` to force a reload.

    Returns
    -------
    Week2Config
        Fully validated, type-safe configuration object.
    """
    env = Week2EnvSettings()

    # Set HF_HOME so diffusers/transformers pick it up
    if env.hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", env.hf_token)
    os.environ.setdefault("HF_HOME", str(Path(env.hf_home).resolve()))

    return _build_config(env)


def reload_config() -> Week2Config:
    """Force reload of config (clears LRU cache)."""
    get_config.cache_clear()
    return get_config()


def get_env_settings() -> Week2EnvSettings:
    """Return the raw environment settings object."""
    return Week2EnvSettings()
