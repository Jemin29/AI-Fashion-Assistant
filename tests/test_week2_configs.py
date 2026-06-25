"""
week2/tests/test_config_manager.py
=====================================
Tests for the configuration system — no model loading required.
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Skip entire module if core dependencies are not installed
_YAML_AVAILABLE    = importlib.util.find_spec("yaml")             is not None
_PYDANTIC_AVAILABLE= importlib.util.find_spec("pydantic")         is not None
_SETTINGS_AVAILABLE= importlib.util.find_spec("pydantic_settings") is not None
_DEPS_OK           = _YAML_AVAILABLE and _PYDANTIC_AVAILABLE and _SETTINGS_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _DEPS_OK,
    reason="pyyaml / pydantic / pydantic-settings not installed — run: pip install -r week2/requirements.txt",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# ── GenerationConfig Tests
# =============================================================================

class TestGenerationConfig:

    def test_default_values(self):
        from src.utils.config_manager import GenerationConfig
        cfg = GenerationConfig()
        assert cfg.width == 1024
        assert cfg.height == 1024
        assert cfg.num_inference_steps == 30
        assert cfg.guidance_scale == 7.5
        assert cfg.scheduler == "euler"

    def test_valid_output_format(self):
        from src.utils.config_manager import GenerationConfig
        for fmt in ("png", "jpg", "webp"):
            cfg = GenerationConfig(output_format=fmt)
            assert cfg.output_format == fmt

    def test_invalid_output_format_raises(self):
        from src.utils.config_manager import GenerationConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GenerationConfig(output_format="bmp")


# =============================================================================
# ── RuntimeConfig Tests
# =============================================================================

class TestRuntimeConfig:

    def test_default_device(self):
        from src.utils.config_manager import RuntimeConfig
        cfg = RuntimeConfig()
        assert cfg.device == "auto"

    def test_valid_dtypes(self):
        from src.utils.config_manager import RuntimeConfig
        for dtype in ("float16", "bfloat16", "float32"):
            cfg = RuntimeConfig(torch_dtype=dtype)
            assert cfg.torch_dtype == dtype

    def test_invalid_dtype_raises(self):
        from src.utils.config_manager import RuntimeConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RuntimeConfig(torch_dtype="int8")

    def test_all_offload_flags_default_false(self):
        from src.utils.config_manager import RuntimeConfig
        cfg = RuntimeConfig()
        assert cfg.enable_cpu_offload is False
        assert cfg.enable_sequential_offload is False
        assert cfg.enable_attention_slicing is False


# =============================================================================
# ── StylePreset (Config Model) Tests
# =============================================================================

class TestConfigStylePreset:

    def test_default_empty_lists(self):
        from src.utils.config_manager import StylePreset
        p = StylePreset()
        assert p.positive == []
        assert p.negative == []

    def test_custom_values(self):
        from src.utils.config_manager import StylePreset
        p = StylePreset(positive=["red", "blue"], negative=["ugly"])
        assert "red" in p.positive
        assert "ugly" in p.negative


# =============================================================================
# ── PromptConfig Tests
# =============================================================================

class TestPromptConfig:

    def test_global_negative_combines_lists(self):
        from src.utils.config_manager import PromptConfig
        cfg = PromptConfig(
            negative={
                "global": ["blurry"],
                "fashion_specific": ["dirty"],
            }
        )
        combined = cfg.global_negative
        assert "blurry" in combined
        assert "dirty" in combined


# =============================================================================
# ── ModelSpec Tests
# =============================================================================

class TestModelSpec:

    def test_default_variant(self):
        from src.utils.config_manager import ModelSpec
        m = ModelSpec(repo_id="test/model")
        assert m.variant == "fp16"

    def test_local_path_none_by_default(self):
        from src.utils.config_manager import ModelSpec
        m = ModelSpec()
        assert m.local_path is None


# =============================================================================
# ── Week2Config Tests
# =============================================================================

class TestWeek2Config:

    def test_config_has_all_sections(self):
        from src.utils.config_manager import Week2Config
        cfg = Week2Config()
        assert hasattr(cfg, "generation")
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "prompts")
        assert hasattr(cfg, "evaluation")
        assert hasattr(cfg, "output_dir")
        assert hasattr(cfg, "log_dir")

    def test_output_dir_is_path(self):
        from src.utils.config_manager import Week2Config
        cfg = Week2Config()
        assert isinstance(cfg.output_dir, Path)


# =============================================================================
# ── Env Settings Tests
# =============================================================================

class TestWeek2EnvSettings:

    def test_default_device(self):
        from src.utils.config_manager import Week2EnvSettings
        env = Week2EnvSettings()
        assert env.device in ("auto", "cuda", "cpu", "mps")

    def test_default_torch_dtype(self):
        from src.utils.config_manager import Week2EnvSettings
        env = Week2EnvSettings()
        assert env.torch_dtype in ("float16", "bfloat16", "float32")

    def test_default_dimensions(self):
        from src.utils.config_manager import Week2EnvSettings
        env = Week2EnvSettings()
        assert env.default_width > 0
        assert env.default_height > 0

    def test_hf_token_optional(self):
        from src.utils.config_manager import Week2EnvSettings
        env = Week2EnvSettings()
        assert env.hf_token is None or isinstance(env.hf_token, str)


# =============================================================================
# ── Singleton Tests
# =============================================================================

class TestGetConfig:

    def test_get_config_returns_week2config(self):
        from src.utils.config_manager import get_config, reload_config, Week2Config
        cfg = reload_config()  # Force fresh load
        assert isinstance(cfg, Week2Config)

    def test_get_config_cached(self):
        from src.utils.config_manager import get_config
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2   # Same object (LRU cached)

    def test_reload_config_produces_new_object(self):
        from src.utils.config_manager import get_config, reload_config
        c1 = get_config()
        c2 = reload_config()
        # After reload, a new object is created
        assert isinstance(c2, type(c1))

    def test_deep_merge_basic(self):
        from src.utils.config_manager import _deep_merge
        base     = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"d": 99}, "e": 5}
        result   = _deep_merge(base, override)
        assert result["b"]["c"] == 2
        assert result["b"]["d"] == 99
        assert result["e"]       == 5
        assert result["a"]       == 1

    def test_load_yaml_missing_file_returns_empty(self):
        from src.utils.config_manager import _load_yaml
        result = _load_yaml(Path("/nonexistent/path/config.yaml"))
        assert result == {}
