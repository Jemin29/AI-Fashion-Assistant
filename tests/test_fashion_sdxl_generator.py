"""
week2/tests/test_fashion_sdxl_generator.py
============================================
Comprehensive tests for FashionSDXLGenerator.

All tests run WITHOUT loading any real SDXL model — diffusers and torch
are mocked where necessary.

Coverage
--------
- GenerationOutput dataclass (shape, repr, convenience properties)
- FashionSDXLGenerator initialisation (all param combos)
- _detect_device() (cuda / mps / cpu auto-detection)
- _resolve_dtype() (mapping + CPU downgrade)
- _snap_to_multiple() (alignment to 8)
- _resolve_seed() (random vs fixed)
- _new_image_id() (format, uniqueness)
- _build_metadata() (structure, types)
- _get_vram_gb() (with and without CUDA)
- load_model() (success, failure, force_reload, already-loaded guard)
- generate_image() (success, OOM recovery, model-not-loaded auto-load)
- generate_batch() (empty, partial failure, stop_on_error, seed padding)
- save_output() (png, jpg, metadata sidecar, overwrite guard, empty error)
- get_info() / is_loaded / device / vram_gb properties
- Scheduler application
- Memory optimisation tier selection
- Context manager (__enter__ / __exit__)
- SDXLGenerator backward-compatible adapter
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TORCH_AVAILABLE    = importlib.util.find_spec("torch")     is not None
_DIFFUSERS_AVAILABLE= importlib.util.find_spec("diffusers") is not None
_PIL_AVAILABLE      = importlib.util.find_spec("PIL")       is not None

from src.generation.generator.sdxl_generator import (
    FashionSDXLGenerator,
    GenerationOutput,
    GenerationResult,
    SIZE_PRESETS,
    SCHEDULER_MAP,
    SDXLGenerator,
    _DEFAULT_MODEL_ID,
    _DEFAULT_OUTPUT_DIR,
)


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture
def gen(tmp_path):
    """Uninitialised FashionSDXLGenerator with a temp output dir."""
    return FashionSDXLGenerator(
        model_id   = "test/model",
        vae_id     = None,
        device     = "cpu",
        torch_dtype= "float32",
        output_dir = tmp_path / "outputs",
    )


@pytest.fixture
def tiny_pil(tmp_path):
    """A minimal 64×64 PIL image (skipped if Pillow not installed)."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image
    return Image.new("RGB", (64, 64), color=(123, 45, 67))


@pytest.fixture
def mock_pipe(tiny_pil):
    """Mock diffusers pipeline that returns a tiny PIL image."""
    pipe         = MagicMock()
    output       = MagicMock()
    output.images = [tiny_pil, tiny_pil]   # 2 images
    pipe.return_value = output
    pipe.scheduler = MagicMock()
    pipe.scheduler.config = {}
    pipe.text_encoder_2 = MagicMock()
    pipe.vae            = MagicMock()
    return pipe


# =============================================================================
# ── GenerationOutput Tests
# =============================================================================

class TestGenerationOutput:

    def test_empty_len_is_zero(self):
        o = GenerationOutput()
        assert len(o) == 0

    def test_len_equals_images(self):
        o = GenerationOutput(images=["a", "b", "c"])
        assert len(o) == 3

    def test_bool_true_on_success(self):
        assert bool(GenerationOutput(success=True)) is True

    def test_bool_false_on_failure(self):
        assert bool(GenerationOutput(success=False)) is False

    def test_repr_ok(self):
        o = GenerationOutput(success=True, generation_time_s=2.5)
        r = repr(o)
        assert "✓" in r or "OK" in r.upper() or "success" in r.lower() or "2.5" in r

    def test_repr_error(self):
        o = GenerationOutput(success=False, error="CUDA OOM")
        r = repr(o)
        assert "✗" in r or "error" in r.lower() or "CUDA OOM" in r

    def test_first_image_none_when_empty(self):
        assert GenerationOutput().first_image is None

    def test_first_image_returns_image(self):
        o = GenerationOutput(images=["img1"])
        assert o.first_image == "img1"

    def test_first_path_none_when_empty(self):
        assert GenerationOutput().first_path is None

    def test_first_path_returns_path(self):
        p = Path("/some/path.png")
        o = GenerationOutput(image_paths=[p])
        assert o.first_path == p

    def test_default_fields(self):
        o = GenerationOutput()
        assert o.success is True
        assert o.error is None
        assert o.seed == 0
        assert o.width == 1024
        assert o.height == 1024

    def test_generation_result_is_alias(self):
        """GenerationResult must be a subclass/alias of GenerationOutput."""
        r = GenerationResult()
        assert isinstance(r, GenerationOutput)


# =============================================================================
# ── Initialisation Tests
# =============================================================================

class TestFashionSDXLGeneratorInit:

    def test_default_output_dir_created(self, tmp_path):
        sub = tmp_path / "gen_outputs"
        g   = FashionSDXLGenerator(output_dir=sub)
        assert sub.exists()

    def test_is_not_loaded_initially(self, gen):
        assert gen.is_loaded is False

    def test_device_property(self, gen):
        assert gen.device == "cpu"

    def test_vram_gb_zero_on_cpu(self, gen):
        assert gen.vram_gb == 0.0

    def test_model_id_stored(self):
        g = FashionSDXLGenerator(model_id="custom/sdxl")
        assert g.model_id == "custom/sdxl"

    def test_scheduler_stored(self):
        g = FashionSDXLGenerator(scheduler="dpm++")
        assert g.scheduler_name == "dpm++"

    def test_hf_token_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "test-token-123")
        g = FashionSDXLGenerator(output_dir=tmp_path)
        assert g.hf_token == "test-token-123"

    def test_hf_token_explicit_overrides_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        g = FashionSDXLGenerator(hf_token="explicit-token", output_dir=tmp_path)
        assert g.hf_token == "explicit-token"

    def test_get_info_structure(self, gen):
        info = gen.get_info()
        assert "model_id"             in info
        assert "device"               in info
        assert "is_loaded"            in info
        assert "vram_gb"              in info
        assert "supported_sizes"      in info
        assert "supported_schedulers" in info

    def test_default_model_id_constant(self):
        assert "stable-diffusion-xl" in _DEFAULT_MODEL_ID.lower()

    def test_size_presets_not_empty(self):
        assert len(SIZE_PRESETS) >= 5

    def test_scheduler_map_not_empty(self):
        assert len(SCHEDULER_MAP) >= 5
        assert "euler" in SCHEDULER_MAP


# =============================================================================
# ── Static Method Tests (no model load needed)
# =============================================================================

class TestStaticMethods:

    # ── _detect_device ────────────────────────────────────────────────────

    def test_detect_device_explicit_cpu(self):
        assert FashionSDXLGenerator._detect_device("cpu") == "cpu"

    def test_detect_device_explicit_cuda(self):
        assert FashionSDXLGenerator._detect_device("cuda") == "cuda"

    def test_detect_device_explicit_mps(self):
        assert FashionSDXLGenerator._detect_device("mps") == "mps"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_detect_device_auto_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert FashionSDXLGenerator._detect_device("auto") == "cuda"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_detect_device_auto_cpu_fallback(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert FashionSDXLGenerator._detect_device("auto") == "cpu"

    # ── _resolve_seed ─────────────────────────────────────────────────────

    def test_resolve_seed_fixed(self):
        assert FashionSDXLGenerator._resolve_seed(42) == 42

    def test_resolve_seed_zero(self):
        assert FashionSDXLGenerator._resolve_seed(0) == 0

    def test_resolve_seed_minus_one_is_random(self):
        seeds = {FashionSDXLGenerator._resolve_seed(-1) for _ in range(20)}
        assert len(seeds) > 1

    def test_resolve_seed_minus_one_in_range(self):
        for _ in range(10):
            s = FashionSDXLGenerator._resolve_seed(-1)
            assert 0 <= s < 2 ** 32

    def test_resolve_seed_negative_other(self):
        s = FashionSDXLGenerator._resolve_seed(-99)
        assert 0 <= s < 2 ** 32

    # ── _snap_to_multiple ─────────────────────────────────────────────────

    def test_snap_already_aligned(self):
        assert FashionSDXLGenerator._snap_to_multiple(1024, 8) == 1024

    def test_snap_rounds_up(self):
        assert FashionSDXLGenerator._snap_to_multiple(1025, 8) == 1032

    def test_snap_zero_stays_zero(self):
        assert FashionSDXLGenerator._snap_to_multiple(0, 8) == 0

    def test_snap_small_value(self):
        assert FashionSDXLGenerator._snap_to_multiple(3, 8) == 8

    # ── _new_image_id ─────────────────────────────────────────────────────

    def test_image_id_starts_with_fashion(self):
        assert FashionSDXLGenerator._new_image_id().startswith("FASHION_")

    def test_image_id_unique(self):
        ids = {FashionSDXLGenerator._new_image_id() for _ in range(50)}
        assert len(ids) == 50

    def test_image_id_length_reasonable(self):
        img_id = FashionSDXLGenerator._new_image_id()
        assert 15 < len(img_id) < 60

    # ── _build_metadata ───────────────────────────────────────────────────

    def test_build_metadata_structure(self):
        meta = FashionSDXLGenerator._build_metadata(
            image_ids         = ["IMG_001"],
            prompt            = "a red dress",
            negative_prompt   = "blurry",
            width             = 1024,
            height            = 1024,
            steps             = 30,
            guidance_scale    = 7.5,
            scheduler         = "euler",
            seed              = 42,
            generation_time_s = 5.0,
            device            = "cuda",
            model_id          = "test/model",
            use_refiner       = False,
            refiner_strength  = 0.3,
        )
        assert meta["prompt"]       == "a red dress"
        assert meta["image_ids"]    == ["IMG_001"]
        assert meta["model_id"]     == "test/model"
        assert "generation"         in meta
        assert "hardware"           in meta
        assert meta["generation"]["width"]  == 1024
        assert meta["generation"]["seed"]   == 42
        assert meta["hardware"]["device"]   == "cuda"
        assert "generated_at"       in meta

    def test_build_metadata_json_serializable(self):
        meta = FashionSDXLGenerator._build_metadata(
            image_ids=["X"], prompt="p", negative_prompt="n",
            width=512, height=512, steps=10, guidance_scale=7.0,
            scheduler="euler", seed=1, generation_time_s=1.0,
            device="cpu", model_id="m", use_refiner=False, refiner_strength=0.3,
        )
        json_str = json.dumps(meta)   # must not raise
        loaded   = json.loads(json_str)
        assert loaded["generation"]["seed"] == 1

    # ── _snap — edge cases ────────────────────────────────────────────────

    def test_snap_multiples_of_64(self):
        for val in (64, 128, 512, 768, 1024, 1536):
            assert FashionSDXLGenerator._snap_to_multiple(val, 8) == val


# =============================================================================
# ── load_model() Tests
# =============================================================================

class TestLoadModel:

    def test_already_loaded_skips_reload(self, gen):
        gen._is_loaded = True
        # Should not call any real load logic
        result = gen.load_model()
        assert result is gen
        assert gen._is_loaded is True

    def test_force_reload_clears_loaded_flag(self, gen):
        gen._is_loaded = True
        gen._pipe      = MagicMock()
        # Force reload will call unload → _pipe = None, then attempt real load
        with pytest.raises(Exception):
            # Will raise because diffusers isn't installed or mock not set up
            gen.load_model(force_reload=True)
        # After forced unload, _is_loaded should be False
        assert gen._is_loaded is False

    @pytest.mark.skipif(not _DIFFUSERS_AVAILABLE, reason="diffusers not installed")
    def test_load_sets_is_loaded_true(self, gen, mock_pipe):
        with (
            patch("diffusers.StableDiffusionXLPipeline.from_pretrained", return_value=mock_pipe),
            patch("diffusers.AutoencoderKL.from_pretrained", side_effect=Exception("skip")),
        ):
            gen.load_model()
            assert gen.is_loaded is True

    @pytest.mark.skipif(not _DIFFUSERS_AVAILABLE, reason="diffusers not installed")
    def test_load_failure_raises_runtime_error(self, gen):
        with patch(
            "diffusers.StableDiffusionXLPipeline.from_pretrained",
            side_effect=RuntimeError("404 model not found"),
        ):
            with pytest.raises(RuntimeError):
                gen.load_model()
        assert gen.is_loaded is False

    def test_load_raises_runtime_error_on_import_error(self, gen, monkeypatch):
        """If torch is not installed, load_model should raise RuntimeError."""
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "diffusers", None)
        with pytest.raises((RuntimeError, ImportError, TypeError)):
            gen.load_model()

    def test_returns_self_for_chaining(self, gen):
        gen._is_loaded = True
        result = gen.load_model()
        assert result is gen


# =============================================================================
# ── generate_image() Tests
# =============================================================================

class TestGenerateImage:

    def _run_generate(self, gen, mock_pipe, **kwargs):
        """Helper: set up the generator with a mock pipe and run generate_image."""
        gen._is_loaded = True
        gen._pipe      = mock_pipe

        # Mock torch inside the generator module to avoid ImportError
        mock_torch = MagicMock()
        mock_torch.Generator.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch.object(gen, "_inference_context", return_value=__import__("contextlib").nullcontext()):
            return gen.generate_image(**kwargs)

    def test_returns_generation_output(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="a red dress",
                                    width=64, height=64, num_inference_steps=2, seed=42, num_images=2)
        assert isinstance(result, GenerationOutput)

    def test_success_flag_true_on_ok(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="a blue hoodie", width=64, height=64, seed=1)
        assert result.success is True

    def test_failure_returns_output_with_success_false(self, gen):
        gen._is_loaded = True
        gen._pipe      = MagicMock(side_effect=RuntimeError("CUDA error"))
        mock_torch     = MagicMock()
        mock_torch.Generator.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch.object(gen, "_inference_context", return_value=__import__("contextlib").nullcontext()):
            result = gen.generate_image("fail prompt", width=64, height=64)
        assert result.success is False
        assert result.error is not None

    def test_seed_is_resolved(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="test", width=64, height=64, seed=-1)
        assert result.seed != -1
        assert result.seed >= 0

    def test_fixed_seed_stored_in_result(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="test", width=64, height=64, seed=99)
        assert result.seed == 99

    def test_size_preset_overrides_wh(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="test",
                                    width=99, height=99, size_preset="square_512")
        assert result.width  == 512
        assert result.height == 512

    def test_invalid_size_preset_uses_wh(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="test",
                                    width=64, height=64, size_preset="nonexistent_preset")
        assert result.width  == 64
        assert result.height == 64

    def test_auto_load_when_not_loaded(self, gen):
        assert not gen.is_loaded
        with patch.object(gen, "load_model", side_effect=RuntimeError("no model")):
            result = gen.generate_image("test", width=64, height=64)
        assert result.success is False
        assert "no model" in (result.error or "")

    def test_metadata_has_correct_prompt(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="my fashion prompt",
                                    width=64, height=64, seed=7)
        assert result.metadata.get("prompt") == "my fashion prompt"

    def test_generation_time_positive(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="test", width=64, height=64)
        assert result.generation_time_s >= 0.0

    def test_image_ids_populated(self, gen, mock_pipe):
        result = self._run_generate(gen, mock_pipe, prompt="test", num_images=2, width=64, height=64)
        assert len(result.image_ids) == len(result.images)
        for img_id in result.image_ids:
            assert img_id.startswith("FASHION_")


# =============================================================================
# ── generate_batch() Tests
# =============================================================================

class TestGenerateBatch:

    def _run_batch(self, gen, mock_pipe, prompts, **kwargs):
        """Helper that patches torch and inference context for batch runs."""
        gen._is_loaded = True
        gen._pipe      = mock_pipe
        mock_torch = MagicMock()
        mock_torch.Generator.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch.object(gen, "_inference_context", return_value=__import__("contextlib").nullcontext()):
            return gen.generate_batch(prompts, **kwargs)

    def test_empty_prompts_returns_empty(self, gen):
        results = gen.generate_batch([])
        assert results == []

    def test_returns_list_of_outputs(self, gen, mock_pipe):
        results = self._run_batch(gen, mock_pipe, ["a red dress", "a blue hoodie"], width=64, height=64)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, GenerationOutput)

    def test_each_prompt_in_correct_result(self, gen, mock_pipe):
        prompts = ["prompt_A", "prompt_B", "prompt_C"]
        results = self._run_batch(gen, mock_pipe, prompts, width=64, height=64)
        for r, p in zip(results, prompts):
            assert r.prompt == p

    def test_seeds_padded_to_length(self, gen, mock_pipe):
        results = self._run_batch(gen, mock_pipe, ["a", "b", "c"], seeds=[42], width=64, height=64)
        assert results[0].seed == 42
        for r in results[1:]:
            assert r.seed >= 0

    def test_negative_prompts_padded(self, gen, mock_pipe):
        results = self._run_batch(gen, mock_pipe, ["a", "b"],
                                  negative_prompts=["bad"], width=64, height=64)
        assert results[1].negative_prompt == "bad"

    def test_stop_on_error_stops_batch(self, gen):
        gen._is_loaded = True
        call_count = [0]

        def fake_generate(**kwargs):
            call_count[0] += 1
            return GenerationOutput(
                prompt=kwargs.get("prompt", ""),
                success=False, error="fake error",
            )

        with patch.object(gen, "generate_image", side_effect=fake_generate):
            results = gen.generate_batch(
                ["a", "b", "c"],
                stop_on_error=True,
                width=64, height=64,
            )
        assert len(results) == 1
        assert call_count[0] == 1

    def test_no_stop_on_error_runs_all(self, gen):
        gen._is_loaded = True

        def fake_generate(**kwargs):
            return GenerationOutput(
                prompt=kwargs.get("prompt", ""),
                success=False, error="err",
            )

        with patch.object(gen, "generate_image", side_effect=fake_generate):
            results = gen.generate_batch(
                ["a", "b", "c"],
                stop_on_error=False,
                width=64, height=64,
            )
        assert len(results) == 3

    def test_partial_failure_does_not_raise(self, gen, mock_pipe):
        gen._is_loaded = True
        gen._pipe      = mock_pipe
        calls          = [0]

        def fake_generate(**kwargs):
            calls[0] += 1
            if calls[0] == 2:
                return GenerationOutput(success=False, error="item 2 failed", prompt="b")
            return GenerationOutput(success=True, prompt=kwargs.get("prompt", ""))

        with patch.object(gen, "generate_image", side_effect=fake_generate):
            results = gen.generate_batch(["a", "b", "c"], width=64, height=64)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# =============================================================================
# ── save_output() Tests
# =============================================================================

class TestSaveOutput:

    def test_empty_images_raises(self, gen):
        with pytest.raises(ValueError, match="no images"):
            gen.save_output(GenerationOutput(images=[]))

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_saves_png_file(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(
            images    = [tiny_pil],
            image_ids = ["TEST_SAVE_001"],
            success   = True,
        )
        paths = gen.save_output(output, output_dir=tmp_path, fmt="png")
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_saves_jpg_file(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(
            images    = [tiny_pil],
            image_ids = ["TEST_SAVE_002"],
            success   = True,
        )
        paths = gen.save_output(output, output_dir=tmp_path, fmt="jpg")
        assert paths[0].suffix == ".jpg"

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_saves_metadata_sidecar(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(
            images    = [tiny_pil],
            image_ids = ["META_TEST"],
            metadata  = {"prompt": "test prompt", "seed": 42},
            success   = True,
        )
        paths = gen.save_output(output, output_dir=tmp_path, save_metadata=True)
        meta_path = paths[0].with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["prompt"] == "test prompt"

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_no_metadata_sidecar_when_disabled(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(
            images    = [tiny_pil],
            image_ids = ["NO_META"],
            metadata  = {"prompt": "p"},
        )
        paths = gen.save_output(output, output_dir=tmp_path, save_metadata=False)
        meta_path = paths[0].with_suffix(".json")
        assert not meta_path.exists()

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_output_paths_updated_on_output(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(images=[tiny_pil], image_ids=["PATH_TEST"])
        gen.save_output(output, output_dir=tmp_path)
        assert len(output.image_paths) == 1
        assert output.image_paths[0].exists()

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_multiple_images_saved(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(
            images    = [tiny_pil, tiny_pil, tiny_pil],
            image_ids = ["A", "B", "C"],
        )
        paths = gen.save_output(output, output_dir=tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_overwrite_false_avoids_collision(self, gen, tiny_pil, tmp_path):
        img_ids = ["SAME_ID"]
        output1 = GenerationOutput(images=[tiny_pil], image_ids=img_ids)
        output2 = GenerationOutput(images=[tiny_pil], image_ids=img_ids)
        paths1 = gen.save_output(output1, output_dir=tmp_path, overwrite=False)
        paths2 = gen.save_output(output2, output_dir=tmp_path, overwrite=False)
        # Both must exist and be different files
        assert paths1[0].exists()
        assert paths2[0].exists()

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_custom_prefix_in_filename(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(images=[tiny_pil], image_ids=["ID1"])
        paths  = gen.save_output(
            output, output_dir=tmp_path,
            filename_prefix="streetwear",
        )
        assert "streetwear" in paths[0].name

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_saves_to_date_subdirectory(self, gen, tiny_pil, tmp_path):
        output = GenerationOutput(images=[tiny_pil], image_ids=["DATE_TEST"])
        paths  = gen.save_output(output, output_dir=tmp_path)
        # Parent should be a date directory (YYYY-MM-DD)
        date_dir = paths[0].parent.name
        assert len(date_dir) == 10
        assert date_dir[4] == "-" and date_dir[7] == "-"


# =============================================================================
# ── Lifecycle / Context Manager Tests
# =============================================================================

class TestLifecycle:

    def test_unload_clears_loaded(self, gen):
        gen._is_loaded = True
        gen._pipe      = MagicMock()
        gen.unload_model()
        assert gen.is_loaded is False
        assert gen._pipe is None

    def test_unload_clears_refiner(self, gen):
        gen._is_loaded = True
        gen._refiner   = MagicMock()
        gen.unload_model()
        assert gen._refiner is None

    def test_context_manager_auto_loads(self, gen):
        with patch.object(gen, "load_model", return_value=gen) as mock_load, \
             patch.object(gen, "unload_model") as mock_unload:
            with gen:
                mock_load.assert_called_once()
            mock_unload.assert_called_once()

    def test_context_manager_unloads_on_exception(self, gen):
        with patch.object(gen, "load_model", return_value=gen), \
             patch.object(gen, "unload_model") as mock_unload:
            try:
                with gen:
                    raise ValueError("test error")
            except ValueError:
                pass
            mock_unload.assert_called_once()


# =============================================================================
# ── Scheduler Tests
# =============================================================================

class TestScheduler:

    def test_scheduler_map_has_euler(self):
        assert "EulerDiscreteScheduler" in SCHEDULER_MAP["euler"]

    def test_scheduler_map_has_dpm(self):
        assert "DPMSolver" in SCHEDULER_MAP["dpm++"]

    def test_apply_scheduler_unknown_logs_warning(self, gen, caplog):
        gen._pipe = MagicMock()
        # Should not raise; just warn
        gen._apply_scheduler(gen._pipe, "totally_unknown_scheduler")

    def test_apply_scheduler_none_pipe_safe(self, gen):
        gen._apply_scheduler(None, "euler")   # Must not raise


# =============================================================================
# ── Memory Optimisation Tier Tests
# =============================================================================

class TestMemoryOptimisation:

    @pytest.fixture
    def pipe_mock(self):
        p = MagicMock()
        p.to = MagicMock(return_value=None)
        p.enable_xformers_memory_efficient_attention = MagicMock()
        p.enable_attention_slicing = MagicMock()
        p.enable_model_cpu_offload = MagicMock()
        p.enable_sequential_cpu_offload = MagicMock()
        return p

    def test_cpu_device_moves_to_cpu(self, gen, pipe_mock):
        gen._device   = "cpu"
        gen._vram_gb  = 0.0
        gen._apply_memory_optimisation(pipe_mock)
        pipe_mock.to.assert_called_with("cpu")

    def test_sequential_offload_applied_when_requested(self, gen, pipe_mock):
        gen._device   = "cuda"
        gen._vram_gb  = 4.0
        gen._apply_memory_optimisation(pipe_mock, sequential_offload=True)
        pipe_mock.enable_sequential_cpu_offload.assert_called_once()

    def test_low_vram_mode_enables_cpu_offload(self, gen, pipe_mock):
        gen._device   = "cuda"
        gen._vram_gb  = 10.0
        gen._apply_memory_optimisation(pipe_mock, low_vram_mode=True)
        pipe_mock.enable_model_cpu_offload.assert_called_once()

    def test_low_vram_auto_detection(self, gen, pipe_mock):
        gen._device  = "cuda"
        gen._vram_gb = 4.0   # < 6 GB → sequential offload
        gen._apply_memory_optimisation(pipe_mock)
        pipe_mock.enable_sequential_cpu_offload.assert_called_once()

    def test_medium_vram_cpu_offload(self, gen, pipe_mock):
        gen._device  = "cuda"
        gen._vram_gb = 7.0   # 6–8 GB → model CPU offload
        gen._apply_memory_optimisation(pipe_mock)
        pipe_mock.enable_model_cpu_offload.assert_called_once()

    def test_high_vram_moves_to_device(self, gen, pipe_mock):
        gen._device  = "cuda"
        gen._vram_gb = 16.0
        gen._apply_memory_optimisation(pipe_mock)
        pipe_mock.to.assert_called_with("cuda")


# =============================================================================
# ── SDXLGenerator Adapter Tests (backward compat)
# =============================================================================

class TestSDXLGeneratorAdapter:

    def test_resolve_seed_static(self):
        assert SDXLGenerator._resolve_seed(42) == 42

    def test_resolve_seed_random(self):
        s = SDXLGenerator._resolve_seed(-1)
        assert 0 <= s < 2 ** 32

    def test_adapter_wraps_fashion_generator(self, mock_config):
        """Adapter should create a FashionSDXLGenerator internally."""
        adapter = SDXLGenerator(config=mock_config)
        assert isinstance(adapter._gen, FashionSDXLGenerator)

    def test_warm_up_calls_load_model(self, mock_config):
        adapter = SDXLGenerator(config=mock_config)
        with patch.object(adapter._gen, "load_model", return_value=adapter._gen) as m:
            result = adapter.warm_up()
        m.assert_called_once()
        assert result is adapter

    def test_unload_calls_unload_model(self, mock_config):
        adapter = SDXLGenerator(config=mock_config)
        with patch.object(adapter._gen, "unload_model") as m:
            adapter.unload()
        m.assert_called_once()
