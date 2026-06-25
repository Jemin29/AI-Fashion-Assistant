"""
week2/tests/test_sdxl_generator.py
=====================================
Tests for SDXLGenerator, ModelManager, SchedulerFactory,
and ImageProcessor — all with mocked diffusers/torch.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_DIFFUSERS_AVAILABLE = importlib.util.find_spec("diffusers") is not None

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# ── SchedulerFactory Tests (no model load)
# =============================================================================

class TestSchedulerFactory:

    def test_available_not_empty(self):
        from src.generation.generator.scheduler_factory import SchedulerFactory
        names = SchedulerFactory.available()
        assert len(names) >= 5

    def test_known_schedulers_listed(self):
        from src.generation.generator.scheduler_factory import SchedulerFactory
        avail = SchedulerFactory.available()
        for name in ("euler", "euler_a", "dpm++", "ddim", "pndm"):
            assert name in avail

    def test_get_info_known(self):
        from src.generation.generator.scheduler_factory import SchedulerFactory
        info = SchedulerFactory.get_info("euler")
        assert "class" in info
        assert "EulerDiscreteScheduler" in info["class"]

    def test_get_info_unknown_raises(self):
        from src.generation.generator.scheduler_factory import SchedulerFactory
        with pytest.raises(KeyError):
            SchedulerFactory.get_info("does_not_exist_xyz")

    @pytest.mark.skipif(not _DIFFUSERS_AVAILABLE, reason="diffusers not installed")
    def test_from_name_unknown_falls_back_to_euler(self):
        """Unknown scheduler name should fall back to euler gracefully."""
        from src.generation.generator.scheduler_factory import SchedulerFactory
        mock_config = MagicMock()
        mock_scheduler = MagicMock()
        with patch("diffusers.EulerDiscreteScheduler") as MockEuler:
            MockEuler.from_config.return_value = MagicMock()
            # Should not raise
            SchedulerFactory.from_name("totally_unknown", mock_config)


# =============================================================================
# ── Image Processor Tests (PIL only, no models)
# =============================================================================

class TestImageProcessor:

    def test_generate_image_id_format(self):
        from src.generation.generator.image_processor import generate_image_id
        img_id = generate_image_id("TEST")
        assert img_id.startswith("TEST_")
        assert len(img_id) > 10

    def test_generate_image_id_unique(self):
        from src.generation.generator.image_processor import generate_image_id
        ids = {generate_image_id() for _ in range(20)}
        assert len(ids) == 20

    def test_build_metadata_structure(self):
        from src.generation.generator.image_processor import build_metadata
        meta = build_metadata(
            image_id         = "GEN_001",
            prompt           = "a red dress",
            negative_prompt  = "blurry",
            width            = 512,
            height           = 512,
            steps            = 20,
            guidance_scale   = 7.5,
            scheduler        = "euler",
            seed             = 42,
            model_id         = "test/model",
            generation_time_s= 1.23,
        )
        assert meta["image_id"]       == "GEN_001"
        assert meta["prompt"]         == "a red dress"
        assert meta["generation"]["width"] == 512
        assert meta["generation"]["seed"]  == 42
        assert meta["timing"]["generation_time_s"] == 1.23

    def test_save_image_creates_file(self, tmp_path, tiny_image):
        if tiny_image is None:
            pytest.skip("Pillow not installed")
        from src.generation.generator.image_processor import save_image
        path = save_image(tiny_image, "TEST_SAVE", tmp_path, fmt="png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_save_metadata_sidecar(self, tmp_path, tiny_image):
        if tiny_image is None:
            pytest.skip("Pillow not installed")
        import json
        from src.generation.generator.image_processor import save_image, save_metadata_sidecar
        img_path = save_image(tiny_image, "TEST_META", tmp_path)
        meta     = {"image_id": "TEST_META", "prompt": "test"}
        sidecar  = save_metadata_sidecar(meta, img_path)
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text())
        assert loaded["image_id"] == "TEST_META"

    def test_resize_image(self, tiny_image):
        if tiny_image is None:
            pytest.skip("Pillow not installed")
        from src.generation.generator.image_processor import resize_image
        resized = resize_image(tiny_image, (128, 128))
        assert resized.size == (128, 128)

    def test_crop_to_square_from_rectangle(self):
        try:
            from PIL import Image
            from src.generation.generator.image_processor import crop_to_square
            img     = Image.new("RGB", (200, 100), color=(0, 0, 0))
            cropped = crop_to_square(img)
            w, h    = cropped.size
            assert w == h == 100
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_images_to_grid(self, tiny_image):
        if tiny_image is None:
            pytest.skip("Pillow not installed")
        from src.generation.generator.image_processor import images_to_grid
        grid = images_to_grid([tiny_image] * 4, cols=2)
        assert grid.size[0] > tiny_image.size[0]

    def test_add_watermark_does_not_crash(self, tiny_image):
        if tiny_image is None:
            pytest.skip("Pillow not installed")
        from src.generation.generator.image_processor import add_watermark
        result = add_watermark(tiny_image, text="Test Watermark")
        assert result is not None
        assert result.size == tiny_image.size


# =============================================================================
# ── GenerationResult Tests
# =============================================================================

class TestGenerationResult:

    def test_empty_result(self):
        from src.generation.generator.sdxl_generator import GenerationResult
        r = GenerationResult()
        assert len(r) == 0
        assert r.success is True

    def test_repr_ok_result(self):
        from src.generation.generator.sdxl_generator import GenerationResult
        r = GenerationResult(success=True, generation_time_s=2.5)
        assert "OK" in repr(r)

    def test_repr_error_result(self):
        from src.generation.generator.sdxl_generator import GenerationResult
        r = GenerationResult(success=False, error="CUDA OOM")
        assert "ERROR" in repr(r)

    def test_len_matches_images(self):
        from src.generation.generator.sdxl_generator import GenerationResult
        r = GenerationResult(images=["img1", "img2"])
        assert len(r) == 2


# =============================================================================
# ── ModelManager Tests (mocked — no real model load)
# =============================================================================

class TestModelManagerResolvers:

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_resolve_device_cuda(self):
        from src.generation.generator.model_manager import ModelManager
        with patch("torch.cuda.is_available", return_value=True):
            device = ModelManager._resolve_device("auto")
            assert device == "cuda"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_resolve_device_cpu_fallback(self):
        from src.generation.generator.model_manager import ModelManager
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = ModelManager._resolve_device("auto")
            assert device == "cpu"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_resolve_device_explicit(self):
        from src.generation.generator.model_manager import ModelManager
        assert ModelManager._resolve_device("cpu")  == "cpu"
        assert ModelManager._resolve_device("cuda") == "cuda"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_resolve_dtype_float16(self):
        import torch
        from src.generation.generator.model_manager import ModelManager
        dtype = ModelManager._resolve_dtype("float16")
        assert dtype == torch.float16

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_resolve_dtype_float32(self):
        import torch
        from src.generation.generator.model_manager import ModelManager
        dtype = ModelManager._resolve_dtype("float32")
        assert dtype == torch.float32

    def test_is_loaded_false_initially(self, mock_config):
        from src.generation.generator.model_manager import ModelManager
        mgr = ModelManager(config=mock_config)
        assert mgr.is_loaded is False


# =============================================================================
# ── SDXLGenerator Seed Tests (no model load)
# =============================================================================

class TestSDXLGeneratorSeed:

    def test_resolve_seed_fixed(self):
        from src.generation.generator.sdxl_generator import SDXLGenerator
        assert SDXLGenerator._resolve_seed(42) == 42

    def test_resolve_seed_random(self):
        from src.generation.generator.sdxl_generator import SDXLGenerator
        seeds = {SDXLGenerator._resolve_seed(-1) for _ in range(10)}
        assert len(seeds) > 1  # Should be different each time

    def test_resolve_seed_random_in_range(self):
        from src.generation.generator.sdxl_generator import SDXLGenerator
        seed = SDXLGenerator._resolve_seed(-1)
        assert 0 <= seed < 2**32
