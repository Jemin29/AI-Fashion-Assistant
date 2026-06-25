"""
week3/tests/test_controlnet_engine.py
=====================================
Unit tests for FashionControlNetEngine.
Runs tests in mock mode and patches diffusers for loader tests.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine, GenerationOutput


# =============================================================================
# ── Helpers / Fixtures
# =============================================================================

@pytest.fixture
def tiny_pil():
    """Create a simple 64x64 PIL image for testing."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


@pytest.fixture
def mock_engine():
    """Return a FashionControlNetEngine forced into mock mode."""
    return FashionControlNetEngine(mock=True)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestFashionControlNetEngine:

    def test_init_defaults(self):
        engine = FashionControlNetEngine(mock=True)
        assert engine.mock is True
        assert engine.device == "cpu"
        assert engine.dtype == "float32"

    def test_resolve_hardware_device_mapping(self):
        # Create an engine with a mock config specifying custom device
        mock_cfg = MagicMock()
        mock_cfg.model.runtime.device = "cpu"
        mock_cfg.model.runtime.torch_dtype = "float16"
        
        # CPU device forces float32 dtype
        engine = FashionControlNetEngine(config=mock_cfg, mock=False)
        assert engine.device == "cpu"
        assert engine.dtype == "float32"

    def test_resolve_hardware_gpu_fallback(self):
        mock_cfg = MagicMock()
        mock_cfg.model.runtime.device = "auto"
        mock_cfg.model.runtime.torch_dtype = "float16"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            engine = FashionControlNetEngine(config=mock_cfg, mock=False)
            assert engine.device == "cpu"
            assert engine.dtype == "float32"

    def test_load_models_mock(self):
        engine = FashionControlNetEngine(mock=True)
        assert engine.load_models("canny") is True
        assert engine._current_type == "canny"
        
        assert engine.load_models("sketch") is True
        assert engine._current_type == "sketch"

    def test_load_models_unknown_raises(self):
        engine = FashionControlNetEngine(mock=True)
        with pytest.raises(ValueError):
            engine.load_models("unknown_control_net_type")

    def test_generate_from_sketch_mock(self, mock_engine, tiny_pil):
        res = mock_engine.generate_from_sketch(
            prompt="A beautiful bohemian shirt",
            sketch_image=tiny_pil,
            conditioning_scale=0.75,
            seed=123
        )
        assert isinstance(res, GenerationOutput)
        assert res.success is True
        assert res.is_mock is True
        assert res.control_type == "sketch"
        assert res.seed == 123
        assert len(res.images) == 1
        assert isinstance(res.images[0], Image.Image)
        assert res.images[0].size == (64, 64)
        assert "generation" in res.metadata
        assert res.metadata["generation"]["conditioning_scale"] == 0.75

    def test_generate_from_pose_mock(self, mock_engine, tiny_pil):
        res = mock_engine.generate_from_pose(
            prompt="A man in a sleek black suit",
            pose_image=tiny_pil,
            seed=42
        )
        assert res.success is True
        assert res.control_type == "pose"
        assert res.seed == 42
        assert len(res.images) == 1

    def test_generate_from_depth_mock(self, mock_engine, tiny_pil):
        res = mock_engine.generate_from_depth(
            prompt="A detailed trench coat outline",
            depth_image=tiny_pil,
            seed=999
        )
        assert res.success is True
        assert res.control_type == "depth"
        assert res.seed == 999
        assert len(res.images) == 1

    def test_save_output_creates_files(self, mock_engine, tiny_pil):
        res = mock_engine.generate_from_sketch(
            prompt="Streetwear pants",
            sketch_image=tiny_pil,
            seed=101
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = mock_engine.save_output(res, output_dir=tmpdir)
            assert len(filepaths) == 1
            
            saved_img_path = filepaths[0]
            assert saved_img_path.exists()
            assert saved_img_path.suffix == ".png"
            
            # Find the JSON metadata sidecar
            json_paths = list(Path(tmpdir).glob("*.json"))
            assert len(json_paths) == 1
            
            # Load and verify JSON contents
            meta = json.loads(json_paths[0].read_text(encoding="utf-8"))
            assert meta["prompt"] == "Streetwear pants"
            assert meta["generation"]["seed"] == 101
            assert meta["generation"]["controlnet_type"] == "sketch"

    def test_unload_clears_models(self, mock_engine):
        mock_engine.load_models("canny")
        assert mock_engine._current_type == "canny"
        
        mock_engine.unload()
        assert mock_engine._current_type is None
        assert mock_engine._pipe is None

    def test_load_models_real_pipeline_calls(self, tiny_pil):
        """Test that the engine loads the real pretrained weights via diffusers API."""
        mock_cfg = MagicMock()
        mock_cfg.model.base.repo_id = "test/base-model"
        mock_cfg.model.vae.repo_id = ""
        mock_cfg.model.runtime.device = "cpu"
        mock_cfg.model.runtime.torch_dtype = "float32"
        mock_cfg.controlnet.enabled = False
        
        # Set up pipeline and modules mocks
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        
        mock_diffusers = MagicMock()
        mock_pipe_class = MagicMock()
        mock_cnet_class = MagicMock()
        
        pipe_instance = MagicMock()
        pipe_instance.scheduler.config = {}
        mock_pipe_class.from_pretrained.return_value = pipe_instance
        mock_cnet_class.from_pretrained.return_value = MagicMock()
        
        mock_diffusers.ControlNetModel = mock_cnet_class
        mock_diffusers.StableDiffusionXLControlNetPipeline = mock_pipe_class
        
        # Patch heavy dependencies so the test runs in clean standard envs
        with patch.dict("sys.modules", {"torch": mock_torch, "diffusers": mock_diffusers}):
            # Force real mode execution (mock=False)
            engine = FashionControlNetEngine(config=mock_cfg, mock=False)
            
            # Call loader
            success = engine.load_models("canny")
            assert success is True
            
            # Verify correct HF repositories were called
            mock_cnet_class.from_pretrained.assert_called_once_with(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype="float32",
                use_safetensors=True
            )
            mock_pipe_class.from_pretrained.assert_called_once_with(
                "test/base-model",
                controlnet=engine._controlnet,
                torch_dtype="float32",
                use_safetensors=True
            )
