"""
week3/tests/test_depth2fashion.py
=================================
Unit tests for Depth2Fashion orchestrator.
Validates loading, single-image generation, batch processing, and output saving.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.controlnet.depth2fashion import Depth2Fashion
from src.controlnet.controlnet.controlnet_engine import GenerationOutput


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def mock_image():
    """Create a 128x128 solid gray PIL image representing a raw photo."""
    return Image.new("RGB", (128, 128), color=(200, 200, 200))


@pytest.fixture
def orchestrator():
    """Return a Depth2Fashion instance forced into mock mode."""
    return Depth2Fashion(mock=True)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestDepth2Fashion:

    def test_init_propagates_mock_flag(self):
        d2f = Depth2Fashion(mock=True)
        assert d2f.mock is True
        assert d2f.engine.mock is True

    def test_preprocess_depth_generates_depth_map(self, orchestrator, mock_image):
        depth_map = orchestrator.preprocess_depth(mock_image)
        assert isinstance(depth_map, Image.Image)
        assert depth_map.size == (128, 128)
        assert depth_map.mode == "RGB"
        
        # Checking that our helper correctly identifies it as a depth map
        assert orchestrator._is_depth_image(depth_map) is True

    def test_is_depth_image_checks_non_depth(self, orchestrator, mock_image):
        # A solid light-gray photo with constant colors might technically count as depth,
        # but let's test a non-grayscale gradient to ensure the heuristic works.
        # Let's create an image with distinct RGB values.
        rainbow = Image.new("RGB", (128, 128))
        pixels = rainbow.load()
        for i in range(128):
            for j in range(128):
                pixels[i, j] = (i * 2, j * 2, 100)
        assert orchestrator._is_depth_image(rainbow) is False

    def test_generate_fashion_mock(self, orchestrator, mock_image):
        res = orchestrator.generate_fashion(
            depth_image=mock_image,
            prompt="A minimalist wool trench coat",
            style="minimalist",
            conditioning_scale=0.85,
            seed=42
        )
        assert isinstance(res, GenerationOutput)
        assert res.success is True
        assert res.control_type == "depth"
        assert res.seed == 42
        assert len(res.images) == 1
        assert res.images[0].size == (128, 128)
        
        # Verify metadata keys
        assert "preprocessor" in res.metadata
        assert res.metadata["preprocessor"]["method"] == "depth"
        assert res.metadata["preprocessor"]["style_preset"] == "minimalist"
        assert "preprocessed_depth" in res.metadata
        assert isinstance(res.metadata["preprocessed_depth"], Image.Image)

    def test_generate_fashion_detects_pre_extracted_depth_map(self, orchestrator, mock_image):
        # 1. Preprocess first to get a depth map
        depth_map = orchestrator.preprocess_depth(mock_image)
        
        # 2. Feed the depth map directly. It should bypass preprocessing.
        res = orchestrator.generate_fashion(
            depth_image=depth_map,
            prompt="A simple black dress"
        )
        assert res.success is True
        assert res.control_type == "depth"

    def test_generate_fashion_with_style_fallback_tags(self, orchestrator, mock_image):
        # Force prompt builder to be None to test simple fallback style tag injection
        orchestrator.prompt_builder = None
        
        res = orchestrator.generate_fashion(
            depth_image=mock_image,
            prompt="A sleek dress",
            style="luxury"
        )
        assert res.success is True
        assert "luxury fashion" in res.prompt
        assert "premium silk" in res.prompt

    def test_generate_batch_mock(self, orchestrator, mock_image):
        images = [mock_image, mock_image]
        prompts = ["a linen shirt", "a silk gown"]
        styles = ["casual", "luxury"]
        
        outputs = orchestrator.generate_batch(
            depth_images=images,
            prompts=prompts,
            styles=styles,
            seed=100
        )
        
        assert len(outputs) == 2
        assert outputs[0].success is True
        assert outputs[0].seed == 100
        assert outputs[1].success is True
        assert outputs[1].seed == 101

    def test_generate_batch_padding(self, orchestrator, mock_image):
        images = [mock_image, mock_image, mock_image]
        prompts = ["a single dress prompt"] # Only 1 prompt provided
        
        outputs = orchestrator.generate_batch(
            depth_images=images,
            prompts=prompts,
            seed=200
        )
        
        assert len(outputs) == 3
        # Prompts list is padded with the last item
        assert outputs[1].prompt == "a single dress prompt"
        assert outputs[2].prompt == "a single dress prompt"

    def test_save_results_creates_three_files(self, orchestrator, mock_image):
        res = orchestrator.generate_fashion(
            depth_image=mock_image,
            prompt="Luxury black oversized hoodie",
            style="luxury",
            seed=999
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = orchestrator.save_results(res, output_dir=tmpdir)
            
            # Should return paths for: design image + preprocessed depth map
            assert len(filepaths) == 2
            
            # Check design image file
            design_path = filepaths[0]
            assert design_path.exists()
            assert "depth_map" not in design_path.name
            
            # Check preprocessed depth map file
            depth_path = filepaths[1]
            assert depth_path.exists()
            assert depth_path.name == f"{design_path.stem}_depth_map.png"
            
            # Find the JSON metadata file on disk
            json_paths = list(Path(tmpdir).glob("*.json"))
            assert len(json_paths) == 1
            
            # Load and verify JSON contents
            with open(json_paths[0], "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["prompt"] == res.prompt
            assert meta["generation"]["seed"] == 999
            assert meta["preprocessor"]["method"] == "depth"

    def test_save_results_empty_or_failed(self, orchestrator):
        failed_res = GenerationOutput(success=False, images=[])
        filepaths = orchestrator.save_results(failed_res)
        assert filepaths == []

    def test_generate_batch_padding_and_null_styles(self, orchestrator, mock_image):
        images = [mock_image, mock_image]
        prompts = ["a simple black dress"] # Only 1 prompt provided
        
        outputs = orchestrator.generate_batch(
            depth_images=images,
            prompts=prompts,
            styles=None, # null styles list
            seed=500
        )
        assert len(outputs) == 2
        assert outputs[1].prompt == "a simple black dress"

    def test_save_results_save_exception(self, orchestrator, mock_image):
        res = orchestrator.generate_fashion(
            depth_image=mock_image,
            prompt="Luxury black oversized hoodie",
            seed=999
        )
        # Mock preprocessed_depth save method to throw an exception
        from unittest.mock import MagicMock
        bad_img = MagicMock()
        bad_img.save.side_effect = Exception("Disk full")
        res.metadata["preprocessed_depth"] = bad_img
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = orchestrator.save_results(res, output_dir=tmpdir)
            assert len(filepaths) == 1

    def test_is_depth_image_exception(self, orchestrator):
        # Pass something that isn't a PIL Image to raise an exception
        assert orchestrator._is_depth_image(None) is False

