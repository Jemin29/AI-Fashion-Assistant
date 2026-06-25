"""
week3/tests/test_sketch2design.py
=================================
Unit tests for Sketch2Design orchestrator.
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

from src.controlnet.controlnet.sketch2design import Sketch2Design
from src.controlnet.controlnet.controlnet_engine import GenerationOutput


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def mock_sketch():
    """Create a 128x128 gray PIL image for testing."""
    return Image.new("RGB", (128, 128), color=(220, 220, 220))


@pytest.fixture
def orchestrator():
    """Return a Sketch2Design instance forced into mock mode."""
    return Sketch2Design(mock=True)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestSketch2Design:

    def test_init_propagates_mock_flag(self):
        s2d = Sketch2Design(mock=True)
        assert s2d.mock is True
        assert s2d.engine.mock is True

    def test_generate_design_mock(self, orchestrator, mock_sketch):
        res = orchestrator.generate_design(
            sketch=mock_sketch,
            prompt="A leather biker jacket",
            style="streetwear",
            method="canny",
            conditioning_scale=0.85,
            seed=42
        )
        assert isinstance(res, GenerationOutput)
        assert res.success is True
        assert res.control_type == "sketch"
        assert res.seed == 42
        assert len(res.images) == 1
        assert res.images[0].size == (128, 128)
        
        # Verify metadata keys
        assert "preprocessor" in res.metadata
        assert res.metadata["preprocessor"]["method"] == "canny"
        assert res.metadata["preprocessor"]["style_preset"] == "streetwear"
        assert "preprocessed_image" in res.metadata
        assert isinstance(res.metadata["preprocessed_image"], Image.Image)

    def test_generate_design_with_style_fallback_tags(self, orchestrator, mock_sketch):
        # Force prompt builder to be None to test simple fallback style tag injection
        orchestrator.prompt_builder = None
        
        res = orchestrator.generate_design(
            sketch=mock_sketch,
            prompt="A minimalist wool coat",
            style="luxury"
        )
        assert res.success is True
        # Check that style tags were appended
        assert "luxury fashion" in res.prompt
        assert "haute couture" in res.prompt

    def test_generate_batch_mock(self, orchestrator, mock_sketch):
        sketches = [mock_sketch, mock_sketch]
        prompts = ["a linen shirt", "a graphic tee"]
        styles = ["casual", "streetwear"]
        
        outputs = orchestrator.generate_batch(
            sketches=sketches,
            prompts=prompts,
            styles=styles,
            seed=100
        )
        
        assert len(outputs) == 2
        assert outputs[0].success is True
        assert outputs[0].seed == 100
        assert outputs[1].success is True
        # Seeds should increment sequentially
        assert outputs[1].seed == 101

    def test_generate_batch_padding(self, orchestrator, mock_sketch):
        sketches = [mock_sketch, mock_sketch, mock_sketch]
        prompts = ["a single dress prompt"] # Only 1 prompt provided
        
        outputs = orchestrator.generate_batch(
            sketches=sketches,
            prompts=prompts,
            seed=200
        )
        
        assert len(outputs) == 3
        # Prompts list is padded with the last item
        assert outputs[1].prompt == "a single dress prompt"
        assert outputs[2].prompt == "a single dress prompt"

    def test_save_results_creates_three_files(self, orchestrator, mock_sketch):
        # Save results should write: 
        # 1. Output generated design image (.png)
        # 2. Preprocessed edge map sketch (.png)
        # 3. Output metadata JSON (.json)
        res = orchestrator.generate_design(
            sketch=mock_sketch,
            prompt="Sporty activewear leggings",
            style="athleisure",
            seed=999
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = orchestrator.save_results(res, output_dir=tmpdir)
            
            # Should return paths for: design image + preprocessed image
            assert len(filepaths) == 2
            
            # Check design image file
            design_path = filepaths[0]
            assert design_path.exists()
            assert "preprocessed" not in design_path.name
            
            # Check preprocessed edge file
            edge_path = filepaths[1]
            assert edge_path.exists()
            assert edge_path.name == f"{design_path.stem}_preprocessed.png"
            
            # Find the JSON metadata file on disk
            json_paths = list(Path(tmpdir).glob("*.json"))
            assert len(json_paths) == 1
            
            # Load and verify JSON contents
            with open(json_paths[0], "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["prompt"] == res.prompt
            assert meta["generation"]["seed"] == 999
            assert meta["preprocessor"]["method"] == "canny"

    def test_save_results_empty_or_failed(self, orchestrator):
        # Create a mock failed output
        failed_res = GenerationOutput(success=False, images=[])
        filepaths = orchestrator.save_results(failed_res)
        assert filepaths == []

    def test_generate_batch_padding_with_fewer_styles(self, orchestrator, mock_sketch):
        sketches = [mock_sketch, mock_sketch]
        prompts = ["p1", "p2"]
        styles = ["streetwear"]  # Fewer styles than sketches
        
        outputs = orchestrator.generate_batch(
            sketches=sketches,
            prompts=prompts,
            styles=styles,
            seed=100
        )
        assert len(outputs) == 2
        assert "streetwear style" in outputs[1].prompt

    def test_save_results_save_exception(self, orchestrator, mock_sketch):
        res = orchestrator.generate_design(
            sketch=mock_sketch,
            prompt="Sporty activewear leggings",
            seed=999
        )
        # Mock preprocessed_image save method to throw an exception
        from unittest.mock import MagicMock
        bad_img = MagicMock()
        bad_img.save.side_effect = Exception("Disk full")
        res.metadata["preprocessed_image"] = bad_img
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = orchestrator.save_results(res, output_dir=tmpdir)
            # Should still return the standard design image paths and not crash!
            assert len(filepaths) == 1

