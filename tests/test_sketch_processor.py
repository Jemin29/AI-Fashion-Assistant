"""
week3/tests/test_sketch_processor.py
=====================================
Unit tests for SketchProcessor.
Validates loading, preprocessing modes, fallbacks, and visualizations.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure project root is in path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.preprocessors.sketch_processor import SketchProcessor


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def processor():
    return SketchProcessor()


@pytest.fixture
def mock_sketch_img():
    """Create a 128x128 gray PIL image with a drawn white circle (representing a sketch)."""
    from PIL import ImageDraw
    img = Image.new("RGB", (128, 128), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw sketch outlines (a circle and lines)
    draw.ellipse([30, 30, 98, 98], outline=(20, 20, 20), width=3)
    draw.line([64, 10, 64, 118], fill=(50, 50, 50), width=2)
    return img


# =============================================================================
# ── Test Suite
# =============================================================================

class TestSketchProcessor:

    def test_load_image_resolves_and_transposes(self, processor, mock_sketch_img):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_sketch.png"
            mock_sketch_img.save(path)
            
            loaded = processor.load_image(path)
            assert isinstance(loaded, Image.Image)
            assert loaded.size == (128, 128)
            assert loaded.mode == "RGB"

    def test_load_image_missing_raises_error(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.load_image("/nonexistent/file/path/xyz.jpg")

    def test_preprocess_canny_runs(self, processor, mock_sketch_img):
        processed = processor.preprocess_sketch(mock_sketch_img, method="canny")
        assert isinstance(processed, Image.Image)
        assert processed.size == (128, 128)
        assert processed.mode == "RGB"  # Standardized for ControlNet

    def test_preprocess_hed_runs_fallback(self, processor, mock_sketch_img):
        processed = processor.preprocess_sketch(mock_sketch_img, method="hed")
        assert isinstance(processed, Image.Image)
        assert processed.size == (128, 128)
        assert processed.mode == "RGB"

    def test_preprocess_lineart_runs_fallback(self, processor, mock_sketch_img):
        processed = processor.preprocess_sketch(mock_sketch_img, method="lineart")
        assert isinstance(processed, Image.Image)
        assert processed.size == (128, 128)
        assert processed.mode == "RGB"

    def test_preprocess_invalid_method_falls_back_to_canny(self, processor, mock_sketch_img):
        processed = processor.preprocess_sketch(mock_sketch_img, method="totally_unknown_method")
        assert isinstance(processed, Image.Image)
        assert processed.size == (128, 128)
        assert processed.mode == "RGB"

    def test_save_processed_image(self, processor, mock_sketch_img):
        processed = processor.preprocess_sketch(mock_sketch_img, method="canny")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "sub" / "output_edge.png"
            saved_path = processor.save_processed_image(processed, out_path)
            
            assert saved_path.exists()
            assert saved_path == out_path
            
            # Verify file contents are valid image
            with Image.open(saved_path) as img:
                assert img.size == (128, 128)

    def test_create_comparison_grid(self, processor, mock_sketch_img):
        processed = processor.preprocess_sketch(mock_sketch_img, method="canny")
        grid = processor.create_comparison_grid(mock_sketch_img, processed, title="Test Grid")
        
        assert isinstance(grid, Image.Image)
        # Expected width: 2 * original_w + 3 * margin = 2 * 128 + 3 * 15 = 256 + 45 = 301
        # Expected height: original_h + header_h + 2 * margin = 128 + 60 + 2 * 15 = 128 + 60 + 30 = 218
        assert grid.size == (301, 218)
        assert grid.mode == "RGB"

    def test_sobel_edge_detection_math_fallback(self, processor, mock_sketch_img):
        # Force Sobel fallback path manually by supplying gray PIL image and calling the math function
        gray_img = mock_sketch_img.convert("L")
        sobel_out = processor._sobel_edge_detection(gray_img, threshold=30)
        
        assert isinstance(sobel_out, Image.Image)
        assert sobel_out.size == (128, 128)
        assert sobel_out.mode in ("L", "1")
