"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_preprocessing.py — Unit Tests: Image Preprocessor
=============================================================================
Tests FashionPreprocessor with synthetic PIL images — no real dataset needed.

Run:
    pytest tests/test_preprocessing.py -v
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image


def _make_synthetic_record(
    width: int = 512,
    height: int = 400,
    dataset_source: str = "fashiongen",
    description: str = "A vibrant floral summer dress with spaghetti straps.",
) -> Dict[str, Any]:
    """Create a synthetic fashion record with a random-color PIL image."""
    image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return {
        "image_id"      : "TEST_0000001",
        "image_array"   : image_array,
        "description"   : description,
        "category"      : "DRESSES",
        "subcategory"   : "MAXI_DRESSES",
        "gender"        : "women",
        "split"         : "train",
        "dataset_source": dataset_source,
    }


class TestFashionPreprocessor:
    """Unit tests for FashionPreprocessor."""

    @pytest.fixture
    def preprocessor(self, tmp_path):
        """Return a default FashionPreprocessor writing to tmp_path."""
        from src.data.preprocessing.image_preprocessor import (
            FashionPreprocessor,
            PreprocessorConfig,
        )

        config = PreprocessorConfig(
            target_size=(256, 256),
            blur_threshold=0.0,  # Disable blur check for synthetic random images
            min_entropy=0.0,     # Disable contrast check
            normalize=True,
        )
        return FashionPreprocessor(output_dir=tmp_path / "processed", config=config)

    def test_process_returns_success(self, preprocessor):
        """Processing a valid synthetic record should succeed."""
        record = _make_synthetic_record()
        result = preprocessor.process(record)
        assert result.success, f"Expected success but got: {result.rejection_reason}"

    def test_process_output_file_exists(self, preprocessor):
        """The processed image file should be saved to disk."""
        record = _make_synthetic_record()
        result = preprocessor.process(record)
        assert result.processed_path is not None
        assert Path(result.processed_path).exists()

    def test_process_normalized_array_shape(self, preprocessor):
        """Normalized array should be float32 with shape (3, 256, 256)."""
        record = _make_synthetic_record()
        result = preprocessor.process(record)
        assert result.normalized_array is not None
        assert result.normalized_array.shape == (3, 256, 256)
        assert result.normalized_array.dtype == np.float32

    def test_process_metadata_dimensions(self, preprocessor):
        """Result metadata should report target output dimensions."""
        record = _make_synthetic_record(width=800, height=600)
        result = preprocessor.process(record)
        assert result.metadata["width"]  == 256
        assert result.metadata["height"] == 256

    def test_process_rejects_too_small_image(self, preprocessor):
        """Images smaller than min_width/min_height should be rejected."""
        record = _make_synthetic_record(width=32, height=32)
        result = preprocessor.process(record)
        assert not result.success
        assert "too small" in result.rejection_reason.lower()

    def test_process_batch_returns_all_results(self, preprocessor):
        """process_batch should return one result per input record."""
        records = [_make_synthetic_record() for _ in range(5)]
        results = preprocessor.process_batch(records)
        assert len(results) == 5

    def test_process_batch_success_rate(self, preprocessor):
        """All valid synthetic records should succeed."""
        records = [_make_synthetic_record() for _ in range(3)]
        results = preprocessor.process_batch(records)
        success_count = sum(1 for r in results if r.success)
        assert success_count == 3

    def test_normalize_channel_first(self, preprocessor):
        """Normalization should convert HWC → CHW layout."""
        from src.data.preprocessing.image_preprocessor import PreprocessorConfig

        cfg = preprocessor.config
        arr = np.ones((256, 256, 3), dtype=np.float32) * 0.5  # mid-grey
        normalized = preprocessor._normalize(arr)
        assert normalized.shape == (3, 256, 256)

    def test_md5_is_deterministic(self, preprocessor, tmp_path):
        """Same file should produce the same MD5 hash every time."""
        from src.data.preprocessing.image_preprocessor import FashionPreprocessor

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"fashion ai test data 12345")
        h1 = FashionPreprocessor._md5(test_file)
        h2 = FashionPreprocessor._md5(test_file)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex digest is 32 chars


class TestPreprocessorConfig:
    """Tests for the PreprocessorConfig dataclass."""

    def test_default_target_size(self):
        from src.data.preprocessing.image_preprocessor import PreprocessorConfig
        cfg = PreprocessorConfig()
        assert cfg.target_size == (256, 256)

    def test_custom_target_size(self):
        from src.data.preprocessing.image_preprocessor import PreprocessorConfig
        cfg = PreprocessorConfig(target_size=(512, 512))
        assert cfg.target_size == (512, 512)

    def test_default_normalization_enabled(self):
        from src.data.preprocessing.image_preprocessor import PreprocessorConfig
        cfg = PreprocessorConfig()
        assert cfg.normalize is True
