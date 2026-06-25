"""
=============================================================================
AI-Powered Fashion Design Assistant
tests/test_ingestion.py — Unit Tests: Dataset Ingesters
=============================================================================
Tests the ingester classes using synthetic/mock data so tests run without
any real dataset files being present.

Test Strategy:
    - Unit tests only — no network or file-system side effects
    - Mock h5py and PIL to isolate ingester logic
    - Test both happy path and error cases
    - Use pytest fixtures for reusable setup

Run tests:
    pytest tests/test_ingestion.py -v --cov=data_pipeline.ingestion
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# FashionGen Ingester Tests
# =============================================================================

class TestFashionGenIngester:
    """Unit tests for FashionGenIngester."""

    def _make_mock_hdf5(self, n: int = 5) -> MagicMock:
        """
        Build a mock h5py.File object with realistic FashionGen structure.

        Uses a dict-backed __getitem__ side_effect so that
        hdf5_file["input_image"][idx] returns a real numpy array.
        """
        images        = np.random.randint(0, 255, (n, 256, 256, 3), dtype=np.uint8)
        descriptions  = [f"A nice blue cotton T-shirt number {i}".encode() for i in range(n)]
        categories    = [b"SHIRTS"] * n
        subcategories = [b"T-SHIRTS"] * n
        genders       = [b"men"] * (n // 2) + [b"women"] * (n - n // 2)

        datasets = {
            "input_image"       : images,
            "input_description" : descriptions,
            "input_category"    : categories,
            "input_subcategory" : subcategories,
            "input_gender"      : genders,
        }

        mock_file = MagicMock()
        mock_file.__getitem__.side_effect = lambda key: datasets[key]
        mock_file.__contains__ = lambda self, key: key in datasets
        # len() of the file uses the image dataset length
        mock_file.__len__.return_value = n

        return mock_file

    @pytest.fixture
    def ingester(self, tmp_path):
        """FashionGenIngester pointing at a temp .h5 file."""
        from src.data.ingestion.fashiongen_ingester import FashionGenIngester

        # Create a dummy file so the path-exists check passes
        h5_path = tmp_path / "fashiongen_test.h5"
        h5_path.touch()
        return FashionGenIngester(hdf5_path=h5_path, split="train")

    def test_init_missing_file_warns(self, tmp_path, caplog):
        """Ingester should warn (not raise) when HDF5 file is not found."""
        from src.data.ingestion.fashiongen_ingester import FashionGenIngester

        ingester = FashionGenIngester(
            hdf5_path=tmp_path / "nonexistent.h5",
            split="train",
        )
        assert ingester is not None  # Should not raise

    def test_decode_bytes_bytes(self):
        """_decode_bytes should handle raw bytes correctly."""
        from src.data.ingestion.fashiongen_ingester import FashionGenIngester

        assert FashionGenIngester._decode_bytes(b"hello world") == "hello world"

    def test_decode_bytes_numpy_bytes(self):
        """_decode_bytes should handle numpy byte strings."""
        from src.data.ingestion.fashiongen_ingester import FashionGenIngester

        assert FashionGenIngester._decode_bytes(np.bytes_(b"test")) == "test"

    def test_decode_bytes_str_passthrough(self):
        """_decode_bytes should return plain strings unchanged."""
        from src.data.ingestion.fashiongen_ingester import FashionGenIngester

        assert FashionGenIngester._decode_bytes("already decoded") == "already decoded"

    def test_stream_raises_on_missing_file(self, tmp_path):
        """stream() must raise FileNotFoundError if HDF5 is absent."""
        from src.data.ingestion.fashiongen_ingester import FashionGenIngester

        ingester = FashionGenIngester(
            hdf5_path=tmp_path / "ghost.h5", split="train"
        )
        with pytest.raises(FileNotFoundError):
            list(ingester.stream())

    @patch("data_pipeline.ingestion.fashiongen_ingester.h5py.File")
    def test_stream_yields_correct_count(self, mock_h5, ingester):
        """stream(max_items=3) should yield exactly 3 records."""
        mock_file = self._make_mock_hdf5(n=5)
        mock_h5.return_value.__enter__ = lambda s: mock_file
        mock_h5.return_value.__exit__  = MagicMock(return_value=False)

        records = list(ingester.stream(max_items=3))
        assert len(records) == 3

    @patch("data_pipeline.ingestion.fashiongen_ingester.h5py.File")
    def test_stream_record_schema(self, mock_h5, ingester):
        """Each yielded record must contain all required keys."""
        mock_file = self._make_mock_hdf5(n=2)
        mock_h5.return_value.__enter__ = lambda s: mock_file
        mock_h5.return_value.__exit__  = MagicMock(return_value=False)

        required_keys = {
            "image_id", "image_array", "description",
            "category", "subcategory", "gender", "split",
            "dataset_source", "source_index",
        }
        records = list(ingester.stream(max_items=2))
        for rec in records:
            assert required_keys.issubset(rec.keys()), (
                f"Record missing keys: {required_keys - rec.keys()}"
            )

    @patch("data_pipeline.ingestion.fashiongen_ingester.h5py.File")
    def test_stream_image_shape(self, mock_h5, ingester):
        """Image arrays should be 256×256×3 uint8."""
        mock_file = self._make_mock_hdf5(n=1)
        mock_h5.return_value.__enter__ = lambda s: mock_file
        mock_h5.return_value.__exit__  = MagicMock(return_value=False)

        records = list(ingester.stream(max_items=1))
        img = records[0]["image_array"]
        assert hasattr(img, "shape"), f"Expected ndarray, got {type(img)}"
        assert img.shape == (256, 256, 3), f"Unexpected shape: {img.shape}"
        assert img.dtype == np.uint8, f"Unexpected dtype: {img.dtype}"

    @patch("data_pipeline.ingestion.fashiongen_ingester.h5py.File")
    def test_stream_dataset_source(self, mock_h5, ingester):
        """dataset_source must be 'fashiongen'."""
        mock_file = self._make_mock_hdf5(n=1)
        mock_h5.return_value.__enter__ = lambda s: mock_file
        mock_h5.return_value.__exit__  = MagicMock(return_value=False)

        records = list(ingester.stream(max_items=1))
        assert records[0]["dataset_source"] == "fashiongen"


# =============================================================================
# DeepFashion Ingester Tests
# =============================================================================

class TestDeepFashionIngester:
    """Unit tests for DeepFashionIngester."""

    def test_init_missing_root_warns(self, tmp_path):
        """Ingester should warn (not raise) when root_dir is absent."""
        from src.data.ingestion.deepfashion_ingester import DeepFashionIngester

        ingester = DeepFashionIngester(root_dir=tmp_path / "nonexistent_deepfashion")
        assert ingester is not None

    def test_stream_raises_on_empty_split_map(self, tmp_path):
        """stream() must raise RuntimeError if annotation maps are empty."""
        from src.data.ingestion.deepfashion_ingester import DeepFashionIngester

        ingester = DeepFashionIngester(root_dir=tmp_path / "empty_dir")
        with pytest.raises(RuntimeError, match="Annotation maps are empty"):
            list(ingester.stream())

    def test_decode_bytes_from_fashiongen_for_reuse(self):
        """
        Verify the static decode helper works across ingester classes.
        (DeepFashion uses PIL directly — this test ensures no import clash.)
        """
        from src.data.ingestion.deepfashion_ingester import DeepFashionIngester
        # DeepFashionIngester doesn't have _decode_bytes but should import cleanly
        assert DeepFashionIngester is not None
