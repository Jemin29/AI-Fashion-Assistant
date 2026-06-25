"""
week4/tests/test_brand_dataset_manager.py
=========================================
Unit tests for the Fashion Brand Dataset Management System.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.lora.datasets.brand_dataset_manager import BrandDatasetManager


class TestBrandDatasetManager:
    """Verify ingestion, duplication detection, validation, and stats aggregation."""

    @pytest.fixture
    def test_image(self):
        """Create a standard 512x512 solid RGB PIL image for testing."""
        return Image.new("RGB", (512, 512), color=(255, 0, 0))

    @pytest.fixture
    def small_image(self):
        """Create a low-resolution image below 512x512 threshold."""
        return Image.new("RGB", (128, 128), color=(0, 255, 0))

    def test_manager_init(self):
        """Verify constructor initializes path layouts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            assert manager.dataset_root.exists()

    def test_ingest_image_success(self, test_image):
        """Verify successful image ingestion and manifest serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            
            raw_meta = {
                "category": "hoodies",
                "color": ["black"],
                "description": "Retro vintage collection."
            }
            
            record = manager.ingest_image(
                brand="nike",
                image=test_image,
                filename="nike_hoodie.jpg",
                raw_metadata=raw_meta
            )
            
            # Assertions on returned record
            assert record["brand"] == "nike"
            assert record["category"] == "hoodies"
            assert "nike_hoodie.jpg" in record["filename"]
            assert record["resolution"] == "512x512"
            
            # Verify file created
            assert (Path(tmpdir) / "nike" / "nike_hoodie.jpg").exists()
            
            # Verify manifest updated
            manifest_path = Path(tmpdir) / "nike_manifest.json"
            assert manifest_path.exists()
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                
            assert "nike_hoodie.jpg" in manifest
            assert manifest["nike_hoodie.jpg"]["category"] == "hoodies"

    def test_ingest_unsupported_brand(self, test_image):
        """Assert raising ValueError for unsupported brand strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            with pytest.raises(ValueError):
                manager.ingest_image(brand="adidas", image=test_image, filename="adidas.jpg")

    def test_ingest_invalid_resolution(self, small_image):
        """Assert raising ValueError for low-resolution inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            with pytest.raises(ValueError):
                manager.ingest_image(brand="gucci", image=small_image, filename="gucci.jpg")

    def test_duplicate_detection(self, test_image):
        """Verify identical image hashes trigger duplication flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            
            # Ingest image 1
            manager.ingest_image(brand="zara", image=test_image, filename="zara_1.jpg")
            
            # Ingest image 2 (exact same pixel bytes)
            record2 = manager.ingest_image(brand="zara", image=test_image, filename="zara_2.jpg")
            
            assert record2["is_duplicate"] is True
            
            duplicates = manager.detect_duplicates(brand="zara")
            assert len(duplicates) == 1
            assert duplicates[0] == ("zara_1.jpg", "zara_2.jpg")

    def test_get_statistics_schema(self, test_image):
        """Verify get_statistics returns the exact dictionary schema structure requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            
            manager.ingest_image(brand="h&m", image=test_image, filename="hm_tee.jpg", raw_metadata={"category": "shirts"})
            manager.ingest_image(brand="h&m", image=test_image, filename="hm_pant.jpg", raw_metadata={"category": "pants"})
            
            stats = manager.get_statistics(brand="h&m")
            
            # Exact schema assertions
            assert stats["brand"] == "h&m"
            assert stats["images"] == 2
            assert sorted(stats["categories"]) == ["pants", "shirts"]

    def test_validate_dataset_auditing(self, test_image):
        """Verify structural auditing catches disk modifications."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            
            manager.ingest_image(brand="nike", image=test_image, filename="nike_fit.jpg")
            
            # Run validate - should find 1 valid file
            res1 = manager.validate_dataset(brand="nike")
            assert res1["total_manifest_records"] == 1
            assert res1["valid_files"] == 1
            assert res1["corrupt_or_missing_files"] == 0
            
            # Physically delete the file
            (Path(tmpdir) / "nike" / "nike_fit.jpg").unlink()
            
            # Run validate again - should catch the deletion
            res2 = manager.validate_dataset(brand="nike")
            assert res2["valid_files"] == 0
            assert res2["corrupt_or_missing_files"] == 1
            assert res2["failures"][0]["filename"] == "nike_fit.jpg"
