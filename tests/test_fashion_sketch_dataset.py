"""
tests/test_fashion_sketch_dataset.py
====================================
Unit tests for FashionSketchDataset.
Checks loading, splitting, matched augmentations, dynamic preprocessors, and error robustness.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.fashion_sketch_dataset import FashionSketchDataset


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def mock_dataset_records():
    """Create a list of 10 mock records."""
    return [
        {
            "image_id": f"IMG_{i:04d}",
            "image_path": f"designs/IMG_{i:04d}.jpg",
            "sketch_path": f"sketches/IMG_{i:04d}.png",
            "prompt": f"A beautiful dress item {i}",
            "category": "dresses",
            "style": "luxury",
            "color": ["Red"]
        }
        for i in range(10)
    ]


@pytest.fixture
def mock_filesystem():
    """Sets up a temporary directory with design images, sketch images, and a manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        design_dir = tmp_path / "designs"
        sketch_dir = tmp_path / "sketches"
        design_dir.mkdir()
        sketch_dir.mkdir()

        # Create dummy images
        for i in range(10):
            img_name = f"IMG_{i:04d}"
            # Design image (solid red)
            design_img = Image.new("RGB", (128, 128), color=(255, 0, 0))
            design_img.save(design_dir / f"{img_name}.jpg")
            
            # Sketch image (solid white)
            sketch_img = Image.new("RGB", (128, 128), color=(255, 255, 255))
            sketch_img.save(sketch_dir / f"{img_name}.png")

        # Create manifest
        records = [
            {
                "image_id": f"IMG_{i:04d}",
                "image_path": f"designs/IMG_{i:04d}.jpg",
                "sketch_path": f"sketches/IMG_{i:04d}.png",
                "prompt": f"A beautiful dress item {i}",
                "category": "dresses",
                "style": "luxury",
                "color": ["Red"]
            }
            for i in range(10)
        ]
        
        manifest_data = {
            "_meta": {"total_records": 10},
            "records": records
        }
        
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)
            
        yield tmp_path, manifest_path, design_dir, sketch_dir


# =============================================================================
# ── Test Suite
# =============================================================================

class TestFashionSketchDataset:

    def test_init_raises_on_invalid_split(self):
        with pytest.raises(ValueError, match="Invalid split name"):
            FashionSketchDataset(manifest_path=[], split="invalid_split")

    def test_init_raises_on_missing_sources(self):
        with pytest.raises(ValueError, match="Must specify either manifest_path or design_dir"):
            FashionSketchDataset(manifest_path=None, design_dir=None)

    def test_deterministic_splitting(self, mock_dataset_records):
        # 10 records with 80% split ratio -> train has 8, val has 2
        train_ds = FashionSketchDataset(
            manifest_path=mock_dataset_records,
            split="train",
            split_ratio=0.8,
            seed=42
        )
        val_ds = FashionSketchDataset(
            manifest_path=mock_dataset_records,
            split="val",
            split_ratio=0.8,
            seed=42
        )
        
        assert len(train_ds) == 8
        assert len(val_ds) == 2
        
        # Verify no overlap between train and val splits
        train_ids = {r["image_id"] for r in train_ds.records}
        val_ids = {r["image_id"] for r in val_ds.records}
        assert train_ids.isdisjoint(val_ids)
        assert len(train_ids.union(val_ids)) == 10

    def test_explicit_split_metadata(self):
        records = [
            {"image_id": "1", "image_path": "a.jpg", "split": "train"},
            {"image_id": "2", "image_path": "b.jpg", "split": "train"},
            {"image_id": "3", "image_path": "c.jpg", "split": "val"},
            {"image_id": "4", "image_path": "d.jpg", "split": "validation"},
        ]
        
        train_ds = FashionSketchDataset(manifest_path=records, split="train")
        val_ds = FashionSketchDataset(manifest_path=records, split="val")
        
        assert len(train_ds) == 2
        assert len(val_ds) == 2
        assert train_ds.records[0]["image_id"] == "1"
        assert val_ds.records[0]["image_id"] == "3"

    def test_paired_loading_from_manifest(self, mock_filesystem):
        root_path, manifest_path, design_dir, sketch_dir = mock_filesystem
        
        ds = FashionSketchDataset(
            manifest_path=manifest_path,
            design_dir=design_dir,
            sketch_dir=sketch_dir,
            split="train",
            split_ratio=1.0,
            target_size=(256, 256)
        )
        
        assert len(ds) == 10
        item = ds[0]
        
        assert "pixel_values" in item
        assert "conditioning_pixel_values" in item
        assert "prompt" in item
        assert "metadata" in item
        
        # Check shapes
        assert item["pixel_values"].shape == (3, 256, 256)
        assert item["conditioning_pixel_values"].shape == (3, 256, 256)
        
        # Check normalization ranges
        # design target image: [-1.0, 1.0]
        assert item["pixel_values"].min() >= -1.0
        assert item["pixel_values"].max() <= 1.0
        
        # sketch conditioning image: [0.0, 1.0]
        assert item["conditioning_pixel_values"].min() >= 0.0
        assert item["conditioning_pixel_values"].max() <= 1.0

    def test_directory_scanning(self, mock_filesystem):
        root_path, _, design_dir, sketch_dir = mock_filesystem
        
        # Initialize by scanning design_dir and sketch_dir directly
        ds = FashionSketchDataset(
            manifest_path=None,
            design_dir=design_dir,
            sketch_dir=sketch_dir,
            split="train",
            split_ratio=1.0
        )
        
        assert len(ds) == 10
        # Ensure it matched paired files
        assert ds.records[0]["sketch_path"] is not None
        assert ds[0]["prompt"] != ""

    def test_dynamic_sketch_extraction(self, mock_filesystem):
        root_path, _, design_dir, _ = mock_filesystem
        
        # Initialize WITHOUT sketch_dir, forcing dynamic edge extraction on the fly
        ds = FashionSketchDataset(
            manifest_path=None,
            design_dir=design_dir,
            sketch_dir=None,
            split="train",
            split_ratio=1.0,
            edge_method="canny"
        )
        
        item = ds[0]
        assert item["conditioning_pixel_values"] is not None
        assert item["conditioning_pixel_values"].shape == (3, 1024, 1024)

    def test_paired_augmentations(self, mock_filesystem):
        root_path, manifest_path, design_dir, sketch_dir = mock_filesystem
        
        # Enable augmentations
        ds = FashionSketchDataset(
            manifest_path=manifest_path,
            design_dir=design_dir,
            sketch_dir=sketch_dir,
            split="train",
            split_ratio=1.0,
            augment=True,
            target_size=(128, 128)
        )
        
        # Load item and verify shape remains standard target size
        item = ds[0]
        assert item["pixel_values"].shape == (3, 128, 128)
        assert item["conditioning_pixel_values"].shape == (3, 128, 128)

    def test_getitem_robustness_on_failure(self, mock_filesystem):
        root_path, manifest_path, design_dir, sketch_dir = mock_filesystem
        
        ds = FashionSketchDataset(
            manifest_path=manifest_path,
            design_dir=design_dir,
            sketch_dir=sketch_dir,
            split="train",
            split_ratio=1.0,
            target_size=(128, 128)
        )
        
        # Delete one of the design images so it fails loading
        bad_image_path = design_dir / "IMG_0000.jpg"
        if bad_image_path.exists():
            bad_image_path.unlink()
            
        # Access index 0, which points to the deleted image.
        # It should catch the failure, log a warning, and return a fallback item instead of crashing.
        item = ds[0]
        assert item is not None
        assert item["pixel_values"].shape == (3, 128, 128)
