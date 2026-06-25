"""
week4/tests/test_style_analyzer.py
==================================
Unit tests for the Brand Style Analysis Engine.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.lora.datasets.brand_dataset_manager import BrandDatasetManager
from src.lora.style_manager.style_analyzer import BrandStyleAnalyzer


class TestBrandStyleAnalyzer:
    """Verify color palettes, silhouettes, design patterns, aesthetics, and registry outputs."""

    @pytest.fixture
    def mock_manager(self):
        """Creates a BrandDatasetManager pointing to a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BrandDatasetManager(dataset_root=tmpdir)
            yield manager

    def test_analyzer_defaults_fallback(self, mock_manager):
        """Verify fallback defaults are returned when manifests are missing."""
        analyzer = BrandStyleAnalyzer(dataset_manager=mock_manager)
        
        # Test Gucci default fallback
        profile = analyzer.analyze_brand(brand="gucci")
        assert profile["style"] == "luxury"
        assert "red" in profile["dominant_colors"]
        assert profile["fit"] == "tailored"
        assert profile["design_language"] == "opulence"

    def test_analyze_brand_with_manifest_parsing(self, mock_manager):
        """Verify that extracting from manifest uses keywords and categories."""
        # Create a mock manifest
        brand_dir = mock_manager.dataset_root / "nike"
        brand_dir.mkdir(parents=True, exist_ok=True)
        
        # Ingest 3 mock images with style indicators in filenames/descriptions
        # Image 1: solid black hoodie
        img_black = Image.new("RGB", (512, 512), color=(0, 0, 0))
        mock_manager.ingest_image(
            brand="nike",
            image=img_black,
            filename="nike_techwear_oversized_hoodie.jpg",
            raw_metadata={
                "category": "hoodies",
                "color": ["black"],
                "description": "Oversized loose technical fit print."
            }
        )
        
        # Image 2: solid white jacket
        img_white = Image.new("RGB", (512, 512), color=(255, 255, 255))
        mock_manager.ingest_image(
            brand="nike",
            image=img_white,
            filename="nike_techwear_oversized_jacket.jpg",
            raw_metadata={
                "category": "jackets",
                "color": ["white"],
                "description": "Oversized technical windbreaker print."
            }
        )

        analyzer = BrandStyleAnalyzer(dataset_manager=mock_manager)
        profile = analyzer.analyze_brand(brand="nike")
        
        # Verified resolved values based on mock items
        assert profile["style"] == "sportswear"
        assert "black" in profile["dominant_colors"] or "white" in profile["dominant_colors"]
        assert profile["silhouette"] == "oversized"
        assert profile["aesthetic"] == "techwear"
        assert profile["pattern"] == "print"
        assert "oversized techwear silhouettes" in profile["style_signature"]

    def test_dominant_color_mapping(self):
        """Verify closest Euclidean distance color mapping works."""
        analyzer = BrandStyleAnalyzer()
        
        # Test solid red mapping
        img_red = Image.new("RGB", (64, 64), color=(255, 5, 5))
        colors = analyzer._get_image_dominant_colors(img_red)
        assert "red" in colors

        # Test solid beige mapping
        img_beige = Image.new("RGB", (64, 64), color=(245, 243, 218))
        colors_beige = analyzer._get_image_dominant_colors(img_beige)
        assert "beige" in colors_beige or "cream" in colors_beige

    def test_generate_profile_registry(self, mock_manager):
        """Verify generate_profile_registry writes a valid consolidated JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "registry_style_profile.json"
            analyzer = BrandStyleAnalyzer(
                dataset_manager=mock_manager,
                profile_output_path=out_file
            )
            
            registry = analyzer.generate_profile_registry(brands=["nike", "gucci"])
            
            assert out_file.exists()
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            assert "nike" in data
            assert "gucci" in data
            assert data["nike"]["style"] == "sportswear"
            assert data["gucci"]["fit"] == "tailored"
