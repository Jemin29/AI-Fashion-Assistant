"""
week4/tests/test_kohya_pipeline.py
==================================
Unit tests for the KohyaPipeline training automation workflow.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.utils.config_manager import get_default_config
from src.lora.datasets.brand_dataset_manager import BrandDatasetManager
from src.lora.trainers.kohya_pipeline import KohyaPipeline


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for datasets and pipeline outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_dataset_manager(temp_workspace):
    """Instantiate a BrandDatasetManager referencing the temp workspace."""
    # Write a dummy manifest and image to test preparation
    mgr = BrandDatasetManager(dataset_root=temp_workspace / "datasets")
    
    # Ingest a mock image for Nike
    img = Image.new("RGB", (512, 512), color=(255, 0, 0))
    mgr.ingest_image(
        brand="nike",
        image=img,
        filename="nike_shoes_01.png",
        raw_metadata={
            "category": "shoes",
            "style_tags": ["sportswear"],
            "color": ["black"],
            "description": "Mock athletic shoe."
        }
    )
    return mgr


class TestKohyaPipeline:
    """Verify Kohya dataset prep, captioning, configuration compilation, and dry-run execution."""

    def test_pipeline_initialization(self, temp_workspace, mock_dataset_manager):
        """Verify pipeline root resolves and creates successfully."""
        pipeline = KohyaPipeline(
            dataset_manager=mock_dataset_manager,
            pipeline_root=temp_workspace / "kohya"
        )
        assert pipeline.pipeline_root.exists()
        assert pipeline.pipeline_root == (temp_workspace / "kohya").resolve()

    def test_prepare_dataset_success(self, temp_workspace, mock_dataset_manager):
        """Verify dataset preparation moves files and builds caption siblings."""
        pipeline = KohyaPipeline(
            dataset_manager=mock_dataset_manager,
            pipeline_root=temp_workspace / "kohya"
        )
        
        brand_dir = pipeline.prepare_dataset(brand="nike", repeats=5, activation_word="nike_test")
        
        assert brand_dir.exists()
        img_dir = brand_dir / "img" / "5_nike_test"
        assert img_dir.exists()
        
        # Verify copied image and created caption file
        copied_image = img_dir / "nike_shoes_01.png"
        caption_file = img_dir / "nike_shoes_01.txt"
        
        assert copied_image.exists()
        assert caption_file.exists()
        
        # Check caption content
        with open(caption_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "nike" in content
        assert "shoes" in content

    def test_prepare_dataset_dummy_fallback(self, temp_workspace):
        """Verify fallback behavior when no images exist in the source dataset."""
        # Create empty dataset manager
        mgr = BrandDatasetManager(dataset_root=temp_workspace / "datasets")
        pipeline = KohyaPipeline(
            dataset_manager=mgr,
            pipeline_root=temp_workspace / "kohya"
        )
        
        # Should generate a dummy image automatically to prevent failures
        brand_dir = pipeline.prepare_dataset(brand="gucci", repeats=2)
        img_dir = brand_dir / "img" / "2_gucci"
        
        # Verify dummy file is present
        assert list(img_dir.glob("*.png"))
        assert list(img_dir.glob("*.txt"))

    def test_generate_config(self, temp_workspace, mock_dataset_manager):
        """Verify dynamic configuration JSON structures are type-correct."""
        cfg = get_default_config()
        # Override some config parameters
        cfg.lora.r = 16
        cfg.lora.alpha = 32
        cfg.trainer.batch_size = 2
        cfg.trainer.learning_rate = 5e-5
        
        pipeline = KohyaPipeline(
            config=cfg,
            dataset_manager=mock_dataset_manager,
            pipeline_root=temp_workspace / "kohya"
        )
        
        config_dict = pipeline.generate_config(brand="nike")
        
        # Check overrides mapping
        assert config_dict["network_dim"] == 16
        assert config_dict["network_alpha"] == 32
        assert config_dict["train_batch_size"] == 2
        assert config_dict["learning_rate"] == 5e-5
        assert config_dict["output_name"] == "nike_style_lora"
        
        # Check file output
        config_file = temp_workspace / "kohya" / "nike" / "training_config.json"
        assert config_file.exists()
        
        with open(config_file, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
        assert saved_config["network_dim"] == 16
        assert saved_config["learning_rate"] == 5e-5

    def test_run_training_dry_run(self, temp_workspace, mock_dataset_manager):
        """Verify automated training command compilation and output weights generation."""
        pipeline = KohyaPipeline(
            dataset_manager=mock_dataset_manager,
            pipeline_root=temp_workspace / "kohya"
        )
        
        res = pipeline.run_training(brand="nike", dry_run=True)
        
        assert res["success"] is True
        assert res["dry_run"] is True
        assert "command" in res
        
        # Check CLI arguments
        cmd_str = " ".join(res["command"])
        assert "accelerate launch" in cmd_str
        assert "--pretrained_model_name_or_path" in cmd_str
        assert "--train_data_dir" in cmd_str
        
        # Check mock safetensors output
        output_model = Path(res["output_model"])
        assert output_model.exists()
        with open(output_model, "rb") as f:
            data = f.read()
        assert b"MOCK_LORA_WEIGHTS_KOHYA_SS" in data

    def test_unsupported_brand_errors(self, temp_workspace, mock_dataset_manager):
        """Verify passing an unsupported brand raises a ValueError."""
        pipeline = KohyaPipeline(
            dataset_manager=mock_dataset_manager,
            pipeline_root=temp_workspace / "kohya"
        )
        
        with pytest.raises(ValueError):
            pipeline.prepare_dataset(brand="adidas")
            
        with pytest.raises(ValueError):
            pipeline.generate_config(brand="puma")
