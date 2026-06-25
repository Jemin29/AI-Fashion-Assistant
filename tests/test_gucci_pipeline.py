"""
week4/tests/test_gucci_pipeline.py
==================================
Unit tests for the Gucci-specific LoRA training pipeline.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from src.lora.trainers.train_gucci_lora import main, run_gucci_training


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for outputs and datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestGucciPipeline:
    """Verify Gucci training runner configurations, validation flows, and output targets."""

    def test_run_gucci_training_dry_run(self, temp_workspace):
        """Verify dry-run mode preprocesses, trains, validates, and exports target weights."""
        res = run_gucci_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            epochs=2,
            batch_size=2,
            learning_rate=2e-4,
            dry_run=True
        )

        assert res["success"] is True
        assert res["dry_run"] is True
        assert "command" in res
        
        # Verify CLI args custom inputs mapped correctly
        cmd_str = " ".join(res["command"])
        assert "--max_train_epochs=2" in cmd_str
        assert "--train_batch_size=2" in cmd_str
        assert "--learning_rate=0.0002" in cmd_str
        assert "--output_name=gucci_style" in cmd_str

        # Verify target weights file has been written and placed
        final_model_path = Path(res["final_model_path"])
        assert final_model_path.exists()
        assert final_model_path.name == "gucci_style.safetensors"

        # Verify validation output exists
        assert "validation_sample" in res
        assert Path(res["validation_sample"]).exists()

    def test_run_gucci_training_raw_ingestion(self, temp_workspace):
        """Verify automated preprocessing ingests and registers new raw images."""
        # Create some raw input images to be automatically preprocessed
        raw_gucci_dir = temp_workspace / "datasets" / "raw_gucci"
        raw_gucci_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a raw mock image to ingest
        img = Image.new("RGB", (600, 600), color=(150, 50, 100))
        img_name = "new_gucci_dress.png"
        img.save(raw_gucci_dir / img_name)

        # Run pipeline
        res = run_gucci_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            dry_run=True
        )

        assert res["success"] is True
        
        # Verify that the manifest lists the ingested file
        manifest_file = temp_workspace / "datasets" / "gucci_manifest.json"
        assert manifest_file.exists()
        
        # Check prepared image exists in the Kohya folder structure
        prepared_dir = temp_workspace / "outputs" / "kohya" / "gucci" / "img"
        sub_folders = list(prepared_dir.glob("10_gucci"))
        assert len(sub_folders) == 1
        assert (sub_folders[0] / img_name).exists()
        assert (sub_folders[0] / f"{Path(img_name).stem}.txt").exists()

    def test_resume_checkpoint_logging(self, temp_workspace):
        """Verify resume checkpoint maps to outputs successfully."""
        mock_checkpoint_path = temp_workspace / "outputs" / "kohya" / "gucci" / "checkpoint-10"
        mock_checkpoint_path.mkdir(parents=True, exist_ok=True)

        res = run_gucci_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            dry_run=True,
            resume_checkpoint=str(mock_checkpoint_path)
        )
        assert res["success"] is True

    def test_cli_entrypoint(self, temp_workspace):
        """Verify script command-line arguments parser works."""
        test_args = [
            "train_gucci_lora.py",
            "--output-dir", str(temp_workspace / "outputs"),
            "--dataset-root", str(temp_workspace / "datasets"),
            "--epochs", "3",
            "--batch-size", "2",
            "--lr", "1e-5"
        ]
        
        with patch.object(sys, "argv", test_args):
            exit_code = main()
            assert exit_code == 0
            
            # Verify target file has been written
            final_model = temp_workspace / "outputs" / "gucci_style.safetensors"
            assert final_model.exists()
