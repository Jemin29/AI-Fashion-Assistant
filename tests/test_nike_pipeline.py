"""
week4/tests/test_nike_pipeline.py
=================================
Unit tests for the Nike-specific LoRA training pipeline.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from src.lora.trainers.train_nike_lora import main, run_nike_training


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for outputs and datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestNikePipeline:
    """Verify Nike training runner configurations, validation flows, and output targets."""

    def test_run_nike_training_dry_run(self, temp_workspace):
        """Verify dry-run mode preprocesses, trains, and exports target weights."""
        res = run_nike_training(
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
        assert "--output_name=nike_style" in cmd_str

        # Verify target file has been written and placed
        final_model_path = Path(res["final_model_path"])
        assert final_model_path.exists()
        assert final_model_path.name == "nike_style.safetensors"

        # Check content of mock weights
        with open(final_model_path, "rb") as f:
            content = f.read()
        assert b"MOCK_LORA_WEIGHTS_KOHYA_SS_WEEK4" in content

    def test_run_nike_training_raw_ingestion(self, temp_workspace):
        """Verify automated preprocessing ingests and registers new raw images."""
        # Create some raw input images to be automatically preprocessed
        raw_nike_dir = temp_workspace / "datasets" / "raw_nike"
        raw_nike_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a raw mock image to ingest
        img = Image.new("RGB", (600, 600), color=(100, 200, 50))
        img_name = "new_nike_shoe.png"
        img.save(raw_nike_dir / img_name)

        # Run pipeline
        res = run_nike_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            dry_run=True
        )

        assert res["success"] is True
        
        # Verify that the manifest lists the ingested file
        manifest_file = temp_workspace / "datasets" / "nike_manifest.json"
        assert manifest_file.exists()
        
        # Check prepared image exists in the Kohya folder structure
        prepared_dir = temp_workspace / "outputs" / "kohya" / "nike" / "img"
        sub_folders = list(prepared_dir.glob("10_nike"))
        assert len(sub_folders) == 1
        assert (sub_folders[0] / img_name).exists()
        assert (sub_folders[0] / f"{Path(img_name).stem}.txt").exists()

    def test_resume_checkpoint_logging(self, temp_workspace):
        """Verify resume checkpoint maps to outputs successfully."""
        mock_checkpoint_path = temp_workspace / "outputs" / "kohya" / "nike" / "checkpoint-10"
        mock_checkpoint_path.mkdir(parents=True, exist_ok=True)

        res = run_nike_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            dry_run=True,
            resume_checkpoint=str(mock_checkpoint_path)
        )
        assert res["success"] is True

    def test_cli_entrypoint(self, temp_workspace):
        """Verify script command-line arguments parser works."""
        test_args = [
            "train_nike_lora.py",
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
            final_model = temp_workspace / "outputs" / "nike_style.safetensors"
            assert final_model.exists()
