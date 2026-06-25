"""
week4/tests/test_zara_pipeline.py
=================================
Unit tests for the Zara-specific LoRA training pipeline and style evaluator.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from src.evaluation.week4_style_evaluator import (
    compute_clip_similarity,
    compute_color_alignment,
    compute_structural_similarity,
)
from src.lora.trainers.train_zara_lora import main, run_zara_training


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for outputs and datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestZaraPipeline:
    """Verify Zara training configurations, validation flows, and output targets."""

    def test_style_evaluator_metrics(self):
        """Verify color alignment, SSIM, and CLIP similarity calculations."""
        # Create solid beige image
        img_beige = Image.new("RGB", (100, 100), color=(245, 245, 220))
        # Create solid red image
        img_red = Image.new("RGB", (100, 100), color=(255, 0, 0))

        # Zara palette: beige, black, cream
        zara_palette = ["beige", "black", "cream"]
        
        # Beige image should align 100% (1.0) with Zara palette
        beige_align = compute_color_alignment(img_beige, zara_palette)
        assert beige_align == 1.0

        # Red image should align 0% (0.0) with Zara palette
        red_align = compute_color_alignment(img_red, zara_palette)
        assert red_align == 0.0

        # SSIM between identical images should be 1.0
        ssim_self = compute_structural_similarity(img_beige, img_beige)
        assert ssim_self == 1.0

        # CLIP prompt similarity should run deterministically
        clip_score = compute_clip_similarity(img_beige, "A trendy Zara outfit design.")
        assert 0.25 <= clip_score <= 0.35

    def test_run_zara_training_dry_run(self, temp_workspace):
        """Verify dry-run mode preprocesses, trains, evaluates, and exports target weights."""
        res = run_zara_training(
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
        assert "--output_name=zara_style" in cmd_str

        # Verify target weights file has been written and placed
        final_model_path = Path(res["final_model_path"])
        assert final_model_path.exists()
        assert final_model_path.name == "zara_style.safetensors"

        # Verify evaluation report file exists
        eval_report_file = temp_workspace / "outputs" / "zara_evaluation.json"
        assert eval_report_file.exists()

        # Check content of evaluation metrics
        assert "evaluation_metrics" in res
        metrics = res["evaluation_metrics"]
        assert "color_palette_alignment" in metrics
        assert "structural_similarity_ssim" in metrics
        assert "prompt_similarity_clip" in metrics

    def test_run_zara_training_raw_ingestion(self, temp_workspace):
        """Verify automated preprocessing ingests and registers new raw images."""
        # Create some raw input images to be automatically preprocessed
        raw_zara_dir = temp_workspace / "datasets" / "raw_zara"
        raw_zara_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a raw mock image to ingest
        img = Image.new("RGB", (600, 600), color=(245, 245, 220))
        img_name = "new_zara_jacket.png"
        img.save(raw_zara_dir / img_name)

        # Run pipeline
        res = run_zara_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            dry_run=True
        )

        assert res["success"] is True
        
        # Verify that the manifest lists the ingested file
        manifest_file = temp_workspace / "datasets" / "zara_manifest.json"
        assert manifest_file.exists()
        
        # Check prepared image exists in the Kohya folder structure
        prepared_dir = temp_workspace / "outputs" / "kohya" / "zara" / "img"
        sub_folders = list(prepared_dir.glob("10_zara"))
        assert len(sub_folders) == 1
        assert (sub_folders[0] / img_name).exists()
        assert (sub_folders[0] / f"{Path(img_name).stem}.txt").exists()

    def test_resume_checkpoint_logging(self, temp_workspace):
        """Verify resume checkpoint maps to outputs successfully."""
        mock_checkpoint_path = temp_workspace / "outputs" / "kohya" / "zara" / "checkpoint-10"
        mock_checkpoint_path.mkdir(parents=True, exist_ok=True)

        res = run_zara_training(
            output_dir=temp_workspace / "outputs",
            dataset_root=temp_workspace / "datasets",
            dry_run=True,
            resume_checkpoint=str(mock_checkpoint_path)
        )
        assert res["success"] is True

    def test_cli_entrypoint(self, temp_workspace):
        """Verify script command-line arguments parser works."""
        test_args = [
            "train_zara_lora.py",
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
            final_model = temp_workspace / "outputs" / "zara_style.safetensors"
            assert final_model.exists()
