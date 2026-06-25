"""
week3/tests/test_controlnet_trainer.py
======================================
Unit tests for FashionControlNetTrainer.
Validates trainer initialization, batch collation, dry-run training loop execution,
checkpoint serializations, and validation outputs.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.controlnet.controlnet_trainer import FashionControlNetTrainer


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

class DummyDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset returning dummy images and prompts for trainer tests."""
    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {
            "pixel_values": torch.randn(3, 64, 64),
            "conditioning_pixel_values": torch.randn(3, 64, 64),
            "prompt": f"A trendy outfit sample {idx}"
        }


@pytest.fixture
def dummy_train_ds():
    return DummyDataset(size=4)


@pytest.fixture
def dummy_val_ds():
    return DummyDataset(size=2)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestFashionControlNetTrainer:

    def test_init_defaults(self, dummy_train_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FashionControlNetTrainer(
                train_dataset=dummy_train_ds,
                batch_size=2,
                learning_rate=2e-5,
                output_dir=tmpdir,
                dry_run=True
            )
            assert trainer.batch_size == 2
            assert trainer.learning_rate == 2e-5
            assert trainer.dry_run is True
            assert Path(tmpdir).exists()

    def test_collate_fn(self, dummy_train_ds):
        trainer = FashionControlNetTrainer(train_dataset=dummy_train_ds, dry_run=True)
        examples = [dummy_train_ds[0], dummy_train_ds[1]]
        batch = trainer._collate_fn(examples)
        
        assert "pixel_values" in batch
        assert "conditioning_pixel_values" in batch
        assert "prompts" in batch
        
        # Dimensions check
        assert batch["pixel_values"].shape == (2, 3, 64, 64)
        assert batch["conditioning_pixel_values"].shape == (2, 3, 64, 64)
        assert len(batch["prompts"]) == 2
        assert batch["prompts"][0] == "A trendy outfit sample 0"

    def test_dry_run_training_loop(self, dummy_train_ds, dummy_val_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            trainer = FashionControlNetTrainer(
                train_dataset=dummy_train_ds,
                val_dataset=dummy_val_ds,
                batch_size=2,
                learning_rate=1e-4,
                num_train_epochs=2,
                gradient_accumulation_steps=1,
                output_dir=out_dir,
                validation_steps=2,     # Trigger validation quickly
                checkpointing_steps=2,   # Trigger checkpointing quickly
                dry_run=True
            )
            
            result = trainer.train()
            
            assert result["global_step"] == 4  # 4 samples, batch_size=2 -> 2 steps per epoch -> 2 epochs * 2 steps = 4 steps
            assert result["average_loss"] > 0.0
            assert result["output_dir"] == out_dir
            
            # Check checkpoint subdirectories are saved
            checkpoint_dir = out_dir / "checkpoint-2"
            assert checkpoint_dir.exists()
            assert (checkpoint_dir / "mock_model.bin").exists()
            
            # Check final model subdirectory is saved
            final_dir = out_dir / "final_controlnet"
            assert final_dir.exists()
            assert (final_dir / "mock_model.bin").exists()
            
            # Check validation image directory and outputs
            val_dir = out_dir / "validation"
            assert val_dir.exists()
            assert (val_dir / "step_2.png").exists()
            assert (val_dir / "step_4.png").exists()
