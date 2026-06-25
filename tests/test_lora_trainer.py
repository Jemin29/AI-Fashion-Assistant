"""
week4/tests/test_lora_trainer.py
================================
Unit tests for the LoraTrainer framework.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.lora.trainers.lora_trainer import LoraTrainer


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

class DummyLoraDataset(torch.utils.data.Dataset):
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
    return DummyLoraDataset(size=4)


@pytest.fixture
def dummy_val_ds():
    return DummyLoraDataset(size=2)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestLoraTrainer:
    """Verify LoRA training dry-runs, checkpoints serialization, and validation audits."""

    def test_init_defaults(self, dummy_train_ds):
        """Verify default assignments work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LoraTrainer(
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
        """Verify batch collation handles shapes and labels correctly."""
        trainer = LoraTrainer(train_dataset=dummy_train_ds, dry_run=True)
        examples = [dummy_train_ds[0], dummy_train_ds[1]]
        batch = trainer._collate_fn(examples)
        
        assert "pixel_values" in batch
        assert "conditioning_pixel_values" in batch
        assert "prompts" in batch
        
        # Dimensions checks
        assert batch["pixel_values"].shape == (2, 3, 64, 64)
        assert batch["conditioning_pixel_values"].shape == (2, 3, 64, 64)
        assert len(batch["prompts"]) == 2
        assert batch["prompts"][0] == "A trendy outfit sample 0"

    def test_dry_run_training_loop(self, dummy_train_ds, dummy_val_ds):
        """Verify that training compiles and runs through epochs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            trainer = LoraTrainer(
                train_dataset=dummy_train_ds,
                val_dataset=dummy_val_ds,
                batch_size=2,
                learning_rate=1e-4,
                num_epochs=2,
                validation_steps=2,
                checkpointing_steps=2,
                output_dir=out_dir,
                dry_run=True
            )
            
            stats = trainer.train()
            assert stats["global_step"] == 4  # 2 epochs * (4 samples / batch_size 2) = 4 steps
            
            # Check validation output files
            assert (out_dir / "validation_samples" / "step_2.png").exists()
            
            # Check checkpoint output files
            assert (out_dir / "checkpoint-2").exists()
            assert (out_dir / "checkpoint-4").exists()
            
            # Check final weights output
            assert (out_dir / "final_lora" / "mock_model.pt").exists()

    def test_checkpoint_saving_and_loading(self, dummy_train_ds):
        """Verify mock checkpoint serialization and deserialization."""
        from accelerate import Accelerator
        acc = Accelerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            trainer = LoraTrainer(
                train_dataset=dummy_train_ds,
                output_dir=out_dir,
                dry_run=True
            )
            
            from src.lora.trainers.lora_trainer import MockLoraModel
            model = MockLoraModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Save step 10 checkpoint
            trainer.save_checkpoint(step=10, accelerator=acc, model=model)
            
            checkpoint_folder = out_dir / "checkpoint-10"
            assert checkpoint_folder.exists()
            assert (checkpoint_folder / "mock_model.pt").exists()
            assert (checkpoint_folder / "checkpoint_meta.json").exists()
            
            # Load back checkpoint
            step, epoch = trainer.load_checkpoint(checkpoint_folder, acc, optimizer)
            assert step == 10
            assert epoch == 1
