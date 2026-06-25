"""
week4/tests/test_logging.py
===========================
Unit tests verifying log creation and multi-sink filters routing.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from loguru import logger

from src.utils.logging_setup import setup_logging


class TestWeek4LoggingSetup:
    """Validate Loguru directory creations and sink routing filters."""

    def test_log_directory_creation_and_sinks(self):
        """Verify that initializing loguru creates directories and sinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            
            # Setup logging sinks
            setup_logging(log_dir=log_path, level="DEBUG")
            
            # Log different levels
            logger.debug("Debug test message")
            logger.info("Info test message")
            logger.warning("Warning test message")
            logger.error("Error test message")
            
            # Log training specific messages
            logger.info("Step 10 | Loss: 0.423 | Epoch 1/10")
            logger.info("Adapter injected successfully")
            
            # Wait for file writes (Loguru writes asynchronously in some pools, though usually sync)
            time.sleep(0.5)
            
            # Verify files were created
            general_log = log_path / "week4_general.log"
            error_log = log_path / "errors.log"
            trainer_log = log_path / "lora_trainer.log"
            
            assert general_log.exists()
            assert error_log.exists()
            assert trainer_log.exists()
            
            # Read files to assert routing
            with open(general_log, "r", encoding="utf-8") as f:
                general_content = f.read()
                
            with open(error_log, "r", encoding="utf-8") as f:
                error_content = f.read()
                
            with open(trainer_log, "r", encoding="utf-8") as f:
                trainer_content = f.read()
                
            # Assertions
            # General log contains everything
            assert "Debug test message" in general_content
            assert "Error test message" in general_content
            assert "Loss: 0.423" in general_content
            
            # Error log contains only error
            assert "Error test message" in error_content
            assert "Debug test message" not in error_content
            assert "Loss: 0.423" not in error_content
            
            # Trainer log contains epoch and adapter logs, but NOT standard logs
            assert "Loss: 0.423" in trainer_content
            assert "Adapter injected" in trainer_content
            assert "Debug test message" not in trainer_content
            assert "Error test message" not in trainer_content
            
            # Clear loguru handlers to release files on Windows
            logger.remove()
