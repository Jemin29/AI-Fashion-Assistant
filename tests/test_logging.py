"""
week5/tests/test_logging.py
===========================
Unit tests verifying log creation and multi-sink filters routing for Week 5.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from loguru import logger

from src.utils.logging_setup import setup_logging


class TestWeek5LoggingSetup:
    """Validate Loguru directory creations and sink routing filters for Week 5."""

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
            
            # Log RAG and retrieval specific messages
            logger.info("Running FAISS vector query for black hoodies")
            logger.info("Retrieved 5 similarity matches from index database")
            
            # Wait for file writes (Loguru writes asynchronously in some pools)
            time.sleep(0.5)
            
            # Verify files were created
            general_log = log_path / "week5_general.log"
            error_log = log_path / "errors.log"
            retrieval_log = log_path / "rag_retrieval.log"
            
            assert general_log.exists()
            assert error_log.exists()
            assert retrieval_log.exists()
            
            # Read files to assert routing
            with open(general_log, "r", encoding="utf-8") as f:
                general_content = f.read()
                
            with open(error_log, "r", encoding="utf-8") as f:
                error_content = f.read()
                
            with open(retrieval_log, "r", encoding="utf-8") as f:
                retrieval_content = f.read()
                
            # Assertions
            # General log contains everything
            assert "Debug test message" in general_content
            assert "Error test message" in general_content
            assert "FAISS vector query" in general_content
            
            # Error log contains only error
            assert "Error test message" in error_content
            assert "Debug test message" not in error_content
            assert "FAISS vector query" not in error_content
            
            # Retrieval log contains query and matches logs, but NOT standard logs
            assert "FAISS vector query" in retrieval_content
            assert "Retrieved 5 similarity matches" in retrieval_content
            assert "Debug test message" not in retrieval_content
            assert "Error test message" not in retrieval_content
            
            # Clear loguru handlers to release file handles on Windows
            logger.remove()
