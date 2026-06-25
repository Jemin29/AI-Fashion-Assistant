"""
week4/tests/test_demo_lora.py
=============================
Unit tests for the week4 demo_lora runner.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import demo_lora


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDemoLoraRunner:
    """Verify execution of the main demo runner under mock arguments."""

    def test_demo_runner_execution(self, temp_workspace):
        """Test that the full demo runner executes in dry-run mode and creates reports."""
        output_dir = temp_workspace / "demo_out"
        
        args = ["demo_lora.py", "--output-dir", str(output_dir)]
        with patch("sys.argv", args):
            exit_code = demo_lora.main()
            
        assert exit_code == 0
        
        # Verify the summary file exists
        summary_file = output_dir / "demo_lora_summary.json"
        assert summary_file.exists()
        
        # Verify individual files were generated
        files_list = list(output_dir.glob("*"))
        assert len(files_list) > 2
