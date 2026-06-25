"""
week5/tests/test_demo_rag.py
============================
Unit tests for the Week 5 Demo Application.
Asserts successful execution and generation of the report output files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.utils.config_manager import get_default_config


class TestDemoRag:
    """Validate demo execution, returncode, and file output generations."""

    def test_demo_execution_and_reports(self):
        """Execute demo_rag.py and assert report files exist and are not empty."""
        config = get_default_config()

        # Target report paths
        report_dir = Path("outputs").resolve()
        eval_report_path = report_dir / "evaluation_report.json"
        exp_runs_path = report_dir / "experiment_runs.json"

        # Delete files if they already exist to ensure fresh generation
        if eval_report_path.exists():
            eval_report_path.unlink()
        if exp_runs_path.exists():
            exp_runs_path.unlink()

        # Run demo_rag.py as a subprocess
        cmd = [sys.executable, str(Path("scripts") / "demo_rag.py")]
        repo_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        
        result = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
        
        # Verify run success
        assert result.returncode == 0, f"Demo run failed: {result.stderr}"

        # Verify reports are generated
        assert eval_report_path.exists(), "evaluation_report.json was not generated."
        assert exp_runs_path.exists(), "experiment_runs.json was not generated."

        # Verify reports contain data
        assert eval_report_path.stat().st_size > 0, "evaluation_report.json is empty."
        assert exp_runs_path.stat().st_size > 0, "experiment_runs.json is empty."
