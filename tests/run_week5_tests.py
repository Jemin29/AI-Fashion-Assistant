"""
week5/tests/run_week5_tests.py
==============================
Week 5 Standalone Test Runner.
Executes pytest on all week5 unit tests with coverage constraints for week5 modules.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# Ensure project root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    """Run all week 5 tests with coverage and enforce 80% threshold."""
    cmd = [
        sys.executable, "-m", "pytest",
        "week5/tests/",
        "--cov=week5",
        "--cov-fail-under=80",
        "--cov-report=term-missing",
        "--strict-markers",
        "-v"
    ]
    
    print(f"Running Week 5 test runner: {' '.join(cmd)}\n")
    
    # Run pytest process
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    
    if result.returncode == 0:
        print("\n[SUCCESS] All Week 5 unit tests passed and met the 80%+ coverage threshold!")
    else:
        print("\n[FAILURE] Some Week 5 tests failed or coverage threshold was not met.")
        
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
