"""
week4/tests/run_week4_tests.py
==============================
Week 4 Standalone Test Runner.
Executes pytest on all week4 unit tests with coverage constraints for week4 modules.
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
    """Run all week 4 tests with coverage and enforce 80% threshold."""
    cmd = [
        sys.executable, "-m", "pytest",
        "week4/tests/",
        "--cov=week4",
        "--cov-fail-under=80",
        "--cov-report=term-missing",
        "--strict-markers",
        "-v"
    ]
    
    print(f"Running Week 4 test runner: {' '.join(cmd)}\n")
    
    # Run pytest process
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    
    if result.returncode == 0:
        print("\n[SUCCESS] All Week 4 unit tests passed and met the 80%+ coverage threshold!")
    else:
        print("\n[FAILURE] Some Week 4 tests failed or coverage threshold was not met.")
        
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
