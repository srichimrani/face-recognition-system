"""
Compare Teachable Machine vs best classical ML on processed test folder.
Run after both models are available.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_testing_suite import run_tests

if __name__ == "__main__":
    run_tests()
