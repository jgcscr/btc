"""Pytest configuration to ensure the project package is importable.

This adds the repository root (which contains the `src/` package) to
`sys.path` so imports like `from src.data...` work reliably when tests
are run from the project root.
"""

import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
