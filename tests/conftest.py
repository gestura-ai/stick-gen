"""Pytest configuration for stick-gen tests.

This file is automatically loaded by pytest and sets up the Python path
to allow imports from the src directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

