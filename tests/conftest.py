"""
conftest.py
===========
pytest configuration and shared session-scoped fixtures.
"""

import sys
from pathlib import Path

# Make sure the project root is on sys.path so imports work from any CWD.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
