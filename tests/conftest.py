"""Pytest configuration and shared fixtures.

This module imports all fixtures from the fixtures submodule to make them
available to all test modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.graphs import *
