"""Scaffold definitions and library for priming.

This submodule provides scaffold patterns organized by complexity tiers
and a library class for managing scaffolds.
"""

from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary
from src.transfer_learning.scaffolds.tier_patterns import (
    TIER1_SCAFFOLDS,
    TIER2_SCAFFOLDS,
    TIER3_SCAFFOLDS,
)

__all__ = [
    "Scaffold",
    "ScaffoldLibrary",
    "TIER1_SCAFFOLDS",
    "TIER2_SCAFFOLDS",
    "TIER3_SCAFFOLDS",
]
