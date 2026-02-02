"""Transfer learning module for scaffold priming and complex molecule generation.

This module provides utilities for priming hierarchical tokenizers (HSENT, HDT, HDTC)
with scaffold tokens to enable zero-shot complex molecule generation.

Submodules:
- scaffolds: Scaffold definitions and library
- primers: Tokenizer-specific primer generators
- generation: Primed generation utilities
"""

from src.transfer_learning.generation.primed_generator import PrimedGenerator
from src.transfer_learning.primers.factory import PrimerFactory
from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary

__all__ = [
    "Scaffold",
    "ScaffoldLibrary",
    "PrimerFactory",
    "PrimedGenerator",
]
