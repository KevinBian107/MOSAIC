"""Transfer learning module for scaffold priming and complex molecule generation.

This module provides utilities for priming hierarchical tokenizers (HSENT, HDT, HDTC)
with scaffold tokens to enable zero-shot complex molecule generation.

Submodules:
- scaffolds: Scaffold definitions and library
- primers: Tokenizer-specific primer generators
- generation: Primed generation utilities
- datasets: Complex molecule datasets
- evaluation: Priming evaluation utilities
"""

from src.transfer_learning.datasets.complex_molecule_dataset import (
    ComplexMoleculeDataset,
    ComplexMoleculeSample,
)
from src.transfer_learning.evaluation.priming_evaluator import PrimingEvaluator
from src.transfer_learning.evaluation.visualization import (
    create_summary_grid,
    visualize_evaluation_results,
    visualize_priming_comparison,
)
from src.transfer_learning.generation.primed_generator import PrimedGenerator
from src.transfer_learning.primers.factory import PrimerFactory
from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary
from src.transfer_learning.scaffolds.murcko_extractor import MurckoExtractor

__all__ = [
    "ComplexMoleculeDataset",
    "ComplexMoleculeSample",
    "MurckoExtractor",
    "PrimedGenerator",
    "PrimerFactory",
    "PrimingEvaluator",
    "Scaffold",
    "ScaffoldLibrary",
    "create_summary_grid",
    "visualize_evaluation_results",
    "visualize_priming_comparison",
]
