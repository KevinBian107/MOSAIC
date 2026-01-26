"""Realistic generation module.

This module provides functionality for unconditional graph generation and
analysis of generation quality by comparing structural patterns between
generated molecules and training data.

Key exports:
    - generate_molecules: Unconditional molecule generation
    - analyze_benzene_substitution: Analyze substitution patterns on benzene
    - analyze_functional_groups: Analyze functional group co-occurrence
    - compare_distributions: Compute distribution similarity metrics
    - plot_combined_analysis: Visualize training vs generated comparison
    - draw_molecule_comparison: Draw actual molecular structures for comparison
"""

from src.realistic_gen.analysis import (
    analyze_benzene_substitution,
    analyze_functional_groups,
    compare_distributions,
    draw_molecule_comparison,
    plot_combined_analysis,
    plot_functional_group_comparison,
    plot_substitution_comparison,
)
from src.realistic_gen.generator import generate_molecules

__all__ = [
    # Generation
    "generate_molecules",
    # Analysis functions
    "analyze_benzene_substitution",
    "analyze_functional_groups",
    "compare_distributions",
    # Visualization
    "plot_substitution_comparison",
    "plot_functional_group_comparison",
    "plot_combined_analysis",
    "draw_molecule_comparison",
]
