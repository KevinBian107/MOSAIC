"""HDTC (Hierarchical DFS-based Tokenizer Compositional) module.

This module provides the HDTCTokenizer for tokenizing molecular graphs
using a two-level functional hierarchy representation.

The tokenizer builds a two-level hierarchy:
- Level 1: Functional communities (rings, functional groups, singletons)
- Level 2: Super-graph showing how communities connect

Key components:
- HDTCTokenizer: Main tokenizer class
- FunctionalHierarchyBuilder: Builds two-level hierarchy (in coarsening module)
- TwoLevelHierarchy: Data structure for hierarchy (in structures module)
"""

from src.tokenizers.hdtc.tokenizer import HDTCTokenizer

__all__ = ["HDTCTokenizer"]
