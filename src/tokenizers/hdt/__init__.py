"""Hierarchical DFS-based Tokenizer (HDT).

This module provides the HDT tokenization scheme that achieves ~45% token
reduction over H-SENT by using DFS traversal with implicit hierarchy encoding.
"""

from src.tokenizers.hdt.tokenizer import HDTTokenizer

__all__ = ["HDTTokenizer"]
