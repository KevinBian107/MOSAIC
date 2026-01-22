"""Graph coarsening strategies for hierarchical decomposition.

This module provides coarsening algorithms that partition graphs into
communities for hierarchical representation.

Available strategies:
- SpectralCoarsening: Modularity-optimized spectral clustering
- MotifAwareSpectralCoarsening: Spectral clustering with motif preservation
- MotifCommunityCoarsening: Direct motif-based community assignment

Usage:
    from src.tokenizers.coarsening import SpectralCoarsening

    coarsener = SpectralCoarsening(min_community_size=4)
    hg = coarsener.build_hierarchy(data)
"""

from src.tokenizers.coarsening.base import CoarseningStrategy
from src.tokenizers.coarsening.motif_aware_spectral import (
    MotifAwareCoarsening,  # Backwards compatibility alias
    MotifAwareSpectralCoarsening,
)
from src.tokenizers.coarsening.motif_community import MotifCommunityCoarsening
from src.tokenizers.coarsening.spectral import SpectralCoarsening

__all__ = [
    # Protocol
    "CoarseningStrategy",
    # Strategies
    "SpectralCoarsening",
    "MotifAwareSpectralCoarsening",
    "MotifAwareCoarsening",  # Backwards compatibility
    "MotifCommunityCoarsening",
]
