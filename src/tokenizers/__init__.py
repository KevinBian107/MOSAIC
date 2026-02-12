"""Graph tokenization schemes for MOSAIC.

This package provides tokenization strategies for converting graphs to token
sequences suitable for transformer-based generation.

Tokenizers:
- SENTTokenizer: Flat SENT tokenization (AutoGraph)
- HSENTTokenizer: Hierarchical SENT with explicit structure
- HDTTokenizer: Hierarchical DFS-based tokenization (most efficient)
- HDTCTokenizer: Compositional tokenization with functional hierarchy

Coarsening strategies:
- SpectralCoarsening: Modularity-optimized spectral clustering (multi-level)
- SimpleSpectralCoarsening: Single-level spectral clustering
- MotifAwareSpectralCoarsening: Spectral clustering with motif preservation
- AffinityCoarsening: Boruvka-based affinity clustering with modularity-optimal cut
- FunctionalHierarchyBuilder: Two-level functional hierarchy for HDTC

Shared utilities:
- structures: Partition, Bipartite, HierarchicalGraph, TwoLevelHierarchy data classes
- ordering: Node ordering strategies (BFS, DFS, etc.)
- visualization: Hierarchy visualization utilities
- motif: Motif detection and affinity computation
"""

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.coarsening import (
    AffinityCoarsening,
    CoarseningStrategy,
    FunctionalHierarchyBuilder,
    HACCoarsening,
    MotifAwareCoarsening,
    MotifAwareSpectralCoarsening,
    MotifCommunityCoarsening,
    SimpleSpectralCoarsening,
    SpectralCoarsening,
)
from src.tokenizers.hdt import HDTTokenizer
from src.tokenizers.hdtc import HDTCTokenizer
from src.tokenizers.hsent import HSENTTokenizer
from src.tokenizers.ordering import OrderingMethod, order_partition_nodes
from src.tokenizers.sent import SENTTokenizer
from src.tokenizers.structures import (
    Bipartite,
    CommunityCommunityEdge,
    FunctionalCommunity,
    HierarchicalGraph,
    Partition,
    TwoLevelHierarchy,
)

__all__ = [
    # Base
    "Tokenizer",
    "BatchConverter",
    # Tokenizers
    "SENTTokenizer",
    "HSENTTokenizer",
    "HDTTokenizer",
    "HDTCTokenizer",
    # Coarsening
    "CoarseningStrategy",
    "SpectralCoarsening",
    "SimpleSpectralCoarsening",
    "MotifAwareSpectralCoarsening",
    "MotifAwareCoarsening",  # Backwards compatibility
    "MotifCommunityCoarsening",
    "AffinityCoarsening",
    "HACCoarsening",  # Backwards compatibility alias
    "FunctionalHierarchyBuilder",
    # Structures
    "Partition",
    "Bipartite",
    "HierarchicalGraph",
    "FunctionalCommunity",
    "CommunityCommunityEdge",
    "TwoLevelHierarchy",
    # Ordering
    "OrderingMethod",
    "order_partition_nodes",
]
