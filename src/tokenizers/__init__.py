"""Graph tokenization schemes for MOSAIC.

This package provides tokenization strategies for converting graphs to token
sequences suitable for transformer-based generation.

Tokenizers:
- SENTTokenizer: Flat SENT tokenization (AutoGraph)
- HSENTTokenizer: Hierarchical SENT with explicit structure
- HDTTokenizer: Hierarchical DFS-based tokenization (most efficient)

Coarsening strategies:
- SpectralCoarsening: Modularity-optimized spectral clustering
- MotifAwareSpectralCoarsening: Spectral clustering with motif preservation

Shared utilities:
- structures: Partition, Bipartite, HierarchicalGraph data classes
- ordering: Node ordering strategies (BFS, DFS, etc.)
- visualization: Hierarchy visualization utilities
- motif: Motif detection and affinity computation
"""

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.coarsening import (
    CoarseningStrategy,
    MotifAwareCoarsening,
    MotifAwareSpectralCoarsening,
    MotifCommunityCoarsening,
    SpectralCoarsening,
)
from src.tokenizers.hdt import HDTTokenizer
from src.tokenizers.hsent import HSENTTokenizer
from src.tokenizers.ordering import OrderingMethod, order_partition_nodes
from src.tokenizers.sent import SENTTokenizer
from src.tokenizers.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
)

__all__ = [
    # Base
    "Tokenizer",
    "BatchConverter",
    # Tokenizers
    "SENTTokenizer",
    "HSENTTokenizer",
    "HDTTokenizer",
    # Coarsening
    "CoarseningStrategy",
    "SpectralCoarsening",
    "MotifAwareSpectralCoarsening",
    "MotifAwareCoarsening",  # Backwards compatibility
    "MotifCommunityCoarsening",
    # Structures
    "Partition",
    "Bipartite",
    "HierarchicalGraph",
    # Ordering
    "OrderingMethod",
    "order_partition_nodes",
]
