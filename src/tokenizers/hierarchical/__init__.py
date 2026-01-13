"""Hierarchical tokenization module.

This module provides hierarchical graph tokenization based on HiGen's
hierarchical decomposition with SENT-style sequential encoding.

Key components:
- HSENTTokenizer: Main tokenizer class
- SpectralCoarsening: Graph coarsening via spectral clustering
- HierarchicalGraph: Data structure for hierarchical representation
- Visualization utilities for inspecting hierarchies
"""

from src.tokenizers.hierarchical.coarsening import (
    CoarseningStrategy,
    SpectralCoarsening,
)
from src.tokenizers.hierarchical.hsent import HSENTTokenizer
from src.tokenizers.hierarchical.ordering import (
    OrderingMethod,
    compute_canonical_order,
    order_partition_nodes,
)
from src.tokenizers.hierarchical.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
    create_empty_hierarchy,
)
from src.tokenizers.hierarchical.visualization import (
    quick_visualize,
    visualize_block_matrix,
    visualize_graph_communities,
    visualize_hierarchy,
    visualize_tokens,
)

__all__ = [
    # Main tokenizer
    "HSENTTokenizer",
    # Coarsening
    "SpectralCoarsening",
    "CoarseningStrategy",
    # Data structures
    "Partition",
    "Bipartite",
    "HierarchicalGraph",
    "create_empty_hierarchy",
    # Ordering
    "order_partition_nodes",
    "compute_canonical_order",
    "OrderingMethod",
    # Visualization
    "visualize_hierarchy",
    "visualize_graph_communities",
    "visualize_block_matrix",
    "visualize_tokens",
    "quick_visualize",
]
