"""Base protocol for graph coarsening strategies.

This module defines the CoarseningStrategy protocol that all coarsening
implementations must follow, enabling interchangeable coarsening approaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from src.tokenizers.structures import HierarchicalGraph


@runtime_checkable
class CoarseningStrategy(Protocol):
    """Protocol for graph coarsening strategies.

    All coarsening implementations must provide methods to partition
    a graph into communities and build hierarchical representations.
    This allows swapping between spectral, motif-based, or hybrid approaches.

    Attributes:
        min_community_size: Minimum nodes before stopping recursion.
    """

    min_community_size: int

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph nodes into communities.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            List of sets, each containing node indices for one community.
        """
        ...

    def build_hierarchy(self, data: Data) -> HierarchicalGraph:
        """Build hierarchical graph representation from partitioning.

        This method should recursively decompose the graph based on
        min_community_size. Partitions with fewer nodes become leaf
        partitions; larger ones are further decomposed.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            HierarchicalGraph representing the decomposed structure.
        """
        ...
