"""Data structures for hierarchical graph representation.

This module defines the core data structures for representing graphs in a
hierarchical decomposition, supporting arbitrary depth controlled by
community size thresholds.

The hierarchy consists of:
- Partitions: Induced subgraphs within communities (diagonal blocks)
- Bipartites: Edges between communities (off-diagonal blocks)
- HierarchicalGraph: Container that can be nested for multi-level hierarchies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class Partition:
    """Induced subgraph of a community (diagonal block).

    A partition represents the internal structure of a single community.
    For multi-level hierarchies, a partition can contain a child hierarchy
    that further decomposes its internal structure.

    Attributes:
        part_id: Unique identifier for this partition within its level.
        global_node_indices: Mapping from local index to global node index.
        edge_index: Internal edges in LOCAL indices [2, num_edges].
        child_hierarchy: Optional nested HierarchicalGraph for deeper levels.
        node_features: Optional node features (e.g., atom types) in LOCAL indices [num_nodes].
    """

    part_id: int
    global_node_indices: list[int]
    edge_index: Tensor
    child_hierarchy: Optional[HierarchicalGraph] = None
    node_features: Optional[Tensor] = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in this partition."""
        return len(self.global_node_indices)

    @property
    def num_edges(self) -> int:
        """Number of edges in this partition (at this level only)."""
        return self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0

    @property
    def is_leaf(self) -> bool:
        """Whether this partition is a leaf (no further decomposition)."""
        return self.child_hierarchy is None

    def local_to_global(self, local_idx: int) -> int:
        """Convert a local node index to global index.

        Args:
            local_idx: Local index within this partition.

        Returns:
            Global node index in the original graph.

        Raises:
            IndexError: If local_idx is out of range.
        """
        return self.global_node_indices[local_idx]

    def global_to_local(self, global_idx: int) -> int:
        """Convert a global node index to local index.

        Args:
            global_idx: Global index in the original graph.

        Returns:
            Local node index within this partition.

        Raises:
            ValueError: If global_idx is not in this partition.
        """
        return self.global_node_indices.index(global_idx)

    def get_all_edges_global(self) -> list[tuple[int, int]]:
        """Get all internal edges in global indices.

        If this partition has a child hierarchy, recursively collects
        all edges from the hierarchy. Otherwise, returns edges from
        this level's edge_index.

        Returns:
            List of (src, dst) tuples in global node indices.
        """
        if self.child_hierarchy is not None:
            return self.child_hierarchy.get_all_edges_global()

        edges = []
        if self.edge_index.numel() > 0:
            ei = self.edge_index.numpy()
            for e in range(ei.shape[1]):
                local_src, local_dst = int(ei[0, e]), int(ei[1, e])
                global_src = self.local_to_global(local_src)
                global_dst = self.local_to_global(local_dst)
                edges.append((global_src, global_dst))
        return edges


@dataclass
class Bipartite:
    """Edges between two communities (off-diagonal block).

    A bipartite represents the connections between two partitions.
    Edge indices use local coordinates relative to each partition.

    Attributes:
        left_part_id: ID of the left (source) partition.
        right_part_id: ID of the right (target) partition.
        edge_index: Edges as [2, num_edges] where row 0 is left local
            indices and row 1 is right local indices.
        edge_features: Optional edge features (e.g., bond types) [num_edges].
    """

    left_part_id: int
    right_part_id: int
    edge_index: Tensor
    edge_features: Optional[Tensor] = None

    @property
    def num_edges(self) -> int:
        """Number of edges in this bipartite."""
        return self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0


@dataclass
class HierarchicalGraph:
    """Hierarchical decomposition of a graph.

    This class represents a graph decomposed into communities (partitions)
    and inter-community edges (bipartites). Supports arbitrary depth through
    nested partitions containing child hierarchies.

    The hierarchy is built top-down but reconstructed bottom-up:
    1. Coarsening: Large graph → partitions → (recursively) smaller partitions
    2. Reconstruction: Leaf partitions → combine with bipartites → full graph

    Attributes:
        partitions: List of partition subgraphs at this level.
        bipartites: List of bipartite edge sets between partitions.
        community_assignment: Mapping from global node index to partition ID.
        num_nodes: Total number of nodes in the original graph.
        num_communities: Number of partitions at this level.
        depth: Depth of this hierarchy (0 for single-level, >0 for nested).
        node_features: Optional global node features (e.g., atom types) [num_nodes].
        edge_features: Optional global edge features as dict {(src, dst): bond_type}.
    """

    partitions: list[Partition]
    bipartites: list[Bipartite]
    community_assignment: list[int]
    node_features: Optional[Tensor] = None
    edge_features: Optional[dict[tuple[int, int], int]] = None
    num_nodes: int = field(init=False)
    num_communities: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived attributes after initialization."""
        self.num_nodes = len(self.community_assignment)
        self.num_communities = len(self.partitions)

    @property
    def depth(self) -> int:
        """Compute the depth of this hierarchy.

        Returns:
            0 if all partitions are leaves, otherwise 1 + max child depth.
        """
        if not self.partitions:
            return 0
        max_child_depth = 0
        for part in self.partitions:
            if part.child_hierarchy is not None:
                child_depth = part.child_hierarchy.depth + 1
                max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth

    def get_partition(self, part_id: int) -> Partition:
        """Get a partition by its ID.

        Args:
            part_id: Partition identifier.

        Returns:
            The partition with the given ID.

        Raises:
            KeyError: If no partition with the given ID exists.
        """
        for p in self.partitions:
            if p.part_id == part_id:
                return p
        raise KeyError(f"Partition {part_id} not found")

    def get_all_edges_global(self) -> list[tuple[int, int]]:
        """Get all edges in global indices.

        Recursively collects edges from all partitions and bipartites.

        Returns:
            List of (src, dst) tuples in global node indices.
        """
        all_edges: list[tuple[int, int]] = []

        # Collect intra-partition edges (diagonal blocks)
        for part in self.partitions:
            all_edges.extend(part.get_all_edges_global())

        # Collect inter-partition edges (off-diagonal blocks)
        for bipart in self.bipartites:
            left_part = self.get_partition(bipart.left_part_id)
            right_part = self.get_partition(bipart.right_part_id)

            if bipart.edge_index.numel() > 0:
                ei = bipart.edge_index.numpy()
                for e in range(ei.shape[1]):
                    left_local = int(ei[0, e])
                    right_local = int(ei[1, e])
                    global_left = left_part.local_to_global(left_local)
                    global_right = right_part.local_to_global(right_local)
                    # Add both directions for undirected graphs
                    all_edges.append((global_left, global_right))
                    all_edges.append((global_right, global_left))

        return all_edges

    def reconstruct(self) -> Data:
        """Reconstruct the original graph from the hierarchical representation.

        This is the key method for validating the roundtrip:
        raw_graph → HierarchicalGraph → reconstruct() → raw_graph

        Returns:
            PyTorch Geometric Data object with edge_index, num_nodes, and optionally
            x (node features) and edge_attr (edge features) if present.
        """
        all_edges = self.get_all_edges_global()

        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            # Remove duplicate edges
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Reconstruct edge attributes if edge_features exist
        edge_attr = None
        if self.edge_features is not None and edge_index.shape[1] > 0:
            edge_attr_list = []
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])
                # Lookup bond type from edge_features dictionary
                bond_type = self.edge_features.get((src, dst), 0)
                edge_attr_list.append(bond_type)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

        return Data(
            edge_index=edge_index,
            num_nodes=self.num_nodes,
            x=self.node_features,  # Will be None if not present
            edge_attr=edge_attr,   # Will be None if not present
        )

    def get_level_info(self) -> dict:
        """Get information about each level of the hierarchy.

        Returns:
            Dictionary with level statistics for debugging/visualization.
        """
        info = {
            "num_communities": self.num_communities,
            "num_bipartites": len(self.bipartites),
            "partition_sizes": [p.num_nodes for p in self.partitions],
            "partition_edges": [p.num_edges for p in self.partitions],
            "bipartite_edges": [b.num_edges for b in self.bipartites],
            "depth": self.depth,
        }
        return info


def create_empty_hierarchy(num_nodes: int) -> HierarchicalGraph:
    """Create an empty hierarchical graph (single partition, no decomposition).

    Useful for graphs too small to decompose or as a fallback.

    Args:
        num_nodes: Number of nodes in the graph.

    Returns:
        HierarchicalGraph with a single partition containing all nodes.
    """
    single_partition = Partition(
        part_id=0,
        global_node_indices=list(range(num_nodes)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        child_hierarchy=None,
    )
    return HierarchicalGraph(
        partitions=[single_partition],
        bipartites=[],
        community_assignment=[0] * num_nodes,
    )
