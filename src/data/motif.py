"""Motif detection algorithms for graphs.

This module provides efficient algorithms for detecting common structural
motifs in graphs, including triangles, cycles, cliques, and stars.
"""

from enum import Enum, auto
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class MotifType(Enum):
    """Enumeration of supported motif types."""

    TRIANGLE = auto()
    FOUR_CYCLE = auto()
    FIVE_CYCLE = auto()
    FOUR_CLIQUE = auto()
    STAR = auto()


class MotifDetector:
    """Detects and labels motifs in graphs.

    This class provides methods to detect various structural motifs in graphs
    and label nodes according to their motif membership.

    Attributes:
        motif_types: List of motif types to detect.
    """

    def __init__(self, motif_types: Optional[list[MotifType]] = None) -> None:
        """Initialize the motif detector.

        Args:
            motif_types: Motif types to detect. If None, detects all types.
        """
        if motif_types is None:
            motif_types = list(MotifType)
        self.motif_types = motif_types

    def detect(self, data: Data) -> dict[str, torch.Tensor | list | int]:
        """Detect motifs in a PyG Data object.

        Args:
            data: PyTorch Geometric Data object with edge_index.

        Returns:
            Dictionary containing:
                - motif_labels: Tensor of shape [num_nodes] with motif IDs (-1 if none)
                - motif_types: List of motif type strings found
                - motif_counts: Dictionary mapping motif type to count
                - num_motifs: Total number of distinct motif instances
        """
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        return self.detect_networkx(G)

    def detect_networkx(self, G: nx.Graph) -> dict[str, torch.Tensor | list | int]:
        """Detect motifs in a NetworkX graph.

        Args:
            G: NetworkX graph.

        Returns:
            Dictionary containing motif detection results.
        """
        num_nodes = G.number_of_nodes()
        motif_labels = torch.full((num_nodes,), -1, dtype=torch.long)
        motif_types_found: list[str] = []
        motif_counts: dict[str, int] = {}
        current_motif_id = 0

        for motif_type in self.motif_types:
            if motif_type == MotifType.TRIANGLE:
                motifs = self._find_triangles(G)
            elif motif_type == MotifType.FOUR_CYCLE:
                motifs = self._find_cycles(G, length=4)
            elif motif_type == MotifType.FIVE_CYCLE:
                motifs = self._find_cycles(G, length=5)
            elif motif_type == MotifType.FOUR_CLIQUE:
                motifs = self._find_cliques(G, size=4)
            elif motif_type == MotifType.STAR:
                motifs = self._find_stars(G, min_leaves=3)
            else:
                continue

            if motifs:
                motif_types_found.append(motif_type.name.lower())
                motif_counts[motif_type.name.lower()] = len(motifs)

                for motif_nodes in motifs:
                    for node in motif_nodes:
                        if node < num_nodes and motif_labels[node] == -1:
                            motif_labels[node] = current_motif_id
                    current_motif_id += 1

        return {
            "motif_labels": motif_labels,
            "motif_types": motif_types_found,
            "motif_counts": motif_counts,
            "num_motifs": current_motif_id,
        }

    def _find_triangles(self, G: nx.Graph) -> list[tuple[int, ...]]:
        """Find all triangles in the graph.

        Args:
            G: NetworkX graph.

        Returns:
            List of tuples, each containing the three nodes forming a triangle.
        """
        triangles = []
        for node in G.nodes():
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors:
                common = neighbors & set(G.neighbors(neighbor))
                for third in common:
                    if node < neighbor < third:
                        triangles.append((node, neighbor, third))
        return triangles

    def _find_cycles(self, G: nx.Graph, length: int) -> list[tuple[int, ...]]:
        """Find all simple cycles of a specific length.

        Args:
            G: NetworkX graph.
            length: Desired cycle length.

        Returns:
            List of tuples, each containing the nodes forming a cycle.
        """
        cycles = []
        try:
            all_cycles = nx.simple_cycles(G.to_directed())
            for cycle in all_cycles:
                if len(cycle) == length:
                    min_idx = cycle.index(min(cycle))
                    normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                    if normalized not in cycles:
                        cycles.append(normalized)
        except nx.NetworkXError:
            pass
        return cycles

    def _find_cliques(self, G: nx.Graph, size: int) -> list[tuple[int, ...]]:
        """Find all cliques of a specific size.

        Args:
            G: NetworkX graph.
            size: Desired clique size.

        Returns:
            List of tuples, each containing the nodes forming a clique.
        """
        cliques = []
        for clique in nx.enumerate_all_cliques(G):
            if len(clique) == size:
                cliques.append(tuple(sorted(clique)))
            elif len(clique) > size:
                break
        return cliques

    def _find_stars(
        self, G: nx.Graph, min_leaves: int = 3
    ) -> list[tuple[int, ...]]:
        """Find all star motifs (hub with multiple leaves).

        Args:
            G: NetworkX graph.
            min_leaves: Minimum number of leaves to qualify as a star.

        Returns:
            List of tuples, where first element is hub and rest are leaves.
        """
        stars = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            leaves = [n for n in neighbors if G.degree(n) == 1]
            if len(leaves) >= min_leaves:
                stars.append(tuple([node] + sorted(leaves)))
        return stars

    def get_motif_counts(self, data: Data) -> dict[str, int]:
        """Get motif counts for a graph.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Dictionary mapping motif type names to counts.
        """
        result = self.detect(data)
        return result["motif_counts"]

    def get_motif_vector(self, data: Data) -> np.ndarray:
        """Get a normalized motif count vector for a graph.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Numpy array of normalized motif counts.
        """
        counts = self.get_motif_counts(data)
        vector = np.zeros(len(MotifType))
        for i, motif_type in enumerate(MotifType):
            vector[i] = counts.get(motif_type.name.lower(), 0)
        num_nodes = data.num_nodes if data.num_nodes else 1
        return vector / num_nodes
