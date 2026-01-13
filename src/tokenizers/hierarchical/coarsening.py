"""Graph coarsening strategies for hierarchical decomposition.

This module implements coarsening algorithms that partition graphs into
communities for hierarchical representation. The primary implementation
uses spectral clustering adapted from HiGen.

The module is designed for extensibility:
- CoarseningStrategy protocol defines the interface
- SpectralCoarsening implements spectral clustering
- MotifAwareCoarsening extends spectral clustering with motif preservation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_dense_adj

from src.tokenizers.hierarchical.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
    create_empty_hierarchy,
)


@runtime_checkable
class CoarseningStrategy(Protocol):
    """Protocol for graph coarsening strategies.

    All coarsening implementations must provide a method to partition
    a graph into communities. This allows swapping between spectral,
    motif-based, or hybrid approaches.
    """

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph nodes into communities.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            List of sets, each containing node indices for one community.
        """
        ...


class SpectralCoarsening:
    """Spectral clustering-based graph coarsening.

    Adapted from HiGen's graph_corsening.py. Uses spectral clustering
    to find communities that maximize modularity.

    Attributes:
        k_min_factor: Factor for minimum cluster count (k_min = sqrt(n) * factor).
        k_max_factor: Factor for maximum cluster count (k_max = sqrt(n) * factor).
        n_init: Number of initializations for spectral clustering.
        min_community_size: Minimum nodes to attempt coarsening (controls depth).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k_min_factor: float = 0.7,
        k_max_factor: float = 1.3,
        n_init: int = 100,
        min_community_size: int = 4,
        seed: int | None = None,
    ) -> None:
        """Initialize the spectral coarsening strategy.

        Args:
            k_min_factor: Multiplier for minimum k (default 0.7).
            k_max_factor: Multiplier for maximum k (default 1.3).
            n_init: Number of spectral clustering initializations.
            min_community_size: Minimum community size to attempt further
                coarsening. Communities smaller than this become leaf partitions.
            seed: Random seed for reproducibility.
        """
        self.k_min_factor = k_min_factor
        self.k_max_factor = k_max_factor
        self.n_init = n_init
        self.min_community_size = min_community_size
        self.seed = seed

    def _compute_modularity(
        self, adj: np.ndarray, partition: dict[int, int]
    ) -> float:
        """Compute modularity score for a partition.

        Modularity measures the quality of a community structure by comparing
        edge density within communities to expected density in a random graph.

        Args:
            adj: Adjacency matrix as numpy array.
            partition: Dictionary mapping node → community ID.

        Returns:
            Modularity score (higher is better, max is 1.0).
        """
        # Try to use python-louvain if available
        try:
            import community as community_louvain
            import networkx as nx

            G = nx.from_numpy_array(adj)
            return community_louvain.modularity(partition, G)
        except ImportError:
            pass

        # Fallback: manual modularity computation
        m = adj.sum() / 2  # Total edge weight
        if m == 0:
            return 0.0

        # Group nodes by community
        communities: dict[int, list[int]] = {}
        for node, comm in partition.items():
            communities.setdefault(comm, []).append(node)

        # Compute modularity: Q = (1/2m) * sum_{ij}[A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
        Q = 0.0
        for nodes in communities.values():
            for i in nodes:
                ki = adj[i].sum()  # Degree of node i
                for j in nodes:
                    kj = adj[j].sum()  # Degree of node j
                    Q += adj[i, j] - (ki * kj) / (2 * m)

        return Q / (2 * m)

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph into communities using spectral clustering.

        Searches for the number of clusters K that maximizes modularity.
        Adapted from HiGen's best_spectral_partition().

        Args:
            data: PyTorch Geometric Data object with edge_index.

        Returns:
            List of sets containing node indices for each community.
        """
        n = data.num_nodes

        # Handle trivial cases
        if n <= 1:
            return [set(range(n))]

        # Build adjacency matrix
        adj = to_dense_adj(data.edge_index, max_num_nodes=n)[0]
        adj = ((adj + adj.t()) / 2).numpy()  # Symmetrize for undirected

        if adj.sum() == 0:
            return [set(range(n))]

        # Compute k range based on graph size (HiGen's formula)
        k_min = max(2, int(np.sqrt(n) * self.k_min_factor))
        k_max = min(n - 1, int(np.sqrt(n) * self.k_max_factor))

        if k_min > k_max:
            k_min = k_max = max(2, min(n - 1, 2))

        # Search for best K by modularity
        best_modularity = -float("inf")
        best_partition: dict[int, int] | None = None

        # Import here to avoid issues if not installed
        from sklearn.cluster import SpectralClustering

        for K in range(k_min, k_max + 1):
            try:
                sc = SpectralClustering(
                    n_clusters=K,
                    affinity="precomputed",
                    n_init=self.n_init,
                    random_state=self.seed,
                    assign_labels="kmeans",
                )
                # Add small diagonal for numerical stability
                labels = sc.fit_predict(adj + np.eye(n) * 1e-6)
                partition = dict(enumerate(labels))

                modularity = self._compute_modularity(adj, partition)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_partition = partition
            except Exception:
                # Skip if clustering fails for this K
                continue

        if best_partition is None:
            # Fallback: single community
            return [set(range(n))]

        # Convert to list of sets
        communities: dict[int, set[int]] = {}
        for node, comm in best_partition.items():
            communities.setdefault(comm, set()).add(node)

        return list(communities.values())

    def build_hierarchy(
        self,
        data: Data,
        recursive: bool = False,
    ) -> HierarchicalGraph:
        """Build hierarchical graph representation from spectral partitioning.

        This is the main entry point for creating hierarchical representations.
        Supports recursive decomposition controlled by min_community_size.

        Args:
            data: PyTorch Geometric Data object.
            recursive: Whether to recursively coarsen large communities.

        Returns:
            HierarchicalGraph representing the decomposed structure.
        """
        n = data.num_nodes

        # Don't coarsen if graph is too small
        if n < self.min_community_size:
            return self._build_single_partition(data)

        # Partition into communities
        communities = self.partition(data)

        # If partitioning produced a single community, don't coarsen
        if len(communities) <= 1:
            return self._build_single_partition(data)

        # Build community assignment mapping
        community_assignment = [0] * n
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                community_assignment[node] = comm_id

        # Extract partitions (diagonal blocks)
        partitions = []
        for comm_id, nodes in enumerate(communities):
            node_list = sorted(nodes)

            # Extract subgraph edges
            if len(node_list) > 0:
                sub_edge_index, _ = subgraph(
                    subset=torch.tensor(node_list, dtype=torch.long),
                    edge_index=data.edge_index,
                    relabel_nodes=True,
                    num_nodes=n,
                )
            else:
                sub_edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Recursively coarsen if community is large enough
            child_hierarchy = None
            if recursive and len(node_list) >= self.min_community_size:
                # Create a Data object for the subgraph
                sub_data = Data(
                    edge_index=sub_edge_index,
                    num_nodes=len(node_list),
                )
                # Recursively build hierarchy for this partition
                child_hg = self.build_hierarchy(sub_data, recursive=True)

                # Only use child hierarchy if it actually decomposed further
                if child_hg.num_communities > 1:
                    # Remap child hierarchy's global indices to parent's local indices
                    child_hierarchy = self._remap_child_hierarchy(
                        child_hg, node_list
                    )

            partitions.append(
                Partition(
                    part_id=comm_id,
                    global_node_indices=node_list,
                    edge_index=sub_edge_index,
                    child_hierarchy=child_hierarchy,
                )
            )

        # Extract bipartites (off-diagonal blocks)
        bipartites = self._extract_bipartites(
            data, communities, partitions
        )

        return HierarchicalGraph(
            partitions=partitions,
            bipartites=bipartites,
            community_assignment=community_assignment,
        )

    def _build_single_partition(self, data: Data) -> HierarchicalGraph:
        """Build a hierarchy with a single partition (no decomposition).

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            HierarchicalGraph with one partition containing all nodes.
        """
        n = data.num_nodes
        partition = Partition(
            part_id=0,
            global_node_indices=list(range(n)),
            edge_index=data.edge_index.clone(),
            child_hierarchy=None,
        )
        return HierarchicalGraph(
            partitions=[partition],
            bipartites=[],
            community_assignment=[0] * n,
        )

    def _extract_bipartites(
        self,
        data: Data,
        communities: list[set[int]],
        partitions: list[Partition],
    ) -> list[Bipartite]:
        """Extract bipartite edge sets between all pairs of communities.

        Args:
            data: Original graph data.
            communities: List of node sets for each community.
            partitions: List of partition objects.

        Returns:
            List of Bipartite objects for non-empty community pairs.
        """
        bipartites = []
        edge_index_np = data.edge_index.numpy()

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                left_nodes = partitions[i].global_node_indices
                right_nodes = partitions[j].global_node_indices
                left_set = set(left_nodes)
                right_set = set(right_nodes)

                # Find edges from left to right community
                bipart_edges = []
                for e in range(edge_index_np.shape[1]):
                    src, dst = int(edge_index_np[0, e]), int(edge_index_np[1, e])

                    if src in left_set and dst in right_set:
                        # Map to local indices
                        local_src = left_nodes.index(src)
                        local_dst = right_nodes.index(dst)
                        bipart_edges.append((local_src, local_dst))

                if bipart_edges:
                    bipart_edge_index = torch.tensor(
                        bipart_edges, dtype=torch.long
                    ).t()
                    bipartites.append(
                        Bipartite(
                            left_part_id=i,
                            right_part_id=j,
                            edge_index=bipart_edge_index,
                        )
                    )

        return bipartites

    def _remap_child_hierarchy(
        self,
        child_hg: HierarchicalGraph,
        parent_node_list: list[int],
    ) -> HierarchicalGraph:
        """Remap a child hierarchy's indices to parent's coordinate system.

        When we recursively build a hierarchy for a subgraph, the child
        uses indices 0..k-1. We need to remap these to the parent's
        local indices.

        Args:
            child_hg: Child hierarchy with local indices.
            parent_node_list: Parent's global node indices (sorted).

        Returns:
            Child hierarchy with remapped global indices.
        """
        # Remap partitions
        remapped_partitions = []
        for part in child_hg.partitions:
            # Child's global indices are 0..k-1, map to parent's global indices
            remapped_global = [
                parent_node_list[idx] for idx in part.global_node_indices
            ]
            # Recursively remap child hierarchy if present
            remapped_child = None
            if part.child_hierarchy is not None:
                remapped_child = self._remap_child_hierarchy(
                    part.child_hierarchy, remapped_global
                )
            remapped_partitions.append(
                Partition(
                    part_id=part.part_id,
                    global_node_indices=remapped_global,
                    edge_index=part.edge_index.clone(),
                    child_hierarchy=remapped_child,
                )
            )

        # Remap community assignment
        remapped_assignment = [
            child_hg.community_assignment[parent_node_list.index(parent_node_list[i])]
            for i in range(len(parent_node_list))
        ]

        return HierarchicalGraph(
            partitions=remapped_partitions,
            bipartites=child_hg.bipartites,  # Bipartites use local indices, no change
            community_assignment=remapped_assignment,
        )


class MotifAwareCoarsening(SpectralCoarsening):
    """Spectral coarsening with motif-aware affinity augmentation.

    Extends SpectralCoarsening to incorporate motif co-membership as
    additional affinity in the spectral clustering input. This encourages
    the clustering algorithm to keep known motifs (e.g., benzene rings,
    functional groups) together in the same partition.

    The augmented affinity matrix is: A' = A + alpha * M
    where M[i,j] = number of motifs containing both atoms i and j.

    Attributes:
        alpha: Weight for motif affinity (0 = standard spectral clustering).
        motif_patterns: SMARTS patterns for motif detection.
        normalize_by_motif_size: Whether to normalize M by motif size.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        motif_patterns: dict[str, str] | None = None,
        normalize_by_motif_size: bool = False,
        k_min_factor: float = 0.7,
        k_max_factor: float = 1.3,
        n_init: int = 100,
        min_community_size: int = 4,
        seed: int | None = None,
    ) -> None:
        """Initialize the motif-aware coarsening strategy.

        Args:
            alpha: Weight for motif affinity matrix. Higher values give
                stronger preference to keeping motifs together.
                0 = standard spectral clustering, 1 = motif co-membership
                treated as equivalent to having an edge.
            motif_patterns: Dictionary mapping motif names to SMARTS patterns.
                Defaults to CLUSTERING_MOTIFS (ring-focused patterns).
            normalize_by_motif_size: If True, normalize each motif's
                contribution by 1/motif_size to prevent large motifs
                from dominating.
            k_min_factor: Factor for minimum cluster count (k_min = sqrt(n) * factor).
            k_max_factor: Factor for maximum cluster count (k_max = sqrt(n) * factor).
            n_init: Number of initializations for spectral clustering.
            min_community_size: Minimum nodes to attempt coarsening.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            k_min_factor=k_min_factor,
            k_max_factor=k_max_factor,
            n_init=n_init,
            min_community_size=min_community_size,
            seed=seed,
        )
        self.alpha = alpha
        self.normalize_by_motif_size = normalize_by_motif_size

        # Import here to avoid circular imports
        from src.tokenizers.hierarchical.motifs import CLUSTERING_MOTIFS

        self.motif_patterns = motif_patterns or CLUSTERING_MOTIFS

        # Cache for detected motifs (available after partition() call)
        self._cached_motifs: list | None = None

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph into communities using motif-augmented affinity.

        Detects motifs in the molecule and augments the adjacency matrix
        with motif co-membership weights before running spectral clustering.

        Args:
            data: PyTorch Geometric Data object with edge_index.
                Should have 'smiles' attribute for motif detection.

        Returns:
            List of sets containing node indices for each community.
        """
        n = data.num_nodes

        # Handle trivial cases
        if n <= 1:
            self._cached_motifs = []
            return [set(range(n))]

        # Build adjacency matrix
        adj = to_dense_adj(data.edge_index, max_num_nodes=n)[0]
        adj = ((adj + adj.t()) / 2).numpy()

        if adj.sum() == 0:
            self._cached_motifs = []
            return [set(range(n))]

        # Detect motifs and augment affinity if alpha > 0
        if self.alpha > 0:
            motifs = self._detect_motifs(data)
            self._cached_motifs = motifs

            if motifs:
                M = self._compute_motif_affinity(n, motifs)
                adj = adj + self.alpha * M
        else:
            self._cached_motifs = []

        # Run spectral clustering on the (possibly augmented) adjacency
        return self._partition_with_adjacency(adj, n)

    def _detect_motifs(self, data: Data) -> list:
        """Detect motif instances in the molecular graph.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            List of MotifInstance objects.
        """
        from src.tokenizers.hierarchical.motifs import detect_motifs_from_data

        return detect_motifs_from_data(data, self.motif_patterns)

    def _compute_motif_affinity(self, n: int, motifs: list) -> np.ndarray:
        """Compute motif co-membership affinity matrix.

        Args:
            n: Number of nodes.
            motifs: List of MotifInstance objects.

        Returns:
            Symmetric affinity matrix of shape (n, n).
        """
        from src.tokenizers.hierarchical.motifs import compute_motif_affinity_matrix

        return compute_motif_affinity_matrix(
            n, motifs, normalize_by_size=self.normalize_by_motif_size
        )

    def _partition_with_adjacency(
        self, adj: np.ndarray, n: int
    ) -> list[set[int]]:
        """Run spectral clustering on a given adjacency matrix.

        This is extracted from the parent's partition() method to allow
        reuse with an augmented adjacency matrix.

        Args:
            adj: Adjacency matrix (possibly augmented with motif affinity).
            n: Number of nodes.

        Returns:
            List of sets containing node indices for each community.
        """
        # Compute k range based on graph size (HiGen's formula)
        k_min = max(2, int(np.sqrt(n) * self.k_min_factor))
        k_max = min(n - 1, int(np.sqrt(n) * self.k_max_factor))

        if k_min > k_max:
            k_min = k_max = max(2, min(n - 1, 2))

        # Search for best K by modularity
        best_modularity = -float("inf")
        best_partition: dict[int, int] | None = None

        from sklearn.cluster import SpectralClustering

        for K in range(k_min, k_max + 1):
            try:
                sc = SpectralClustering(
                    n_clusters=K,
                    affinity="precomputed",
                    n_init=self.n_init,
                    random_state=self.seed,
                    assign_labels="kmeans",
                )
                # Add small diagonal for numerical stability
                labels = sc.fit_predict(adj + np.eye(n) * 1e-6)
                partition = dict(enumerate(labels))

                # Use original adjacency for modularity computation
                # (we want modularity of the actual graph, not augmented)
                modularity = self._compute_modularity(adj, partition)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_partition = partition
            except Exception:
                continue

        if best_partition is None:
            return [set(range(n))]

        # Convert to list of sets
        communities: dict[int, set[int]] = {}
        for node, comm in best_partition.items():
            communities.setdefault(comm, set()).add(node)

        return list(communities.values())

    def get_motif_cohesion(self, communities: list[set[int]] | None = None) -> float:
        """Compute motif cohesion for the last partition.

        Returns the fraction of detected motifs that are entirely
        contained within a single community.

        Args:
            communities: Optional partition to evaluate. If None, uses
                the result from the last partition() call (requires
                calling partition() first and caching the result).

        Returns:
            Fraction of motifs kept intact (0.0 to 1.0).
            Returns 1.0 if no motifs were detected.
        """
        from src.tokenizers.hierarchical.motifs import compute_motif_cohesion

        if self._cached_motifs is None:
            return 1.0

        if communities is None:
            # Caller should provide communities or we return 1.0
            return 1.0

        return compute_motif_cohesion(communities, self._cached_motifs)

    @property
    def cached_motifs(self) -> list | None:
        """Return the motifs detected in the last partition() call."""
        return self._cached_motifs
