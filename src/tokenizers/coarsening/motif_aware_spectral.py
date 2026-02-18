"""Motif-aware spectral clustering for graph coarsening.

This module extends spectral coarsening with motif affinity augmentation
to preserve structural motifs (rings, functional groups) during partitioning.
"""

from __future__ import annotations

import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from src.tokenizers.coarsening.spectral import SpectralCoarsening
from src.tokenizers.motif import (
    CLUSTERING_MOTIFS,
    compute_motif_affinity_matrix,
    compute_motif_cohesion,
    detect_motifs_from_data,
)


class MotifAwareSpectralCoarsening(SpectralCoarsening):
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
        orig_adj = adj.copy()
        if self.alpha > 0:
            motifs = self._detect_motifs(data)
            self._cached_motifs = motifs

            if motifs:
                M = self._compute_motif_affinity(n, motifs)
                adj = adj + self.alpha * M
        else:
            self._cached_motifs = []

        # Run spectral clustering on the (possibly augmented) adjacency,
        # but evaluate modularity on the original graph structure
        return self._partition_with_adjacency(adj, n, orig_adj)

    def _detect_motifs(self, data: Data) -> list:
        """Detect motif instances in the molecular graph.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            List of MotifInstance objects.
        """
        return detect_motifs_from_data(data, self.motif_patterns)

    def _compute_motif_affinity(self, n: int, motifs: list) -> np.ndarray:
        """Compute motif co-membership affinity matrix.

        Args:
            n: Number of nodes.
            motifs: List of MotifInstance objects.

        Returns:
            Symmetric affinity matrix of shape (n, n).
        """
        return compute_motif_affinity_matrix(
            n, motifs, normalize_by_size=self.normalize_by_motif_size
        )

    def _partition_with_adjacency(
        self,
        adj: np.ndarray,
        n: int,
        orig_adj: np.ndarray | None = None,
    ) -> list[set[int]]:
        """Run spectral clustering on a given adjacency matrix.

        This is extracted from the parent's partition() method to allow
        reuse with an augmented adjacency matrix.

        Args:
            adj: Adjacency matrix (possibly augmented with motif affinity).
            n: Number of nodes.
            orig_adj: Original (unaugmented) adjacency for modularity
                evaluation. If None, uses adj.

        Returns:
            List of sets containing node indices for each community.
        """
        if orig_adj is None:
            orig_adj = adj
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
                modularity = self._compute_modularity(orig_adj, partition)
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

    def get_motif_cohesion(
        self, communities: list[set[int]] | None = None
    ) -> float:
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


# Backwards compatibility alias
MotifAwareCoarsening = MotifAwareSpectralCoarsening
