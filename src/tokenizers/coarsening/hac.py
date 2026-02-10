"""Hierarchical Agglomerative Clustering (HAC) based graph coarsening.

This module implements HAC coarsening, which partitions graphs into communities
using bottom-up agglomerative clustering with a connectivity constraint.
Uses sklearn's AgglomerativeClustering with the graph adjacency matrix as
the connectivity constraint, ensuring only adjacent nodes are merged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_dense_adj

from src.tokenizers.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
)


class HACCoarsening:
    """Hierarchical Agglomerative Clustering based graph coarsening.

    Uses sklearn's AgglomerativeClustering with adjacency-based connectivity
    constraint. Each node is represented by its row in the adjacency matrix
    (neighborhood pattern), and only adjacent nodes can be merged.

    Attributes:
        linkage: Linkage criterion for merging clusters.
        feature_type: Node feature type for computing distances.
        k_min_factor: Factor for minimum cluster count (k_min = sqrt(n) * factor).
        k_max_factor: Factor for maximum cluster count (k_max = sqrt(n) * factor).
        min_community_size: Minimum nodes to attempt coarsening (controls depth).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        linkage: str = "ward",
        feature_type: str = "adjacency",
        k_min_factor: float = 0.7,
        k_max_factor: float = 1.3,
        min_community_size: int = 4,
        seed: int | None = None,
    ) -> None:
        """Initialize the HAC coarsening strategy.

        Args:
            linkage: Linkage criterion for merging clusters. Options:
                - "ward": Minimize within-cluster variance (default, requires
                  euclidean metric)
                - "complete": Maximum distance between cluster members
                - "average": Mean distance between cluster members
                - "single": Minimum distance between cluster members
            feature_type: Node feature representation. Options:
                - "adjacency": Use adjacency matrix rows as features (default)
            k_min_factor: Multiplier for minimum k (default 0.7).
            k_max_factor: Multiplier for maximum k (default 1.3).
            min_community_size: Minimum community size to attempt further
                coarsening. Communities smaller than this become leaf partitions.
            seed: Random seed for reproducibility (unused by HAC but kept for
                interface consistency).
        """
        self.linkage = linkage
        self.feature_type = feature_type
        self.k_min_factor = k_min_factor
        self.k_max_factor = k_max_factor
        self.min_community_size = min_community_size
        self.seed = seed

    def _compute_modularity(self, adj: np.ndarray, partition: dict[int, int]) -> float:
        """Compute modularity score for a partition.

        Vectorized implementation matching SpectralCoarsening._compute_modularity.

        Args:
            adj: Adjacency matrix as numpy array.
            partition: Dictionary mapping node -> community ID.

        Returns:
            Modularity score (higher is better, max is 1.0).
        """
        m = adj.sum() / 2
        if m == 0:
            return 0.0

        n = len(partition)
        num_communities = max(partition.values()) + 1

        # Create community membership matrix
        membership = np.zeros((n, num_communities))
        for node, comm in partition.items():
            membership[node, comm] = 1

        # Compute degree vector
        degrees = adj.sum(axis=1)

        # Vectorized modularity: Q = sum_c [ e_c/m - (d_c/(2m))^2 ]
        Q = 0.0
        for c in range(num_communities):
            nodes_in_c = membership[:, c] == 1
            # Edges within community
            e_c = adj[nodes_in_c][:, nodes_in_c].sum() / 2
            # Sum of degrees in community
            d_c = degrees[nodes_in_c].sum()
            Q += e_c / m - (d_c / (2 * m)) ** 2

        return Q

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph into communities using agglomerative clustering.

        Searches for the number of clusters K that maximizes modularity,
        using adjacency-constrained HAC.

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

        # Use adjacency matrix rows as node features
        features = adj.copy()

        # Build connectivity matrix (scipy sparse) for adjacency constraint
        from scipy.sparse import csr_matrix

        connectivity = csr_matrix(adj)

        # Compute k range based on graph size (same formula as SpectralCoarsening)
        k_min = max(2, int(np.sqrt(n) * self.k_min_factor))
        k_max = min(n - 1, int(np.sqrt(n) * self.k_max_factor))

        if k_min > k_max:
            k_min = k_max = max(2, min(n - 1, 2))

        # Search for best K by modularity
        best_modularity = -float("inf")
        best_partition: dict[int, int] | None = None

        from sklearn.cluster import AgglomerativeClustering

        for K in range(k_min, k_max + 1):
            try:
                hac = AgglomerativeClustering(
                    n_clusters=K,
                    linkage=self.linkage,
                    connectivity=connectivity,
                )
                labels = hac.fit_predict(features)
                partition_dict = dict(enumerate(labels))

                modularity = self._compute_modularity(adj, partition_dict)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_partition = partition_dict
            except ValueError:
                # Skip if clustering fails for this K (e.g., K > connected components)
                continue

        if best_partition is None:
            # Fallback: single community
            return [set(range(n))]

        # Convert to list of sets
        communities: dict[int, set[int]] = {}
        for node, comm in best_partition.items():
            communities.setdefault(comm, set()).add(node)

        return list(communities.values())

    def build_hierarchy(self, data: Data) -> HierarchicalGraph:
        """Build hierarchical graph representation from HAC partitioning.

        This is the main entry point for creating hierarchical representations.
        Always uses recursive decomposition controlled by min_community_size.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            HierarchicalGraph representing the decomposed structure.
        """
        n = data.num_nodes

        # Extract node and edge features if present
        node_features_global = (
            data.x if hasattr(data, "x") and data.x is not None else None
        )
        edge_features_global = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            # Build edge feature dictionary for efficient lookup
            edge_features_global = {}
            for i in range(data.edge_index.shape[1]):
                src = int(data.edge_index[0, i])
                dst = int(data.edge_index[1, i])
                bond_type = int(data.edge_attr[i])
                edge_features_global[(src, dst)] = bond_type

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
                sub_edge_index, sub_edge_mapping = subgraph(
                    subset=torch.tensor(node_list, dtype=torch.long),
                    edge_index=data.edge_index,
                    relabel_nodes=True,
                    num_nodes=n,
                )
            else:
                sub_edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Extract partition node features (LOCAL indices)
            part_node_features = None
            if node_features_global is not None:
                part_node_features = node_features_global[node_list]

            # Extract partition edge features for subgraph edges
            part_edge_features_dict = None
            if edge_features_global is not None and sub_edge_index.numel() > 0:
                part_edge_features_dict = {}
                for i in range(sub_edge_index.shape[1]):
                    local_src = int(sub_edge_index[0, i])
                    local_dst = int(sub_edge_index[1, i])
                    global_src = node_list[local_src]
                    global_dst = node_list[local_dst]
                    if (global_src, global_dst) in edge_features_global:
                        # Store with LOCAL indices
                        part_edge_features_dict[(local_src, local_dst)] = (
                            edge_features_global[(global_src, global_dst)]
                        )

            # Recursively coarsen if community is large enough
            child_hierarchy = None
            if len(node_list) >= self.min_community_size:
                # Create a Data object for the subgraph with features
                sub_data = Data(
                    edge_index=sub_edge_index,
                    num_nodes=len(node_list),
                )
                # Add features if present
                if part_node_features is not None:
                    sub_data.x = part_node_features
                if part_edge_features_dict is not None and sub_edge_index.numel() > 0:
                    # Convert dict to tensor for sub_data
                    sub_edge_attr_list = []
                    for i in range(sub_edge_index.shape[1]):
                        local_src = int(sub_edge_index[0, i])
                        local_dst = int(sub_edge_index[1, i])
                        bond = part_edge_features_dict.get((local_src, local_dst), 0)
                        sub_edge_attr_list.append(bond)
                    sub_data.edge_attr = torch.tensor(
                        sub_edge_attr_list, dtype=torch.long
                    )

                # Recursively build hierarchy for this partition
                child_hg = self.build_hierarchy(sub_data)

                # Only use child hierarchy if it actually decomposed further
                if child_hg.num_communities > 1:
                    # Remap child hierarchy's global indices to parent's local indices
                    child_hierarchy = self._remap_child_hierarchy(child_hg, node_list)

            partitions.append(
                Partition(
                    part_id=comm_id,
                    global_node_indices=node_list,
                    edge_index=sub_edge_index,
                    child_hierarchy=child_hierarchy,
                    node_features=part_node_features,
                )
            )

        # Extract bipartites (off-diagonal blocks) with edge features
        bipartites = self._extract_bipartites(
            data, communities, partitions, edge_features_global
        )

        return HierarchicalGraph(
            partitions=partitions,
            bipartites=bipartites,
            community_assignment=community_assignment,
            node_features=node_features_global,
            edge_features=edge_features_global,
        )

    def _build_single_partition(self, data: Data) -> HierarchicalGraph:
        """Build a hierarchy with a single partition (no decomposition).

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            HierarchicalGraph with one partition containing all nodes.
        """
        n = data.num_nodes

        # Extract features if present
        node_features_global = (
            data.x if hasattr(data, "x") and data.x is not None else None
        )
        edge_features_global = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_features_global = {}
            for i in range(data.edge_index.shape[1]):
                src = int(data.edge_index[0, i])
                dst = int(data.edge_index[1, i])
                bond_type = int(data.edge_attr[i])
                edge_features_global[(src, dst)] = bond_type

        partition = Partition(
            part_id=0,
            global_node_indices=list(range(n)),
            edge_index=data.edge_index.clone(),
            child_hierarchy=None,
            node_features=node_features_global,
        )
        return HierarchicalGraph(
            partitions=[partition],
            bipartites=[],
            community_assignment=[0] * n,
            node_features=node_features_global,
            edge_features=edge_features_global,
        )

    def _extract_bipartites(
        self,
        data: Data,
        communities: list[set[int]],
        partitions: list[Partition],
        edge_features_global: Optional[dict[tuple[int, int], int]] = None,
    ) -> list[Bipartite]:
        """Extract bipartite edge sets between all pairs of communities.

        Uses dict-based global-to-local index mapping for O(1) lookups
        instead of list.index() O(n) lookups.

        Args:
            data: Original graph data.
            communities: List of node sets for each community.
            partitions: List of partition objects.
            edge_features_global: Optional edge feature dictionary.

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

                # Build O(1) lookup dicts (improvement over SpectralCoarsening)
                left_global_to_local = {g: idx for idx, g in enumerate(left_nodes)}
                right_global_to_local = {g: idx for idx, g in enumerate(right_nodes)}

                # Find edges from left to right community
                bipart_edges = []
                bipart_edge_features = [] if edge_features_global is not None else None

                for e in range(edge_index_np.shape[1]):
                    src, dst = int(edge_index_np[0, e]), int(edge_index_np[1, e])

                    if src in left_set and dst in right_set:
                        # Map to local indices using dict (O(1))
                        local_src = left_global_to_local[src]
                        local_dst = right_global_to_local[dst]
                        bipart_edges.append((local_src, local_dst))

                        # Extract edge feature if present
                        if edge_features_global is not None:
                            bond_type = edge_features_global.get((src, dst), 0)
                            bipart_edge_features.append(bond_type)

                if bipart_edges:
                    bipart_edge_index = torch.tensor(bipart_edges, dtype=torch.long).t()

                    # Convert edge features to tensor if present
                    bipart_edge_attr = None
                    if bipart_edge_features is not None:
                        bipart_edge_attr = torch.tensor(
                            bipart_edge_features, dtype=torch.long
                        )

                    bipartites.append(
                        Bipartite(
                            left_part_id=i,
                            right_part_id=j,
                            edge_index=bipart_edge_index,
                            edge_features=bipart_edge_attr,
                        )
                    )

        return bipartites

    def _remap_child_hierarchy(
        self,
        child_hg: HierarchicalGraph,
        parent_node_list: list[int],
    ) -> HierarchicalGraph:
        """Remap a child hierarchy's top-level indices to parent's coordinate system.

        When we recursively build a hierarchy for a subgraph, the child uses indices
        0..k-1. This function remaps only the top-level partition indices to the
        parent's coordinate system. Nested child hierarchies remain in their local
        coordinate systems.

        Args:
            child_hg: Child hierarchy with local indices 0..k-1.
            parent_node_list: Parent's global node indices (sorted, length k).

        Returns:
            Child hierarchy with remapped top-level indices.
        """
        # Remap partitions
        remapped_partitions = []
        for part in child_hg.partitions:
            remapped_global = [
                parent_node_list[idx] for idx in part.global_node_indices
            ]
            remapped_partitions.append(
                Partition(
                    part_id=part.part_id,
                    global_node_indices=remapped_global,
                    edge_index=part.edge_index.clone(),
                    child_hierarchy=part.child_hierarchy,
                )
            )

        # Remap community assignment
        remapped_assignment = list(child_hg.community_assignment)

        return HierarchicalGraph(
            partitions=remapped_partitions,
            bipartites=child_hg.bipartites,
            community_assignment=remapped_assignment,
        )
