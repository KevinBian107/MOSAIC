"""Affinity-based graph coarsening via Boruvka's algorithm.

This module implements affinity coarsening, which partitions graphs into
communities using Boruvka-style greedy merging with modularity-optimal
tree cutting. Replaces the previous sklearn-based HAC approach with a
proper graph-native algorithm that uses edge weights directly as affinity.

Based on: Bateni et al., "Affinity Clustering: Hierarchical Clustering
at Scale", NeurIPS 2017.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from src.tokenizers.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
)

# Bond type index → affinity weight (from src/data/molecular.py BOND_TYPES)
# Index 0: single, 1: double, 2: triple, 3: aromatic, 4: unknown
BOND_WEIGHT_MAP: dict[int, float] = {
    0: 1.0,  # single bond
    1: 2.0,  # double bond
    2: 3.0,  # triple bond
    3: 1.5,  # aromatic bond
    4: 1.0,  # unknown bond type
}


class _UnionFind:
    """Disjoint-set (union-find) data structure with path compression.

    Attributes:
        parent: Parent array for each element.
        rank: Rank array for union-by-rank.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if a merge occurred."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


class AffinityCoarsening:
    """Affinity-based graph coarsening using Boruvka's algorithm.

    Uses edge weights directly as affinity scores. Each Boruvka round
    merges every cluster with its highest-affinity neighbor. The merge
    tree is then cut at the level maximizing modularity.

    Attributes:
        min_community_size: Minimum nodes to attempt coarsening (controls depth).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        min_community_size: int = 4,
        seed: int | None = None,
    ) -> None:
        """Initialize the affinity coarsening strategy.

        Args:
            min_community_size: Minimum community size to attempt further
                coarsening. Communities smaller than this become leaf partitions.
            seed: Random seed for reproducibility.
        """
        self.min_community_size = min_community_size
        self.seed = seed

    def _build_weighted_adj(self, data: Data) -> np.ndarray:
        """Build weighted adjacency matrix from graph data.

        Uses bond type edge attributes to set edge weights. Falls back
        to uniform weight 1.0 when edge_attr is not available.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Weighted adjacency matrix as numpy array.
        """
        n = data.num_nodes
        adj = np.zeros((n, n), dtype=np.float64)

        if data.edge_index.numel() == 0:
            return adj

        ei = data.edge_index.numpy()
        has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None

        for e in range(ei.shape[1]):
            src, dst = int(ei[0, e]), int(ei[1, e])
            if src == dst:
                continue

            if has_edge_attr:
                bond_type = int(data.edge_attr[e])
                weight = BOND_WEIGHT_MAP.get(bond_type, 1.0)
            else:
                weight = 1.0

            # Symmetrize (take max weight for each direction)
            adj[src, dst] = max(adj[src, dst], weight)
            adj[dst, src] = max(adj[dst, src], weight)

        return adj

    def _boruvka_merge_tree(self, adj: np.ndarray) -> list[list[tuple[int, int]]]:
        """Build merge tree using greedy affinity merging.

        At each step, compute inter-cluster affinities (sum of edge weights
        between clusters) and merge the pair with highest affinity. Each
        merge is recorded as a separate level, giving fine-grained control
        over the tree cut.

        Args:
            adj: Weighted adjacency matrix.

        Returns:
            List of levels, each containing one (cluster_a, cluster_b) merge.
        """
        n = adj.shape[0]
        uf = _UnionFind(n)
        merge_levels: list[list[tuple[int, int]]] = []

        for _ in range(n - 1):  # At most n-1 merges
            # Compute inter-cluster affinities
            cluster_affinity: dict[tuple[int, int], float] = {}
            for i in range(n):
                ri = uf.find(i)
                for j in range(i + 1, n):
                    if adj[i, j] <= 0:
                        continue
                    rj = uf.find(j)
                    if ri == rj:
                        continue
                    pair = (min(ri, rj), max(ri, rj))
                    cluster_affinity[pair] = cluster_affinity.get(pair, 0.0) + adj[i, j]

            if not cluster_affinity:
                break

            # Find the pair with maximum affinity
            best_pair = max(cluster_affinity, key=cluster_affinity.get)
            uf.union(best_pair[0], best_pair[1])
            merge_levels.append([best_pair])

        return merge_levels

    def _cut_at_level(
        self, n: int, merge_levels: list[list[tuple[int, int]]], cut_level: int
    ) -> dict[int, int]:
        """Replay merges up to cut_level and return partition.

        Args:
            n: Number of nodes.
            merge_levels: Merge tree from Boruvka.
            cut_level: Number of levels to replay (0 = each node is own cluster).

        Returns:
            Dictionary mapping node → community ID.
        """
        uf = _UnionFind(n)
        for level_idx in range(min(cut_level, len(merge_levels))):
            for a, b in merge_levels[level_idx]:
                uf.union(a, b)

        # Map roots to contiguous IDs
        root_to_id: dict[int, int] = {}
        partition: dict[int, int] = {}
        for i in range(n):
            root = uf.find(i)
            if root not in root_to_id:
                root_to_id[root] = len(root_to_id)
            partition[i] = root_to_id[root]

        return partition

    def _compute_modularity(self, adj: np.ndarray, partition: dict[int, int]) -> float:
        """Compute modularity score for a partition.

        Q = sum_c [e_c/m - (d_c/(2m))^2]

        Args:
            adj: Weighted adjacency matrix.
            partition: Dictionary mapping node → community ID.

        Returns:
            Modularity score (higher is better, max is 1.0).
        """
        m = adj.sum() / 2
        if m == 0:
            return 0.0

        num_communities = max(partition.values()) + 1
        degrees = adj.sum(axis=1)

        Q = 0.0
        for c in range(num_communities):
            nodes_in_c = np.array([i for i, cid in partition.items() if cid == c])
            if len(nodes_in_c) == 0:
                continue
            e_c = adj[np.ix_(nodes_in_c, nodes_in_c)].sum() / 2
            d_c = degrees[nodes_in_c].sum()
            Q += e_c / m - (d_c / (2 * m)) ** 2

        return Q

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph into communities using affinity clustering.

        Runs Boruvka's algorithm to build a merge tree, then cuts at the
        level maximizing modularity.

        Args:
            data: PyTorch Geometric Data object with edge_index.

        Returns:
            List of sets containing node indices for each community.
        """
        n = data.num_nodes

        # Handle trivial cases
        if n <= 1:
            return [set(range(n))]

        adj = self._build_weighted_adj(data)

        if adj.sum() == 0:
            return [set(range(n))]

        # Build merge tree via Boruvka
        merge_levels = self._boruvka_merge_tree(adj)

        if not merge_levels:
            return [set(range(n))]

        # Find optimal cut level by modularity
        best_modularity = -float("inf")
        best_partition: dict[int, int] | None = None

        # Test cutting at each level (0 = all singletons, len = fully merged)
        for cut_level in range(len(merge_levels) + 1):
            part = self._cut_at_level(n, merge_levels, cut_level)
            num_clusters = max(part.values()) + 1

            # Skip trivial partitions (all singletons or all in one cluster)
            if num_clusters == n or num_clusters == 1:
                continue

            mod = self._compute_modularity(adj, part)
            if mod > best_modularity:
                best_modularity = mod
                best_partition = part

        if best_partition is None:
            return [set(range(n))]

        # Convert to list of sets
        communities: dict[int, set[int]] = {}
        for node, comm in best_partition.items():
            communities.setdefault(comm, set()).add(node)

        return list(communities.values())

    def build_hierarchy(self, data: Data) -> HierarchicalGraph:
        """Build hierarchical graph representation from affinity partitioning.

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
                feat = data.edge_attr[i]
                bond_type = int(feat.argmax()) if feat.dim() > 0 and feat.numel() > 1 else int(feat)
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
                feat = data.edge_attr[i]
                bond_type = int(feat.argmax()) if feat.dim() > 0 and feat.numel() > 1 else int(feat)
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

        Uses dict-based global-to-local index mapping for O(1) lookups.

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

                # Build O(1) lookup dicts
                left_global_to_local = {g: idx for idx, g in enumerate(left_nodes)}
                right_global_to_local = {g: idx for idx, g in enumerate(right_nodes)}

                # Find edges from left to right community
                bipart_edges = []
                bipart_edge_features = [] if edge_features_global is not None else None

                for e in range(edge_index_np.shape[1]):
                    src, dst = int(edge_index_np[0, e]), int(edge_index_np[1, e])

                    if src in left_set and dst in right_set:
                        local_src = left_global_to_local[src]
                        local_dst = right_global_to_local[dst]
                        bipart_edges.append((local_src, local_dst))

                        if edge_features_global is not None:
                            bond_type = edge_features_global.get((src, dst), 0)
                            bipart_edge_features.append(bond_type)

                if bipart_edges:
                    bipart_edge_index = torch.tensor(bipart_edges, dtype=torch.long).t()

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
        """Remap a child hierarchy's indices to parent's coordinate system.

        Recursively remaps all levels of nesting so that global_node_indices
        at every depth use the parent's coordinate system.

        Args:
            child_hg: Child hierarchy with local indices 0..k-1.
            parent_node_list: Parent's global node indices (sorted, length k).

        Returns:
            Child hierarchy with fully remapped indices at all depths.
        """
        remapped_partitions = []
        for part in child_hg.partitions:
            remapped_global = [
                parent_node_list[idx] for idx in part.global_node_indices
            ]

            # Recursively remap nested child hierarchies
            remapped_child = None
            if part.child_hierarchy is not None:
                remapped_child = self._remap_child_hierarchy(
                    part.child_hierarchy, parent_node_list
                )

            remapped_partitions.append(
                Partition(
                    part_id=part.part_id,
                    global_node_indices=remapped_global,
                    edge_index=part.edge_index.clone(),
                    child_hierarchy=remapped_child,
                    node_features=part.node_features,
                )
            )

        remapped_assignment = list(child_hg.community_assignment)

        return HierarchicalGraph(
            partitions=remapped_partitions,
            bipartites=child_hg.bipartites,
            community_assignment=remapped_assignment,
        )


# Backward compatibility alias
HACCoarsening = AffinityCoarsening
