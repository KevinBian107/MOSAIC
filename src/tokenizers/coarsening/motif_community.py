"""Motif-based community assignment for graph coarsening.

This module implements direct motif-based community assignment, where atoms
in the same structural motif (e.g., ring system) are assigned to the same
community without using spectral clustering.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data

from src.tokenizers.motif import (
    CLUSTERING_MOTIFS,
    MotifInstance,
    detect_motifs_from_data,
)
from src.tokenizers.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
)


class MotifCommunityCoarsening:
    """Direct motif-based community assignment.

    Assigns atoms to communities based on detected structural motifs.
    Atoms belonging to the same ring system or functional group are
    placed in the same community. Atoms not in any motif form singleton
    communities or are grouped with their neighbors.

    This approach:
    - Guarantees motif preservation (100% cohesion)
    - Creates variable-sized communities based on motif structure
    - May create many small communities for molecules with few motifs

    Attributes:
        motif_patterns: SMARTS patterns for motif detection.
        min_community_size: Minimum nodes for coarsening (controls depth).
        merge_singletons: Whether to merge singleton communities with neighbors.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        motif_patterns: dict[str, str] | None = None,
        min_community_size: int = 4,
        merge_singletons: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the motif community coarsening strategy.

        Args:
            motif_patterns: Dictionary mapping motif names to SMARTS patterns.
                Defaults to CLUSTERING_MOTIFS (ring-focused patterns).
            min_community_size: Minimum community size for further decomposition.
            merge_singletons: Whether to merge singleton communities (atoms not
                in any motif) with their neighbors.
            seed: Random seed for reproducibility.
        """
        self.motif_patterns = motif_patterns or CLUSTERING_MOTIFS
        self.min_community_size = min_community_size
        self.merge_singletons = merge_singletons
        self.seed = seed

    def partition(self, data: Data) -> list[set[int]]:
        """Partition graph into communities based on detected motifs.

        Args:
            data: PyTorch Geometric Data object with edge_index.
                Should have 'smiles' attribute for motif detection.

        Returns:
            List of sets containing node indices for each community.
        """
        n = data.num_nodes

        # Handle trivial cases
        if n <= 1:
            return [set(range(n))]

        # Detect motifs
        motifs = detect_motifs_from_data(data, self.motif_patterns)

        if not motifs:
            # No motifs detected - return single community
            return [set(range(n))]

        # Assign atoms to communities based on motif membership
        communities = self._assign_communities(n, motifs, data)

        # Optionally merge singleton communities
        if self.merge_singletons:
            communities = self._merge_singletons(communities, data)

        return communities

    def _assign_communities(
        self,
        n: int,
        motifs: list[MotifInstance],
        data: Data,
    ) -> list[set[int]]:
        """Assign atoms to communities based on motif membership.

        Uses a union-find approach: atoms in overlapping motifs are merged
        into the same community.

        Args:
            n: Number of nodes.
            motifs: List of detected motif instances.
            data: Original graph data.

        Returns:
            List of sets containing node indices for each community.
        """
        # Union-find for merging overlapping motifs
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union atoms in the same motif
        for motif in motifs:
            atoms = sorted(motif.atom_indices)
            if len(atoms) < 2:
                continue
            for i in range(1, len(atoms)):
                union(atoms[0], atoms[i])

        # Group atoms by their root
        communities_dict: dict[int, set[int]] = {}
        for node in range(n):
            root = find(node)
            if root not in communities_dict:
                communities_dict[root] = set()
            communities_dict[root].add(node)

        return list(communities_dict.values())

    def _merge_singletons(
        self,
        communities: list[set[int]],
        data: Data,
    ) -> list[set[int]]:
        """Merge singleton communities with their neighbors.

        Args:
            communities: Initial community assignment.
            data: Graph data for adjacency information.

        Returns:
            Communities with singletons merged.
        """
        # Build adjacency
        adj: dict[int, set[int]] = {i: set() for i in range(data.num_nodes)}
        if data.edge_index.numel() > 0:
            ei = data.edge_index.numpy()
            for e in range(ei.shape[1]):
                src, dst = int(ei[0, e]), int(ei[1, e])
                adj[src].add(dst)
                adj[dst].add(src)

        # Map node to community index
        node_to_comm: dict[int, int] = {}
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = comm_idx

        # Find singletons and merge with neighbor's community
        merged = [set(c) for c in communities]  # Copy
        to_remove: set[int] = set()

        for comm_idx, comm in enumerate(communities):
            if len(comm) == 1:
                node = next(iter(comm))
                neighbors = adj.get(node, set())

                # Find a non-singleton neighbor's community
                for neighbor in neighbors:
                    neighbor_comm = node_to_comm.get(neighbor)
                    if neighbor_comm is not None and len(communities[neighbor_comm]) > 1:
                        # Merge this singleton into neighbor's community
                        merged[neighbor_comm].add(node)
                        to_remove.add(comm_idx)
                        break

        # Remove empty communities
        result = [c for i, c in enumerate(merged) if i not in to_remove]
        return result

    def build_hierarchy(self, data: Data) -> HierarchicalGraph:
        """Build hierarchical graph representation from motif partitioning.

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
                from torch_geometric.utils import subgraph

                sub_edge_index, _ = subgraph(
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

            partitions.append(
                Partition(
                    part_id=comm_id,
                    global_node_indices=node_list,
                    edge_index=sub_edge_index,
                    child_hierarchy=None,
                    node_features=part_node_features,
                )
            )

        # Extract bipartites (off-diagonal blocks)
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

                bipart_edges = []
                bipart_edge_features = (
                    [] if edge_features_global is not None else None
                )

                for e in range(edge_index_np.shape[1]):
                    src, dst = int(edge_index_np[0, e]), int(edge_index_np[1, e])

                    if src in left_set and dst in right_set:
                        local_src = left_nodes.index(src)
                        local_dst = right_nodes.index(dst)
                        bipart_edges.append((local_src, local_dst))

                        if edge_features_global is not None:
                            bond_type = edge_features_global.get((src, dst), 0)
                            bipart_edge_features.append(bond_type)

                if bipart_edges:
                    bipart_edge_index = torch.tensor(
                        bipart_edges, dtype=torch.long
                    ).t()

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
