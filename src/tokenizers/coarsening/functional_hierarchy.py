"""Functional hierarchy builder for HDTC tokenization.

This module provides the FunctionalHierarchyBuilder class that constructs
a two-level functional hierarchy from a molecular graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from src.tokenizers.motif.functional_detection import (
    FunctionalGroupDetector,
    FunctionalGroupInstance,
)
from src.tokenizers.motif.functional_patterns import (
    FUNCTIONAL_GROUP_PATTERNS,
    RING_PATTERNS,
)
from src.tokenizers.structures import (
    CommunityCommunityEdge,
    FunctionalCommunity,
    TwoLevelHierarchy,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data


class FunctionalHierarchyBuilder:
    """Builder for two-level functional hierarchies.

    Constructs a TwoLevelHierarchy from a molecular graph by:
    1. Detecting functional groups (rings and functional groups)
    2. Assigning atoms to communities (detected groups + singletons for remaining)
    3. Extracting internal edges per community
    4. Building super-edges between communities

    Attributes:
        include_rings: Whether to detect ring structures.
        ring_patterns: Custom ring SMARTS patterns.
        functional_patterns: Custom functional group patterns.
        detector: FunctionalGroupDetector instance.
    """

    def __init__(
        self,
        include_rings: bool = True,
        ring_patterns: dict[str, str] | None = None,
        functional_patterns: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the functional hierarchy builder.

        Args:
            include_rings: Whether to detect ring structures.
            ring_patterns: Custom ring SMARTS patterns. Defaults to RING_PATTERNS.
            functional_patterns: Custom functional group patterns.
                Defaults to FUNCTIONAL_GROUP_PATTERNS.
        """
        self.include_rings = include_rings
        self.ring_patterns = (
            ring_patterns if ring_patterns is not None else RING_PATTERNS
        )
        self.functional_patterns = (
            functional_patterns
            if functional_patterns is not None
            else FUNCTIONAL_GROUP_PATTERNS
        )
        self.detector = FunctionalGroupDetector(
            include_rings=include_rings,
            ring_patterns=self.ring_patterns,
            functional_patterns=self.functional_patterns,
        )

    def build(self, data: Data) -> TwoLevelHierarchy:
        """Build a two-level functional hierarchy from a graph.

        Args:
            data: PyTorch Geometric Data object with edge_index and num_nodes.
                Optionally contains 'smiles' for functional group detection
                and 'x' for node features.

        Returns:
            TwoLevelHierarchy representing the functional decomposition.
        """
        num_atoms = data.num_nodes

        # Build adjacency for edge lookup
        adjacency = self._build_adjacency(data)

        # Detect functional groups
        smiles = getattr(data, "smiles", None)
        if smiles is not None:
            detected_groups = self.detector.detect(smiles)
        else:
            detected_groups = []

        # Assign atoms to communities
        communities, atom_to_community = self._assign_communities(
            num_atoms, detected_groups, adjacency, data
        )

        # Build super-edges between communities
        super_edges = self._build_super_edges(adjacency, atom_to_community, data)

        # Extract node features
        node_features = data.x if hasattr(data, "x") and data.x is not None else None

        # Build edge features dictionary
        edge_features = self._build_edge_features(data)

        return TwoLevelHierarchy(
            communities=communities,
            super_edges=super_edges,
            atom_to_community=atom_to_community,
            num_atoms=num_atoms,
            node_features=node_features,
            edge_features=edge_features,
        )

    def _build_adjacency(self, data: Data) -> dict[int, set[int]]:
        """Build adjacency list from edge_index.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Dictionary mapping node index to set of neighbors.
        """
        adj: dict[int, set[int]] = defaultdict(set)

        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.numpy()
            for i in range(ei.shape[1]):
                src, dst = int(ei[0, i]), int(ei[1, i])
                if src != dst:
                    adj[src].add(dst)
                    adj[dst].add(src)

        return adj

    def _assign_communities(
        self,
        num_atoms: int,
        detected_groups: list[FunctionalGroupInstance],
        adjacency: dict[int, set[int]],
        data: Data,
    ) -> tuple[list[FunctionalCommunity], list[int]]:
        """Assign atoms to communities.

        Each detected functional group becomes a community.
        Remaining atoms become singleton communities.

        Args:
            num_atoms: Total number of atoms.
            detected_groups: List of detected functional groups.
            adjacency: Adjacency list of the graph.
            data: Original Data object for node features.

        Returns:
            Tuple of (list of FunctionalCommunity, atom_to_community mapping).
        """
        communities: list[FunctionalCommunity] = []
        atom_to_community: list[int] = [-1] * num_atoms
        assigned_atoms: set[int] = set()
        community_id = 0

        # Create communities from detected functional groups
        for group in detected_groups:
            atom_indices = sorted(group.atom_indices)

            # Get internal edges
            internal_edges = self._get_internal_edges(atom_indices, adjacency)

            # Determine community type
            if group.pattern_type == "ring":
                community_type = "ring"
            else:
                community_type = "functional"

            # Get node features for this community
            node_features = None
            if hasattr(data, "x") and data.x is not None:
                indices_tensor = torch.tensor(atom_indices, dtype=torch.long)
                node_features = data.x[indices_tensor]

            community = FunctionalCommunity(
                community_id=community_id,
                community_type=community_type,
                group_name=group.name,
                atom_indices=atom_indices,
                internal_edges=internal_edges,
                node_features=node_features,
            )
            communities.append(community)

            # Update assignment
            for atom in atom_indices:
                atom_to_community[atom] = community_id
                assigned_atoms.add(atom)

            community_id += 1

        # Create singleton communities for remaining atoms
        for atom in range(num_atoms):
            if atom not in assigned_atoms:
                # Get node features for this atom
                node_features = None
                if hasattr(data, "x") and data.x is not None:
                    node_features = data.x[atom : atom + 1]

                community = FunctionalCommunity(
                    community_id=community_id,
                    community_type="singleton",
                    group_name="singleton",
                    atom_indices=[atom],
                    internal_edges=[],
                    node_features=node_features,
                )
                communities.append(community)
                atom_to_community[atom] = community_id
                community_id += 1

        return communities, atom_to_community

    def _get_internal_edges(
        self,
        atom_indices: list[int],
        adjacency: dict[int, set[int]],
    ) -> list[tuple[int, int]]:
        """Get edges that are internal to a set of atoms.

        Args:
            atom_indices: List of atom indices in the community.
            adjacency: Adjacency list of the graph.

        Returns:
            List of (src, dst) tuples for internal edges.
        """
        atom_set = set(atom_indices)
        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for atom in atom_indices:
            for neighbor in adjacency.get(atom, set()):
                if neighbor in atom_set:
                    edge = (min(atom, neighbor), max(atom, neighbor))
                    if edge not in seen:
                        # Add both directions
                        edges.append((atom, neighbor))
                        edges.append((neighbor, atom))
                        seen.add(edge)

        return edges

    def _build_super_edges(
        self,
        adjacency: dict[int, set[int]],
        atom_to_community: list[int],
        data: Data,
    ) -> list[CommunityCommunityEdge]:
        """Build super-edges between communities.

        Args:
            adjacency: Adjacency list of the graph.
            atom_to_community: Mapping from atom index to community ID.
            data: Original Data object for edge features.

        Returns:
            List of CommunityCommunityEdge objects.
        """
        super_edges: list[CommunityCommunityEdge] = []
        seen: set[tuple[int, int, int, int]] = set()

        # Build edge features dict for lookup
        edge_features_dict = self._build_edge_features(data)

        for atom, neighbors in adjacency.items():
            src_comm = atom_to_community[atom]
            for neighbor in neighbors:
                dst_comm = atom_to_community[neighbor]

                # Only consider inter-community edges
                if src_comm != dst_comm:
                    # Use canonical key to avoid duplicates
                    if src_comm < dst_comm:
                        key = (src_comm, dst_comm, atom, neighbor)
                    else:
                        key = (dst_comm, src_comm, neighbor, atom)

                    if key not in seen:
                        # Get bond type if available
                        bond_type = 0
                        if edge_features_dict:
                            bond_type = edge_features_dict.get((atom, neighbor), 0)
                            if bond_type == 0:
                                bond_type = edge_features_dict.get((neighbor, atom), 0)

                        super_edge = CommunityCommunityEdge(
                            source_community=src_comm,
                            target_community=dst_comm,
                            source_atom=atom,
                            target_atom=neighbor,
                            bond_type=bond_type,
                        )
                        super_edges.append(super_edge)
                        seen.add(key)

        return super_edges

    def _build_edge_features(self, data: Data) -> dict[tuple[int, int], int] | None:
        """Build edge features dictionary from Data object.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Dictionary mapping (src, dst) to bond type, or None if not available.
        """
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            return None

        if data.edge_index is None or data.edge_index.numel() == 0:
            return None

        edge_features: dict[tuple[int, int], int] = {}
        ei = data.edge_index.numpy()
        ea = data.edge_attr.numpy()

        for i in range(ei.shape[1]):
            src, dst = int(ei[0, i]), int(ei[1, i])
            bond_type = int(ea[i]) if ea.ndim == 1 else int(ea[i, 0])
            edge_features[(src, dst)] = bond_type

        return edge_features
