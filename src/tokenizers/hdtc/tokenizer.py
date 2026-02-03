"""HDTC (HDT Compositional) Tokenizer.

This module implements HDTC, a compositional tokenizer that builds a two-level
functional hierarchy:
- Level 1: Functional communities (rings, functional groups, singletons)
- Level 2: Super-graph showing how communities connect

Key differences from HDT:
- Uses functional group detection instead of spectral clustering
- Two-level hierarchy (flat) instead of recursive hierarchy
- Explicit super-graph representation
- Guarantees functional group preservation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.coarsening.functional_hierarchy import FunctionalHierarchyBuilder
from src.tokenizers.motif.functional_patterns import (
    FUNCTIONAL_GROUP_PATTERNS,
    RING_PATTERNS,
)
from src.tokenizers.ordering import OrderingMethod, order_partition_nodes
from src.tokenizers.structures import (
    CommunityCommunityEdge,
    FunctionalCommunity,
    Partition,
    TwoLevelHierarchy,
)


class HDTCTokenizer(Tokenizer):
    """HDTC tokenizer for compositional graph-to-sequence conversion.

    Uses a two-level functional hierarchy with explicit community and
    super-graph encoding.

    Token vocabulary:
        0: SOS (start of sequence)
        1: EOS (end of sequence)
        2: PAD (padding)
        3: COMM_START (start community block)
        4: COMM_END (end community block)
        5: LEDGE (left edge bracket)
        6: REDGE (right edge bracket)
        7: SUPER_START (start super-graph block)
        8: SUPER_END (end super-graph block)
        9: TYPE_RING (community type: ring)
        10: TYPE_FUNC (community type: functional group)
        11: TYPE_SINGLETON (community type: singleton)
        12+: Node indices (offset by IDX_OFFSET)

    Sequence format:
        [SOS]
        {COMM_START TYPE_X <comm_id> <atoms...> COMM_END}*
        [SUPER_START <src_comm> <dst_comm> <src_atom> <dst_atom> ... SUPER_END]
        [EOS]

    Attributes:
        node_order: Ordering method for nodes within communities.
        max_length: Maximum sequence length (-1 for unlimited).
        truncation_length: Length for truncation during batching.
        include_rings: Whether to detect ring structures.
        labeled_graph: Whether to encode node/edge features.
        hierarchy_builder: FunctionalHierarchyBuilder instance.
    """

    # Tokenizer type identifier
    tokenizer_type: str = "hdtc"

    # Special token IDs
    SOS: int = 0
    EOS: int = 1
    PAD: int = 2
    COMM_START: int = 3
    COMM_END: int = 4
    LEDGE: int = 5
    REDGE: int = 6
    SUPER_START: int = 7
    SUPER_END: int = 8
    TYPE_RING: int = 9
    TYPE_FUNC: int = 10
    TYPE_SINGLETON: int = 11
    IDX_OFFSET: int = 12

    # Token names for debugging
    SPECIAL_TOKEN_NAMES = {
        0: "SOS",
        1: "EOS",
        2: "PAD",
        3: "COMM_START",
        4: "COMM_END",
        5: "LEDGE",
        6: "REDGE",
        7: "SUPER_START",
        8: "SUPER_END",
        9: "TYPE_RING",
        10: "TYPE_FUNC",
        11: "TYPE_SINGLETON",
    }

    # Community type to token mapping
    COMMUNITY_TYPE_TOKENS = {
        "ring": 9,
        "functional": 10,
        "singleton": 11,
    }

    # Token to community type mapping
    TOKEN_TO_COMMUNITY_TYPE = {
        9: "ring",
        10: "functional",
        11: "singleton",
    }

    # Override base class attributes
    sos: int = 0
    eos: int = 1
    pad: int = 2

    def __init__(
        self,
        node_order: OrderingMethod = "BFS",
        max_length: int = -1,
        truncation_length: Optional[int] = None,
        undirected: bool = True,
        seed: Optional[int] = None,
        include_rings: bool = True,
        ring_patterns: Optional[dict[str, str]] = None,
        functional_patterns: Optional[dict[str, tuple[str, str]]] = None,
        labeled_graph: bool = False,
    ) -> None:
        """Initialize the HDTC tokenizer.

        Args:
            node_order: Ordering method for nodes within communities.
            max_length: Maximum sequence length (-1 for unlimited).
            truncation_length: Length for truncation during batching.
            undirected: Whether to treat graphs as undirected.
            seed: Random seed for reproducibility.
            include_rings: Whether to detect ring structures.
            ring_patterns: Custom ring SMARTS patterns.
            functional_patterns: Custom functional group patterns.
            labeled_graph: Whether to encode node/edge features.
        """
        self.node_order = node_order
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.undirected = undirected
        self.seed = seed
        self.include_rings = include_rings

        self.ring_patterns = (
            ring_patterns if ring_patterns is not None else RING_PATTERNS
        )
        self.functional_patterns = (
            functional_patterns
            if functional_patterns is not None
            else FUNCTIONAL_GROUP_PATTERNS
        )

        # Initialize hierarchy builder
        self.hierarchy_builder = FunctionalHierarchyBuilder(
            include_rings=include_rings,
            ring_patterns=self.ring_patterns,
            functional_patterns=self.functional_patterns,
        )

        self.max_num_nodes: Optional[int] = None

        # Labeled graph support
        self.labeled_graph = labeled_graph
        self.num_node_types: int = 0
        self.num_edge_types: int = 0
        self.node_idx_offset: int = 0
        self.edge_idx_offset: int = 0

    def set_num_nodes(self, max_num_nodes: int) -> None:
        """Set maximum number of nodes for vocabulary sizing.

        Args:
            max_num_nodes: Maximum nodes in any graph.
        """
        if self.max_num_nodes is None or self.max_num_nodes < max_num_nodes:
            self.max_num_nodes = max_num_nodes

    def set_num_node_and_edge_types(
        self, num_node_types: int, num_edge_types: int
    ) -> None:
        """Set the number of node and edge types for labeled graphs.

        Args:
            num_node_types: Number of distinct node types.
            num_edge_types: Number of distinct edge types.

        Raises:
            ValueError: If labeled_graph=False.
        """
        if not self.labeled_graph:
            raise ValueError("Cannot set node/edge types when labeled_graph=False")

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Vocabulary layout:
        # [special tokens (12)] + [node IDs (max_num_nodes)] + [atom types] + [bond types]
        self.node_idx_offset = self.IDX_OFFSET + self.max_num_nodes
        self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size.

        Unlabeled: IDX_OFFSET (12) + max_num_nodes
        Labeled: IDX_OFFSET (12) + max_num_nodes + num_node_types + num_edge_types
        """
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes() first")

        base_size = self.IDX_OFFSET + self.max_num_nodes
        if self.labeled_graph:
            return base_size + self.num_node_types + self.num_edge_types
        return base_size

    # =====================================================================
    # TOKENIZATION: Graph -> Tokens
    # =====================================================================

    def tokenize(self, data: Data) -> Tensor:
        """Tokenize graph via two-level functional hierarchy.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            1D tensor of token indices.
        """
        hierarchy = self.hierarchy_builder.build(data)
        return self.tokenize_hierarchy(hierarchy)

    def tokenize_hierarchy(self, hierarchy: TwoLevelHierarchy) -> Tensor:
        """Tokenize a TwoLevelHierarchy.

        Args:
            hierarchy: TwoLevelHierarchy to tokenize.

        Returns:
            1D tensor of token indices.
        """
        tokens: list[int] = [self.SOS]
        visited_atoms: list[int] = []

        # Build full adjacency for back-edge detection
        full_adj = self._build_full_adjacency(hierarchy)

        # Serialize each community
        for comm in hierarchy.communities:
            self._serialize_community(comm, tokens, visited_atoms, full_adj, hierarchy)

        # Serialize super-graph
        self._serialize_super_graph(hierarchy.super_edges, tokens)

        tokens.append(self.EOS)

        # Truncate if needed
        if self.max_length > 0 and len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.EOS]

        # Validate token IDs are within vocab bounds
        if self.max_num_nodes is not None:
            vocab_size = self.vocab_size
            for tok in tokens:
                if tok >= vocab_size:
                    raise ValueError(
                        f"Token ID {tok} exceeds vocab_size {vocab_size}. "
                        f"max_num_nodes={self.max_num_nodes}, "
                        f"num_node_types={self.num_node_types}, "
                        f"num_edge_types={self.num_edge_types}"
                    )

        return torch.tensor(tokens, dtype=torch.long)

    def _build_full_adjacency(
        self, hierarchy: TwoLevelHierarchy
    ) -> dict[int, set[int]]:
        """Build full adjacency from hierarchy.

        Args:
            hierarchy: TwoLevelHierarchy.

        Returns:
            Dictionary mapping atom index to set of neighbors.
        """
        adj: dict[int, set[int]] = defaultdict(set)

        # Add internal edges from each community
        for comm in hierarchy.communities:
            for src, dst in comm.internal_edges:
                adj[src].add(dst)
                adj[dst].add(src)

        # Add super-edges
        for se in hierarchy.super_edges:
            adj[se.source_atom].add(se.target_atom)
            adj[se.target_atom].add(se.source_atom)

        return adj

    def _serialize_community(
        self,
        comm: FunctionalCommunity,
        tokens: list[int],
        visited_atoms: list[int],
        full_adj: dict[int, set[int]],
        hierarchy: TwoLevelHierarchy,
    ) -> None:
        """Serialize a community to tokens.

        Format: COMM_START TYPE_X <comm_id> <atoms with back-edges> COMM_END

        Args:
            comm: FunctionalCommunity to serialize.
            tokens: Token list to append to.
            visited_atoms: List of already visited atoms.
            full_adj: Full adjacency map.
            hierarchy: Parent hierarchy for feature lookup.
        """
        tokens.append(self.COMM_START)

        # Emit community type
        type_token = self.COMMUNITY_TYPE_TOKENS.get(
            comm.community_type, self.TYPE_SINGLETON
        )
        tokens.append(type_token)

        # Emit community ID
        tokens.append(self.IDX_OFFSET + comm.community_id)

        # Create a temporary partition for node ordering
        # Map global indices to local indices
        global_to_local = {g: i for i, g in enumerate(comm.atom_indices)}

        # Convert internal edges to local indices
        local_edges = []
        for src, dst in comm.internal_edges:
            if src in global_to_local and dst in global_to_local:
                local_edges.append((global_to_local[src], global_to_local[dst]))

        if local_edges:
            edge_index = torch.tensor(local_edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        temp_partition = Partition(
            part_id=comm.community_id,
            global_node_indices=list(comm.atom_indices),
            edge_index=edge_index,
            child_hierarchy=None,
            node_features=comm.node_features,
        )

        # Get canonical node ordering
        node_order = order_partition_nodes(temp_partition, self.node_order, self.seed)

        # Serialize atoms with back-edges
        for local_idx in node_order:
            global_idx = comm.atom_indices[local_idx]
            tokens.append(self.IDX_OFFSET + global_idx)

            # Add atom type if labeled graph
            if self.labeled_graph and comm.node_features is not None:
                if local_idx < len(comm.node_features):
                    atom_type = int(comm.node_features[local_idx])
                    tokens.append(self.node_idx_offset + atom_type)

            # Find back-edges to previously visited atoms
            back_edges = self._find_back_edges(global_idx, visited_atoms, full_adj)

            if back_edges:
                tokens.append(self.LEDGE)
                for target_global in back_edges:
                    tokens.append(self.IDX_OFFSET + target_global)

                    # Add bond type if labeled graph
                    if self.labeled_graph and hierarchy.edge_features:
                        bond_type = hierarchy.edge_features.get(
                            (global_idx, target_global), 0
                        )
                        if bond_type == 0:
                            bond_type = hierarchy.edge_features.get(
                                (target_global, global_idx), 0
                            )
                        tokens.append(self.edge_idx_offset + bond_type)

                tokens.append(self.REDGE)

            visited_atoms.append(global_idx)

        tokens.append(self.COMM_END)

    def _find_back_edges(
        self,
        global_idx: int,
        visited_atoms: list[int],
        full_adj: dict[int, set[int]],
    ) -> list[int]:
        """Find edges from current atom to previously visited atoms.

        Args:
            global_idx: Global index of current atom.
            visited_atoms: List of already visited atom indices.
            full_adj: Full adjacency map.

        Returns:
            Sorted list of global indices of back-edge targets.
        """
        neighbors = full_adj.get(global_idx, set())
        visited_set = set(visited_atoms)
        back_targets = [n for n in neighbors if n in visited_set]
        return sorted(back_targets)

    def _serialize_super_graph(
        self,
        super_edges: list[CommunityCommunityEdge],
        tokens: list[int],
    ) -> None:
        """Serialize the super-graph edges.

        Format: SUPER_START [<src_comm> <dst_comm> <src_atom> <dst_atom>]* SUPER_END

        Args:
            super_edges: List of super-edges.
            tokens: Token list to append to.
        """
        if not super_edges:
            return

        tokens.append(self.SUPER_START)

        for se in super_edges:
            tokens.append(self.IDX_OFFSET + se.source_community)
            tokens.append(self.IDX_OFFSET + se.target_community)
            tokens.append(self.IDX_OFFSET + se.source_atom)
            tokens.append(self.IDX_OFFSET + se.target_atom)

        tokens.append(self.SUPER_END)

    # =====================================================================
    # DECODING: Tokens -> Graph
    # =====================================================================

    def decode(self, tokens: Tensor) -> Data:
        """Decode tokens to graph via two-level hierarchy.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            PyTorch Geometric Data object.
        """
        hierarchy = self.parse_tokens(tokens)
        return hierarchy.reconstruct()

    def parse_tokens(self, tokens: Tensor) -> TwoLevelHierarchy:
        """Parse HDTC tokens back to TwoLevelHierarchy.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            Reconstructed TwoLevelHierarchy.
        """
        # Remove special tokens (PAD, SOS, EOS)
        mask = (tokens != self.PAD) & (tokens != self.SOS) & (tokens != self.EOS)
        tokens_list = tokens[mask].tolist()

        if not tokens_list:
            return TwoLevelHierarchy([], [], [], 0)

        communities: list[FunctionalCommunity] = []
        super_edges: list[CommunityCommunityEdge] = []
        all_atoms: set[int] = set()
        all_edges: list[tuple[int, int]] = []
        node_features_dict: dict[int, int] = {}
        edge_features_dict: dict[tuple[int, int], int] = {}

        idx = 0
        while idx < len(tokens_list):
            tok = tokens_list[idx]

            if tok == self.COMM_START:
                comm, idx = self._parse_community(
                    tokens_list,
                    idx,
                    all_atoms,
                    all_edges,
                    node_features_dict,
                    edge_features_dict,
                )
                if comm is not None:
                    communities.append(comm)

            elif tok == self.SUPER_START:
                super_edges, idx = self._parse_super_graph(tokens_list, idx)

            else:
                idx += 1

        # Build atom_to_community mapping
        num_atoms = max(all_atoms) + 1 if all_atoms else 0
        atom_to_community = [-1] * num_atoms
        for comm in communities:
            for atom in comm.atom_indices:
                atom_to_community[atom] = comm.community_id

        # Build node features tensor
        node_features = None
        if self.labeled_graph and node_features_dict:
            node_features = torch.zeros(num_atoms, dtype=torch.long)
            for atom_idx, atom_type in node_features_dict.items():
                node_features[atom_idx] = atom_type

        return TwoLevelHierarchy(
            communities=communities,
            super_edges=super_edges,
            atom_to_community=atom_to_community,
            num_atoms=num_atoms,
            node_features=node_features,
            edge_features=edge_features_dict if self.labeled_graph else None,
        )

    def _parse_community(
        self,
        tokens: list[int],
        idx: int,
        all_atoms: set[int],
        all_edges: list[tuple[int, int]],
        node_features_dict: dict[int, int],
        edge_features_dict: dict[tuple[int, int], int],
    ) -> tuple[Optional[FunctionalCommunity], int]:
        """Parse a community block from tokens.

        Args:
            tokens: Token list.
            idx: Current index (at COMM_START).
            all_atoms: Set to add atoms to.
            all_edges: List to add edges to.
            node_features_dict: Dict to add node features to.
            edge_features_dict: Dict to add edge features to.

        Returns:
            Tuple of (FunctionalCommunity or None, next index).
        """
        if tokens[idx] != self.COMM_START:
            return None, idx

        idx += 1  # Skip COMM_START

        # Parse community type
        if idx >= len(tokens):
            return None, idx

        type_token = tokens[idx]
        community_type = self.TOKEN_TO_COMMUNITY_TYPE.get(type_token, "singleton")
        idx += 1

        # Parse community ID
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx

        community_id = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse atoms and edges
        atom_indices: list[int] = []
        internal_edges: list[tuple[int, int]] = []
        current_atom: Optional[int] = None

        while idx < len(tokens) and tokens[idx] != self.COMM_END:
            tok = tokens[idx]

            if tok >= self.IDX_OFFSET:
                # Check if this is a node ID or feature token
                if self.labeled_graph and tok >= self.node_idx_offset:
                    # This is an atom type token
                    if current_atom is not None:
                        atom_type = tok - self.node_idx_offset
                        node_features_dict[current_atom] = atom_type
                    idx += 1
                elif self.labeled_graph and tok >= self.edge_idx_offset:
                    # This is a bond type token - skip (handled in LEDGE block)
                    idx += 1
                else:
                    # This is a node ID
                    atom_idx = tok - self.IDX_OFFSET
                    atom_indices.append(atom_idx)
                    all_atoms.add(atom_idx)
                    current_atom = atom_idx
                    idx += 1

            elif tok == self.LEDGE:
                # Back-edge block
                idx += 1
                while idx < len(tokens) and tokens[idx] != self.REDGE:
                    if tokens[idx] >= self.IDX_OFFSET:
                        if self.labeled_graph and tokens[idx] >= self.edge_idx_offset:
                            # Bond type token - already handled
                            idx += 1
                        elif self.labeled_graph and tokens[idx] >= self.node_idx_offset:
                            # Atom type in wrong place - skip
                            idx += 1
                        else:
                            # Target node
                            target = tokens[idx] - self.IDX_OFFSET
                            if current_atom is not None:
                                # Add edge
                                internal_edges.append((current_atom, target))
                                internal_edges.append((target, current_atom))
                                all_edges.append((current_atom, target))
                                all_edges.append((target, current_atom))

                                # Check for bond type
                                idx += 1
                                if self.labeled_graph:
                                    if (
                                        idx < len(tokens)
                                        and tokens[idx] >= self.edge_idx_offset
                                    ):
                                        bond_type = tokens[idx] - self.edge_idx_offset
                                        edge_features_dict[(current_atom, target)] = (
                                            bond_type
                                        )
                                        edge_features_dict[(target, current_atom)] = (
                                            bond_type
                                        )
                                        idx += 1
                            else:
                                idx += 1
                    else:
                        idx += 1
                idx += 1  # Skip REDGE

            else:
                idx += 1

        # Skip COMM_END
        if idx < len(tokens) and tokens[idx] == self.COMM_END:
            idx += 1

        # Determine group name based on type
        group_name = community_type if community_type != "singleton" else "singleton"

        # Build node features tensor for this community
        node_features = None
        if self.labeled_graph and node_features_dict:
            node_features = torch.zeros(len(atom_indices), dtype=torch.long)
            for local_idx, global_idx in enumerate(atom_indices):
                if global_idx in node_features_dict:
                    node_features[local_idx] = node_features_dict[global_idx]

        return (
            FunctionalCommunity(
                community_id=community_id,
                community_type=community_type,
                group_name=group_name,
                atom_indices=atom_indices,
                internal_edges=internal_edges,
                node_features=node_features,
            ),
            idx,
        )

    def _parse_super_graph(
        self,
        tokens: list[int],
        idx: int,
    ) -> tuple[list[CommunityCommunityEdge], int]:
        """Parse super-graph block from tokens.

        Args:
            tokens: Token list.
            idx: Current index (at SUPER_START).

        Returns:
            Tuple of (list of super-edges, next index).
        """
        if tokens[idx] != self.SUPER_START:
            return [], idx

        idx += 1  # Skip SUPER_START

        super_edges: list[CommunityCommunityEdge] = []

        while idx + 3 < len(tokens) and tokens[idx] != self.SUPER_END:
            if tokens[idx] >= self.IDX_OFFSET:
                src_comm = tokens[idx] - self.IDX_OFFSET
                dst_comm = tokens[idx + 1] - self.IDX_OFFSET
                src_atom = tokens[idx + 2] - self.IDX_OFFSET
                dst_atom = tokens[idx + 3] - self.IDX_OFFSET

                super_edges.append(
                    CommunityCommunityEdge(
                        source_community=src_comm,
                        target_community=dst_comm,
                        source_atom=src_atom,
                        target_atom=dst_atom,
                        bond_type=0,
                    )
                )
                idx += 4
            else:
                idx += 1

        # Skip SUPER_END
        if idx < len(tokens) and tokens[idx] == self.SUPER_END:
            idx += 1

        return super_edges, idx

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================

    def batch_converter(self) -> Callable[[Sequence[Tensor]], Tensor]:
        """Return the batch conversion function.

        Returns:
            BatchConverter instance.
        """
        return BatchConverter(self, self.truncation_length)

    def tokens_to_string(self, tokens: Tensor) -> str:
        """Convert token tensor to human-readable string for debugging.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            String representation of the token sequence.
        """
        parts: list[str] = []
        tokens_list = tokens.tolist()

        for tok in tokens_list:
            if tok in self.SPECIAL_TOKEN_NAMES:
                parts.append(f"[{self.SPECIAL_TOKEN_NAMES[tok]}]")
            elif tok >= self.IDX_OFFSET:
                parts.append(str(tok - self.IDX_OFFSET))
            else:
                parts.append(f"?{tok}")

        return " ".join(parts)
