"""Hierarchical SENT (H-SENT) tokenizer.

This module implements a hierarchical extension of SENT tokenization that
combines HiGen's hierarchical graph construction with SENT-style sequential
encoding for transformer-based generation.

Key features:
- Hierarchical structure encoded via nested [LCOM ... RCOM] blocks
- SENT-style back-edge encoding within partitions
- Bipartite edges encoded as pairs between communities
- Supports arbitrary depth via recursive tokenization
"""

import warnings
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.coarsening import (
    AffinityCoarsening,
    MotifAwareCoarsening,
    MotifCommunityCoarsening,
    SimpleSpectralCoarsening,
    SpectralCoarsening,
)
from src.tokenizers.ordering import OrderingMethod, order_partition_nodes
from src.tokenizers.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
)


class HSENTTokenizer(Tokenizer):
    """Hierarchical SENT tokenizer for graph-to-sequence conversion.

    Combines HiGen's hierarchical graph construction with SENT-style
    sequential encoding. Supports arbitrary hierarchy depth controlled
    by min_community_size parameter.

    Token vocabulary:
        0: SOS (start of sequence)
        1: EOS (end of sequence)
        2: PAD (padding)
        3: RESET (component restart, from SENT)
        4: LADJ (left back-edge bracket)
        5: RADJ (right back-edge bracket)
        6: LCOM (start community block)
        7: RCOM (end community block)
        8: LBIP (start bipartite block)
        9: RBIP (end bipartite block)
        10: SEP (separator)
        11+: Node indices (offset by IDX_OFFSET)

    Attributes:
        node_order: Ordering method for nodes within partitions.
        max_length: Maximum sequence length (-1 for unlimited).
        truncation_length: Length for truncation during batching.
        undirected: Whether to treat graphs as undirected.
        seed: Random seed for reproducibility.
        min_community_size: Minimum size for community decomposition.
        coarsener: Coarsening instance for hierarchy building.
        max_num_nodes: Maximum nodes (determines vocab size).
        motif_aware: Whether to use motif-aware coarsening.
        motif_alpha: Weight for motif affinity (if motif_aware=True).
    """

    # Tokenizer type identifier
    tokenizer_type: str = "hsent"

    # Special token IDs
    SOS: int = 0
    EOS: int = 1
    PAD: int = 2
    RESET: int = 3
    LADJ: int = 4
    RADJ: int = 5
    LCOM: int = 6
    RCOM: int = 7
    LBIP: int = 8
    RBIP: int = 9
    SEP: int = 10
    IDX_OFFSET: int = 11

    # Token names for debugging
    SPECIAL_TOKEN_NAMES = {
        0: "SOS",
        1: "EOS",
        2: "PAD",
        3: "RESET",
        4: "LADJ",
        5: "RADJ",
        6: "LCOM",
        7: "RCOM",
        8: "LBIP",
        9: "RBIP",
        10: "SEP",
    }

    # Override base class attributes
    sos: int = 0
    eos: int = 1
    pad: int = 2

    # Type alias for coarsening strategy
    CoarseningStrategyType = Literal[
        "spectral", "motif_aware_spectral", "motif_community", "hac"
    ]

    def __init__(
        self,
        node_order: OrderingMethod = "BFS",
        max_length: int = -1,
        truncation_length: Optional[int] = None,
        undirected: bool = True,
        seed: Optional[int] = None,
        min_community_size: int = 4,
        k_min_factor: float = 0.7,
        k_max_factor: float = 1.3,
        n_init: int = 10,
        coarsening_strategy: Optional[CoarseningStrategyType] = None,
        motif_aware: bool = False,
        motif_alpha: float = 1.0,
        motif_patterns: Optional[dict[str, str]] = None,
        normalize_by_motif_size: bool = False,
        labeled_graph: bool = False,
    ) -> None:
        """Initialize the H-SENT tokenizer.

        Args:
            node_order: Ordering method for nodes within partitions.
            max_length: Maximum sequence length (-1 for unlimited).
            truncation_length: Length for truncation during batching.
            undirected: Whether to treat graphs as undirected.
            seed: Random seed for reproducibility.
            min_community_size: Minimum community size for decomposition.
                Communities smaller than this become leaf partitions.
            k_min_factor: Factor for minimum cluster count (default 0.7, optimized).
            k_max_factor: Factor for maximum cluster count (default 1.3, optimized).
            n_init: Number of spectral clustering initializations (default 10, optimized).
            coarsening_strategy: Strategy for graph coarsening. Options:
                - "spectral": Standard spectral clustering with optimizations (default)
                  Uses n_init=10, k=[0.7,1.3] for 25x speedup with equivalent quality
                - "motif_aware_spectral": Spectral clustering with motif preservation
                - "motif_community": Direct motif-based community assignment
                - "hac": Boruvka-based affinity clustering with modularity-optimal cut
                - "simple_spectral": DEPRECATED - Single-level spectral (no hierarchy)
            motif_aware: DEPRECATED. Use coarsening_strategy="motif_aware_spectral".
            motif_alpha: Weight for motif affinity matrix (only used with
                motif-aware strategies). Higher values = stronger motif preservation.
            motif_patterns: Custom SMARTS patterns for motif detection.
            normalize_by_motif_size: Normalize motif contributions by 1/motif_size.
            labeled_graph: Whether to encode node/edge features (atom/bond types).
        """
        self.node_order = node_order
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.undirected = undirected
        self.seed = seed
        self.min_community_size = min_community_size
        self.motif_alpha = motif_alpha

        # Handle backwards compatibility for motif_aware parameter
        if coarsening_strategy is None:
            if motif_aware:
                warnings.warn(
                    "motif_aware parameter is deprecated. "
                    "Use coarsening_strategy='motif_aware_spectral' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                coarsening_strategy = "motif_aware_spectral"
            else:
                coarsening_strategy = "spectral"

        self.coarsening_strategy = coarsening_strategy
        self.motif_aware = coarsening_strategy in (
            "motif_aware_spectral",
            "motif_community",
        )

        # Select coarsening strategy
        if coarsening_strategy == "motif_aware_spectral":
            self.coarsener = MotifAwareCoarsening(
                alpha=motif_alpha,
                motif_patterns=motif_patterns,
                normalize_by_motif_size=normalize_by_motif_size,
                k_min_factor=k_min_factor,
                k_max_factor=k_max_factor,
                n_init=n_init,
                min_community_size=min_community_size,
                seed=seed,
            )
        elif coarsening_strategy == "motif_community":
            self.coarsener = MotifCommunityCoarsening(
                motif_patterns=motif_patterns,
                min_community_size=min_community_size,
                seed=seed,
            )
        elif coarsening_strategy == "hac":
            self.coarsener = AffinityCoarsening(
                min_community_size=min_community_size,
                seed=seed,
            )
        elif coarsening_strategy == "simple_spectral":
            self.coarsener = SimpleSpectralCoarsening(
                k_min_factor=k_min_factor,
                k_max_factor=k_max_factor,
                n_init=n_init,
                seed=seed,
            )
        else:  # Default: spectral
            self.coarsener = SpectralCoarsening(
                k_min_factor=k_min_factor,
                k_max_factor=k_max_factor,
                n_init=n_init,
                min_community_size=min_community_size,
                seed=seed,
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

        This method must be called when labeled_graph=True before tokenization.
        Updates offsets for node and edge type tokens in the vocabulary.

        Args:
            num_node_types: Number of distinct node types (e.g., atom types).
            num_edge_types: Number of distinct edge types (e.g., bond types).

        Raises:
            ValueError: If labeled_graph=False.
        """
        if not self.labeled_graph:
            raise ValueError("Cannot set node/edge types when labeled_graph=False")

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Offsets in vocabulary:
        # [SOS, EOS, PAD, RESET, LADJ, RADJ, LCOM, RCOM, LBIP, RBIP, SEP] = 11 special tokens
        # Then: [0, 1, 2, ..., max_num_nodes-1] = node IDs
        # Then: [atom_type_0, ..., atom_type_{num_node_types-1}] = atom types
        # Then: [bond_type_0, ..., bond_type_{num_edge_types-1}] = bond types
        self.node_idx_offset = self.IDX_OFFSET + self.max_num_nodes
        self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size.

        Unlabeled: IDX_OFFSET (11) + max_num_nodes
        Labeled: IDX_OFFSET (11) + max_num_nodes + num_node_types + num_edge_types
        """
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes() first")

        base_size = self.IDX_OFFSET + self.max_num_nodes
        if self.labeled_graph:
            return base_size + self.num_node_types + self.num_edge_types
        return base_size

    # =====================================================================
    # TOKENIZATION: Graph → Tokens
    # =====================================================================

    def tokenize(self, data: Data) -> Tensor:
        """Tokenize graph via hierarchical representation.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            1D tensor of token indices.
        """
        hg = self.coarsener.build_hierarchy(data)
        return self.tokenize_hierarchy(hg)

    def tokenize_hierarchy(self, hg: HierarchicalGraph) -> Tensor:
        """Tokenize a HierarchicalGraph to token sequence.

        Separated from tokenize() for testing the roundtrip at the
        hierarchy level.

        Args:
            hg: HierarchicalGraph to tokenize.

        Returns:
            1D tensor of token indices.
        """
        tokens: list[int] = [self.SOS]

        # Encode number of communities at this level
        tokens.append(self.IDX_OFFSET + hg.num_communities)

        # Encode each partition (recursively if nested)
        for part in hg.partitions:
            tokens.extend(self._tokenize_partition(part, root_hg=hg))

        # Encode each bipartite
        for bipart in hg.bipartites:
            tokens.extend(self._tokenize_bipartite(bipart))

        tokens.append(self.EOS)

        # Truncate if needed
        if self.max_length > 0 and len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.EOS]

        return torch.tensor(tokens, dtype=torch.long)

    def _tokenize_partition(
        self, part: Partition, root_hg: Optional[HierarchicalGraph] = None
    ) -> list[int]:
        """Encode a partition using SENT-style walk with back-edges.

        If the partition has a child hierarchy, recursively encode it.
        Otherwise, encode the partition's edges using SENT back-edge format.

        Token format: [LCOM] [part_id] [num_nodes] [global_idx_0] ... [global_idx_n-1] [SEP] [SENT...] [RCOM]

        Args:
            part: Partition to tokenize.
            root_hg: Root HierarchicalGraph for edge feature lookup.

        Returns:
            List of token indices for this partition.
        """
        tokens: list[int] = [
            self.LCOM,
            self.IDX_OFFSET + part.part_id,
            self.IDX_OFFSET + part.num_nodes,
        ]

        # Encode global node indices
        for global_idx in part.global_node_indices:
            tokens.append(self.IDX_OFFSET + global_idx)
        tokens.append(self.SEP)  # Separator before SENT content

        if part.num_nodes == 0:
            tokens.append(self.RCOM)
            return tokens

        # If partition has child hierarchy, recursively encode it
        if part.child_hierarchy is not None:
            # Encode child's structure (without SOS/EOS)
            child_tokens = self._tokenize_child_hierarchy(
                part.child_hierarchy, root_hg=root_hg
            )
            tokens.extend(child_tokens)
        else:
            # Leaf partition: encode with SENT-style back-edges
            tokens.extend(self._tokenize_partition_sent(part, root_hg=root_hg))

        tokens.append(self.RCOM)
        return tokens

    def _tokenize_child_hierarchy(
        self, hg: HierarchicalGraph, root_hg: Optional[HierarchicalGraph] = None
    ) -> list[int]:
        """Encode a nested hierarchy (without SOS/EOS).

        Args:
            hg: Child HierarchicalGraph.
            root_hg: Root HierarchicalGraph for edge feature lookup.

        Returns:
            List of token indices.
        """
        tokens: list[int] = []

        # Number of sub-communities
        tokens.append(self.IDX_OFFSET + hg.num_communities)

        # Encode each sub-partition
        for part in hg.partitions:
            tokens.extend(self._tokenize_partition(part, root_hg=root_hg))

        # Encode each sub-bipartite
        for bipart in hg.bipartites:
            tokens.extend(self._tokenize_bipartite(bipart))

        return tokens

    def _tokenize_partition_sent(
        self, part: Partition, root_hg: Optional[HierarchicalGraph] = None
    ) -> list[int]:
        """Encode partition edges using SENT-style back-edges.

        For labeled graphs, interleaves atom and bond type tokens:
        - After each node: append atom type token
        - After each back-edge target: append bond type token

        Args:
            part: Leaf partition to tokenize.
            root_hg: Root HierarchicalGraph for edge feature lookup.

        Returns:
            List of token indices (node sequence with back-edges and features).
        """
        tokens: list[int] = []

        # Get canonical node ordering
        node_order = order_partition_nodes(part, self.node_order, self.seed)

        # Build adjacency for back-edge detection
        adj: dict[int, set[int]] = {i: set() for i in range(part.num_nodes)}
        if part.edge_index.numel() > 0:
            ei = part.edge_index.numpy()
            for e in range(ei.shape[1]):
                src, dst = int(ei[0, e]), int(ei[1, e])
                if src != dst:
                    adj[src].add(dst)
                    if self.undirected:
                        adj[dst].add(src)

        # Encode nodes with back-edges
        visited: set[int] = set()
        order_to_idx = {n: i for i, n in enumerate(node_order)}

        for node in node_order:
            tokens.append(self.IDX_OFFSET + node)

            # Add atom type if labeled graph
            if self.labeled_graph:
                atom_type = self._get_node_feature(part, node)
                atom_token = self.node_idx_offset + atom_type
                tokens.append(atom_token)

            visited.add(node)

            # Find back-edges to previously visited neighbors
            back_edges = sorted(
                order_to_idx[n] for n in adj[node] if n in visited and n != node
            )

            if back_edges:
                tokens.append(self.LADJ)
                for be in back_edges:
                    tokens.append(self.IDX_OFFSET + be)

                    # Add bond type if labeled graph
                    if self.labeled_graph:
                        # Get the actual node index from order
                        target_node = node_order[be]
                        # Get global indices for edge lookup
                        src_global = part.global_node_indices[node]
                        dst_global = part.global_node_indices[target_node]
                        bond_type = self._get_edge_feature_global(
                            root_hg, src_global, dst_global
                        )
                        bond_token = self.edge_idx_offset + bond_type
                        tokens.append(bond_token)

                tokens.append(self.RADJ)

        return tokens

    def _tokenize_bipartite(self, bipart: Bipartite) -> list[int]:
        """Encode bipartite edges as pairs.

        Args:
            bipart: Bipartite to tokenize.

        Returns:
            List of token indices.
        """
        tokens: list[int] = [
            self.LBIP,
            self.IDX_OFFSET + bipart.left_part_id,
            self.IDX_OFFSET + bipart.right_part_id,
            self.IDX_OFFSET + bipart.num_edges,
        ]

        # Encode edge pairs with optional bond types
        if bipart.edge_index.numel() > 0:
            ei = bipart.edge_index.numpy()
            for e in range(ei.shape[1]):
                tokens.append(self.IDX_OFFSET + int(ei[0, e]))
                tokens.append(self.IDX_OFFSET + int(ei[1, e]))

                # Add bond type if labeled graph and edge features available
                if self.labeled_graph and bipart.edge_features is not None:
                    bond_type = int(bipart.edge_features[e])
                    bond_token = self.edge_idx_offset + bond_type
                    tokens.append(bond_token)

        tokens.append(self.RBIP)
        return tokens

    def _get_node_feature(self, part: Partition, local_idx: int) -> int:
        """Get node feature (atom type) for a node in a partition.

        Args:
            part: The partition containing the node.
            local_idx: Local index of the node within the partition.

        Returns:
            Node feature value (atom type), or 0 if not found.
        """
        if part.node_features is not None and local_idx < len(part.node_features):
            return int(part.node_features[local_idx])
        return 0  # Default atom type

    def _get_edge_feature_global(
        self, root_hg: Optional[HierarchicalGraph], src_global: int, dst_global: int
    ) -> int:
        """Get edge feature (bond type) for an edge using global indices.

        Args:
            root_hg: Root HierarchicalGraph containing edge_features.
            src_global: Source node global index.
            dst_global: Target node global index.

        Returns:
            Edge feature value (bond type), or 0 if not found.
        """
        if root_hg is None or root_hg.edge_features is None:
            return 0

        # Try both directions since graph may be undirected
        bond_type = root_hg.edge_features.get((src_global, dst_global), None)
        if bond_type is not None:
            return bond_type

        # Try reverse direction
        bond_type = root_hg.edge_features.get((dst_global, src_global), None)
        if bond_type is not None:
            return bond_type

        return 0  # Default bond type

    # =====================================================================
    # DECODING: Tokens → Graph
    # =====================================================================

    def decode(self, tokens: Tensor) -> Data:
        """Decode tokens to graph via hierarchical representation.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            PyTorch Geometric Data object.
        """
        hg = self.parse_tokens(tokens)
        return hg.reconstruct()

    def parse_tokens(self, tokens: Tensor) -> HierarchicalGraph:
        """Parse tokens back to HierarchicalGraph.

        Separated from decode() for testing the roundtrip at the
        hierarchy level.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            Reconstructed HierarchicalGraph.
        """
        # Remove special tokens
        mask = (tokens != self.PAD) & (tokens != self.SOS) & (tokens != self.EOS)
        tokens_list = tokens[mask].tolist()

        if not tokens_list:
            return HierarchicalGraph([], [], [])

        idx = 0

        # Parse number of communities
        if tokens_list[idx] >= self.IDX_OFFSET:
            num_communities = tokens_list[idx] - self.IDX_OFFSET
            idx += 1
        else:
            num_communities = 0

        # Tracking dictionaries for labeled graphs
        node_features_dict: dict[int, int] = {}  # global_idx -> atom_type
        edge_features_dict: dict[tuple[int, int], int] = {}  # (src, dst) -> bond_type

        # Parse partitions - track global node offset
        partitions: list[Partition] = []
        global_node_offset = 0
        while idx < len(tokens_list) and tokens_list[idx] == self.LCOM:
            part, idx = self._parse_partition(
                tokens_list,
                idx + 1,
                global_node_offset,
                node_features_dict,
                edge_features_dict,
            )
            if part is not None:
                partitions.append(part)
                global_node_offset += part.num_nodes

        # Parse bipartites
        bipartites: list[Bipartite] = []
        while idx < len(tokens_list) and tokens_list[idx] == self.LBIP:
            bipart, idx = self._parse_bipartite(tokens_list, idx + 1)
            if bipart is not None:
                bipartites.append(bipart)

                # Add bipartite edge features to global edge_features_dict
                if self.labeled_graph and bipart.edge_features is not None:
                    # Get the partitions to convert local to global indices
                    if bipart.left_part_id < len(
                        partitions
                    ) and bipart.right_part_id < len(partitions):
                        left_part = partitions[bipart.left_part_id]
                        right_part = partitions[bipart.right_part_id]

                        # Add each bipartite edge's bond type to the global dict
                        ei = bipart.edge_index.numpy()
                        for e in range(ei.shape[1]):
                            left_local = int(ei[0, e])
                            right_local = int(ei[1, e])
                            if (
                                left_local < left_part.num_nodes
                                and right_local < right_part.num_nodes
                            ):
                                left_global = left_part.local_to_global(left_local)
                                right_global = right_part.local_to_global(right_local)
                                bond_type = int(bipart.edge_features[e])

                                # Add both directions (undirected graph)
                                edge_features_dict[(left_global, right_global)] = (
                                    bond_type
                                )
                                edge_features_dict[(right_global, left_global)] = (
                                    bond_type
                                )

        # Reconstruct community assignment
        community_assignment = self._build_community_assignment(partitions)

        # Convert node features to tensor
        node_features = None
        if self.labeled_graph and node_features_dict:
            max_node_idx = max(node_features_dict.keys())
            node_features = torch.zeros(max_node_idx + 1, dtype=torch.long)
            for node_idx, atom_type in node_features_dict.items():
                node_features[node_idx] = atom_type

        return HierarchicalGraph(
            partitions,
            bipartites,
            community_assignment,
            node_features=node_features,
            edge_features=edge_features_dict if self.labeled_graph else None,
        )

    def _parse_partition(
        self,
        tokens: list[int],
        start: int,
        global_node_offset: int = 0,
        node_features_dict: Optional[dict[int, int]] = None,
        edge_features_dict: Optional[dict[tuple[int, int], int]] = None,
    ) -> tuple[Optional[Partition], int]:
        """Parse a [LCOM ... RCOM] block.

        Token format: [LCOM] [part_id] [num_nodes] [global_idx_0] ... [global_idx_n-1] [SEP] [SENT...] [RCOM]

        Args:
            tokens: Token list.
            start: Starting index (after LCOM).
            global_node_offset: Offset to add to local indices for global indices (fallback).
            node_features_dict: Dictionary to populate with node features (global idx -> atom type).
            edge_features_dict: Dictionary to populate with edge features ((src, dst) -> bond type).

        Returns:
            Tuple of (parsed Partition or None, next index).
        """
        idx = start

        # Initialize dicts if not provided
        if node_features_dict is None:
            node_features_dict = {}
        if edge_features_dict is None:
            edge_features_dict = {}

        # Parse part_id
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        part_id = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse num_nodes
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        num_nodes = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse global node indices (until SEP)
        global_indices: list[int] = []
        while idx < len(tokens) and tokens[idx] != self.SEP:
            if tokens[idx] >= self.IDX_OFFSET:
                global_indices.append(tokens[idx] - self.IDX_OFFSET)
            idx += 1

        # Skip SEP if present
        if idx < len(tokens) and tokens[idx] == self.SEP:
            idx += 1

        # If no global indices found, use fallback (sequential from offset)
        if not global_indices:
            global_indices = list(
                range(global_node_offset, global_node_offset + num_nodes)
            )

        # Check if this is a nested hierarchy or leaf partition
        # Nested if we see a number followed by LCOM
        child_hierarchy = None

        if idx < len(tokens) and tokens[idx] >= self.IDX_OFFSET:
            # Could be sub-community count or first node of SENT
            lookahead = idx + 1
            if lookahead < len(tokens) and tokens[lookahead] == self.LCOM:
                # Nested hierarchy
                child_hierarchy, idx = self._parse_child_hierarchy(
                    tokens,
                    idx,
                    global_node_offset,
                    node_features_dict,
                    edge_features_dict,
                )
                # Find RCOM
                while idx < len(tokens) and tokens[idx] != self.RCOM:
                    idx += 1
                if idx < len(tokens):
                    idx += 1  # Skip RCOM

                # Create partition with child hierarchy
                partition = Partition(
                    part_id=part_id,
                    global_node_indices=global_indices,
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    child_hierarchy=child_hierarchy,
                )
                return partition, idx

        # Leaf partition: parse SENT-style encoding
        # NOTE: We only create edges from back-edge brackets, NOT from sequential
        # walk positions. This is because BFS/DFS ordering may visit non-adjacent
        # nodes consecutively (e.g., siblings in BFS tree).
        edges: list[tuple[int, int]] = []
        in_bracket = False
        bracket_nodes: list[int] = []
        node_sequence: list[int] = []

        while idx < len(tokens):
            tok = tokens[idx]

            if tok == self.RCOM:
                idx += 1
                break
            elif tok == self.LCOM:
                # Nested partition - shouldn't happen in leaf, but handle it
                break
            elif tok == self.LADJ:
                in_bracket = True
                bracket_nodes = []
                idx += 1
            elif tok == self.RADJ:
                in_bracket = False
                # Add back-edges: edges from current node to previously visited nodes
                if node_sequence:
                    current_node_local = node_sequence[-1]
                    if current_node_local < len(global_indices):
                        for back_pos in bracket_nodes:
                            if back_pos < len(node_sequence):
                                back_node_local = node_sequence[back_pos]
                                if back_node_local < len(global_indices):
                                    edges.append((current_node_local, back_node_local))
                                    edges.append((back_node_local, current_node_local))
                idx += 1
            elif tok >= self.IDX_OFFSET:
                val = tok - self.IDX_OFFSET

                if in_bracket:
                    # This is a back-edge target position
                    bracket_nodes.append(val)
                    idx += 1

                    # Read bond type if labeled graph
                    if self.labeled_graph:
                        if idx < len(tokens) and tokens[idx] >= self.edge_idx_offset:
                            bond_token = tokens[idx]
                            bond_type = bond_token - self.edge_idx_offset
                            # Store the bond type for this back-edge
                            if node_sequence and val < len(node_sequence):
                                current_node_local = node_sequence[-1]
                                back_node_local = node_sequence[val]
                                if current_node_local < len(
                                    global_indices
                                ) and back_node_local < len(global_indices):
                                    current_node_global = global_indices[
                                        current_node_local
                                    ]
                                    back_node_global = global_indices[back_node_local]
                                    edge_features_dict[
                                        (current_node_global, back_node_global)
                                    ] = bond_type
                                    edge_features_dict[
                                        (back_node_global, current_node_global)
                                    ] = bond_type
                            idx += 1
                else:
                    # This is a node in the sequence
                    node_sequence.append(val)
                    idx += 1

                    # Read atom type if labeled graph
                    if self.labeled_graph:
                        if (
                            idx < len(tokens)
                            and tokens[idx] >= self.node_idx_offset
                            and tokens[idx] < self.edge_idx_offset
                        ):
                            atom_token = tokens[idx]
                            atom_type = atom_token - self.node_idx_offset
                            # Store atom type using GLOBAL index
                            if val < len(global_indices):
                                global_idx = global_indices[val]
                                node_features_dict[global_idx] = atom_type
                            idx += 1
            else:
                idx += 1

        # Build edge index
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Extract node features for this partition (in LOCAL indices)
        node_features = None
        if self.labeled_graph and node_features_dict:
            node_features = torch.zeros(len(global_indices), dtype=torch.long)
            for local_idx, global_idx in enumerate(global_indices):
                if global_idx in node_features_dict:
                    node_features[local_idx] = node_features_dict[global_idx]

        partition = Partition(
            part_id=part_id,
            global_node_indices=global_indices,
            edge_index=edge_index,
            child_hierarchy=None,
            node_features=node_features,
        )

        return partition, idx

    def _parse_child_hierarchy(
        self,
        tokens: list[int],
        start: int,
        global_node_offset: int = 0,
        node_features_dict: Optional[dict[int, int]] = None,
        edge_features_dict: Optional[dict[tuple[int, int], int]] = None,
    ) -> tuple[Optional[HierarchicalGraph], int]:
        """Parse a nested hierarchy within a partition.

        Args:
            tokens: Token list.
            start: Starting index (at sub-community count).
            global_node_offset: Offset to add to local indices for global indices.
            node_features_dict: Dictionary to populate with node features.
            edge_features_dict: Dictionary to populate with edge features.

        Returns:
            Tuple of (parsed HierarchicalGraph or None, next index).
        """
        idx = start

        # Initialize dicts if not provided
        if node_features_dict is None:
            node_features_dict = {}
        if edge_features_dict is None:
            edge_features_dict = {}

        # Parse number of sub-communities
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        num_sub_communities = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse sub-partitions - track offset within this hierarchy
        partitions: list[Partition] = []
        child_offset = global_node_offset
        while idx < len(tokens) and tokens[idx] == self.LCOM:
            part, idx = self._parse_partition(
                tokens, idx + 1, child_offset, node_features_dict, edge_features_dict
            )
            if part is not None:
                partitions.append(part)
                child_offset += part.num_nodes

        # Parse sub-bipartites
        bipartites: list[Bipartite] = []
        while idx < len(tokens) and tokens[idx] == self.LBIP:
            bipart, idx = self._parse_bipartite(tokens, idx + 1)
            if bipart is not None:
                bipartites.append(bipart)

                # Propagate bipartite edge features to global edge_features_dict
                if self.labeled_graph and bipart.edge_features is not None:
                    if bipart.left_part_id < len(
                        partitions
                    ) and bipart.right_part_id < len(partitions):
                        left_part = partitions[bipart.left_part_id]
                        right_part = partitions[bipart.right_part_id]

                        ei = bipart.edge_index.numpy()
                        for e in range(ei.shape[1]):
                            left_local = int(ei[0, e])
                            right_local = int(ei[1, e])
                            if (
                                left_local < left_part.num_nodes
                                and right_local < right_part.num_nodes
                            ):
                                left_global = left_part.local_to_global(left_local)
                                right_global = right_part.local_to_global(right_local)
                                bond_type = int(bipart.edge_features[e])

                                edge_features_dict[(left_global, right_global)] = (
                                    bond_type
                                )
                                edge_features_dict[(right_global, left_global)] = (
                                    bond_type
                                )

        # Build community assignment for child
        community_assignment = self._build_community_assignment(partitions)

        return HierarchicalGraph(partitions, bipartites, community_assignment), idx

    def _parse_bipartite(
        self, tokens: list[int], start: int
    ) -> tuple[Optional[Bipartite], int]:
        """Parse a [LBIP ... RBIP] block.

        Args:
            tokens: Token list.
            start: Starting index (after LBIP).

        Returns:
            Tuple of (parsed Bipartite or None, next index).
        """
        idx = start

        # Parse left_part_id
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        left_id = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse right_part_id
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        right_id = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse num_edges (skip it, we count from pairs)
        if idx < len(tokens) and tokens[idx] >= self.IDX_OFFSET:
            idx += 1

        # Parse edge pairs with optional bond types
        edges: list[tuple[int, int]] = []
        edge_features_list: list[int] = []

        while idx < len(tokens):
            tok = tokens[idx]

            if tok == self.RBIP:
                idx += 1
                break
            elif tok >= self.IDX_OFFSET:
                left_local = tok - self.IDX_OFFSET
                idx += 1

                if idx < len(tokens) and tokens[idx] >= self.IDX_OFFSET:
                    right_local = tokens[idx] - self.IDX_OFFSET
                    edges.append((left_local, right_local))
                    idx += 1

                    # Read bond type if labeled graph
                    if self.labeled_graph:
                        if idx < len(tokens) and tokens[idx] >= self.edge_idx_offset:
                            bond_token = tokens[idx]
                            bond_type = bond_token - self.edge_idx_offset
                            edge_features_list.append(bond_type)
                            idx += 1
                        else:
                            edge_features_list.append(0)  # Default bond type
            else:
                idx += 1

        if not edges:
            return None, idx

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Create edge features tensor if labeled graph
        edge_features = None
        if self.labeled_graph and edge_features_list:
            edge_features = torch.tensor(edge_features_list, dtype=torch.long)

        return Bipartite(left_id, right_id, edge_index, edge_features), idx

    def _build_community_assignment(self, partitions: list[Partition]) -> list[int]:
        """Build community assignment from partitions.

        Args:
            partitions: List of partitions.

        Returns:
            List mapping node index to partition ID.
        """
        total_nodes = sum(p.num_nodes for p in partitions)
        assignment = [0] * total_nodes

        offset = 0
        for part in partitions:
            for i in range(part.num_nodes):
                if offset + i < len(assignment):
                    assignment[offset + i] = part.part_id
            offset += part.num_nodes

        return assignment

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
        for tok in tokens.tolist():
            if tok in self.SPECIAL_TOKEN_NAMES:
                parts.append(f"[{self.SPECIAL_TOKEN_NAMES[tok]}]")
            else:
                parts.append(str(tok - self.IDX_OFFSET))
        return " ".join(parts)
