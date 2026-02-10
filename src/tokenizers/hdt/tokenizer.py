"""Hierarchical DFS-based Tokenizer (HDT).

This module implements HDT, a simplified DFS-based tokenization scheme that
achieves ~45% token reduction over H-SENT by:
1. Using ENTER/EXIT tokens to encode hierarchy structure implicitly via DFS nesting
2. Eliminating bipartite blocks entirely - cross-community edges become back-edges
3. Using a single EXIT token instead of multiple closing brackets

Key insight: HDT serializes the same HierarchicalGraph data structure as H-SENT,
but uses DFS traversal to implicitly encode the hierarchy rather than explicit
partition and bipartite blocks.
"""

import warnings
from collections import defaultdict
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.coarsening import (
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


class HDTTokenizer(Tokenizer):
    """Hierarchical DFS-based tokenizer for graph-to-sequence conversion.

    Uses DFS traversal through hierarchy levels with implicit structure
    encoding via ENTER/EXIT special tokens.

    Key differences from H-SENT:
    - No bipartite blocks - cross-community edges are back-edges
    - Single EXIT token instead of matching brackets
    - ~45% fewer tokens for typical graphs
    - Smaller vocabulary (IDX_OFFSET=7 vs 11)

    Token vocabulary:
        0: SOS (start of sequence)
        1: EOS (end of sequence)
        2: PAD (padding)
        3: ENTER (enter super node, followed by level+IDX_OFFSET, id+IDX_OFFSET)
        4: EXIT (exit current super node)
        5: LEDGE (left edge bracket)
        6: REDGE (right edge bracket)
        7+: Node indices (offset by IDX_OFFSET)

    Sequence format for entering a super node:
        [ENTER] <level + IDX_OFFSET> <local_id + IDX_OFFSET>

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
    tokenizer_type: str = "hdt"

    # Special token IDs
    SOS: int = 0
    EOS: int = 1
    PAD: int = 2
    ENTER: int = 3
    EXIT: int = 4
    LEDGE: int = 5
    REDGE: int = 6
    IDX_OFFSET: int = 7

    # Token names for debugging
    SPECIAL_TOKEN_NAMES = {
        0: "SOS",
        1: "EOS",
        2: "PAD",
        3: "ENTER",
        4: "EXIT",
        5: "LEDGE",
        6: "REDGE",
    }

    # Override base class attributes
    sos: int = 0
    eos: int = 1
    pad: int = 2

    # Type alias for coarsening strategy
    CoarseningStrategyType = Literal["spectral", "motif_aware_spectral", "motif_community"]

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
        """Initialize the HDT tokenizer.

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
        # [SOS, EOS, PAD, ENTER, EXIT, LEDGE, REDGE] = 7 special tokens
        # Then: [0, 1, 2, ..., max_num_nodes-1] = node IDs
        # Then: [atom_type_0, ..., atom_type_{num_node_types-1}] = atom types
        # Then: [bond_type_0, ..., bond_type_{num_edge_types-1}] = bond types
        self.node_idx_offset = self.IDX_OFFSET + self.max_num_nodes
        self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size.

        Unlabeled: IDX_OFFSET (7) + max_num_nodes
        Labeled: IDX_OFFSET (7) + max_num_nodes + num_node_types + num_edge_types
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
        """Tokenize graph via hierarchical DFS representation.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            1D tensor of token indices.
        """
        hg = self.coarsener.build_hierarchy(data)
        return self.tokenize_hierarchy(hg)

    def tokenize_hierarchy(self, hg: HierarchicalGraph) -> Tensor:
        """Tokenize a HierarchicalGraph using DFS traversal.

        Separated from tokenize() for testing the roundtrip at the
        hierarchy level.

        Args:
            hg: HierarchicalGraph to tokenize.

        Returns:
            1D tensor of token indices.
        """
        tokens: list[int] = [self.SOS]
        visited_atoms: list[int] = []

        # Build full adjacency including bipartite edges
        full_adj = self._build_full_adjacency(hg)

        # DFS through hierarchy starting at root level
        self._dfs_serialize(
            hg=hg,
            level=0,
            local_id=0,
            tokens=tokens,
            visited_atoms=visited_atoms,
            full_adj=full_adj,
            root_hg=hg,  # Pass root hierarchy for edge feature lookup
        )

        tokens.append(self.EOS)

        # Truncate if needed
        if self.max_length > 0 and len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.EOS]

        # Validate tokens are within vocab bounds
        if self.max_num_nodes is not None:
            vocab_size = self.vocab_size
            for i, tok in enumerate(tokens):
                if tok >= vocab_size:
                    raise ValueError(
                        f"HDT Token overflow at position {i}: token={tok} >= vocab_size={vocab_size}. "
                        f"max_num_nodes={self.max_num_nodes}, IDX_OFFSET={self.IDX_OFFSET}, "
                        f"node_idx_offset={getattr(self, 'node_idx_offset', 'N/A')}, "
                        f"edge_idx_offset={getattr(self, 'edge_idx_offset', 'N/A')}, "
                        f"labeled_graph={self.labeled_graph}"
                    )

        return torch.tensor(tokens, dtype=torch.long)

    def _build_full_adjacency(
        self, hg: HierarchicalGraph
    ) -> dict[int, set[int]]:
        """Build adjacency including bipartite edges.

        This creates a full adjacency map that includes both intra-partition
        edges and inter-partition (bipartite) edges. This is needed because
        HDT encodes bipartite edges as back-edges to previously visited atoms.

        Args:
            hg: HierarchicalGraph with partitions and bipartites.

        Returns:
            Dictionary mapping global node index to set of neighbors.
        """
        adj: dict[int, set[int]] = defaultdict(set)

        # Add intra-partition edges (recursively if nested)
        for part in hg.partitions:
            self._add_partition_edges(part, adj)

        # Add bipartite edges
        for bipart in hg.bipartites:
            left_part = hg.get_partition(bipart.left_part_id)
            right_part = hg.get_partition(bipart.right_part_id)

            if bipart.edge_index.numel() > 0:
                ei = bipart.edge_index.numpy()
                for e in range(ei.shape[1]):
                    left_local = int(ei[0, e])
                    right_local = int(ei[1, e])
                    left_global = left_part.local_to_global(left_local)
                    right_global = right_part.local_to_global(right_local)
                    adj[left_global].add(right_global)
                    adj[right_global].add(left_global)

        return adj

    def _add_partition_edges(
        self, part: Partition, adj: dict[int, set[int]]
    ) -> None:
        """Add edges from a partition to the adjacency map.

        Handles nested hierarchies recursively.

        Args:
            part: Partition to add edges from.
            adj: Adjacency map to update.
        """
        if part.child_hierarchy is not None:
            # Recursively add from child hierarchy
            for child_part in part.child_hierarchy.partitions:
                self._add_partition_edges(child_part, adj)
            # Add bipartite edges from child hierarchy
            for bipart in part.child_hierarchy.bipartites:
                left_part = part.child_hierarchy.get_partition(bipart.left_part_id)
                right_part = part.child_hierarchy.get_partition(bipart.right_part_id)
                if bipart.edge_index.numel() > 0:
                    ei = bipart.edge_index.numpy()
                    for e in range(ei.shape[1]):
                        left_local = int(ei[0, e])
                        right_local = int(ei[1, e])
                        left_global = left_part.local_to_global(left_local)
                        right_global = right_part.local_to_global(right_local)
                        adj[left_global].add(right_global)
                        adj[right_global].add(left_global)
        else:
            # Leaf partition - add direct edges
            if part.edge_index.numel() > 0:
                ei = part.edge_index.numpy()
                for e in range(ei.shape[1]):
                    src_local = int(ei[0, e])
                    dst_local = int(ei[1, e])
                    if src_local != dst_local:
                        src_global = part.local_to_global(src_local)
                        dst_global = part.local_to_global(dst_local)
                        adj[src_global].add(dst_global)
                        adj[dst_global].add(src_global)

    def _dfs_serialize(
        self,
        hg: HierarchicalGraph,
        level: int,
        local_id: int,
        tokens: list[int],
        visited_atoms: list[int],
        full_adj: dict[int, set[int]],
        root_hg: HierarchicalGraph,
    ) -> None:
        """Recursive DFS serialization of hierarchical graph.

        Each partition is wrapped with ENTER/EXIT tokens to preserve partition
        boundaries during roundtrip. The traversal follows the pattern:
        Parent -> Child1 -> (back to Parent) -> Child2 -> (back to Parent) -> ...

        Args:
            hg: HierarchicalGraph at current level.
            level: Current hierarchy level (0 = root).
            local_id: Local ID within parent (0 for root).
            tokens: Token list to append to.
            visited_atoms: List of already visited atom indices.
            full_adj: Full adjacency map including bipartite edges.
            root_hg: Root HierarchicalGraph for edge feature lookup.
        """
        # Emit: [ENTER] <level> <local_id>
        tokens.append(self.ENTER)
        tokens.append(self.IDX_OFFSET + level)
        tokens.append(self.IDX_OFFSET + local_id)

        # Process each partition - ALWAYS wrap with ENTER/EXIT to preserve
        # partition boundaries during roundtrip
        for part in hg.partitions:
            # Emit ENTER for this partition
            tokens.append(self.ENTER)
            tokens.append(self.IDX_OFFSET + level + 1)
            tokens.append(self.IDX_OFFSET + part.part_id)

            if part.child_hierarchy is not None:
                # Recurse into nested hierarchy's partitions
                self._serialize_nested_partitions(
                    hg=part.child_hierarchy,
                    level=level + 1,
                    tokens=tokens,
                    visited_atoms=visited_atoms,
                    full_adj=full_adj,
                    root_hg=root_hg,
                )
            else:
                # Leaf partition: emit atoms with back-edges
                self._serialize_atoms(part, tokens, visited_atoms, full_adj, root_hg)

            # Emit EXIT for this partition (back to parent level)
            tokens.append(self.EXIT)

        # Emit exit token for the current level
        tokens.append(self.EXIT)

    def _serialize_nested_partitions(
        self,
        hg: HierarchicalGraph,
        level: int,
        tokens: list[int],
        visited_atoms: list[int],
        full_adj: dict[int, set[int]],
        root_hg: HierarchicalGraph,
    ) -> None:
        """Serialize partitions within a nested hierarchy.

        This handles the case where a partition has a child_hierarchy.
        Each partition in the child hierarchy gets its own ENTER/EXIT wrapper.

        Args:
            hg: Child HierarchicalGraph to serialize.
            level: Current hierarchy level.
            tokens: Token list to append to.
            visited_atoms: List of already visited atom indices.
            full_adj: Full adjacency map including bipartite edges.
            root_hg: Root HierarchicalGraph for edge feature lookup.
        """
        for part in hg.partitions:
            # Emit ENTER for this partition
            tokens.append(self.ENTER)
            tokens.append(self.IDX_OFFSET + level + 1)
            tokens.append(self.IDX_OFFSET + part.part_id)

            if part.child_hierarchy is not None:
                # Recurse deeper
                self._serialize_nested_partitions(
                    hg=part.child_hierarchy,
                    level=level + 1,
                    tokens=tokens,
                    visited_atoms=visited_atoms,
                    full_adj=full_adj,
                    root_hg=root_hg,
                )
            else:
                # Leaf partition: emit atoms
                self._serialize_atoms(part, tokens, visited_atoms, full_adj, root_hg)

            # Emit EXIT for this partition
            tokens.append(self.EXIT)

    def _serialize_atoms(
        self,
        part: Partition,
        tokens: list[int],
        visited_atoms: list[int],
        full_adj: dict[int, set[int]],
        root_hg: HierarchicalGraph,
    ) -> None:
        """Serialize atoms with back-edges (including cross-community).

        For labeled graphs, interleaves atom and bond type tokens:
        - After each node ID: append atom type token
        - After each back-edge target: append bond type token

        Args:
            part: Leaf partition to serialize.
            tokens: Token list to append to.
            visited_atoms: List of already visited atom indices.
            full_adj: Full adjacency map including bipartite edges.
            root_hg: Root HierarchicalGraph for edge feature lookup.
        """
        # Get canonical node ordering
        node_order = order_partition_nodes(part, self.node_order, self.seed)

        for local_idx in node_order:
            global_idx = part.global_node_indices[local_idx]
            tokens.append(self.IDX_OFFSET + global_idx)

            # Add atom type if labeled graph
            if self.labeled_graph:
                atom_type = self._get_node_feature(part, local_idx)
                atom_token = self.node_idx_offset + atom_type
                tokens.append(atom_token)

            # Find ALL edges to previously visited atoms
            # This includes both intra-partition AND cross-partition edges
            back_edges = self._find_back_edges(global_idx, visited_atoms, full_adj)

            if back_edges:
                tokens.append(self.LEDGE)
                for target_global in back_edges:
                    tokens.append(self.IDX_OFFSET + target_global)

                    # Add bond type if labeled graph
                    if self.labeled_graph:
                        bond_type = self._get_edge_feature(
                            root_hg, global_idx, target_global
                        )
                        bond_token = self.edge_idx_offset + bond_type
                        tokens.append(bond_token)

                tokens.append(self.REDGE)

            visited_atoms.append(global_idx)

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

    def _get_edge_feature(
        self, root_hg: HierarchicalGraph, src_global: int, dst_global: int
    ) -> int:
        """Get edge feature (bond type) for an edge.

        This method looks up edge features from the root hierarchy's edge_features
        dictionary using global indices.

        Args:
            root_hg: Root HierarchicalGraph containing edge_features.
            src_global: Source node global index.
            dst_global: Target node global index.

        Returns:
            Edge feature value (bond type), or 0 if not found.
        """
        if root_hg.edge_features is not None:
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
    # DECODING: Tokens -> Graph
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
        """Parse HDT tokens back to HierarchicalGraph.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            Reconstructed HierarchicalGraph.
        """
        # Remove special tokens (PAD, SOS, EOS)
        mask = (tokens != self.PAD) & (tokens != self.SOS) & (tokens != self.EOS)
        tokens_list = tokens[mask].tolist()

        if not tokens_list:
            return HierarchicalGraph([], [], [])

        # Parse the hierarchy
        idx = 0
        atoms_visited: list[int] = []
        all_edges: list[tuple[int, int]] = []
        node_features_dict: dict[int, int] = {}  # global_idx -> atom_type
        edge_features_dict: dict[tuple[int, int], int] = {}  # (src, dst) -> bond_type

        hg, idx = self._parse_hierarchy(
            tokens_list,
            idx,
            atoms_visited,
            all_edges,
            node_features_dict,
            edge_features_dict,
        )

        if hg is None:
            return HierarchicalGraph([], [], [])

        return hg

    def _parse_hierarchy(
        self,
        tokens: list[int],
        idx: int,
        atoms_visited: list[int],
        all_edges: list[tuple[int, int]],
        node_features_dict: dict[int, int],
        edge_features_dict: dict[tuple[int, int], int],
    ) -> tuple[Optional[HierarchicalGraph], int]:
        """Recursively parse hierarchy from DFS token stream.

        Args:
            tokens: Token list.
            idx: Current index in token list.
            atoms_visited: Global list of visited atoms.
            all_edges: Accumulator for all edges found.
            node_features_dict: Dictionary mapping node index to atom type.
            edge_features_dict: Dictionary mapping edge to bond type.

        Returns:
            Tuple of (parsed HierarchicalGraph or None, next index).
        """
        if idx >= len(tokens):
            return None, idx

        # Expect ENTER token
        if tokens[idx] != self.ENTER:
            return None, idx

        idx += 1

        # Parse level and local_id
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        level = tokens[idx] - self.IDX_OFFSET
        idx += 1

        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        local_id = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Collect partitions and atoms at this level
        partitions: list[Partition] = []
        current_atoms: list[int] = []
        current_edges: list[tuple[int, int]] = []
        part_counter = 0

        while idx < len(tokens) and tokens[idx] != self.EXIT:
            tok = tokens[idx]

            if tok == self.ENTER:
                # Save current atoms as partition if any
                if current_atoms:
                    part = self._make_partition(
                        part_counter, current_atoms, current_edges, node_features_dict
                    )
                    partitions.append(part)
                    part_counter += 1
                    current_atoms = []
                    current_edges = []

                # Recurse into nested hierarchy
                child_hg, idx = self._parse_hierarchy(
                    tokens,
                    idx,
                    atoms_visited,
                    all_edges,
                    node_features_dict,
                    edge_features_dict,
                )
                if child_hg is not None:
                    # Create a partition that contains this child hierarchy
                    child_global_indices = self._collect_global_indices(child_hg)
                    part = Partition(
                        part_id=part_counter,
                        global_node_indices=child_global_indices,
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                        child_hierarchy=child_hg,
                    )
                    partitions.append(part)
                    part_counter += 1

            elif tok >= self.IDX_OFFSET:
                # Atom token - check if this is a node ID or atom type
                # Order: check edge_idx_offset first (largest), then node_idx_offset
                if self.labeled_graph and tok >= self.edge_idx_offset:
                    # Bond type token in wrong place - skip
                    idx += 1
                elif self.labeled_graph and tok >= self.node_idx_offset:
                    # Atom type token in wrong place - skip
                    idx += 1
                else:
                    # This is a node ID
                    global_idx = tok - self.IDX_OFFSET
                    current_atoms.append(global_idx)
                    atoms_visited.append(global_idx)
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
                            node_features_dict[global_idx] = atom_type
                            idx += 1

            elif tok == self.LEDGE:
                # Back-edge block
                idx += 1
                while idx < len(tokens) and tokens[idx] != self.REDGE:
                    if tokens[idx] >= self.IDX_OFFSET:
                        # Check if this is a target node ID or feature token
                        # Order: check edge_idx_offset first (largest)
                        if (
                            self.labeled_graph
                            and tokens[idx] >= self.edge_idx_offset
                        ):
                            # Bond type for the previous target - already processed
                            idx += 1
                        elif (
                            self.labeled_graph
                            and tokens[idx] >= self.node_idx_offset
                        ):
                            # Atom type in wrong place - skip
                            idx += 1
                        else:
                            # This is a target node ID
                            target = tokens[idx] - self.IDX_OFFSET
                            # Current atom is the last one added
                            if current_atoms:
                                src = current_atoms[-1]
                                current_edges.append((src, target))
                                current_edges.append((target, src))
                                all_edges.append((src, target))
                                all_edges.append((target, src))

                                # Read bond type if labeled graph
                                idx += 1
                                if self.labeled_graph:
                                    if (
                                        idx < len(tokens)
                                        and tokens[idx] >= self.edge_idx_offset
                                    ):
                                        bond_token = tokens[idx]
                                        bond_type = bond_token - self.edge_idx_offset
                                        edge_features_dict[(src, target)] = bond_type
                                        edge_features_dict[(target, src)] = bond_type
                                        idx += 1
                            else:
                                idx += 1
                    else:
                        idx += 1
                idx += 1  # Skip REDGE

            else:
                idx += 1

        # Skip EXIT token
        if idx < len(tokens) and tokens[idx] == self.EXIT:
            idx += 1

        # Finalize last partition
        if current_atoms:
            part = self._make_partition(
                part_counter, current_atoms, current_edges, node_features_dict
            )
            partitions.append(part)

        # Build HierarchicalGraph
        # For HDT, bipartites are reconstructed from cross-partition edges
        bipartites = self._infer_bipartites(partitions, all_edges)

        # Build community assignment
        community_assignment = self._build_community_assignment(partitions)

        # Convert node features to tensor
        node_features = None
        if self.labeled_graph and node_features_dict:
            max_node_idx = max(node_features_dict.keys())
            node_features = torch.zeros(max_node_idx + 1, dtype=torch.long)
            for node_idx, atom_type in node_features_dict.items():
                node_features[node_idx] = atom_type

        return (
            HierarchicalGraph(
                partitions,
                bipartites,
                community_assignment,
                node_features=node_features,
                edge_features=edge_features_dict if self.labeled_graph else None,
            ),
            idx,
        )

    def _make_partition(
        self,
        part_id: int,
        global_atoms: list[int],
        edges: list[tuple[int, int]],
        node_features_dict: dict[int, int],
    ) -> Partition:
        """Create a partition from atoms and edges.

        Args:
            part_id: Partition ID.
            global_atoms: Global atom indices in this partition.
            edges: Edges as (global_src, global_dst) tuples.
            node_features_dict: Dictionary mapping node index to atom type.

        Returns:
            Partition object with local edge indices and node features.
        """
        global_to_local = {g: i for i, g in enumerate(global_atoms)}
        global_set = set(global_atoms)

        # Convert to local indices, keeping only intra-partition edges
        local_edges = []
        for src, dst in edges:
            if src in global_set and dst in global_set:
                local_edges.append((global_to_local[src], global_to_local[dst]))

        if local_edges:
            edge_index = torch.tensor(local_edges, dtype=torch.long).t()
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Extract node features for this partition (in LOCAL indices)
        node_features = None
        if self.labeled_graph and node_features_dict:
            node_features = torch.zeros(len(global_atoms), dtype=torch.long)
            for local_idx, global_idx in enumerate(global_atoms):
                if global_idx in node_features_dict:
                    node_features[local_idx] = node_features_dict[global_idx]

        return Partition(
            part_id=part_id,
            global_node_indices=global_atoms,
            edge_index=edge_index,
            child_hierarchy=None,
            node_features=node_features,
        )

    def _collect_global_indices(self, hg: HierarchicalGraph) -> list[int]:
        """Collect all global node indices from a hierarchy.

        Args:
            hg: HierarchicalGraph to collect from.

        Returns:
            List of all global node indices.
        """
        indices = []
        for part in hg.partitions:
            if part.child_hierarchy is not None:
                indices.extend(self._collect_global_indices(part.child_hierarchy))
            else:
                indices.extend(part.global_node_indices)
        return indices

    def _infer_bipartites(
        self,
        partitions: list[Partition],
        all_edges: list[tuple[int, int]],
    ) -> list[Bipartite]:
        """Infer bipartite edges from cross-partition edges.

        Args:
            partitions: List of partitions.
            all_edges: All edges found during parsing.

        Returns:
            List of Bipartite objects.
        """
        if len(partitions) <= 1:
            return []

        # Map global index to partition ID
        global_to_part: dict[int, int] = {}
        for part in partitions:
            for global_idx in part.global_node_indices:
                global_to_part[global_idx] = part.part_id

        # Group edges by partition pair
        bipart_edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

        for src, dst in all_edges:
            src_part = global_to_part.get(src)
            dst_part = global_to_part.get(dst)

            if src_part is None or dst_part is None:
                continue

            if src_part != dst_part:
                # Cross-partition edge
                pair = (min(src_part, dst_part), max(src_part, dst_part))
                bipart_edges[pair].append((src, dst))

        # Create bipartite objects
        bipartites = []
        for (left_id, right_id), edges in bipart_edges.items():
            left_part = next(p for p in partitions if p.part_id == left_id)
            right_part = next(p for p in partitions if p.part_id == right_id)

            left_global_to_local = {
                g: i for i, g in enumerate(left_part.global_node_indices)
            }
            right_global_to_local = {
                g: i for i, g in enumerate(right_part.global_node_indices)
            }

            local_edges = []
            seen = set()
            for src, dst in edges:
                if src in left_global_to_local and dst in right_global_to_local:
                    local_src = left_global_to_local[src]
                    local_dst = right_global_to_local[dst]
                    edge_key = (local_src, local_dst)
                    if edge_key not in seen:
                        local_edges.append((local_src, local_dst))
                        seen.add(edge_key)
                elif src in right_global_to_local and dst in left_global_to_local:
                    # Edge goes other direction - swap
                    local_src = left_global_to_local[dst]
                    local_dst = right_global_to_local[src]
                    edge_key = (local_src, local_dst)
                    if edge_key not in seen:
                        local_edges.append((local_src, local_dst))
                        seen.add(edge_key)

            if local_edges:
                edge_index = torch.tensor(local_edges, dtype=torch.long).t()
                bipartites.append(Bipartite(left_id, right_id, edge_index))

        return bipartites

    def _build_community_assignment(
        self, partitions: list[Partition]
    ) -> list[int]:
        """Build community assignment from partitions.

        Args:
            partitions: List of partitions.

        Returns:
            List mapping node index to partition ID.
        """
        # Find total nodes
        all_indices = []
        for part in partitions:
            all_indices.extend(part.global_node_indices)

        if not all_indices:
            return []

        max_idx = max(all_indices)
        assignment = [0] * (max_idx + 1)

        for part in partitions:
            for global_idx in part.global_node_indices:
                assignment[global_idx] = part.part_id

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
        tokens_list = tokens.tolist()
        i = 0

        while i < len(tokens_list):
            tok = tokens_list[i]

            if tok in self.SPECIAL_TOKEN_NAMES:
                name = self.SPECIAL_TOKEN_NAMES[tok]
                if tok == self.ENTER:
                    # Parse following level and id
                    parts.append(f"[{name}]")
                    if i + 1 < len(tokens_list) and tokens_list[i + 1] >= self.IDX_OFFSET:
                        level = tokens_list[i + 1] - self.IDX_OFFSET
                        parts.append(f"L{level}")
                        i += 1
                    if i + 1 < len(tokens_list) and tokens_list[i + 1] >= self.IDX_OFFSET:
                        local_id = tokens_list[i + 1] - self.IDX_OFFSET
                        parts.append(f":{local_id}")
                        i += 1
                else:
                    parts.append(f"[{name}]")
            else:
                parts.append(str(tok - self.IDX_OFFSET))
            i += 1

        return " ".join(parts)
