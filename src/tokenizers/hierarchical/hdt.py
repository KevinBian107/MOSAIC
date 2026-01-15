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

from collections import defaultdict
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.hierarchical.coarsening import (
    MotifAwareCoarsening,
    SpectralCoarsening,
)
from src.tokenizers.hierarchical.ordering import OrderingMethod, order_partition_nodes
from src.tokenizers.hierarchical.structures import (
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
        n_init: int = 100,
        motif_aware: bool = False,
        motif_alpha: float = 1.0,
        motif_patterns: Optional[dict[str, str]] = None,
        normalize_by_motif_size: bool = False,
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
            k_min_factor: Factor for minimum cluster count in spectral clustering.
            k_max_factor: Factor for maximum cluster count in spectral clustering.
            n_init: Number of spectral clustering initializations.
            motif_aware: Whether to use motif-aware coarsening.
            motif_alpha: Weight for motif affinity matrix (only used if
                motif_aware=True).
            motif_patterns: Custom SMARTS patterns for motif detection.
            normalize_by_motif_size: Whether to normalize motif contributions
                by 1/motif_size.
        """
        self.node_order = node_order
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.undirected = undirected
        self.seed = seed
        self.min_community_size = min_community_size
        self.motif_aware = motif_aware
        self.motif_alpha = motif_alpha

        # Select coarsening strategy
        if motif_aware:
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
        else:
            self.coarsener = SpectralCoarsening(
                k_min_factor=k_min_factor,
                k_max_factor=k_max_factor,
                n_init=n_init,
                min_community_size=min_community_size,
                seed=seed,
            )

        self.max_num_nodes: Optional[int] = None

    def set_num_nodes(self, max_num_nodes: int) -> None:
        """Set maximum number of nodes for vocabulary sizing.

        Args:
            max_num_nodes: Maximum nodes in any graph.
        """
        if self.max_num_nodes is None or self.max_num_nodes < max_num_nodes:
            self.max_num_nodes = max_num_nodes

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes() first")
        return self.IDX_OFFSET + self.max_num_nodes

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
        hg = self.coarsener.build_hierarchy(data, recursive=False)
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
        )

        tokens.append(self.EOS)

        # Truncate if needed
        if self.max_length > 0 and len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.EOS]

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
    ) -> None:
        """Recursive DFS serialization of hierarchical graph.

        Args:
            hg: HierarchicalGraph at current level.
            level: Current hierarchy level (0 = root).
            local_id: Local ID within parent (0 for root).
            tokens: Token list to append to.
            visited_atoms: List of already visited atom indices.
            full_adj: Full adjacency map including bipartite edges.
        """
        # Emit: [ENTER] <level> <local_id>
        tokens.append(self.ENTER)
        tokens.append(self.IDX_OFFSET + level)
        tokens.append(self.IDX_OFFSET + local_id)

        # Process each partition
        for part in hg.partitions:
            if part.child_hierarchy is not None:
                # Recurse into nested hierarchy
                self._dfs_serialize(
                    hg=part.child_hierarchy,
                    level=level + 1,
                    local_id=part.part_id,
                    tokens=tokens,
                    visited_atoms=visited_atoms,
                    full_adj=full_adj,
                )
            else:
                # Leaf partition: emit atoms with back-edges
                self._serialize_atoms(part, tokens, visited_atoms, full_adj)

        # Emit exit token
        tokens.append(self.EXIT)

    def _serialize_atoms(
        self,
        part: Partition,
        tokens: list[int],
        visited_atoms: list[int],
        full_adj: dict[int, set[int]],
    ) -> None:
        """Serialize atoms with back-edges (including cross-community).

        Args:
            part: Leaf partition to serialize.
            tokens: Token list to append to.
            visited_atoms: List of already visited atom indices.
            full_adj: Full adjacency map including bipartite edges.
        """
        # Get canonical node ordering
        node_order = order_partition_nodes(part, self.node_order, self.seed)

        for local_idx in node_order:
            global_idx = part.global_node_indices[local_idx]
            tokens.append(self.IDX_OFFSET + global_idx)

            # Find ALL edges to previously visited atoms
            # This includes both intra-partition AND cross-partition edges
            back_edges = self._find_back_edges(global_idx, visited_atoms, full_adj)

            if back_edges:
                tokens.append(self.LEDGE)
                for target_global in back_edges:
                    tokens.append(self.IDX_OFFSET + target_global)
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

        hg, idx = self._parse_hierarchy(tokens_list, idx, atoms_visited, all_edges)

        if hg is None:
            return HierarchicalGraph([], [], [])

        return hg

    def _parse_hierarchy(
        self,
        tokens: list[int],
        idx: int,
        atoms_visited: list[int],
        all_edges: list[tuple[int, int]],
    ) -> tuple[Optional[HierarchicalGraph], int]:
        """Recursively parse hierarchy from DFS token stream.

        Args:
            tokens: Token list.
            idx: Current index in token list.
            atoms_visited: Global list of visited atoms.
            all_edges: Accumulator for all edges found.

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
                        part_counter, current_atoms, current_edges
                    )
                    partitions.append(part)
                    part_counter += 1
                    current_atoms = []
                    current_edges = []

                # Recurse into nested hierarchy
                child_hg, idx = self._parse_hierarchy(
                    tokens, idx, atoms_visited, all_edges
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
                # Atom token
                global_idx = tok - self.IDX_OFFSET
                current_atoms.append(global_idx)
                atoms_visited.append(global_idx)
                idx += 1

            elif tok == self.LEDGE:
                # Back-edge block
                idx += 1
                while idx < len(tokens) and tokens[idx] != self.REDGE:
                    if tokens[idx] >= self.IDX_OFFSET:
                        target = tokens[idx] - self.IDX_OFFSET
                        # Current atom is the last one added
                        if current_atoms:
                            src = current_atoms[-1]
                            current_edges.append((src, target))
                            current_edges.append((target, src))
                            all_edges.append((src, target))
                            all_edges.append((target, src))
                    idx += 1
                idx += 1  # Skip REDGE

            else:
                idx += 1

        # Skip EXIT token
        if idx < len(tokens) and tokens[idx] == self.EXIT:
            idx += 1

        # Finalize last partition
        if current_atoms:
            part = self._make_partition(part_counter, current_atoms, current_edges)
            partitions.append(part)

        # Build HierarchicalGraph
        # For HDT, bipartites are reconstructed from cross-partition edges
        bipartites = self._infer_bipartites(partitions, all_edges)

        # Build community assignment
        community_assignment = self._build_community_assignment(partitions)

        return HierarchicalGraph(partitions, bipartites, community_assignment), idx

    def _make_partition(
        self,
        part_id: int,
        global_atoms: list[int],
        edges: list[tuple[int, int]],
    ) -> Partition:
        """Create a partition from atoms and edges.

        Args:
            part_id: Partition ID.
            global_atoms: Global atom indices in this partition.
            edges: Edges as (global_src, global_dst) tuples.

        Returns:
            Partition object with local edge indices.
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

        return Partition(
            part_id=part_id,
            global_node_indices=global_atoms,
            edge_index=edge_index,
            child_hierarchy=None,
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
