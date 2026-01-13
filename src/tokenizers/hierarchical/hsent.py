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

from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.hierarchical.coarsening import SpectralCoarsening
from src.tokenizers.hierarchical.ordering import OrderingMethod, order_partition_nodes
from src.tokenizers.hierarchical.structures import (
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
        coarsener: SpectralCoarsening instance for hierarchy building.
        max_num_nodes: Maximum nodes (determines vocab size).
    """

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
            k_min_factor: Factor for minimum cluster count in spectral clustering.
            k_max_factor: Factor for maximum cluster count in spectral clustering.
            n_init: Number of spectral clustering initializations.
        """
        self.node_order = node_order
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.undirected = undirected
        self.seed = seed
        self.min_community_size = min_community_size

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
    # TOKENIZATION: Graph → Tokens
    # =====================================================================

    def tokenize(self, data: Data) -> Tensor:
        """Tokenize graph via hierarchical representation.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            1D tensor of token indices.
        """
        hg = self.coarsener.build_hierarchy(data, recursive=False)
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
            tokens.extend(self._tokenize_partition(part))

        # Encode each bipartite
        for bipart in hg.bipartites:
            tokens.extend(self._tokenize_bipartite(bipart))

        tokens.append(self.EOS)

        # Truncate if needed
        if self.max_length > 0 and len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.EOS]

        return torch.tensor(tokens, dtype=torch.long)

    def _tokenize_partition(self, part: Partition) -> list[int]:
        """Encode a partition using SENT-style walk with back-edges.

        If the partition has a child hierarchy, recursively encode it.
        Otherwise, encode the partition's edges using SENT back-edge format.

        Token format: [LCOM] [part_id] [num_nodes] [global_idx_0] ... [global_idx_n-1] [SEP] [SENT...] [RCOM]

        Args:
            part: Partition to tokenize.

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
            child_tokens = self._tokenize_child_hierarchy(part.child_hierarchy)
            tokens.extend(child_tokens)
        else:
            # Leaf partition: encode with SENT-style back-edges
            tokens.extend(self._tokenize_partition_sent(part))

        tokens.append(self.RCOM)
        return tokens

    def _tokenize_child_hierarchy(self, hg: HierarchicalGraph) -> list[int]:
        """Encode a nested hierarchy (without SOS/EOS).

        Args:
            hg: Child HierarchicalGraph.

        Returns:
            List of token indices.
        """
        tokens: list[int] = []

        # Number of sub-communities
        tokens.append(self.IDX_OFFSET + hg.num_communities)

        # Encode each sub-partition
        for part in hg.partitions:
            tokens.extend(self._tokenize_partition(part))

        # Encode each sub-bipartite
        for bipart in hg.bipartites:
            tokens.extend(self._tokenize_bipartite(bipart))

        return tokens

    def _tokenize_partition_sent(self, part: Partition) -> list[int]:
        """Encode partition edges using SENT-style back-edges.

        Args:
            part: Leaf partition to tokenize.

        Returns:
            List of token indices (node sequence with back-edges).
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
            visited.add(node)

            # Find back-edges to previously visited neighbors
            back_edges = sorted(
                order_to_idx[n]
                for n in adj[node]
                if n in visited and n != node
            )

            if back_edges:
                tokens.append(self.LADJ)
                tokens.extend(self.IDX_OFFSET + be for be in back_edges)
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

        # Encode edge pairs
        if bipart.edge_index.numel() > 0:
            ei = bipart.edge_index.numpy()
            for e in range(ei.shape[1]):
                tokens.append(self.IDX_OFFSET + int(ei[0, e]))
                tokens.append(self.IDX_OFFSET + int(ei[1, e]))

        tokens.append(self.RBIP)
        return tokens

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

        # Parse partitions - track global node offset
        partitions: list[Partition] = []
        global_node_offset = 0
        while idx < len(tokens_list) and tokens_list[idx] == self.LCOM:
            part, idx = self._parse_partition(tokens_list, idx + 1, global_node_offset)
            if part is not None:
                partitions.append(part)
                global_node_offset += part.num_nodes

        # Parse bipartites
        bipartites: list[Bipartite] = []
        while idx < len(tokens_list) and tokens_list[idx] == self.LBIP:
            bipart, idx = self._parse_bipartite(tokens_list, idx + 1)
            if bipart is not None:
                bipartites.append(bipart)

        # Reconstruct community assignment
        community_assignment = self._build_community_assignment(partitions)

        return HierarchicalGraph(partitions, bipartites, community_assignment)

    def _parse_partition(
        self, tokens: list[int], start: int, global_node_offset: int = 0
    ) -> tuple[Optional[Partition], int]:
        """Parse a [LCOM ... RCOM] block.

        Token format: [LCOM] [part_id] [num_nodes] [global_idx_0] ... [global_idx_n-1] [SEP] [SENT...] [RCOM]

        Args:
            tokens: Token list.
            start: Starting index (after LCOM).
            global_node_offset: Offset to add to local indices for global indices (fallback).

        Returns:
            Tuple of (parsed Partition or None, next index).
        """
        idx = start

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
            global_indices = list(range(global_node_offset, global_node_offset + num_nodes))

        # Check if this is a nested hierarchy or leaf partition
        # Nested if we see a number followed by LCOM
        child_hierarchy = None

        if idx < len(tokens) and tokens[idx] >= self.IDX_OFFSET:
            # Could be sub-community count or first node of SENT
            lookahead = idx + 1
            if lookahead < len(tokens) and tokens[lookahead] == self.LCOM:
                # Nested hierarchy
                child_hierarchy, idx = self._parse_child_hierarchy(
                    tokens, idx, global_node_offset
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
                    current_node = node_sequence[-1]
                    for back_pos in bracket_nodes:
                        if back_pos < len(node_sequence):
                            back_node = node_sequence[back_pos]
                            # Edge between current node and back node (in LOCAL indices)
                            edges.append((current_node, back_node))
                            edges.append((back_node, current_node))
                idx += 1
            elif tok >= self.IDX_OFFSET:
                val = tok - self.IDX_OFFSET

                if in_bracket:
                    bracket_nodes.append(val)
                else:
                    node_sequence.append(val)
                    # NO sequential edges - back-edges capture all adjacencies
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

        partition = Partition(
            part_id=part_id,
            global_node_indices=global_indices,
            edge_index=edge_index,
            child_hierarchy=None,
        )

        return partition, idx

    def _parse_child_hierarchy(
        self, tokens: list[int], start: int, global_node_offset: int = 0
    ) -> tuple[Optional[HierarchicalGraph], int]:
        """Parse a nested hierarchy within a partition.

        Args:
            tokens: Token list.
            start: Starting index (at sub-community count).
            global_node_offset: Offset to add to local indices for global indices.

        Returns:
            Tuple of (parsed HierarchicalGraph or None, next index).
        """
        idx = start

        # Parse number of sub-communities
        if idx >= len(tokens) or tokens[idx] < self.IDX_OFFSET:
            return None, idx
        num_sub_communities = tokens[idx] - self.IDX_OFFSET
        idx += 1

        # Parse sub-partitions - track offset within this hierarchy
        partitions: list[Partition] = []
        child_offset = global_node_offset
        while idx < len(tokens) and tokens[idx] == self.LCOM:
            part, idx = self._parse_partition(tokens, idx + 1, child_offset)
            if part is not None:
                partitions.append(part)
                child_offset += part.num_nodes

        # Parse sub-bipartites
        bipartites: list[Bipartite] = []
        while idx < len(tokens) and tokens[idx] == self.LBIP:
            bipart, idx = self._parse_bipartite(tokens, idx + 1)
            if bipart is not None:
                bipartites.append(bipart)

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

        # Parse edge pairs
        edges: list[tuple[int, int]] = []
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
            else:
                idx += 1

        if not edges:
            return None, idx

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return Bipartite(left_id, right_id, edge_index), idx

    def _build_community_assignment(
        self, partitions: list[Partition]
    ) -> list[int]:
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
