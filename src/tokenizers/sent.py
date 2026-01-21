"""SENT (Sequence of Edge-indicating Neighborhoods) tokenizer.

This module implements the SENT tokenization scheme from AutoGraph, which
converts graphs to token sequences via random walk with back-edge encoding.
"""

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.sent_utils import (
    sample_labeled_sent_from_graph,
    get_graph_from_labeled_sent,
)


class SENTTokenizer(Tokenizer):
    """SENT tokenizer for graph-to-sequence conversion.

    SENT represents graphs as sequences by performing a random walk traversal
    and encoding back-edges (connections to previously visited nodes) using
    special bracket tokens.

    Attributes:
        sos: Start-of-sequence token (0).
        reset: New trail/component start token (1).
        ladj: Left bracket for back-edge list (2).
        radj: Right bracket for back-edge list (3).
        eos: End-of-sequence token (4).
        pad: Padding token (5).
        idx_offset: Offset for node index tokens.
        max_num_nodes: Maximum number of nodes (determines vocab size).
        max_length: Maximum sequence length (-1 for unlimited).
        truncation_length: Length for truncating during batching.
    """

    tokenizer_type: str = "sent"
    sos: int = 0
    reset: int = 1
    ladj: int = 2
    radj: int = 3
    eos: int = 4
    pad: int = 5
    special_toks = ["sos", "reset", "ladj", "radj", "eos", "pad"]

    def __init__(
        self,
        max_length: int = -1,
        truncation_length: Optional[int] = None,
        undirected: bool = True,
        append_eos: bool = True,
        seed: Optional[int] = None,
        labeled_graph: bool = False,
    ) -> None:
        """Initialize the SENT tokenizer.

        Args:
            max_length: Maximum sequence length (-1 for unlimited).
            truncation_length: Length for truncation during batching.
            undirected: Whether to treat graphs as undirected.
            append_eos: Whether to append EOS token.
            seed: Random seed for walk sampling.
            labeled_graph: Whether to use labeled SENT (encode node/edge types).
        """
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.undirected = undirected
        self.append_eos = append_eos
        self.idx_offset = len(self.special_toks)
        self.max_num_nodes: Optional[int] = None
        self.rng = np.random.RandomState(seed)

        # Labeled graph support (AutoGraph format)
        self.labeled_graph = labeled_graph
        self.num_node_types = 0
        self.num_edge_types = 0
        self.node_idx_offset: Optional[int] = None
        self.edge_idx_offset: Optional[int] = None

    def set_num_nodes(self, max_num_nodes: int) -> None:
        """Set the maximum number of nodes.

        Args:
            max_num_nodes: Maximum nodes in any graph (determines vocab size).
        """
        if self.max_num_nodes is None or self.max_num_nodes < max_num_nodes:
            self.max_num_nodes = max_num_nodes

    def set_num_node_and_edge_types(self, num_node_types: int, num_edge_types: int) -> None:
        """Set number of node and edge types for labeled graphs.

        Args:
            num_node_types: Number of node types (e.g., atom types).
            num_edge_types: Number of edge types (e.g., bond types).
        """
        if self.labeled_graph:
            self.num_node_types = num_node_types
            self.num_edge_types = num_edge_types
            self.node_idx_offset = self.idx_offset + self.max_num_nodes
            self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes() first")
        if self.labeled_graph:
            if self.edge_idx_offset is None:
                raise ValueError("Call set_num_node_and_edge_types() first for labeled graphs")
            return self.edge_idx_offset + self.num_edge_types
        return self.idx_offset + self.max_num_nodes

    def tokenize(self, data: Data) -> torch.Tensor:
        """Convert a graph to a SENT token sequence.

        The algorithm performs a random walk traversal, encoding:
        - Each new node as its index (offset by idx_offset)
        - For labeled graphs: node types and edge types
        - Back-edges as [ladj, ...indices..., radj] brackets

        Args:
            data: PyTorch Geometric Data object with edge_index (and x, edge_attr for labeled).

        Returns:
            1D tensor of token indices.
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        if self.labeled_graph:
            # Use AutoGraph's labeled SENT (matches tokenizer.py lines 74-88)
            walk, _ = sample_labeled_sent_from_graph(
                edge_index=edge_index,
                node_labels=data.x,  # Integer labels
                edge_labels=data.edge_attr,  # Integer labels
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_nodes=num_nodes,
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
                rng=self.rng,
            )
            walk = torch.from_numpy(walk)
        else:
            # Use original MOSAIC unlabeled SENT
            adj = self._build_adjacency(edge_index, num_nodes)
            walk = self._sample_walk(adj, num_nodes)

        # Add SOS/EOS tokens
        tokens_list = [self.sos]
        tokens_list.extend(walk.tolist() if isinstance(walk, torch.Tensor) else walk)
        if self.append_eos:
            tokens_list.append(self.eos)

        return torch.tensor(tokens_list, dtype=torch.long)

    def _build_adjacency(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> dict[int, set[int]]:
        """Build adjacency list from edge index.

        Args:
            edge_index: Edge index tensor [2, num_edges].
            num_nodes: Number of nodes.

        Returns:
            Dictionary mapping node to set of neighbors.
        """
        adj: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        edge_index_np = edge_index.numpy()

        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            adj[src].add(dst)
            if self.undirected:
                adj[dst].add(src)

        return adj

    def _sample_walk(
        self, adj: dict[int, set[int]], num_nodes: int
    ) -> list[int]:
        """Sample a SENT walk through the graph.

        Args:
            adj: Adjacency list.
            num_nodes: Number of nodes.

        Returns:
            List of token indices (without SOS/EOS).
        """
        tokens: list[int] = []
        visited: set[int] = set()
        node_map: dict[int, int] = {}
        current_idx = 0

        unvisited = set(range(num_nodes))
        current: Optional[int] = None
        prev: Optional[int] = None

        while unvisited:
            if current is None:
                if tokens and tokens[-1] != self.reset:
                    tokens.append(self.reset)
                current = self.rng.choice(list(unvisited))
                node_map[current] = current_idx
                tokens.append(self.idx_offset + current_idx)
                current_idx += 1
                visited.add(current)
                unvisited.discard(current)
                prev = None
                continue

            neighbors = adj[current]
            unvisited_neighbors = [n for n in neighbors if n not in visited]

            if unvisited_neighbors:
                next_node = self.rng.choice(unvisited_neighbors)
                node_map[next_node] = current_idx
                tokens.append(self.idx_offset + current_idx)
                current_idx += 1

                back_edges = sorted(
                    node_map[n]
                    for n in neighbors
                    if n in visited and n != prev
                )
                if back_edges:
                    tokens.append(self.ladj)
                    tokens.extend(self.idx_offset + be for be in back_edges)
                    tokens.append(self.radj)

                visited.add(next_node)
                unvisited.discard(next_node)
                prev = current
                current = next_node
            else:
                visited_neighbors = [n for n in neighbors if n in visited]
                if visited_neighbors and self.rng.random() < 0.5:
                    current = self.rng.choice(visited_neighbors)
                    prev = None
                else:
                    current = None
                    prev = None

            if self.max_length > 0 and len(tokens) >= self.max_length - 2:
                break

        return tokens

    def decode(self, tokens: torch.Tensor) -> Data:
        """Decode a token sequence back to a graph.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            PyTorch Geometric Data object with integer labels (if labeled) or empty features.
        """
        tokens = tokens[
            (tokens != self.pad) & (tokens != self.sos) & (tokens != self.eos)
        ]

        if self.labeled_graph:
            # Use AutoGraph's labeled SENT decoder (matches tokenizer.py lines 132-149)
            edge_index, node_labels, edge_labels = get_graph_from_labeled_sent(
                walk_index=tokens,
                idx_offset=self.idx_offset,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
            )

            num_nodes = max(edge_index.flatten().max() + 1 if edge_index.numel() > 0 else 0, len(node_labels))

            return Data(
                x=node_labels,
                edge_index=edge_index,
                edge_attr=edge_labels,
                num_nodes=num_nodes,
            )
        else:
            # Original MOSAIC unlabeled decode
            tokens = tokens.cpu().numpy()

            edges: list[tuple[int, int]] = []
            node_idx_to_real: dict[int, int] = {}
            current_idx = 0
            prev_node: Optional[int] = None
            in_bracket = False
            bracket_nodes: list[int] = []

            i = 0
            while i < len(tokens):
                tok = tokens[i]

                if tok == self.reset:
                    prev_node = None
                    i += 1
                    continue

                if tok == self.ladj:
                    in_bracket = True
                    bracket_nodes = []
                    i += 1
                    continue

                if tok == self.radj:
                    in_bracket = False
                    if prev_node is not None:
                        current_real = node_idx_to_real.get(current_idx - 1)
                        if current_real is not None:
                            for bn in bracket_nodes:
                                back_real = node_idx_to_real.get(bn)
                                if back_real is not None:
                                    edges.append((current_real, back_real))
                                    if self.undirected:
                                        edges.append((back_real, current_real))
                    i += 1
                    continue

                if tok >= self.idx_offset:
                    node_idx = tok - self.idx_offset

                    if in_bracket:
                        bracket_nodes.append(node_idx)
                    else:
                        if node_idx not in node_idx_to_real:
                            node_idx_to_real[node_idx] = len(node_idx_to_real)

                        real_node = node_idx_to_real[node_idx]

                        if prev_node is not None:
                            edges.append((prev_node, real_node))
                            if self.undirected:
                                edges.append((real_node, prev_node))

                        prev_node = real_node
                        current_idx = node_idx + 1

                i += 1

            num_nodes = len(node_idx_to_real)
            if num_nodes == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                num_nodes = 1
            else:
                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)

            return Data(edge_index=edge_index, num_nodes=num_nodes)

    def batch_converter(self) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        """Return the batch conversion function.

        Returns:
            BatchConverter instance.
        """
        return BatchConverter(self, self.truncation_length)
