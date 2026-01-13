"""Node ordering strategies for partition tokenization.

This module provides algorithms for determining the order in which nodes
within a partition are tokenized. Different orderings can affect the
efficiency and quality of the learned representations.

Adapted from HiGen's graph_reorder.py.
"""

from typing import Literal

import networkx as nx
import numpy as np
from torch import Tensor

from src.tokenizers.hierarchical.structures import Partition


OrderingMethod = Literal["BFS", "DFS", "BFSAC", "BFSDC"]


def order_partition_nodes(
    partition: Partition,
    method: OrderingMethod = "BFS",
    seed: int | None = None,
) -> list[int]:
    """Compute canonical node ordering within a partition.

    The ordering determines the sequence in which nodes appear in the
    tokenized representation. Different orderings can capture different
    structural properties.

    Args:
        partition: Partition to order nodes for.
        method: Ordering algorithm to use:
            - "BFS": Breadth-first search from highest-degree node
            - "DFS": Depth-first search from highest-degree node
            - "BFSAC": Modified BFS with ascending cutset weight
            - "BFSDC": Modified BFS with descending cutset weight
        seed: Random seed for tie-breaking.

    Returns:
        List of LOCAL node indices in canonical order.
    """
    n = partition.num_nodes

    if n == 0:
        return []
    if n == 1:
        return [0]

    # Build NetworkX graph from partition edges
    G = _build_networkx_graph(partition)

    # Apply ordering method
    if method == "BFS":
        return _bfs_order(G, seed)
    elif method == "DFS":
        return _dfs_order(G, seed)
    elif method in ("BFSAC", "BFSDC"):
        return _bfs_modified(G, method, seed)
    else:
        raise ValueError(f"Unknown ordering method: {method}")


def _build_networkx_graph(partition: Partition) -> nx.Graph:
    """Build NetworkX graph from partition edges.

    Args:
        partition: Partition with edge_index.

    Returns:
        NetworkX Graph object.
    """
    G = nx.Graph()
    G.add_nodes_from(range(partition.num_nodes))

    if partition.edge_index.numel() > 0:
        edge_index = partition.edge_index.numpy()
        for e in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            if src != dst:
                if G.has_edge(src, dst):
                    # Increment weight for multi-edges
                    G[src][dst]["weight"] = G[src][dst].get("weight", 1) + 1
                else:
                    G.add_edge(src, dst, weight=1)

    return G


def _bfs_order(G: nx.Graph, seed: int | None = None) -> list[int]:
    """BFS ordering starting from highest-degree node.

    Args:
        G: NetworkX graph.
        seed: Random seed for tie-breaking.

    Returns:
        List of node indices in BFS order.
    """
    if len(G) == 0:
        return []

    rng = np.random.RandomState(seed)

    # Find starting node (highest degree, ties broken randomly)
    max_degree = max(G.degree(n) for n in G.nodes())
    candidates = [n for n in G.nodes() if G.degree(n) == max_degree]
    start = candidates[rng.randint(len(candidates))]

    visited: set[int] = set()
    order: list[int] = []
    queue: list[int] = [start]

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        # Add unvisited neighbors to queue
        neighbors = sorted(n for n in G.neighbors(node) if n not in visited)
        queue.extend(neighbors)

    # Handle disconnected components
    for node in G.nodes():
        if node not in visited:
            visited.add(node)
            order.append(node)

    return order


def _dfs_order(G: nx.Graph, seed: int | None = None) -> list[int]:
    """DFS ordering starting from highest-degree node.

    Args:
        G: NetworkX graph.
        seed: Random seed for tie-breaking.

    Returns:
        List of node indices in DFS order.
    """
    if len(G) == 0:
        return []

    rng = np.random.RandomState(seed)

    # Find starting node (highest degree, ties broken randomly)
    max_degree = max(G.degree(n) for n in G.nodes())
    candidates = [n for n in G.nodes() if G.degree(n) == max_degree]
    start = candidates[rng.randint(len(candidates))]

    visited: set[int] = set()
    order: list[int] = []
    stack: list[int] = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        # Add unvisited neighbors to stack (reverse sorted for consistent order)
        neighbors = sorted(
            (n for n in G.neighbors(node) if n not in visited),
            reverse=True,
        )
        stack.extend(neighbors)

    # Handle disconnected components
    for node in G.nodes():
        if node not in visited:
            visited.add(node)
            order.append(node)

    return order


def _bfs_modified(
    G: nx.Graph,
    method: Literal["BFSAC", "BFSDC"],
    seed: int | None = None,
) -> list[int]:
    """Modified BFS with cutset weight preference.

    Adapted from HiGen's bfs_modified_visit(). This ordering considers
    the "cutset weight" - the number of edges from a node to already
    visited nodes.

    Args:
        G: NetworkX graph.
        method: Either "BFSAC" (ascending cutset) or "BFSDC" (descending cutset).
        seed: Random seed for tie-breaking.

    Returns:
        List of node indices in modified BFS order.
    """
    if len(G) == 0:
        return []

    rng = np.random.RandomState(seed)

    # Initialize node info
    info: dict[int, dict] = {}
    for node in G.nodes():
        # Get self-loop weight if any
        self_weight = 0
        if G.has_edge(node, node):
            self_weight = G[node][node].get("weight", 1)

        info[node] = {
            "cutset_weight": self_weight,  # Weight of edges to visited nodes
            "degree": G.degree(node),
            "visited": False,
        }

    def sort_key(node: int) -> tuple:
        """Sort key for selecting next node."""
        i = info[node]
        if i["visited"]:
            return (float("inf"), float("inf"))

        if method == "BFSAC":
            # Ascending: prefer lower cutset weight, then higher degree
            return (i["cutset_weight"], -i["degree"])
        else:  # BFSDC
            # Descending: prefer higher cutset weight, then lower degree
            return (-i["cutset_weight"], i["degree"])

    result: list[int] = []

    while any(not i["visited"] for i in info.values()):
        # Find starting node for this component
        unvisited = [n for n in info if not info[n]["visited"]]

        # For tie-breaking, add small random perturbation
        if seed is not None:
            rng.shuffle(unvisited)

        source = min(unvisited, key=sort_key)
        queue = [source]

        while any(not info[n]["visited"] for n in queue):
            # Select next node from queue
            queue_unvisited = [n for n in queue if not info[n]["visited"]]
            if not queue_unvisited:
                break

            current = min(queue_unvisited, key=sort_key)

            result.append(current)
            info[current]["visited"] = True

            # Update cutset weights for neighbors
            for neighbor in G.neighbors(current):
                if not info[neighbor]["visited"]:
                    weight = G[current][neighbor].get("weight", 1)
                    info[neighbor]["cutset_weight"] += weight
                    if neighbor not in queue:
                        queue.append(neighbor)

    return result


def compute_canonical_order(
    edge_index: Tensor,
    num_nodes: int,
    method: OrderingMethod = "BFS",
    seed: int | None = None,
) -> list[int]:
    """Compute canonical ordering directly from edge_index.

    Convenience function that doesn't require creating a Partition object.

    Args:
        edge_index: Edge index tensor [2, num_edges].
        num_nodes: Number of nodes.
        method: Ordering method to use.
        seed: Random seed for tie-breaking.

    Returns:
        List of node indices in canonical order.
    """
    import torch

    # Create a temporary partition
    temp_partition = Partition(
        part_id=0,
        global_node_indices=list(range(num_nodes)),
        edge_index=edge_index,
        child_hierarchy=None,
    )
    return order_partition_nodes(temp_partition, method, seed)
