"""Graph fixtures for testing.

This module provides common graph structures for testing motif detection,
tokenization, and evaluation metrics.
"""

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


@pytest.fixture
def triangle_graph() -> Data:
    """A simple triangle (3-clique) graph."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
    )
    return Data(edge_index=edge_index, num_nodes=3)


@pytest.fixture
def square_graph() -> Data:
    """A 4-cycle (square) graph."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    return Data(edge_index=edge_index, num_nodes=4)


@pytest.fixture
def star_graph() -> Data:
    """A star graph with 5 leaves."""
    edges = []
    for i in range(1, 6):
        edges.extend([[0, i], [i, 0]])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


@pytest.fixture
def complete_graph_k4() -> Data:
    """A complete graph on 4 nodes (4-clique)."""
    G = nx.complete_graph(4)
    data = from_networkx(G)
    return Data(edge_index=data.edge_index, num_nodes=4)


@pytest.fixture
def erdos_renyi_graph() -> Data:
    """A small Erdos-Renyi random graph."""
    G = nx.erdos_renyi_graph(20, 0.3, seed=42)
    data = from_networkx(G)
    return Data(edge_index=data.edge_index, num_nodes=20)


@pytest.fixture
def barabasi_albert_graph() -> Data:
    """A small Barabasi-Albert preferential attachment graph."""
    G = nx.barabasi_albert_graph(20, 2, seed=42)
    data = from_networkx(G)
    return Data(edge_index=data.edge_index, num_nodes=20)


@pytest.fixture
def disconnected_graph() -> Data:
    """A graph with two disconnected components."""
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
    )
    return Data(edge_index=edge_index, num_nodes=4)


@pytest.fixture
def empty_graph() -> Data:
    """An empty graph with no edges."""
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=5)


@pytest.fixture
def sample_graph_list() -> list[Data]:
    """A list of diverse sample graphs for testing."""
    graphs = []

    G1 = nx.erdos_renyi_graph(30, 0.2, seed=0)
    data1 = from_networkx(G1)
    graphs.append(Data(edge_index=data1.edge_index, num_nodes=30))

    G2 = nx.barabasi_albert_graph(30, 3, seed=0)
    data2 = from_networkx(G2)
    graphs.append(Data(edge_index=data2.edge_index, num_nodes=30))

    G3 = nx.watts_strogatz_graph(30, 4, 0.3, seed=0)
    data3 = from_networkx(G3)
    graphs.append(Data(edge_index=data3.edge_index, num_nodes=30))

    return graphs
