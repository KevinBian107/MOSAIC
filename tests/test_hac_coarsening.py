"""Tests for affinity-based graph coarsening (AffinityCoarsening).

Tests cover:
- Partition coverage and validity
- Trivial and small graph cases
- Bond weight affinity computation
- Hierarchy structure and recursion
- Roundtrip tests with HSENT and HDT tokenizers
- Tokenizer coarsener wiring
- Modularity computation
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers import HDTTokenizer, HSENTTokenizer
from src.tokenizers.coarsening.hac import BOND_WEIGHT_MAP, AffinityCoarsening

# Optional RDKit import
try:
    from src.data.molecular import smiles_to_graph

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ===========================================================================
# Test Fixtures
# ===========================================================================


def edges_equal(g1: Data, g2: Data) -> bool:
    """Check if two graphs have the same edges (ignoring order and direction).

    For undirected graphs, (a, b) and (b, a) are considered the same edge.
    """

    def edge_set(g: Data) -> frozenset[tuple[int, int]]:
        if g.edge_index.numel() == 0:
            return frozenset()
        ei = g.edge_index.numpy()
        return frozenset(
            (min(int(ei[0, i]), int(ei[1, i])), max(int(ei[0, i]), int(ei[1, i])))
            for i in range(ei.shape[1])
            if ei[0, i] != ei[1, i]
        )

    return edge_set(g1) == edge_set(g2)


@pytest.fixture
def two_triangles() -> Data:
    """Two triangles connected by a bridge edge: {0,1,2} -- {3,4,5}."""
    edges = [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (0, 2),
        (2, 0),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (3, 5),
        (5, 3),
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


@pytest.fixture
def larger_graph() -> Data:
    """12-node graph: three 4-cliques connected by bridges."""
    edges = []
    for clique_start in [0, 4, 8]:
        for i in range(clique_start, clique_start + 4):
            for j in range(clique_start, clique_start + 4):
                if i != j:
                    edges.append((i, j))
    # Bridge edges
    edges.extend([(3, 4), (4, 3), (7, 8), (8, 7)])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=12)


@pytest.fixture
def benzene_data() -> Data:
    """Benzene (6-node cycle)."""
    edges = []
    for i in range(6):
        j = (i + 1) % 6
        edges.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


@pytest.fixture
def weighted_graph() -> Data:
    """4-node graph with different bond types for affinity testing."""
    # 0 --double-- 1
    # |            |
    # single      single
    # |            |
    # 2 --triple-- 3
    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)]
    edge_attr = [1, 1, 0, 0, 0, 0, 2, 2]  # double, single, single, triple
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Data(
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
        num_nodes=4,
    )


# ===========================================================================
# Test Partition Coverage
# ===========================================================================


class TestPartitionCoverage:
    """Test that affinity partitions cover all nodes without overlap."""

    def test_all_nodes_assigned(self, two_triangles: Data) -> None:
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(two_triangles)

        all_nodes = set()
        for comm in communities:
            all_nodes |= comm

        assert all_nodes == set(range(two_triangles.num_nodes))

    def test_no_overlap(self, two_triangles: Data) -> None:
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(two_triangles)

        seen: set[int] = set()
        for comm in communities:
            overlap = seen & comm
            assert len(overlap) == 0, f"Overlap found: {overlap}"
            seen |= comm

    def test_all_nodes_assigned_larger(self, larger_graph: Data) -> None:
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(larger_graph)

        all_nodes = set()
        for comm in communities:
            all_nodes |= comm

        assert all_nodes == set(range(larger_graph.num_nodes))


# ===========================================================================
# Test Trivial Cases
# ===========================================================================


class TestTrivialCases:
    """Test behavior on trivial and edge-case graphs."""

    def test_single_node(self) -> None:
        data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=1,
        )
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(data)
        assert len(communities) == 1
        assert communities[0] == {0}

    def test_no_edges(self) -> None:
        data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=5,
        )
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(data)
        # Should return single community (no edges to split on)
        assert len(communities) == 1

    def test_small_graph_below_min_community_size(self) -> None:
        """Graph smaller than min_community_size should be single partition."""
        edges = [(0, 1), (1, 0)]
        data = Data(
            edge_index=torch.tensor(edges, dtype=torch.long).t(),
            num_nodes=2,
        )
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(data)
        assert hg.num_communities == 1


# ===========================================================================
# Test Partition Quality
# ===========================================================================


class TestPartitionQuality:
    """Test that partitions are meaningful."""

    def test_two_triangles_finds_communities(self, two_triangles: Data) -> None:
        """Two triangles should partition into 2+ communities."""
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(two_triangles)
        assert len(communities) >= 2

    def test_three_cliques_finds_communities(self, larger_graph: Data) -> None:
        """Three cliques should partition into 3 communities."""
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(larger_graph)
        assert len(communities) >= 2


# ===========================================================================
# Test Bond Weight Affinity
# ===========================================================================


class TestBondWeightAffinity:
    """Test that edge weights from bond types are used correctly."""

    def test_bond_weight_map_values(self) -> None:
        """Verify the BOND_WEIGHT_MAP has correct entries."""
        assert BOND_WEIGHT_MAP[0] == 1.0  # single
        assert BOND_WEIGHT_MAP[1] == 2.0  # double
        assert BOND_WEIGHT_MAP[2] == 3.0  # triple
        assert BOND_WEIGHT_MAP[3] == 1.5  # aromatic
        assert BOND_WEIGHT_MAP[4] == 1.0  # unknown

    def test_weighted_adjacency_uses_bond_types(self, weighted_graph: Data) -> None:
        """Weighted adjacency should reflect bond types."""
        coarsener = AffinityCoarsening(seed=42)
        adj = coarsener._build_weighted_adj(weighted_graph)

        # 0-1: double bond (weight 2.0)
        assert adj[0, 1] == 2.0
        assert adj[1, 0] == 2.0

        # 0-2: single bond (weight 1.0)
        assert adj[0, 2] == 1.0

        # 2-3: triple bond (weight 3.0)
        assert adj[2, 3] == 3.0
        assert adj[3, 2] == 3.0

    def test_unweighted_graph_uses_unit_weights(self, two_triangles: Data) -> None:
        """Graph without edge_attr should have uniform weight 1.0."""
        coarsener = AffinityCoarsening(seed=42)
        adj = coarsener._build_weighted_adj(two_triangles)

        # All edges should have weight 1.0
        for i in range(6):
            for j in range(6):
                if adj[i, j] > 0:
                    assert adj[i, j] == 1.0


# ===========================================================================
# Test Hierarchy Structure
# ===========================================================================


class TestHierarchyStructure:
    """Test HierarchicalGraph structure produced by affinity coarsening."""

    def test_hierarchy_has_partitions_and_bipartites(self, two_triangles: Data) -> None:
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(two_triangles)

        assert len(hg.partitions) >= 1
        assert len(hg.community_assignment) == two_triangles.num_nodes

    def test_hierarchy_larger_graph(self, larger_graph: Data) -> None:
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(larger_graph)

        assert hg.num_communities >= 2
        total_nodes = sum(p.num_nodes for p in hg.partitions)
        assert total_nodes == larger_graph.num_nodes

    def test_community_assignment_valid(self, larger_graph: Data) -> None:
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(larger_graph)

        for i, comm_id in enumerate(hg.community_assignment):
            assert 0 <= comm_id < hg.num_communities


# ===========================================================================
# Test Modularity
# ===========================================================================


class TestModularity:
    """Test modularity computation."""

    def test_modularity_positive_for_good_partition(self, larger_graph: Data) -> None:
        """Non-trivial graph should have positive modularity."""
        coarsener = AffinityCoarsening(min_community_size=4, seed=42)
        adj = coarsener._build_weighted_adj(larger_graph)

        communities = coarsener.partition(larger_graph)
        partition_dict: dict[int, int] = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                partition_dict[node] = comm_id

        modularity = coarsener._compute_modularity(adj, partition_dict)
        assert modularity > 0

    def test_modularity_zero_for_no_edges(self) -> None:
        coarsener = AffinityCoarsening()
        adj = np.zeros((5, 5))
        partition = {i: 0 for i in range(5)}
        assert coarsener._compute_modularity(adj, partition) == 0.0


# ===========================================================================
# Test Roundtrip: HSENT + Affinity
# ===========================================================================


class TestHSENTRoundtrip:
    """Roundtrip tests: tokenize → decode should preserve edges."""

    def test_roundtrip_larger_graph(self, larger_graph: Data) -> None:
        tokenizer = HSENTTokenizer(
            coarsening_strategy="hac", seed=42, min_community_size=4
        )
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(larger_graph)
        decoded = tokenizer.decode(tokens)

        assert edges_equal(larger_graph, decoded)

    def test_roundtrip_benzene(self, benzene_data: Data) -> None:
        tokenizer = HSENTTokenizer(
            coarsening_strategy="hac", seed=42, min_community_size=4
        )
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(benzene_data)
        decoded = tokenizer.decode(tokens)

        assert edges_equal(benzene_data, decoded)

    @pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
    @pytest.mark.parametrize(
        "smiles",
        ["c1ccccc1", "CC(=O)O", "CCO", "c1ccc2ccccc2c1"],
        ids=["benzene", "acetic_acid", "ethanol", "naphthalene"],
    )
    def test_roundtrip_molecular(self, smiles: str) -> None:
        data = smiles_to_graph(smiles, labeled=False)
        # Strip multi-dim features for unlabeled tokenizer
        clean = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
        tokenizer = HSENTTokenizer(
            coarsening_strategy="hac", seed=42, min_community_size=4
        )
        tokenizer.set_num_nodes(clean.num_nodes + 5)

        tokens = tokenizer.tokenize(clean)
        decoded = tokenizer.decode(tokens)

        assert edges_equal(clean, decoded)


# ===========================================================================
# Test Roundtrip: HDT + Affinity
# ===========================================================================


class TestHDTRoundtrip:
    """Roundtrip tests for HDT tokenizer with affinity coarsening."""

    def test_roundtrip_larger_graph(self, larger_graph: Data) -> None:
        tokenizer = HDTTokenizer(
            coarsening_strategy="hac", seed=42, min_community_size=4
        )
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(larger_graph)
        decoded = tokenizer.decode(tokens)

        assert edges_equal(larger_graph, decoded)

    def test_roundtrip_benzene(self, benzene_data: Data) -> None:
        tokenizer = HDTTokenizer(
            coarsening_strategy="hac", seed=42, min_community_size=4
        )
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(benzene_data)
        decoded = tokenizer.decode(tokens)

        assert edges_equal(benzene_data, decoded)

    @pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
    @pytest.mark.parametrize(
        "smiles",
        ["c1ccccc1", "CC(=O)O", "CCO", "c1ccc2ccccc2c1"],
        ids=["benzene", "acetic_acid", "ethanol", "naphthalene"],
    )
    def test_roundtrip_molecular(self, smiles: str) -> None:
        data = smiles_to_graph(smiles, labeled=False)
        # Strip multi-dim features for unlabeled tokenizer
        clean = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
        tokenizer = HDTTokenizer(
            coarsening_strategy="hac", seed=42, min_community_size=4
        )
        tokenizer.set_num_nodes(clean.num_nodes + 5)

        tokens = tokenizer.tokenize(clean)
        decoded = tokenizer.decode(tokens)

        assert edges_equal(clean, decoded)


# ===========================================================================
# Test Tokenizer Wiring
# ===========================================================================


class TestTokenizerWiring:
    """Test that tokenizers correctly instantiate AffinityCoarsening."""

    def test_hsent_creates_affinity_coarsener(self) -> None:
        tokenizer = HSENTTokenizer(coarsening_strategy="hac", seed=42)
        assert isinstance(tokenizer.coarsener, AffinityCoarsening)

    def test_hdt_creates_affinity_coarsener(self) -> None:
        tokenizer = HDTTokenizer(coarsening_strategy="hac", seed=42)
        assert isinstance(tokenizer.coarsener, AffinityCoarsening)

    def test_hsent_passes_params(self) -> None:
        tokenizer = HSENTTokenizer(
            coarsening_strategy="hac",
            min_community_size=8,
            seed=42,
        )
        assert isinstance(tokenizer.coarsener, AffinityCoarsening)
        assert tokenizer.coarsener.min_community_size == 8

    def test_hdt_passes_params(self) -> None:
        tokenizer = HDTTokenizer(
            coarsening_strategy="hac",
            min_community_size=6,
            seed=42,
        )
        assert isinstance(tokenizer.coarsener, AffinityCoarsening)
        assert tokenizer.coarsener.min_community_size == 6
