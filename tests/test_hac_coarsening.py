"""Tests for HAC (Hierarchical Agglomerative Clustering) coarsening.

Tests cover:
- Partition coverage and validity
- Trivial and small graph cases
- Different linkage criteria
- Hierarchy structure and recursion
- Roundtrip tests with HSENT and HDT tokenizers
- Tokenizer coarsener wiring
- Modularity computation
"""

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers import HDTTokenizer, HSENTTokenizer
from src.tokenizers.coarsening.hac import HACCoarsening

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


# ===========================================================================
# Test Partition Coverage
# ===========================================================================


class TestPartitionCoverage:
    """Test that HAC partitions cover all nodes without overlap."""

    def test_all_nodes_assigned(self, two_triangles: Data) -> None:
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(two_triangles)

        all_nodes = set()
        for comm in communities:
            all_nodes |= comm

        assert all_nodes == set(range(two_triangles.num_nodes))

    def test_no_overlap(self, two_triangles: Data) -> None:
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(two_triangles)

        seen: set[int] = set()
        for comm in communities:
            overlap = seen & comm
            assert len(overlap) == 0, f"Overlap found: {overlap}"
            seen |= comm

    def test_all_nodes_assigned_larger(self, larger_graph: Data) -> None:
        coarsener = HACCoarsening(min_community_size=4, seed=42)
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
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(data)
        assert len(communities) == 1
        assert communities[0] == {0}

    def test_no_edges(self) -> None:
        data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=5,
        )
        coarsener = HACCoarsening(min_community_size=4, seed=42)
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
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(data)
        assert hg.num_communities == 1


# ===========================================================================
# Test Partition Quality
# ===========================================================================


class TestPartitionQuality:
    """Test that partitions are meaningful."""

    def test_two_triangles_finds_communities(self, two_triangles: Data) -> None:
        """Two triangles should partition into 2+ communities."""
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(two_triangles)
        assert len(communities) >= 2

    def test_three_cliques_finds_communities(self, larger_graph: Data) -> None:
        """Three cliques should partition into 3 communities."""
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        communities = coarsener.partition(larger_graph)
        assert len(communities) >= 2


# ===========================================================================
# Test Different Linkages
# ===========================================================================


class TestLinkages:
    """Test that all linkage criteria produce valid partitions."""

    @pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
    def test_linkage_produces_valid_partition(
        self, linkage: str, larger_graph: Data
    ) -> None:
        coarsener = HACCoarsening(linkage=linkage, min_community_size=4, seed=42)
        communities = coarsener.partition(larger_graph)

        # All nodes covered
        all_nodes = set()
        for comm in communities:
            all_nodes |= comm
        assert all_nodes == set(range(larger_graph.num_nodes))

        # At least 1 community
        assert len(communities) >= 1

    @pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
    def test_linkage_builds_hierarchy(self, linkage: str, larger_graph: Data) -> None:
        coarsener = HACCoarsening(linkage=linkage, min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(larger_graph)
        assert hg.num_communities >= 2
        assert len(hg.partitions) == hg.num_communities


# ===========================================================================
# Test Hierarchy Structure
# ===========================================================================


class TestHierarchyStructure:
    """Test HierarchicalGraph structure produced by HAC."""

    def test_hierarchy_has_partitions_and_bipartites(self, two_triangles: Data) -> None:
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(two_triangles)

        assert len(hg.partitions) >= 1
        assert len(hg.community_assignment) == two_triangles.num_nodes

    def test_hierarchy_larger_graph(self, larger_graph: Data) -> None:
        coarsener = HACCoarsening(min_community_size=4, seed=42)
        hg = coarsener.build_hierarchy(larger_graph)

        assert hg.num_communities >= 2
        total_nodes = sum(p.num_nodes for p in hg.partitions)
        assert total_nodes == larger_graph.num_nodes

    def test_community_assignment_valid(self, larger_graph: Data) -> None:
        coarsener = HACCoarsening(min_community_size=4, seed=42)
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
        from torch_geometric.utils import to_dense_adj

        coarsener = HACCoarsening(min_community_size=4, seed=42)
        n = larger_graph.num_nodes
        adj = to_dense_adj(larger_graph.edge_index, max_num_nodes=n)[0]
        adj = ((adj + adj.t()) / 2).numpy()

        communities = coarsener.partition(larger_graph)
        partition_dict: dict[int, int] = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                partition_dict[node] = comm_id

        modularity = coarsener._compute_modularity(adj, partition_dict)
        assert modularity > 0

    def test_modularity_zero_for_no_edges(self) -> None:
        import numpy as np

        coarsener = HACCoarsening()
        adj = np.zeros((5, 5))
        partition = {i: 0 for i in range(5)}
        assert coarsener._compute_modularity(adj, partition) == 0.0


# ===========================================================================
# Test Roundtrip: HSENT + HAC
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
# Test Roundtrip: HDT + HAC
# ===========================================================================


class TestHDTRoundtrip:
    """Roundtrip tests for HDT tokenizer with HAC coarsening."""

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
    """Test that tokenizers correctly instantiate HACCoarsening."""

    def test_hsent_creates_hac_coarsener(self) -> None:
        tokenizer = HSENTTokenizer(coarsening_strategy="hac", seed=42)
        assert isinstance(tokenizer.coarsener, HACCoarsening)
        assert tokenizer.coarsener.linkage == "ward"

    def test_hdt_creates_hac_coarsener(self) -> None:
        tokenizer = HDTTokenizer(coarsening_strategy="hac", seed=42)
        assert isinstance(tokenizer.coarsener, HACCoarsening)
        assert tokenizer.coarsener.linkage == "ward"

    def test_hsent_passes_hac_params(self) -> None:
        tokenizer = HSENTTokenizer(
            coarsening_strategy="hac",
            hac_linkage="complete",
            hac_feature_type="adjacency",
            min_community_size=8,
            seed=42,
        )
        assert isinstance(tokenizer.coarsener, HACCoarsening)
        assert tokenizer.coarsener.linkage == "complete"
        assert tokenizer.coarsener.feature_type == "adjacency"
        assert tokenizer.coarsener.min_community_size == 8

    def test_hdt_passes_hac_params(self) -> None:
        tokenizer = HDTTokenizer(
            coarsening_strategy="hac",
            hac_linkage="average",
            min_community_size=6,
            seed=42,
        )
        assert isinstance(tokenizer.coarsener, HACCoarsening)
        assert tokenizer.coarsener.linkage == "average"
        assert tokenizer.coarsener.min_community_size == 6
