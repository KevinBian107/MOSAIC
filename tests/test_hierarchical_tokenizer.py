"""Tests for hierarchical tokenizer. THE ROUNDTRIP TEST IS THE KEY.

The implementation is correct if and only if:
    raw_graph == tokenizer.decode(tokenizer.tokenize(raw_graph))

Or equivalently:
    raw_graph == hg.reconstruct()
    where hg = parse_tokens(tokenize_hierarchy(build_hierarchy(raw_graph)))
"""

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers import (
    HSENTTokenizer,
    HierarchicalGraph,
    Partition,
    Bipartite,
    SpectralCoarsening,
    order_partition_nodes,
)

# Optional imports for molecular tests
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
    """Two triangles (0-1-2) and (3-4-5) connected by edge (2-3)."""
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


@pytest.fixture
def benzene() -> Data:
    """6-node cycle (benzene ring)."""
    edges = [(i, (i + 1) % 6) for i in range(6)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


@pytest.fixture
def complete_4() -> Data:
    """Complete graph K4."""
    edges = [(i, j) for i in range(4) for j in range(4) if i < j]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=4)


@pytest.fixture
def path_graph() -> Data:
    """Simple path: 0-1-2-3-4."""
    edges = [(i, i + 1) for i in range(4)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=5)


@pytest.fixture
def disconnected_small() -> Data:
    """Two disconnected edges: 0-1 and 2-3."""
    edges = [(0, 1), (2, 3)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=4)


@pytest.fixture
def star_graph() -> Data:
    """Star graph with center node 0 connected to nodes 1-5."""
    edges = [(0, i) for i in range(1, 6)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


@pytest.fixture
def larger_graph() -> Data:
    """Larger graph for testing multi-level hierarchy (12 nodes, 3 communities)."""
    # Community 0: nodes 0-3 (clique)
    comm0_edges = [(i, j) for i in range(4) for j in range(4) if i < j]
    # Community 1: nodes 4-7 (clique)
    comm1_edges = [(i, j) for i in range(4, 8) for j in range(4, 8) if i < j]
    # Community 2: nodes 8-11 (clique)
    comm2_edges = [(i, j) for i in range(8, 12) for j in range(8, 12) if i < j]
    # Inter-community edges
    inter_edges = [(3, 4), (7, 8), (11, 0)]

    all_edges = comm0_edges + comm1_edges + comm2_edges + inter_edges
    edge_list = [(s, d) for s, d in all_edges] + [(d, s) for s, d in all_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=12)


# ===========================================================================
# Core Roundtrip Tests - THE CRITICAL TESTS
# ===========================================================================


class TestRoundtrip:
    """THE CRITICAL TESTS: Roundtrip must preserve the graph exactly."""

    def test_roundtrip_two_triangles(self, two_triangles: Data) -> None:
        """raw → hierarchical → tokens → hierarchical → raw must be identical."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        # Forward: raw → hierarchical → tokens
        hg1 = tokenizer.coarsener.build_hierarchy(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hg1)

        # Backward: tokens → hierarchical → raw
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        # THE KEY ASSERTION
        assert edges_equal(two_triangles, reconstructed), (
            f"Roundtrip failed!\n"
            f"Original edges: {two_triangles.edge_index.tolist()}\n"
            f"Reconstructed edges: {reconstructed.edge_index.tolist()}"
        )

    def test_roundtrip_benzene(self, benzene: Data) -> None:
        """Roundtrip test for benzene ring."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(benzene)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(benzene, reconstructed)

    def test_roundtrip_complete_4(self, complete_4: Data) -> None:
        """Roundtrip test for complete graph K4."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(complete_4)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(complete_4, reconstructed)

    def test_roundtrip_path(self, path_graph: Data) -> None:
        """Roundtrip test for path graph."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(path_graph)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(path_graph, reconstructed)

    def test_roundtrip_star(self, star_graph: Data) -> None:
        """Roundtrip test for star graph."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(star_graph)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(star_graph, reconstructed)

    def test_roundtrip_disconnected(self, disconnected_small: Data) -> None:
        """Roundtrip test for disconnected graph."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(disconnected_small)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(disconnected_small, reconstructed)

    def test_roundtrip_larger_graph(self, larger_graph: Data) -> None:
        """Roundtrip test for larger graph with clear community structure."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(20)

        hg1 = tokenizer.coarsener.build_hierarchy(larger_graph)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(larger_graph, reconstructed)

    def test_direct_tokenize_decode(self, two_triangles: Data) -> None:
        """Test using tokenize() and decode() directly."""
        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(two_triangles, reconstructed)


# ===========================================================================
# Hierarchical Structure Tests
# ===========================================================================


class TestHierarchicalGraphReconstruct:
    """Test HierarchicalGraph.reconstruct() in isolation."""

    def test_reconstruct_simple(self) -> None:
        """Manual hierarchical graph reconstruction."""
        # Two partitions: {0,1} and {2,3}
        # Partition 0: edge 0-1
        # Partition 1: edge 0-1 (local) = 2-3 (global)
        # Bipartite: 1-0 (local) = 1-2 (global)

        p0 = Partition(
            part_id=0,
            global_node_indices=[0, 1],
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            child_hierarchy=None,
        )
        p1 = Partition(
            part_id=1,
            global_node_indices=[2, 3],
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            child_hierarchy=None,
        )
        bp = Bipartite(
            left_part_id=0,
            right_part_id=1,
            edge_index=torch.tensor([[1], [0]], dtype=torch.long),
        )

        hg = HierarchicalGraph([p0, p1], [bp], [0, 0, 1, 1])
        g = hg.reconstruct()

        # Should have: (0,1), (2,3), (1,2) edges
        assert g.num_nodes == 4
        assert g.edge_index.shape[1] > 0

    def test_reconstruct_empty(self) -> None:
        """Test reconstruction of empty hierarchy."""
        hg = HierarchicalGraph([], [], [])
        g = hg.reconstruct()
        assert g.num_nodes == 0
        assert g.edge_index.shape[1] == 0


# ===========================================================================
# Coarsening Tests
# ===========================================================================


class TestSpectralCoarsening:
    """Test coarsening produces valid partitions."""

    def test_all_nodes_assigned(self, two_triangles: Data) -> None:
        """Every node must be in exactly one community."""
        coarsener = SpectralCoarsening(seed=42, min_community_size=2)
        communities = coarsener.partition(two_triangles)

        all_nodes: set[int] = set()
        for comm in communities:
            assert len(comm & all_nodes) == 0, "Overlapping communities!"
            all_nodes.update(comm)

        assert all_nodes == set(range(6)), "Not all nodes assigned!"

    def test_hierarchy_covers_all_edges(self, two_triangles: Data) -> None:
        """Partitions + bipartites must cover all edges."""
        coarsener = SpectralCoarsening(seed=42, min_community_size=2)
        hg = coarsener.build_hierarchy(two_triangles)

        # Reconstruct and check edge count
        reconstructed = hg.reconstruct()
        original_edge_set = set(
            (min(int(two_triangles.edge_index[0, i]), int(two_triangles.edge_index[1, i])),
             max(int(two_triangles.edge_index[0, i]), int(two_triangles.edge_index[1, i])))
            for i in range(two_triangles.edge_index.shape[1])
            if two_triangles.edge_index[0, i] != two_triangles.edge_index[1, i]
        )
        reconstructed_edge_set = set(
            (min(int(reconstructed.edge_index[0, i]), int(reconstructed.edge_index[1, i])),
             max(int(reconstructed.edge_index[0, i]), int(reconstructed.edge_index[1, i])))
            for i in range(reconstructed.edge_index.shape[1])
            if reconstructed.edge_index[0, i] != reconstructed.edge_index[1, i]
        )

        assert original_edge_set == reconstructed_edge_set

    def test_small_graph_single_partition(self) -> None:
        """Small graphs should produce single partition."""
        # Graph with 3 nodes (below default min_community_size)
        edges = [(0, 1), (1, 2)]
        edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        small_graph = Data(edge_index=edge_index, num_nodes=3)

        coarsener = SpectralCoarsening(seed=42, min_community_size=4)
        hg = coarsener.build_hierarchy(small_graph)

        # Should be single partition
        assert hg.num_communities == 1


# ===========================================================================
# Ordering Tests
# ===========================================================================


class TestOrdering:
    """Test node ordering within partitions."""

    def test_ordering_returns_all_nodes(self) -> None:
        """Ordering must return all nodes exactly once."""
        # Create a simple partition
        partition = Partition(
            part_id=0,
            global_node_indices=[0, 1, 2, 3],
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
            ),
            child_hierarchy=None,
        )

        order = order_partition_nodes(partition, method="BFS", seed=42)

        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_ordering_deterministic(self) -> None:
        """Same seed should produce same ordering."""
        partition = Partition(
            part_id=0,
            global_node_indices=[0, 1, 2, 3],
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
            ),
            child_hierarchy=None,
        )

        order1 = order_partition_nodes(partition, method="BFS", seed=42)
        order2 = order_partition_nodes(partition, method="BFS", seed=42)

        assert order1 == order2


# ===========================================================================
# Tokenizer Interface Tests
# ===========================================================================


class TestHSENTTokenizer:
    """Test tokenizer interface and properties."""

    def test_vocab_size_requires_num_nodes(self) -> None:
        """vocab_size should raise error before set_num_nodes."""
        tokenizer = HSENTTokenizer()
        with pytest.raises(ValueError):
            _ = tokenizer.vocab_size

    def test_vocab_size_after_set_num_nodes(self) -> None:
        """vocab_size should return correct value after set_num_nodes."""
        tokenizer = HSENTTokenizer()
        tokenizer.set_num_nodes(100)
        # IDX_OFFSET (11) + max_num_nodes (100)
        assert tokenizer.vocab_size == 111

    def test_tokenize_produces_valid_tokens(self, two_triangles: Data) -> None:
        """Tokenize should produce valid token sequence."""
        tokenizer = HSENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)

        assert tokens[0] == tokenizer.SOS
        assert tokens[-1] == tokenizer.EOS
        assert all(0 <= t < tokenizer.vocab_size for t in tokens)

    def test_callable_interface(self, two_triangles: Data) -> None:
        """Tokenizer should be callable."""
        tokenizer = HSENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens1 = tokenizer.tokenize(two_triangles)
        tokens2 = tokenizer(two_triangles)

        assert torch.equal(tokens1, tokens2)

    def test_tokens_to_string(self, two_triangles: Data) -> None:
        """tokens_to_string should produce readable output."""
        tokenizer = HSENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        string = tokenizer.tokens_to_string(tokens)

        assert "[SOS]" in string
        assert "[EOS]" in string


# ===========================================================================
# Molecular Roundtrip Tests - Real MOSES/QM9 molecules
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMolecularRoundtrip:
    """Roundtrip tests with actual molecular graphs from MOSES."""

    # Common drug-like molecules for testing
    TEST_MOLECULES = [
        ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("benzene", "c1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("pyridine", "c1ccncc1"),
        ("ethanol", "CCO"),
        ("acetic_acid", "CC(=O)O"),
        ("phenol", "Oc1ccccc1"),
        ("aniline", "Nc1ccccc1"),
    ]

    @pytest.mark.parametrize("name,smiles", TEST_MOLECULES)
    def test_molecular_roundtrip(self, name: str, smiles: str) -> None:
        """Test roundtrip for real molecular graphs."""
        data = smiles_to_graph(smiles)
        assert data is not None, f"Failed to parse {name}: {smiles}"

        tokenizer = HSENTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(max(50, data.num_nodes))

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(data, reconstructed), (
            f"Roundtrip failed for {name}!\n"
            f"Original nodes: {data.num_nodes}, edges: {data.edge_index.shape[1]}\n"
            f"Reconstructed nodes: {reconstructed.num_nodes}, "
            f"edges: {reconstructed.edge_index.shape[1]}"
        )

    def test_moses_style_molecules(self) -> None:
        """Test with typical MOSES-style molecules."""
        # Representative molecules from MOSES dataset
        moses_samples = [
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen variant
            "Cc1ccc(C)c(O)c1",  # Cresol
            "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
            "c1ccc2c(c1)ccc1ccccc12",  # Phenanthrene
            "CC1=CC(=O)c2ccccc2C1=O",  # Menadione
        ]

        tokenizer = HSENTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(100)

        for smiles in moses_samples:
            data = smiles_to_graph(smiles)
            if data is None:
                continue

            tokens = tokenizer.tokenize(data)
            reconstructed = tokenizer.decode(tokens)

            assert edges_equal(data, reconstructed), f"Failed for {smiles}"

    def test_hierarchy_depth_for_molecules(self) -> None:
        """Verify hierarchy depth is reasonable for molecular graphs."""
        data = smiles_to_graph("CC1=CC(=O)c2ccccc2C1=O")  # Menadione
        assert data is not None

        tokenizer = HSENTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(50)

        hg = tokenizer.coarsener.build_hierarchy(data)

        # Molecular graphs should have reasonable decomposition
        assert hg.num_communities >= 1
        assert hg.num_communities <= data.num_nodes

        # Check level info
        info = hg.get_level_info()
        assert info["depth"] >= 0
