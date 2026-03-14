"""Tests for HDTC (HDT Compositional) tokenizer.

THE ROUNDTRIP TEST IS THE KEY.

The implementation is correct if and only if:
    raw_graph == tokenizer.decode(tokenizer.tokenize(raw_graph))

Or equivalently:
    raw_graph == hierarchy.reconstruct()
    where hierarchy = parse_tokens(tokenize_hierarchy(build(raw_graph)))
"""

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers.hdtc import HDTCTokenizer

# Optional imports
try:
    from src.data.molecular import smiles_to_graph

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ===========================================================================
# Test Fixtures
# ===========================================================================


def edges_equal(g1: Data, g2: Data) -> bool:
    """Check if two graphs have the same edges (ignoring order and direction)."""

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
    """Larger graph for testing (12 nodes, 3 cliques)."""
    comm0_edges = [(i, j) for i in range(4) for j in range(4) if i < j]
    comm1_edges = [(i, j) for i in range(4, 8) for j in range(4, 8) if i < j]
    comm2_edges = [(i, j) for i in range(8, 12) for j in range(8, 12) if i < j]
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
        """Roundtrip test for two triangles."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(two_triangles, reconstructed), (
            f"Roundtrip failed!\n"
            f"Original edges: {two_triangles.edge_index.tolist()}\n"
            f"Reconstructed edges: {reconstructed.edge_index.tolist()}"
        )

    def test_roundtrip_benzene(self, benzene: Data) -> None:
        """Roundtrip test for benzene ring."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(benzene)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(benzene, reconstructed)

    def test_roundtrip_complete_4(self, complete_4: Data) -> None:
        """Roundtrip test for complete graph K4."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(complete_4)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(complete_4, reconstructed)

    def test_roundtrip_path(self, path_graph: Data) -> None:
        """Roundtrip test for path graph."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(path_graph)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(path_graph, reconstructed)

    def test_roundtrip_star(self, star_graph: Data) -> None:
        """Roundtrip test for star graph."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(star_graph)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(star_graph, reconstructed)

    def test_roundtrip_disconnected(self, disconnected_small: Data) -> None:
        """Roundtrip test for disconnected graph."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(disconnected_small)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(disconnected_small, reconstructed)

    def test_roundtrip_larger_graph(self, larger_graph: Data) -> None:
        """Roundtrip test for larger graph."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(larger_graph)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(larger_graph, reconstructed)


# ===========================================================================
# Tokenizer Interface Tests
# ===========================================================================


class TestHDTCTokenizer:
    """Test tokenizer interface and properties."""

    def test_vocab_size_requires_num_nodes(self) -> None:
        """vocab_size should raise error before set_num_nodes."""
        tokenizer = HDTCTokenizer()
        with pytest.raises(ValueError):
            _ = tokenizer.vocab_size

    def test_vocab_size_after_set_num_nodes(self) -> None:
        """vocab_size should return correct value after set_num_nodes."""
        tokenizer = HDTCTokenizer()
        tokenizer.set_num_nodes(100)
        # IDX_OFFSET (12) + max_num_nodes (100)
        assert tokenizer.vocab_size == 112

    def test_tokenize_produces_valid_tokens(self, two_triangles: Data) -> None:
        """Tokenize should produce valid token sequence."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)

        assert tokens[0] == tokenizer.SOS
        assert tokens[-1] == tokenizer.EOS
        assert all(0 <= t < tokenizer.vocab_size for t in tokens)

    def test_callable_interface(self, two_triangles: Data) -> None:
        """Tokenizer should be callable."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens1 = tokenizer.tokenize(two_triangles)
        tokens2 = tokenizer(two_triangles)

        assert torch.equal(tokens1, tokens2)

    def test_tokens_to_string(self, two_triangles: Data) -> None:
        """tokens_to_string should produce readable output."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        string = tokenizer.tokens_to_string(tokens)

        assert "[SOS]" in string
        assert "[EOS]" in string
        assert "[COMM_START]" in string
        assert "[COMM_END]" in string

    def test_special_token_constants(self) -> None:
        """Verify HDTC special token constants are correct."""
        tokenizer = HDTCTokenizer()

        assert tokenizer.SOS == 0
        assert tokenizer.EOS == 1
        assert tokenizer.PAD == 2
        assert tokenizer.COMM_START == 3
        assert tokenizer.COMM_END == 4
        assert tokenizer.LEDGE == 5
        assert tokenizer.REDGE == 6
        assert tokenizer.SUPER_START == 7
        assert tokenizer.SUPER_END == 8
        assert tokenizer.TYPE_RING == 9
        assert tokenizer.TYPE_FUNC == 10
        assert tokenizer.TYPE_SINGLETON == 11
        assert tokenizer.IDX_OFFSET == 12


# ===========================================================================
# HDTC-Specific Tests
# ===========================================================================


class TestHDTCSpecific:
    """HDTC-specific tests for compositional structure."""

    def test_community_blocks_present(self, two_triangles: Data) -> None:
        """Tokens should contain COMM_START/COMM_END blocks."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        tokens_list = tokens.tolist()

        comm_start_count = sum(1 for t in tokens_list if t == tokenizer.COMM_START)
        comm_end_count = sum(1 for t in tokens_list if t == tokenizer.COMM_END)

        assert comm_start_count > 0, "Expected COMM_START tokens"
        assert comm_start_count == comm_end_count, "COMM_START/END counts should match"

    def test_community_type_tokens(self, two_triangles: Data) -> None:
        """Tokens should contain community type markers."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        tokens_list = tokens.tolist()

        # Should have at least one type token
        type_tokens = [tokenizer.TYPE_RING, tokenizer.TYPE_FUNC, tokenizer.TYPE_SINGLETON]
        type_count = sum(1 for t in tokens_list if t in type_tokens)
        assert type_count > 0, "Expected community type tokens"

    def test_super_graph_block(self, two_triangles: Data) -> None:
        """Tokens should contain SUPER_START/SUPER_END if there are super-edges."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        tokens_list = tokens.tolist()

        # Check if SUPER_START is present
        has_super_start = tokenizer.SUPER_START in tokens_list
        has_super_end = tokenizer.SUPER_END in tokens_list

        # Either both present or both absent
        assert has_super_start == has_super_end

    def test_back_edges_in_communities(self, complete_4: Data) -> None:
        """Complete graph should have back-edges within communities."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(complete_4)
        tokens_list = tokens.tolist()

        ledge_count = sum(1 for t in tokens_list if t == tokenizer.LEDGE)
        assert ledge_count > 0, "Expected LEDGE tokens for back-edges"


# ===========================================================================
# Hierarchy Parsing Tests
# ===========================================================================


class TestHierarchyParsing:
    """Tests for hierarchy parsing from tokens."""

    def test_parse_tokens_preserves_communities(self, two_triangles: Data) -> None:
        """Parsing should preserve community structure."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        hierarchy1 = tokenizer.hierarchy_builder.build(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hierarchy1)
        hierarchy2 = tokenizer.parse_tokens(tokens)

        # Same number of communities
        assert len(hierarchy1.communities) == len(hierarchy2.communities)

        # Same total atoms
        atoms1 = sum(c.num_atoms for c in hierarchy1.communities)
        atoms2 = sum(c.num_atoms for c in hierarchy2.communities)
        assert atoms1 == atoms2

    def test_parse_empty_tokens(self) -> None:
        """Parsing empty tokens should return empty hierarchy."""
        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = torch.tensor([tokenizer.SOS, tokenizer.EOS], dtype=torch.long)
        hierarchy = tokenizer.parse_tokens(tokens)

        assert hierarchy.num_communities == 0
        assert hierarchy.num_atoms == 0


# ===========================================================================
# Molecular Roundtrip Tests
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMolecularRoundtrip:
    """Roundtrip tests with actual molecular graphs."""

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

        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(max(50, data.num_nodes))

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(data, reconstructed), (
            f"Roundtrip failed for {name}!\n"
            f"Original nodes: {data.num_nodes}, edges: {data.edge_index.shape[1]}\n"
            f"Reconstructed nodes: {reconstructed.num_nodes}, "
            f"edges: {reconstructed.edge_index.shape[1]}"
        )

    def test_ring_detection_in_tokens(self) -> None:
        """Benzene should be tokenized with ring community type."""
        data = smiles_to_graph("c1ccccc1")
        assert data is not None

        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(data)
        tokens_list = tokens.tolist()

        # Should have TYPE_RING token
        assert tokenizer.TYPE_RING in tokens_list


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_graph(self) -> None:
        """Test with empty graph."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=0)

        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 0

    def test_single_node(self) -> None:
        """Test with single node graph."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=1)

        tokenizer = HDTCTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 1

    def test_max_length_truncation(self, larger_graph: Data) -> None:
        """Test that max_length truncation works."""
        tokenizer = HDTCTokenizer(seed=42, max_length=20)
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(larger_graph)

        assert len(tokens) <= 20
        assert tokens[-1] == tokenizer.EOS


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestConfiguration:
    """Tests for tokenizer configuration."""

    def test_disable_rings(self, benzene: Data) -> None:
        """Test with ring detection disabled."""
        tokenizer = HDTCTokenizer(seed=42, include_rings=False)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(benzene)
        tokens_list = tokens.tolist()

        # Should not have TYPE_RING token (since rings not detected)
        # All atoms become singletons
        type_singleton_count = sum(
            1 for t in tokens_list if t == tokenizer.TYPE_SINGLETON
        )
        assert type_singleton_count > 0

    def test_custom_patterns(self, benzene: Data) -> None:
        """Test with custom ring patterns."""
        tokenizer = HDTCTokenizer(
            seed=42,
            include_rings=True,
            ring_patterns={"my_ring": "c1ccccc1"},
            functional_patterns={},
        )
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(benzene)
        reconstructed = tokenizer.decode(tokens)

        # Should still roundtrip correctly
        assert edges_equal(benzene, reconstructed)
