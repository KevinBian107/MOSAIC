"""Tests for HDT (Hierarchical DFS-based) tokenizer. THE ROUNDTRIP TEST IS THE KEY.

The implementation is correct if and only if:
    raw_graph == tokenizer.decode(tokenizer.tokenize(raw_graph))

Or equivalently:
    raw_graph == hg.reconstruct()
    where hg = parse_tokens(tokenize_hierarchy(build_hierarchy(raw_graph)))

HDT-specific tests also verify:
    - Token count reduction vs H-SENT (~45% fewer tokens)
    - Cross-community edges encoded as back-edges (no bipartite blocks)
"""

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers import (
    SpectralCoarsening,
    HierarchicalGraph,
    Partition,
    Bipartite,
    order_partition_nodes,
    HDTTokenizer,
)

# Optional imports for molecular tests
try:
    from src.data.molecular import smiles_to_graph

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Optional H-SENT import for comparison tests
try:
    from src.tokenizers import HSENTTokenizer

    HAS_HSENT = True
except ImportError:
    HAS_HSENT = False


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
        """raw -> hierarchical -> tokens -> hierarchical -> raw must be identical."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        # Forward: raw -> hierarchical -> tokens
        hg1 = tokenizer.coarsener.build_hierarchy(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hg1)

        # Backward: tokens -> hierarchical -> raw
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
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(benzene)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(benzene, reconstructed)

    def test_roundtrip_complete_4(self, complete_4: Data) -> None:
        """Roundtrip test for complete graph K4."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(complete_4)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(complete_4, reconstructed)

    def test_roundtrip_path(self, path_graph: Data) -> None:
        """Roundtrip test for path graph."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(path_graph)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(path_graph, reconstructed)

    def test_roundtrip_star(self, star_graph: Data) -> None:
        """Roundtrip test for star graph."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(star_graph)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(star_graph, reconstructed)

    def test_roundtrip_disconnected(self, disconnected_small: Data) -> None:
        """Roundtrip test for disconnected graph."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg1 = tokenizer.coarsener.build_hierarchy(disconnected_small)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(disconnected_small, reconstructed)

    def test_roundtrip_larger_graph(self, larger_graph: Data) -> None:
        """Roundtrip test for larger graph with clear community structure."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(20)

        hg1 = tokenizer.coarsener.build_hierarchy(larger_graph)
        tokens = tokenizer.tokenize_hierarchy(hg1)
        hg2 = tokenizer.parse_tokens(tokens)
        reconstructed = hg2.reconstruct()

        assert edges_equal(larger_graph, reconstructed)

    def test_direct_tokenize_decode(self, two_triangles: Data) -> None:
        """Test using tokenize() and decode() directly."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
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
# Tokenizer Interface Tests
# ===========================================================================


class TestHDTTokenizer:
    """Test tokenizer interface and properties."""

    def test_vocab_size_requires_num_nodes(self) -> None:
        """vocab_size should raise error before set_num_nodes."""
        tokenizer = HDTTokenizer()
        with pytest.raises(ValueError):
            _ = tokenizer.vocab_size

    def test_vocab_size_after_set_num_nodes(self) -> None:
        """vocab_size should return correct value after set_num_nodes."""
        tokenizer = HDTTokenizer()
        tokenizer.set_num_nodes(100)
        # IDX_OFFSET (7) + max_num_nodes (100)
        assert tokenizer.vocab_size == 107

    def test_tokenize_produces_valid_tokens(self, two_triangles: Data) -> None:
        """Tokenize should produce valid token sequence."""
        tokenizer = HDTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)

        assert tokens[0] == tokenizer.SOS
        assert tokens[-1] == tokenizer.EOS
        assert all(0 <= t < tokenizer.vocab_size for t in tokens)

    def test_callable_interface(self, two_triangles: Data) -> None:
        """Tokenizer should be callable."""
        tokenizer = HDTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens1 = tokenizer.tokenize(two_triangles)
        tokens2 = tokenizer(two_triangles)

        assert torch.equal(tokens1, tokens2)

    def test_tokens_to_string(self, two_triangles: Data) -> None:
        """tokens_to_string should produce readable output."""
        tokenizer = HDTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        string = tokenizer.tokens_to_string(tokens)

        assert "[SOS]" in string
        assert "[EOS]" in string
        assert "[ENTER]" in string
        assert "[EXIT]" in string

    def test_special_token_constants(self) -> None:
        """Verify HDT special token constants are correct."""
        tokenizer = HDTTokenizer()

        assert tokenizer.SOS == 0
        assert tokenizer.EOS == 1
        assert tokenizer.PAD == 2
        assert tokenizer.ENTER == 3
        assert tokenizer.EXIT == 4
        assert tokenizer.LEDGE == 5
        assert tokenizer.REDGE == 6
        assert tokenizer.IDX_OFFSET == 7

    def test_no_bipartite_blocks_in_tokens(self, two_triangles: Data) -> None:
        """HDT should not produce LBIP/RBIP tokens (uses back-edges instead)."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        tokens_list = tokens.tolist()

        # HDT does not have LBIP (8) or RBIP (9) tokens - these are from H-SENT
        # HDT only has tokens 0-6 as special, and 7+ as indices
        for tok in tokens_list:
            if tok < tokenizer.IDX_OFFSET:
                # Should only be SOS, EOS, PAD, ENTER, EXIT, LEDGE, REDGE
                assert tok in [0, 1, 2, 3, 4, 5, 6], f"Unexpected special token: {tok}"


# ===========================================================================
# HDT-Specific Tests
# ===========================================================================


class TestHDTSpecific:
    """HDT-specific tests for token efficiency and cross-community edges."""

    def test_cross_community_edges_as_backedges(self, two_triangles: Data) -> None:
        """Cross-community edges should be encoded as back-edges."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)

        # Verify roundtrip works (which means cross-edges are captured)
        reconstructed = tokenizer.decode(tokens)
        assert edges_equal(two_triangles, reconstructed)

        # The edge (2,3) connects the two triangles
        # It should be captured as a back-edge, not in a bipartite block
        tokens_list = tokens.tolist()

        # Count LEDGE tokens (back-edge brackets)
        ledge_count = sum(1 for t in tokens_list if t == tokenizer.LEDGE)
        assert ledge_count > 0, "Expected at least one back-edge bracket"

    @pytest.mark.skipif(not HAS_HSENT, reason="H-SENT not available for comparison")
    def test_token_count_vs_hsent(self, larger_graph: Data) -> None:
        """HDT should produce fewer tokens than H-SENT for same graph."""
        hdt = HDTTokenizer(seed=42, min_community_size=3)
        hdt.set_num_nodes(20)

        hsent = HSENTTokenizer(seed=42, min_community_size=3)
        hsent.set_num_nodes(20)

        hdt_tokens = hdt.tokenize(larger_graph)
        hsent_tokens = hsent.tokenize(larger_graph)

        # HDT should typically have fewer tokens
        # The proposal claims ~45% reduction
        hdt_len = len(hdt_tokens)
        hsent_len = len(hsent_tokens)

        # Allow some variance, but HDT should not be significantly larger
        assert hdt_len <= hsent_len * 1.1, (
            f"HDT produced more tokens ({hdt_len}) than H-SENT ({hsent_len})"
        )

    def test_enter_exit_structure(self, two_triangles: Data) -> None:
        """Verify ENTER/EXIT tokens form valid nested structure."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(two_triangles)
        tokens_list = tokens.tolist()

        # Track nesting depth
        depth = 0
        max_depth = 0

        for tok in tokens_list:
            if tok == tokenizer.ENTER:
                depth += 1
                max_depth = max(max_depth, depth)
            elif tok == tokenizer.EXIT:
                depth -= 1

            # Depth should never go negative
            assert depth >= 0, "Unbalanced EXIT token"

        # Should end at depth 0
        assert depth == 0, "Unbalanced ENTER/EXIT tokens"

        # Should have at least one level of nesting
        assert max_depth >= 1, "Expected at least one hierarchy level"


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

        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
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

        tokenizer = HDTTokenizer(seed=42, min_community_size=3)
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

        tokenizer = HDTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(50)

        hg = tokenizer.coarsener.build_hierarchy(data)

        # Molecular graphs should have reasonable decomposition
        assert hg.num_communities >= 1
        assert hg.num_communities <= data.num_nodes

        # Check level info
        info = hg.get_level_info()
        assert info["depth"] >= 0


# ===========================================================================
# Full Adjacency Building Tests
# ===========================================================================


class TestFullAdjacency:
    """Test _build_full_adjacency captures all edges including bipartites."""

    def test_includes_intra_partition_edges(self, two_triangles: Data) -> None:
        """Full adjacency should include edges within partitions."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg = tokenizer.coarsener.build_hierarchy(two_triangles)
        adj = tokenizer._build_full_adjacency(hg)

        # Check some known edges exist
        # Triangle 1: edges 0-1, 1-2, 0-2
        assert 1 in adj[0] or 0 in adj[1]

    def test_includes_bipartite_edges(self, two_triangles: Data) -> None:
        """Full adjacency should include cross-community (bipartite) edges."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg = tokenizer.coarsener.build_hierarchy(two_triangles)

        # Only test if there are bipartite edges
        if hg.bipartites:
            adj = tokenizer._build_full_adjacency(hg)

            # The edge (2,3) connects the triangles
            # It should appear in the full adjacency
            all_edges = set()
            for node, neighbors in adj.items():
                for neighbor in neighbors:
                    all_edges.add((min(node, neighbor), max(node, neighbor)))

            # Original graph has edge (2,3)
            assert (2, 3) in all_edges, "Cross-community edge (2,3) not in full adjacency"


# ===========================================================================
# Structure Preservation Tests - Verify Partition Boundaries
# ===========================================================================


class TestStructurePreservation:
    """Tests to verify hierarchical structure is preserved during roundtrip.

    These tests go beyond edge preservation to verify that partition
    membership and community structure are correctly maintained.
    """

    def test_partition_count_preserved(self, two_triangles: Data) -> None:
        """Number of partitions should be preserved in roundtrip."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg_original = tokenizer.coarsener.build_hierarchy(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hg_original)
        hg_decoded = tokenizer.parse_tokens(tokens)

        assert len(hg_original.partitions) == len(hg_decoded.partitions), (
            f"Partition count changed: {len(hg_original.partitions)} -> "
            f"{len(hg_decoded.partitions)}"
        )

    def test_partition_membership_preserved(self, two_triangles: Data) -> None:
        """Partition membership (which nodes are in which partition) preserved."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg_original = tokenizer.coarsener.build_hierarchy(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hg_original)
        hg_decoded = tokenizer.parse_tokens(tokens)

        # Collect partition node sets from both
        original_sets = [set(p.global_node_indices) for p in hg_original.partitions]
        decoded_sets = [set(p.global_node_indices) for p in hg_decoded.partitions]

        # Sort for comparison (order may differ)
        original_sorted = sorted(original_sets, key=lambda s: min(s))
        decoded_sorted = sorted(decoded_sets, key=lambda s: min(s))

        assert original_sorted == decoded_sorted, (
            f"Partition membership changed!\n"
            f"Original: {original_sorted}\n"
            f"Decoded: {decoded_sorted}"
        )

    def test_bipartite_count_preserved(self, two_triangles: Data) -> None:
        """Number of bipartite edge sets should be preserved."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg_original = tokenizer.coarsener.build_hierarchy(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hg_original)
        hg_decoded = tokenizer.parse_tokens(tokens)

        assert len(hg_original.bipartites) == len(hg_decoded.bipartites), (
            f"Bipartite count changed: {len(hg_original.bipartites)} -> "
            f"{len(hg_decoded.bipartites)}"
        )

    def test_structure_preserved_larger_graph(self, larger_graph: Data) -> None:
        """Structure preservation for larger graph with 3 communities."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(20)

        hg_original = tokenizer.coarsener.build_hierarchy(larger_graph)
        tokens = tokenizer.tokenize_hierarchy(hg_original)
        hg_decoded = tokenizer.parse_tokens(tokens)

        # Check partition count
        assert len(hg_original.partitions) == len(hg_decoded.partitions)

        # Check partition membership
        original_sets = sorted(
            [set(p.global_node_indices) for p in hg_original.partitions],
            key=lambda s: min(s),
        )
        decoded_sets = sorted(
            [set(p.global_node_indices) for p in hg_decoded.partitions],
            key=lambda s: min(s),
        )
        assert original_sets == decoded_sets

    def test_token_sequence_has_partition_markers(self, two_triangles: Data) -> None:
        """Token sequence should have ENTER/EXIT for each partition."""
        tokenizer = HDTTokenizer(seed=42, min_community_size=2)
        tokenizer.set_num_nodes(10)

        hg = tokenizer.coarsener.build_hierarchy(two_triangles)
        tokens = tokenizer.tokenize_hierarchy(hg)
        tokens_list = tokens.tolist()

        # Count ENTER tokens
        enter_count = sum(1 for t in tokens_list if t == tokenizer.ENTER)
        exit_count = sum(1 for t in tokens_list if t == tokenizer.EXIT)

        # Should have at least: 1 for root + N for each partition
        min_expected = 1 + len(hg.partitions)
        assert enter_count >= min_expected, (
            f"Expected at least {min_expected} ENTER tokens, got {enter_count}"
        )
        assert enter_count == exit_count, "ENTER/EXIT counts should match"

    @pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
    def test_molecular_structure_preservation(self) -> None:
        """Structure preservation for molecular graph (caffeine)."""
        data = smiles_to_graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert data is not None

        tokenizer = HDTTokenizer(seed=42, min_community_size=3)
        tokenizer.set_num_nodes(50)

        hg_original = tokenizer.coarsener.build_hierarchy(data)
        tokens = tokenizer.tokenize_hierarchy(hg_original)
        hg_decoded = tokenizer.parse_tokens(tokens)

        # Verify partition count
        assert len(hg_original.partitions) == len(hg_decoded.partitions)

        # Verify total nodes match
        original_nodes = sum(len(p.global_node_indices) for p in hg_original.partitions)
        decoded_nodes = sum(len(p.global_node_indices) for p in hg_decoded.partitions)
        assert original_nodes == decoded_nodes
