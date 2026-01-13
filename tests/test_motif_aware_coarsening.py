"""Tests for motif-aware coarsening.

Tests cover:
- Motif detection from SMILES
- Motif affinity matrix computation
- MotifAwareCoarsening partitioning
- Motif cohesion metrics
- Integration with HSENTTokenizer
- Roundtrip property preservation
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers.hierarchical import (
    CLUSTERING_MOTIFS,
    HSENTTokenizer,
    MotifAwareCoarsening,
    MotifInstance,
    SpectralCoarsening,
    compute_motif_affinity_matrix,
    compute_motif_cohesion,
    detect_motifs_from_data,
    detect_motifs_from_smiles,
    get_motif_summary,
)

# Optional RDKit import
try:
    from rdkit import Chem

    from src.data.molecular import smiles_to_graph

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ===========================================================================
# Test Fixtures
# ===========================================================================


def make_data_with_smiles(smiles: str) -> Data | None:
    """Create a Data object with SMILES attribute."""
    if not HAS_RDKIT:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())
    data.smiles = smiles
    return data


def edges_equal(g1: Data, g2: Data) -> bool:
    """Check if two graphs have the same edges (ignoring order)."""

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


# ===========================================================================
# Test Motif Detection
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMotifDetection:
    """Tests for motif detection from SMILES."""

    def test_detect_benzene(self) -> None:
        """Detect benzene ring in phenol."""
        motifs = detect_motifs_from_smiles("Oc1ccccc1")

        benzene_motifs = [m for m in motifs if m.name == "benzene"]
        assert len(benzene_motifs) == 1
        assert len(benzene_motifs[0].atom_indices) == 6

    def test_detect_naphthalene(self) -> None:
        """Detect fused rings in naphthalene."""
        motifs = detect_motifs_from_smiles("c1ccc2ccccc2c1")

        # Should find naphthalene pattern
        naphthalene_motifs = [m for m in motifs if m.name == "naphthalene"]
        assert len(naphthalene_motifs) >= 1

        # Should also find benzene patterns
        benzene_motifs = [m for m in motifs if m.name == "benzene"]
        assert len(benzene_motifs) >= 2

    def test_detect_pyridine(self) -> None:
        """Detect pyridine ring."""
        motifs = detect_motifs_from_smiles("c1ccncc1")

        pyridine_motifs = [m for m in motifs if m.name == "pyridine"]
        assert len(pyridine_motifs) == 1
        assert len(pyridine_motifs[0].atom_indices) == 6

    def test_no_motifs_in_alkane(self) -> None:
        """No ring motifs in linear alkane."""
        motifs = detect_motifs_from_smiles("CCCCCC")

        # Should not find any ring motifs
        ring_motifs = [
            m for m in motifs if m.name in ["benzene", "cyclohexane", "cyclopentane"]
        ]
        assert len(ring_motifs) == 0

    def test_detect_from_data(self) -> None:
        """Detect motifs from Data object with smiles attribute."""
        data = make_data_with_smiles("c1ccccc1")
        assert data is not None

        motifs = detect_motifs_from_data(data)
        benzene_motifs = [m for m in motifs if m.name == "benzene"]
        assert len(benzene_motifs) == 1

    def test_detect_from_data_no_smiles(self) -> None:
        """Return empty list if Data has no smiles attribute."""
        data = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            num_nodes=2,
        )
        # No smiles attribute

        motifs = detect_motifs_from_data(data)
        assert motifs == []

    def test_motif_instance_overlap(self) -> None:
        """Test MotifInstance overlap detection."""
        m1 = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        m2 = MotifInstance("benzene", frozenset({4, 5, 6, 7, 8, 9}), "c1ccccc1")
        m3 = MotifInstance("benzene", frozenset({10, 11, 12, 13, 14, 15}), "c1ccccc1")

        assert m1.overlaps_with(m2)  # Share atoms 4, 5
        assert not m1.overlaps_with(m3)  # No shared atoms

    def test_get_motif_summary(self) -> None:
        """Test motif summary generation."""
        motifs = detect_motifs_from_smiles("c1ccc2ccccc2c1")  # Naphthalene
        summary = get_motif_summary(motifs)

        assert "benzene" in summary
        assert summary["benzene"] >= 2


# ===========================================================================
# Test Motif Affinity Matrix
# ===========================================================================


class TestMotifAffinityMatrix:
    """Tests for motif affinity matrix computation."""

    def test_single_motif_creates_clique(self) -> None:
        """Single motif should create a clique in affinity matrix."""
        motif = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        M = compute_motif_affinity_matrix(10, [motif])

        # All pairs within motif should have affinity 1
        for i in range(6):
            for j in range(6):
                assert M[i, j] == 1.0

        # Pairs outside motif should have affinity 0
        assert M[0, 6] == 0.0
        assert M[6, 7] == 0.0

    def test_overlapping_motifs_additive(self) -> None:
        """Overlapping motifs should have additive affinity."""
        m1 = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        m2 = MotifInstance("benzene", frozenset({4, 5, 6, 7, 8, 9}), "c1ccccc1")
        M = compute_motif_affinity_matrix(10, [m1, m2])

        # Atoms 4,5 are in both motifs
        assert M[4, 5] == 2.0
        assert M[4, 4] == 2.0
        assert M[5, 5] == 2.0

        # Other pairs in single motif
        assert M[0, 1] == 1.0
        assert M[6, 7] == 1.0

    def test_normalized_by_size(self) -> None:
        """Test normalization by motif size."""
        motif = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        M = compute_motif_affinity_matrix(10, [motif], normalize_by_size=True)

        # Each pair should have weight 1/6
        assert np.isclose(M[0, 1], 1.0 / 6.0)

    def test_empty_motifs(self) -> None:
        """Empty motif list should return zero matrix."""
        M = compute_motif_affinity_matrix(10, [])
        assert np.allclose(M, 0.0)

    def test_symmetry(self) -> None:
        """Affinity matrix should be symmetric."""
        motif = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        M = compute_motif_affinity_matrix(10, [motif])

        assert np.allclose(M, M.T)


# ===========================================================================
# Test Motif Cohesion
# ===========================================================================


class TestMotifCohesion:
    """Tests for motif cohesion metric."""

    def test_all_intact(self) -> None:
        """All motifs in single partition = perfect cohesion."""
        motif = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        communities = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}]

        cohesion = compute_motif_cohesion(communities, [motif])
        assert cohesion == 1.0

    def test_motif_split(self) -> None:
        """Split motif should reduce cohesion."""
        motif = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        communities = [{0, 1, 2}, {3, 4, 5, 6, 7, 8, 9}]

        cohesion = compute_motif_cohesion(communities, [motif])
        assert cohesion == 0.0

    def test_partial_cohesion(self) -> None:
        """Test partial cohesion with multiple motifs."""
        m1 = MotifInstance("benzene", frozenset({0, 1, 2, 3, 4, 5}), "c1ccccc1")
        m2 = MotifInstance("benzene", frozenset({6, 7, 8, 9, 10, 11}), "c1ccccc1")

        # m1 split, m2 intact
        communities = [{0, 1, 2, 6, 7, 8, 9, 10, 11}, {3, 4, 5}]

        cohesion = compute_motif_cohesion(communities, [m1, m2])
        assert cohesion == 0.5  # 1 of 2 motifs intact

    def test_no_motifs(self) -> None:
        """No motifs should return perfect cohesion."""
        cohesion = compute_motif_cohesion([{0, 1, 2}], [])
        assert cohesion == 1.0


# ===========================================================================
# Test MotifAwareCoarsening
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMotifAwareCoarsening:
    """Tests for MotifAwareCoarsening class."""

    def test_alpha_zero_equals_spectral(self) -> None:
        """With alpha=0, should behave like SpectralCoarsening."""
        data = make_data_with_smiles("c1ccc(-c2ccccc2)cc1")  # Biphenyl
        assert data is not None

        # Same seed for determinism
        spectral = SpectralCoarsening(seed=42, min_community_size=2)
        motif_aware = MotifAwareCoarsening(alpha=0.0, seed=42, min_community_size=2)

        communities_spectral = spectral.partition(data)
        communities_motif = motif_aware.partition(data)

        # Should produce same partitions
        assert len(communities_spectral) == len(communities_motif)

    def test_detects_motifs(self) -> None:
        """Should detect and cache motifs."""
        data = make_data_with_smiles("c1ccccc1")
        assert data is not None

        coarsener = MotifAwareCoarsening(alpha=1.0, seed=42)
        coarsener.partition(data)

        assert coarsener.cached_motifs is not None
        assert len(coarsener.cached_motifs) > 0

    def test_cohesion_metric(self) -> None:
        """Should compute cohesion metric after partition."""
        data = make_data_with_smiles("c1ccccc1")
        assert data is not None

        coarsener = MotifAwareCoarsening(alpha=1.0, seed=42, min_community_size=2)
        communities = coarsener.partition(data)

        cohesion = coarsener.get_motif_cohesion(communities)
        assert 0.0 <= cohesion <= 1.0

    def test_high_alpha_improves_cohesion(self) -> None:
        """High alpha should generally improve motif cohesion vs low alpha."""
        # Biphenyl: two benzene rings connected
        data = make_data_with_smiles("c1ccc(-c2ccccc2)cc1")
        assert data is not None

        # Low alpha coarsener
        coarsener_low = MotifAwareCoarsening(
            alpha=0.1, seed=42, min_community_size=2
        )
        communities_low = coarsener_low.partition(data)
        cohesion_low = coarsener_low.get_motif_cohesion(communities_low)

        # High alpha coarsener
        coarsener_high = MotifAwareCoarsening(
            alpha=5.0, seed=42, min_community_size=2
        )
        communities_high = coarsener_high.partition(data)
        cohesion_high = coarsener_high.get_motif_cohesion(communities_high)

        # High alpha should have equal or better cohesion
        # (not strictly better because both might achieve perfect cohesion)
        assert cohesion_high >= cohesion_low, (
            f"High alpha cohesion ({cohesion_high}) < low alpha ({cohesion_low})"
        )

    def test_no_smiles_fallback(self) -> None:
        """Should fall back gracefully when no SMILES available."""
        # Create data without smiles attribute
        data = Data(
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
            ),
            num_nodes=3,
        )

        coarsener = MotifAwareCoarsening(alpha=1.0, seed=42, min_community_size=2)
        communities = coarsener.partition(data)

        # Should still produce valid partition
        assert len(communities) >= 1
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        assert all_nodes == {0, 1, 2}

        # No motifs cached
        assert coarsener.cached_motifs == []


# ===========================================================================
# Test HSENTTokenizer Integration
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestHSENTTokenizerMotifAware:
    """Tests for HSENTTokenizer with motif_aware flag."""

    def test_motif_aware_flag(self) -> None:
        """Tokenizer should use MotifAwareCoarsening when flag is set."""
        tokenizer = HSENTTokenizer(motif_aware=True, seed=42)

        assert isinstance(tokenizer.coarsener, MotifAwareCoarsening)
        assert tokenizer.motif_aware is True

    def test_default_spectral(self) -> None:
        """Tokenizer should use SpectralCoarsening by default."""
        tokenizer = HSENTTokenizer(seed=42)

        assert isinstance(tokenizer.coarsener, SpectralCoarsening)
        assert not isinstance(tokenizer.coarsener, MotifAwareCoarsening)

    def test_alpha_parameter(self) -> None:
        """Should pass alpha parameter to coarsener."""
        tokenizer = HSENTTokenizer(motif_aware=True, motif_alpha=2.5, seed=42)

        assert isinstance(tokenizer.coarsener, MotifAwareCoarsening)
        assert tokenizer.coarsener.alpha == 2.5

    def test_roundtrip_with_motif_aware(self) -> None:
        """Roundtrip should still work with motif-aware coarsening."""
        data = make_data_with_smiles("Oc1ccccc1")  # Phenol
        assert data is not None

        tokenizer = HSENTTokenizer(
            motif_aware=True, motif_alpha=1.0, seed=42, min_community_size=2
        )
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(data, reconstructed)


# ===========================================================================
# Test Roundtrip Preservation (Critical Tests)
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMotifAwareRoundtrip:
    """Critical roundtrip tests - must preserve graph structure."""

    TEST_MOLECULES = [
        ("benzene", "c1ccccc1"),
        ("phenol", "Oc1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("biphenyl", "c1ccc(-c2ccccc2)cc1"),
        ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ]

    @pytest.mark.parametrize("name,smiles", TEST_MOLECULES)
    def test_roundtrip_motif_aware(self, name: str, smiles: str) -> None:
        """Roundtrip must preserve graph for motif-aware tokenization."""
        data = make_data_with_smiles(smiles)
        assert data is not None, f"Failed to parse {name}"

        tokenizer = HSENTTokenizer(
            motif_aware=True, motif_alpha=1.0, seed=42, min_community_size=2
        )
        tokenizer.set_num_nodes(max(50, data.num_nodes + 10))

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(data, reconstructed), f"Roundtrip failed for {name}"

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_roundtrip_various_alpha(self, alpha: float) -> None:
        """Roundtrip should work for various alpha values."""
        data = make_data_with_smiles("c1ccc2ccccc2c1")  # Naphthalene
        assert data is not None

        tokenizer = HSENTTokenizer(
            motif_aware=True, motif_alpha=alpha, seed=42, min_community_size=2
        )
        tokenizer.set_num_nodes(20)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert edges_equal(data, reconstructed), f"Roundtrip failed for alpha={alpha}"
