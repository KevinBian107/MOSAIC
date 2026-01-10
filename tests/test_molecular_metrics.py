"""Tests for molecular evaluation metrics."""

import pytest

from src.evaluation.molecular_metrics import (
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_snn,
    compute_fragment_similarity,
    compute_scaffold_similarity,
    get_brics_fragments,
    MolecularMetrics,
)
from src.evaluation.motif_distribution import (
    get_functional_group_counts,
    get_motif_counts,
    get_ring_system_info,
    MotifDistributionMetric,
    MOLECULAR_MOTIFS,
)


# Test SMILES
VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccccc1O"]
INVALID_SMILES = ["not_valid", "xyz123"]
MIXED_SMILES = ["CCO", "not_valid", "c1ccccc1"]


class TestValidity:
    """Tests for validity metric."""

    def test_all_valid(self) -> None:
        """Test validity is 1.0 for all valid SMILES."""
        result = compute_validity(VALID_SMILES)
        assert result == 1.0

    def test_all_invalid(self) -> None:
        """Test validity is 0.0 for all invalid SMILES."""
        result = compute_validity(INVALID_SMILES)
        assert result == 0.0

    def test_mixed(self) -> None:
        """Test validity for mixed valid/invalid."""
        result = compute_validity(MIXED_SMILES)
        assert 0 < result < 1

    def test_empty_list(self) -> None:
        """Test validity for empty list."""
        result = compute_validity([])
        assert result == 0.0


class TestUniqueness:
    """Tests for uniqueness metric."""

    def test_all_unique(self) -> None:
        """Test uniqueness is 1.0 for all unique SMILES."""
        result = compute_uniqueness(VALID_SMILES)
        assert result == 1.0

    def test_duplicates(self) -> None:
        """Test uniqueness < 1.0 for duplicates."""
        smiles = ["CCO", "CCO", "c1ccccc1"]
        result = compute_uniqueness(smiles)
        assert result < 1.0

    def test_canonicalization(self) -> None:
        """Test that equivalent SMILES are detected as duplicates."""
        # Different representations of ethanol
        smiles = ["CCO", "OCC", "C(C)O"]
        result = compute_uniqueness(smiles)
        # All should canonicalize to same SMILES
        assert result < 1.0


class TestNovelty:
    """Tests for novelty metric."""

    def test_all_novel(self) -> None:
        """Test novelty is 1.0 when none in reference."""
        generated = ["CCO", "c1ccccc1"]
        reference = ["CCCO", "c1ccc2ccccc2c1"]
        result = compute_novelty(generated, reference)
        assert result == 1.0

    def test_none_novel(self) -> None:
        """Test novelty is 0.0 when all in reference."""
        smiles = ["CCO", "c1ccccc1"]
        result = compute_novelty(smiles, smiles)
        assert result == 0.0

    def test_partial_novelty(self) -> None:
        """Test partial novelty."""
        generated = ["CCO", "CCCO"]
        reference = ["CCO"]
        result = compute_novelty(generated, reference)
        assert 0 < result < 1


class TestSNN:
    """Tests for nearest neighbor similarity."""

    def test_identical_sets(self) -> None:
        """Test SNN is 1.0 for identical sets."""
        result = compute_snn(VALID_SMILES, VALID_SMILES)
        assert result == 1.0

    def test_similar_molecules(self) -> None:
        """Test SNN > 0 for similar molecules."""
        generated = ["CCO"]
        reference = ["CCCO"]  # Similar to ethanol
        result = compute_snn(generated, reference)
        assert result > 0


class TestFragmentSimilarity:
    """Tests for BRICS fragment similarity."""

    def test_identical_sets(self) -> None:
        """Test fragment similarity is 1.0 for identical sets."""
        result = compute_fragment_similarity(VALID_SMILES, VALID_SMILES)
        assert result == 1.0

    def test_get_brics_fragments(self) -> None:
        """Test BRICS fragment extraction."""
        frags = get_brics_fragments("c1ccccc1O")  # Phenol
        assert len(frags) > 0

    def test_invalid_smiles_empty_fragments(self) -> None:
        """Test invalid SMILES returns empty set."""
        frags = get_brics_fragments("not_valid")
        assert len(frags) == 0


class TestScaffoldSimilarity:
    """Tests for Bemis-Murcko scaffold similarity."""

    def test_identical_sets(self) -> None:
        """Test scaffold similarity is 1.0 for identical sets."""
        result = compute_scaffold_similarity(VALID_SMILES, VALID_SMILES)
        assert result == 1.0


class TestMolecularMetrics:
    """Tests for MolecularMetrics class."""

    def test_init(self) -> None:
        """Test initialization."""
        metrics = MolecularMetrics(VALID_SMILES)
        assert len(metrics.reference_smiles) == len(VALID_SMILES)

    def test_compute_returns_all_metrics(self) -> None:
        """Test compute returns all expected metrics."""
        metrics = MolecularMetrics(VALID_SMILES)
        result = metrics.compute(VALID_SMILES)

        assert "validity" in result
        assert "uniqueness" in result
        assert "novelty" in result
        assert "snn" in result
        assert "frag_similarity" in result
        assert "scaff_similarity" in result
        assert "internal_diversity" in result

    def test_callable_interface(self) -> None:
        """Test __call__ works."""
        metrics = MolecularMetrics(VALID_SMILES)
        result = metrics(VALID_SMILES)
        assert "validity" in result


class TestMotifCounts:
    """Tests for motif counting functions."""

    def test_get_motif_counts_benzene(self) -> None:
        """Test benzene detection in phenol."""
        counts = get_motif_counts("c1ccccc1O")
        assert "benzene" in counts
        assert counts["benzene"] >= 1

    def test_get_motif_counts_hydroxyl(self) -> None:
        """Test hydroxyl detection in ethanol."""
        counts = get_motif_counts("CCO")
        assert "hydroxyl" in counts

    def test_get_functional_group_counts(self) -> None:
        """Test functional group counting."""
        counts = get_functional_group_counts("c1ccccc1O")
        # Should detect aromatic and hydroxyl
        assert len(counts) > 0

    def test_get_ring_system_info(self) -> None:
        """Test ring system information."""
        info = get_ring_system_info("c1ccccc1")
        assert info["num_rings"] == 1
        assert info["num_aromatic_rings"] == 1


class TestMotifDistributionMetric:
    """Tests for MotifDistributionMetric class."""

    def test_init(self) -> None:
        """Test initialization."""
        metric = MotifDistributionMetric(VALID_SMILES)
        assert len(metric.reference_smiles) == len(VALID_SMILES)

    def test_compute_returns_mmd_values(self) -> None:
        """Test compute returns MMD values."""
        metric = MotifDistributionMetric(VALID_SMILES)
        result = metric.compute(VALID_SMILES)

        # Check that at least some metrics are present
        assert any(k.startswith("motif_") for k in result.keys())

    def test_compute_self_reference_low_mmd(self) -> None:
        """Test MMD is low when comparing to self."""
        metric = MotifDistributionMetric(VALID_SMILES)
        result = metric.compute(VALID_SMILES)

        # MMD should be very low (close to 0) for self-comparison
        for value in result.values():
            assert value < 0.5

    def test_get_motif_summary(self) -> None:
        """Test motif summary generation."""
        metric = MotifDistributionMetric(VALID_SMILES)
        summary = metric.get_motif_summary(VALID_SMILES)

        assert "functional_groups" in summary
        assert "smarts_motifs" in summary
        assert "ring_systems" in summary
        assert "brics_fragments" in summary
