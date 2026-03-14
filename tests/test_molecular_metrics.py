"""Tests for molecular evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.molecular_metrics import (
    MolecularMetrics,
    compute_fragment_similarity,
    compute_novelty,
    compute_scaffold_similarity,
    compute_snn,
    compute_uniqueness,
    compute_validity,
    get_brics_fragments,
)
from src.evaluation.motif_distribution import (
    MOLECULAR_MOTIFS,
    MotifCooccurrenceMetric,
    MotifDistributionMetric,
    MotifHistogramMetric,
    compute_cooccurrence_matrix,
    compute_motif_histogram,
    get_functional_group_counts,
    get_motif_counts,
    get_ring_system_info,
    kl_divergence,
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


class TestMotifHistogramHelpers:
    """Tests for motif histogram helper functions."""

    def test_compute_motif_histogram_benzene(self) -> None:
        """Test histogram computation for benzene motif."""
        # Molecules: benzene (1), naphthalene (2 fused rings count as 1 benzene match)
        smiles = ["c1ccccc1", "CCO", "c1ccccc1O"]
        hist = compute_motif_histogram(smiles, "benzene", max_count=5)

        assert len(hist) == 6  # 0, 1, 2, 3, 4, 5
        assert hist.sum() == pytest.approx(1.0)  # Normalized
        # CCO has 0 benzenes, others have >= 1
        assert hist[0] > 0

    def test_compute_motif_histogram_empty(self) -> None:
        """Test histogram for empty list."""
        hist = compute_motif_histogram([], "benzene", max_count=5)
        assert hist.sum() == 0

    def test_kl_divergence_identical(self) -> None:
        """Test KL divergence is 0 for identical distributions."""
        p = np.array([0.25, 0.5, 0.25])
        result = kl_divergence(p, p)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_different(self) -> None:
        """Test KL divergence is positive for different distributions."""
        p = np.array([0.9, 0.1, 0.0])
        q = np.array([0.1, 0.1, 0.8])
        result = kl_divergence(p, q)
        assert result > 0


class TestMotifHistogramMetric:
    """Tests for MotifHistogramMetric class."""

    def test_init_precomputes_histograms(self) -> None:
        """Test initialization precomputes reference histograms."""
        metric = MotifHistogramMetric(VALID_SMILES)
        assert len(metric._ref_histograms) == len(MOLECULAR_MOTIFS)

    def test_compute_returns_distances(self) -> None:
        """Test compute returns per-motif and aggregate distances."""
        metric = MotifHistogramMetric(VALID_SMILES)
        result = metric.compute(VALID_SMILES)

        assert "motif_hist_mean" in result
        assert "motif_hist_max" in result
        # Should have per-motif distances
        assert any(k.startswith("motif_hist_") and k not in ["motif_hist_mean", "motif_hist_max"]
                   for k in result.keys())

    def test_self_comparison_low_distance(self) -> None:
        """Test distance is low when comparing to self."""
        metric = MotifHistogramMetric(VALID_SMILES)
        result = metric.compute(VALID_SMILES)

        # KL divergence should be ~0 for identical distributions
        assert result["motif_hist_mean"] < 0.1

    def test_different_distributions_higher_distance(self) -> None:
        """Test distance is higher for different distributions."""
        # Reference with benzene rings
        ref = ["c1ccccc1", "c1ccccc1O", "c1ccc2ccccc2c1"]
        # Generated without benzene rings
        gen = ["CCO", "CCCO", "CCCCO"]

        metric = MotifHistogramMetric(ref)
        result = metric.compute(gen)

        # Benzene histogram should show significant difference
        assert result["motif_hist_benzene"] > 0.1

    def test_wasserstein_distance_option(self) -> None:
        """Test Wasserstein distance function option."""
        metric = MotifHistogramMetric(VALID_SMILES, distance_fn="wasserstein")
        result = metric.compute(VALID_SMILES)

        assert "motif_hist_mean" in result
        assert result["motif_hist_mean"] >= 0

    def test_callable_interface(self) -> None:
        """Test __call__ works."""
        metric = MotifHistogramMetric(VALID_SMILES)
        result = metric(VALID_SMILES)
        assert "motif_hist_mean" in result

    def test_invalid_smiles_handling(self) -> None:
        """Test handling of invalid SMILES."""
        metric = MotifHistogramMetric(VALID_SMILES)
        result = metric.compute(["INVALID", "", "not_valid"])

        assert result["motif_hist_mean"] == float("inf")


class TestMotifCooccurrenceHelpers:
    """Tests for motif co-occurrence helper functions."""

    def test_compute_cooccurrence_matrix_shape(self) -> None:
        """Test co-occurrence matrix has correct shape."""
        motif_names = ["benzene", "hydroxyl", "carbonyl"]
        matrix = compute_cooccurrence_matrix(VALID_SMILES, motif_names)

        assert matrix.shape == (3, 3)

    def test_compute_cooccurrence_matrix_diagonal(self) -> None:
        """Test diagonal is 1.0 for present motifs."""
        # Phenol has benzene, so P(benzene|benzene) = 1.0
        smiles = ["c1ccccc1O"] * 10
        motif_names = ["benzene", "hydroxyl"]
        matrix = compute_cooccurrence_matrix(smiles, motif_names)

        # Diagonal should be 1.0 for motifs that are present
        assert matrix[0, 0] == 1.0  # P(benzene|benzene)
        assert matrix[1, 1] == 1.0  # P(hydroxyl|hydroxyl)

    def test_compute_cooccurrence_phenol(self) -> None:
        """Test co-occurrence for phenol (benzene + hydroxyl)."""
        smiles = ["c1ccccc1O"] * 10  # All phenol
        motif_names = ["benzene", "hydroxyl"]
        matrix = compute_cooccurrence_matrix(smiles, motif_names)

        # P(hydroxyl|benzene) should be 1.0
        assert matrix[0, 1] == 1.0
        # P(benzene|hydroxyl) should be 1.0
        assert matrix[1, 0] == 1.0

    def test_compute_cooccurrence_empty(self) -> None:
        """Test co-occurrence for empty list."""
        matrix = compute_cooccurrence_matrix([], ["benzene", "hydroxyl"])
        assert matrix.shape == (2, 2)
        assert matrix.sum() == 0


class TestMotifCooccurrenceMetric:
    """Tests for MotifCooccurrenceMetric class."""

    def test_init_computes_reference(self) -> None:
        """Test initialization computes reference matrix."""
        metric = MotifCooccurrenceMetric(VALID_SMILES)
        assert metric._ref_cooccur.shape[0] == len(metric.motif_names)

    def test_compute_returns_distances(self) -> None:
        """Test compute returns distance metrics."""
        metric = MotifCooccurrenceMetric(VALID_SMILES)
        result = metric.compute(VALID_SMILES)

        assert "motif_cooccur_frobenius" in result
        assert "motif_cooccur_mean_abs" in result

    def test_self_comparison_low_distance(self) -> None:
        """Test distance is low when comparing to self."""
        metric = MotifCooccurrenceMetric(VALID_SMILES)
        result = metric.compute(VALID_SMILES)

        assert result["motif_cooccur_frobenius"] < 0.1
        assert result["motif_cooccur_mean_abs"] < 0.1

    def test_different_patterns_higher_distance(self) -> None:
        """Test distance is higher for different co-occurrence patterns."""
        # Reference: phenols (benzene + hydroxyl co-occur)
        ref = ["c1ccccc1O", "c1ccc(O)cc1", "c1ccccc1O"]
        # Generated: separate motifs (no co-occurrence)
        gen = ["c1ccccc1", "CCO", "CCCO"]

        metric = MotifCooccurrenceMetric(ref, motif_names=["benzene", "hydroxyl"])
        result = metric.compute(gen)

        # Should show difference in co-occurrence
        assert result["motif_cooccur_frobenius"] > 0

    def test_callable_interface(self) -> None:
        """Test __call__ works."""
        metric = MotifCooccurrenceMetric(VALID_SMILES)
        result = metric(VALID_SMILES)
        assert "motif_cooccur_frobenius" in result

    def test_get_cooccurrence_summary(self) -> None:
        """Test co-occurrence summary generation."""
        metric = MotifCooccurrenceMetric(VALID_SMILES)
        summary = metric.get_cooccurrence_summary(VALID_SMILES, top_k=5)

        assert "top_pairs" in summary
        assert isinstance(summary["top_pairs"], list)

    def test_invalid_smiles_handling(self) -> None:
        """Test handling of invalid SMILES."""
        metric = MotifCooccurrenceMetric(VALID_SMILES)
        result = metric.compute(["INVALID", "", "not_valid"])

        assert result["motif_cooccur_frobenius"] == float("inf")
