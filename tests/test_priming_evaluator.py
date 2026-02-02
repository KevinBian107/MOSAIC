"""Tests for priming evaluator."""

import pytest
from rdkit import Chem

from src.data.molecular import smiles_to_graph
from src.transfer_learning.datasets.complex_molecule_dataset import (
    ComplexMoleculeSample,
)
from src.transfer_learning.evaluation.priming_evaluator import PrimingEvaluator


class TestPrimingEvaluator:
    """Tests for PrimingEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> PrimingEvaluator:
        """Create a priming evaluator."""
        return PrimingEvaluator()

    @pytest.fixture
    def benzene_sample(self) -> ComplexMoleculeSample:
        """Create a benzene sample for testing."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        scaffold = Chem.MolFromSmiles("c1ccccc1")
        return ComplexMoleculeSample(
            molecule=mol,
            molecule_smiles="c1ccccc1",
            scaffold=scaffold,
            scaffold_smiles="c1ccccc1",
        )

    @pytest.fixture
    def toluene_sample(self) -> ComplexMoleculeSample:
        """Create a toluene sample for testing."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        scaffold = Chem.MolFromSmiles("c1ccccc1")  # Scaffold is benzene
        return ComplexMoleculeSample(
            molecule=mol,
            molecule_smiles="Cc1ccccc1",
            scaffold=scaffold,
            scaffold_smiles="c1ccccc1",
        )

    def test_init_default_values(self) -> None:
        """Evaluator should initialize with default values."""
        evaluator = PrimingEvaluator()
        assert evaluator.fingerprint_radius == 2
        assert evaluator.fingerprint_nbits == 2048

    def test_init_custom_values(self) -> None:
        """Evaluator should accept custom values."""
        evaluator = PrimingEvaluator(fingerprint_radius=3, fingerprint_nbits=1024)
        assert evaluator.fingerprint_radius == 3
        assert evaluator.fingerprint_nbits == 1024

    def test_compare_to_target_identical(self, evaluator: PrimingEvaluator) -> None:
        """Identical molecules should have similarity 1.0."""
        result = evaluator.compare_to_target("c1ccccc1", "c1ccccc1")

        assert result["valid"] is True
        assert result["tanimoto_similarity"] == 1.0
        assert result["exact_match"] is True

    def test_compare_to_target_similar(self, evaluator: PrimingEvaluator) -> None:
        """Similar molecules should have non-zero similarity."""
        result = evaluator.compare_to_target("c1ccccc1", "Cc1ccccc1")

        assert result["valid"] is True
        # Morgan fingerprints may give lower similarity for small differences
        assert result["tanimoto_similarity"] > 0.0
        assert result["exact_match"] is False

    def test_compare_to_target_invalid_smiles(
        self, evaluator: PrimingEvaluator
    ) -> None:
        """Invalid SMILES should return valid=False."""
        result = evaluator.compare_to_target("invalid", "c1ccccc1")
        assert result["valid"] is False

        result = evaluator.compare_to_target("c1ccccc1", "invalid")
        assert result["valid"] is False

    def test_evaluate_sample_empty_graphs(
        self, evaluator: PrimingEvaluator, benzene_sample: ComplexMoleculeSample
    ) -> None:
        """Evaluating with no graphs should return zero metrics."""
        results = evaluator.evaluate_sample(benzene_sample, [])

        assert results["n_generated"] == 0
        assert results["n_valid"] == 0
        assert results["valid_rate"] == 0

    def test_evaluate_sample_with_valid_graphs(
        self, evaluator: PrimingEvaluator, benzene_sample: ComplexMoleculeSample
    ) -> None:
        """Evaluating with valid graphs should compute metrics."""
        # Create graphs that decode to valid SMILES
        graph1 = smiles_to_graph("c1ccccc1", labeled=True)  # Benzene
        graph2 = smiles_to_graph("Cc1ccccc1", labeled=True)  # Toluene

        results = evaluator.evaluate_sample(benzene_sample, [graph1, graph2])

        assert results["n_generated"] == 2
        assert results["n_valid"] == 2
        assert results["valid_rate"] == 1.0
        assert len(results["tanimoto_similarities"]) == 2
        assert results["tanimoto_mean"] > 0

    def test_evaluate_sample_scaffold_preservation(
        self, evaluator: PrimingEvaluator, benzene_sample: ComplexMoleculeSample
    ) -> None:
        """Evaluating should check scaffold preservation."""
        # Benzene contains benzene scaffold
        graph1 = smiles_to_graph("c1ccccc1", labeled=True)
        # Toluene also contains benzene scaffold
        graph2 = smiles_to_graph("Cc1ccccc1", labeled=True)
        # Cyclohexane does NOT contain benzene scaffold (aromatic vs aliphatic)
        graph3 = smiles_to_graph("C1CCCCC1", labeled=True)

        results = evaluator.evaluate_sample(benzene_sample, [graph1, graph2, graph3])

        # At least the benzene should be detected as containing scaffold
        assert any(results["scaffold_preservation"])

    def test_contains_scaffold_true(self, evaluator: PrimingEvaluator) -> None:
        """Should detect when scaffold is present."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")  # Toluene
        scaffold = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        assert evaluator._contains_scaffold(mol, scaffold) is True

    def test_contains_scaffold_false(self, evaluator: PrimingEvaluator) -> None:
        """Should detect when scaffold is not present."""
        mol = Chem.MolFromSmiles("CCCCCC")  # Hexane
        scaffold = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        assert evaluator._contains_scaffold(mol, scaffold) is False

    def test_aggregate_results(self, evaluator: PrimingEvaluator) -> None:
        """Should aggregate results across samples."""
        results_list = [
            {
                "scaffold_preservation_rate": 0.8,
                "tanimoto_mean": 0.7,
                "tanimoto_max": 0.9,
                "valid_rate": 1.0,
                "atom_count_ratio_mean": 1.1,
            },
            {
                "scaffold_preservation_rate": 0.6,
                "tanimoto_mean": 0.5,
                "tanimoto_max": 0.8,
                "valid_rate": 0.8,
                "atom_count_ratio_mean": 0.9,
            },
        ]

        aggregated = evaluator._aggregate_results(results_list)

        assert "scaffold_preservation_mean" in aggregated
        assert "tanimoto_mean" in aggregated
        assert "valid_rate_mean" in aggregated
        assert aggregated["scaffold_preservation_mean"] == pytest.approx(0.7)
        assert aggregated["tanimoto_mean"] == pytest.approx(0.6)
        assert aggregated["valid_rate_mean"] == pytest.approx(0.9)


class TestPrimingEvaluatorFingerprints:
    """Tests for fingerprint functionality."""

    @pytest.fixture
    def evaluator(self) -> PrimingEvaluator:
        """Create a priming evaluator."""
        return PrimingEvaluator()

    def test_get_fingerprint(self, evaluator: PrimingEvaluator) -> None:
        """Should generate fingerprint for valid molecule."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        fp = evaluator._get_fingerprint(mol)

        assert fp is not None
        # Fingerprint should have expected number of bits
        assert len(fp) == evaluator.fingerprint_nbits

    def test_fingerprint_similarity(self, evaluator: PrimingEvaluator) -> None:
        """Similar molecules should have similar fingerprints."""
        from rdkit import DataStructs

        mol1 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        mol2 = Chem.MolFromSmiles("Cc1ccccc1")  # Toluene
        mol3 = Chem.MolFromSmiles("CCCCCC")  # Hexane

        fp1 = evaluator._get_fingerprint(mol1)
        fp2 = evaluator._get_fingerprint(mol2)
        fp3 = evaluator._get_fingerprint(mol3)

        sim_benzene_toluene = DataStructs.TanimotoSimilarity(fp1, fp2)
        sim_benzene_hexane = DataStructs.TanimotoSimilarity(fp1, fp3)

        # Benzene-toluene should be more similar than benzene-hexane
        assert sim_benzene_toluene > sim_benzene_hexane


class TestVisualization:
    """Tests for visualization functions."""

    def test_visualize_priming_comparison_returns_image(self) -> None:
        """visualize_priming_comparison should return PIL Image when no path."""
        from src.transfer_learning.evaluation.visualization import (
            visualize_priming_comparison,
        )

        result = visualize_priming_comparison(
            scaffold_smiles="c1ccccc1",
            generated_smiles=["Cc1ccccc1", "CCc1ccccc1"],
            target_smiles="CCCc1ccccc1",
        )

        assert result is not None
        # Check it's a PIL Image
        assert hasattr(result, "save")
        assert hasattr(result, "size")

    def test_visualize_priming_comparison_saves_file(self, tmp_path) -> None:
        """visualize_priming_comparison should save to file when path given."""
        from src.transfer_learning.evaluation.visualization import (
            visualize_priming_comparison,
        )

        output_path = tmp_path / "test.png"

        result = visualize_priming_comparison(
            scaffold_smiles="c1ccccc1",
            generated_smiles=["Cc1ccccc1"],
            target_smiles="CCc1ccccc1",
            output_path=output_path,
        )

        assert result is None  # Returns None when saving
        assert output_path.exists()

    def test_visualize_evaluation_results(self, tmp_path) -> None:
        """visualize_evaluation_results should create multiple images."""
        from src.transfer_learning.evaluation.visualization import (
            visualize_evaluation_results,
        )

        results = [
            {
                "scaffold_smiles": "c1ccccc1",
                "target_smiles": "Cc1ccccc1",
                "valid_smiles": ["Cc1ccccc1", "CCc1ccccc1"],
                "tanimoto_mean": 0.8,
                "scaffold_preservation_rate": 1.0,
            },
            {
                "scaffold_smiles": "c1ccc2ccccc2c1",
                "target_smiles": "Cc1ccc2ccccc2c1",
                "valid_smiles": ["Cc1ccc2ccccc2c1"],
                "tanimoto_mean": 0.9,
                "scaffold_preservation_rate": 1.0,
            },
        ]

        saved_paths = visualize_evaluation_results(
            results,
            tmp_path,
            max_samples=2,
        )

        assert len(saved_paths) == 2
        for path in saved_paths:
            assert path.exists()

    def test_create_summary_grid(self, tmp_path) -> None:
        """create_summary_grid should create a summary image."""
        from src.transfer_learning.evaluation.visualization import create_summary_grid

        results = [
            {
                "scaffold_smiles": "c1ccccc1",
                "target_smiles": "Cc1ccccc1",
                "valid_smiles": ["Cc1ccccc1"],
                "tanimoto_max": 0.8,
            },
        ]

        output_path = tmp_path / "summary.png"
        create_summary_grid(results, output_path, n_samples=1)

        assert output_path.exists()
