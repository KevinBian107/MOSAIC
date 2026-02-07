"""Tests for COCONUT data loader."""

import pytest
from rdkit import Chem

from src.data.coconut_loader import CoconutLoader


class TestCoconutLoader:
    """Tests for CoconutLoader class."""

    @pytest.fixture
    def loader(self) -> CoconutLoader:
        """Create a COCONUT loader with default settings."""
        return CoconutLoader(
            min_atoms=10,
            max_atoms=50,
            min_rings=2,
            min_scaffold_atoms=5,
        )

    def test_init_default_values(self) -> None:
        """Loader should initialize with default values."""
        loader = CoconutLoader()
        assert loader.min_atoms == 30
        assert loader.max_atoms == 100
        assert loader.min_rings == 4
        assert loader.min_scaffold_atoms == 15

    def test_init_custom_values(self) -> None:
        """Loader should accept custom values."""
        loader = CoconutLoader(
            min_atoms=20,
            max_atoms=80,
            min_rings=3,
            min_scaffold_atoms=10,
        )
        assert loader.min_atoms == 20
        assert loader.max_atoms == 80
        assert loader.min_rings == 3
        assert loader.min_scaffold_atoms == 10

    def test_filter_by_complexity_small_molecule(self) -> None:
        """Small molecules should fail complexity filter."""
        loader = CoconutLoader(min_atoms=30, min_rings=4)
        # Benzene is too small
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert not loader.filter_by_complexity(mol)

    def test_filter_by_complexity_large_molecule(self) -> None:
        """Very large molecules should fail max atoms filter."""
        # Anthracene has 14 atoms and 3 rings
        loader = CoconutLoader(
            min_atoms=10, max_atoms=15, min_rings=2, min_scaffold_atoms=5
        )
        mol = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")  # Anthracene
        assert loader.filter_by_complexity(mol)

        # Naphthacene (4 fused rings, 18 atoms) should fail max_atoms=15
        loader2 = CoconutLoader(
            min_atoms=10, max_atoms=15, min_rings=2, min_scaffold_atoms=5
        )
        mol2 = Chem.MolFromSmiles("c1ccc2cc3cc4ccccc4cc3cc2c1")  # Naphthacene
        assert mol2 is not None  # Verify valid SMILES
        assert not loader2.filter_by_complexity(mol2)

    def test_filter_by_complexity_ring_count(self) -> None:
        """Molecules with few rings should fail."""
        loader = CoconutLoader(min_atoms=5, max_atoms=100, min_rings=3)
        # Benzene has 1 ring
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert not loader.filter_by_complexity(mol)

        # Naphthalene has 2 rings
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        assert not loader.filter_by_complexity(mol)

    def test_filter_accepts_complex_molecule(self) -> None:
        """Complex molecules with enough rings should pass."""
        # Relaxed requirements for testing
        loader = CoconutLoader(
            min_atoms=5,
            max_atoms=100,
            min_rings=2,
            min_scaffold_atoms=5,
        )
        # Naphthalene should pass
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        assert loader.filter_by_complexity(mol)

    def test_get_complexity_metrics(self, loader: CoconutLoader) -> None:
        """Loader should calculate complexity metrics."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # Naphthalene
        metrics = loader.get_complexity_metrics(mol)

        assert metrics["num_atoms"] == 10
        assert metrics["num_rings"] == 2
        assert metrics["num_aromatic_rings"] == 2
        assert "molecular_weight" in metrics
        assert "scaffold_num_atoms" in metrics

    def test_load_molecules_file_not_found(self) -> None:
        """Loader should raise FileNotFoundError for missing file."""
        loader = CoconutLoader(data_file="nonexistent_file.smi")
        with pytest.raises(FileNotFoundError):
            loader.load_molecules()

    def test_load_molecules_from_file(self, tmp_path) -> None:
        """Loader should load molecules from file."""
        # Create a test SMILES file with complex molecules
        smiles_file = tmp_path / "test.smi"
        smiles_file.write_text(
            "c1ccc2cc3ccccc3cc2c1\n"  # Anthracene (14 atoms, 3 rings)
            "c1cc2ccc3cccc4ccc(c1)c2c34\n"  # Pyrene (16 atoms, 4 rings)
        )

        loader = CoconutLoader(
            min_atoms=10,
            max_atoms=20,
            min_rings=3,
            min_scaffold_atoms=5,
            data_file=str(smiles_file),
        )
        molecules = loader.load_molecules()
        assert len(molecules) == 2

    def test_load_molecules_with_limit(self, tmp_path) -> None:
        """Loader should respect n_samples limit."""
        smiles_file = tmp_path / "test.smi"
        smiles_file.write_text(
            "c1ccc2cc3ccccc3cc2c1\n"  # Anthracene
            "c1cc2ccc3cccc4ccc(c1)c2c34\n"  # Pyrene
            "c1cc2ccc3cc4ccccc4cc3c2c1\n"  # Naphthacene
        )

        loader = CoconutLoader(
            min_atoms=10,
            max_atoms=30,
            min_rings=3,
            min_scaffold_atoms=5,
            data_file=str(smiles_file),
        )
        molecules = loader.load_molecules(n_samples=2)
        assert len(molecules) == 2

    def test_load_smiles(self, tmp_path) -> None:
        """Loader should return SMILES strings."""
        smiles_file = tmp_path / "test.smi"
        smiles_file.write_text("c1ccc2cc3ccccc3cc2c1\n")

        loader = CoconutLoader(
            min_atoms=10,
            max_atoms=20,
            min_rings=3,
            min_scaffold_atoms=5,
            data_file=str(smiles_file),
        )
        smiles_list = loader.load_smiles()
        assert len(smiles_list) == 1
        assert isinstance(smiles_list[0], str)

    def test_load_skips_invalid_smiles(self, tmp_path) -> None:
        """Loader should skip invalid SMILES."""
        smiles_file = tmp_path / "test.smi"
        smiles_file.write_text(
            "c1ccc2cc3ccccc3cc2c1\n"  # Valid
            "invalid_smiles\n"  # Invalid
            "c1cc2ccc3cccc4ccc(c1)c2c34\n"  # Valid
        )

        loader = CoconutLoader(
            min_atoms=10,
            max_atoms=20,
            min_rings=3,
            min_scaffold_atoms=5,
            data_file=str(smiles_file),
        )
        molecules = loader.load_molecules()
        assert len(molecules) == 2

    def test_load_skips_comments(self, tmp_path) -> None:
        """Loader should skip comment lines."""
        smiles_file = tmp_path / "test.smi"
        smiles_file.write_text(
            "# This is a comment\n"
            "c1ccc2cc3ccccc3cc2c1\n"
            "\n"  # Empty line
            "c1cc2ccc3cccc4ccc(c1)c2c34\n"
        )

        loader = CoconutLoader(
            min_atoms=10,
            max_atoms=20,
            min_rings=3,
            min_scaffold_atoms=5,
            data_file=str(smiles_file),
        )
        molecules = loader.load_molecules()
        assert len(molecules) == 2
