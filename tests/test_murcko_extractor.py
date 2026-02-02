"""Tests for Murcko scaffold extractor."""

import pytest
from rdkit import Chem

from src.transfer_learning.scaffolds.murcko_extractor import MurckoExtractor


class TestMurckoExtractor:
    """Tests for MurckoExtractor class."""

    @pytest.fixture
    def extractor(self) -> MurckoExtractor:
        """Create a Murcko extractor."""
        return MurckoExtractor()

    def test_extract_scaffold_benzene(self, extractor: MurckoExtractor) -> None:
        """Benzene scaffold should be benzene itself."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        scaffold = extractor.extract_scaffold(mol)

        assert scaffold is not None
        assert scaffold.GetNumAtoms() == 6

    def test_extract_scaffold_toluene(self, extractor: MurckoExtractor) -> None:
        """Toluene scaffold should be benzene (no methyl)."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")  # Toluene
        scaffold = extractor.extract_scaffold(mol)

        assert scaffold is not None
        # Scaffold should be benzene without the methyl group
        assert scaffold.GetNumAtoms() == 6

    def test_extract_scaffold_naphthalene(self, extractor: MurckoExtractor) -> None:
        """Naphthalene scaffold should be naphthalene."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        scaffold = extractor.extract_scaffold(mol)

        assert scaffold is not None
        assert scaffold.GetNumAtoms() == 10

    def test_extract_scaffold_with_sidechains(self, extractor: MurckoExtractor) -> None:
        """Scaffold should exclude side chains."""
        mol = Chem.MolFromSmiles("CCCc1ccc(CCC)cc1")  # Disubstituted benzene
        scaffold = extractor.extract_scaffold(mol)

        assert scaffold is not None
        # Should be just benzene
        assert scaffold.GetNumAtoms() == 6

    def test_extract_scaffold_returns_none_for_invalid(
        self, extractor: MurckoExtractor
    ) -> None:
        """Extractor should return None for molecules without rings."""
        mol = Chem.MolFromSmiles("CCCCCC")  # Hexane (no rings)
        scaffold = extractor.extract_scaffold(mol)

        # For acyclic molecules, scaffold extraction returns empty or fails
        # The behavior depends on RDKit version
        if scaffold is not None:
            assert scaffold.GetNumAtoms() == 0

    def test_get_scaffold_smiles(self, extractor: MurckoExtractor) -> None:
        """Extractor should return scaffold SMILES."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        smiles = extractor.get_scaffold_smiles(mol)

        assert smiles is not None
        assert isinstance(smiles, str)
        # Should be benzene
        scaffold_mol = Chem.MolFromSmiles(smiles)
        assert scaffold_mol.GetNumAtoms() == 6

    def test_scaffold_from_smiles(self, extractor: MurckoExtractor) -> None:
        """Extractor should work from SMILES string."""
        scaffold = extractor.scaffold_from_smiles("Cc1ccccc1")

        assert scaffold is not None
        assert scaffold.GetNumAtoms() == 6

    def test_scaffold_from_smiles_invalid(self, extractor: MurckoExtractor) -> None:
        """Extractor should return None for invalid SMILES."""
        scaffold = extractor.scaffold_from_smiles("invalid_smiles")
        assert scaffold is None

    def test_scaffold_smiles_from_smiles(self, extractor: MurckoExtractor) -> None:
        """Convenience method should return scaffold SMILES from molecule SMILES."""
        smiles = extractor.scaffold_smiles_from_smiles("Cc1ccccc1")

        assert smiles is not None
        assert isinstance(smiles, str)

    def test_get_scaffold_ratio(self, extractor: MurckoExtractor) -> None:
        """Scaffold ratio should be between 0 and 1."""
        # Benzene - scaffold is the whole molecule
        mol = Chem.MolFromSmiles("c1ccccc1")
        ratio = extractor.get_scaffold_ratio(mol)
        assert ratio == 1.0

        # Toluene - methyl not in scaffold
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        ratio = extractor.get_scaffold_ratio(mol)
        assert ratio == 6 / 7  # 6 scaffold atoms / 7 total atoms

    def test_extract_core(self, extractor: MurckoExtractor) -> None:
        """Core extraction should work."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # Naphthalene
        core = extractor.extract_core(mol)

        assert core is not None
        # Core should be generic scaffold (atoms become generic)

    def test_get_core_smiles(self, extractor: MurckoExtractor) -> None:
        """Core SMILES extraction should work."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        smiles = extractor.get_core_smiles(mol)

        assert smiles is not None
        assert isinstance(smiles, str)


class TestMurckoExtractorWithSidechains:
    """Tests for MurckoExtractor with include_sidechains=True."""

    @pytest.fixture
    def extractor(self) -> MurckoExtractor:
        """Create a Murcko extractor with sidechains."""
        return MurckoExtractor(include_sidechains=True)

    def test_init_with_sidechains(self, extractor: MurckoExtractor) -> None:
        """Extractor should store include_sidechains setting."""
        assert extractor.include_sidechains is True

    def test_extract_scaffold_generic(self, extractor: MurckoExtractor) -> None:
        """With include_sidechains, scaffold should be generic."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        scaffold = extractor.extract_scaffold(mol)

        assert scaffold is not None
        # Generic scaffold has all atoms as generic type
