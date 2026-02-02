"""Tests for scaffold library and tier patterns."""

import pytest
from rdkit import Chem

from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary
from src.transfer_learning.scaffolds.tier_patterns import (
    ALL_SCAFFOLDS,
    SCAFFOLD_TIERS,
    TIER1_SCAFFOLDS,
    TIER2_SCAFFOLDS,
    TIER3_SCAFFOLDS,
)


class TestTierPatterns:
    """Tests for tier pattern definitions."""

    def test_tier1_scaffolds_exist(self) -> None:
        """Tier 1 scaffolds should be defined."""
        assert len(TIER1_SCAFFOLDS) > 0
        assert "benzene" in TIER1_SCAFFOLDS
        assert "pyridine" in TIER1_SCAFFOLDS

    def test_tier2_scaffolds_exist(self) -> None:
        """Tier 2 scaffolds should be defined."""
        assert len(TIER2_SCAFFOLDS) > 0
        assert "naphthalene" in TIER2_SCAFFOLDS
        assert "indole" in TIER2_SCAFFOLDS

    def test_tier3_scaffolds_exist(self) -> None:
        """Tier 3 scaffolds should be defined."""
        assert len(TIER3_SCAFFOLDS) > 0
        assert "carbazole" in TIER3_SCAFFOLDS
        assert "pyrene" in TIER3_SCAFFOLDS

    def test_all_scaffolds_have_smiles(self) -> None:
        """All scaffolds should have valid SMILES."""
        for name, info in ALL_SCAFFOLDS.items():
            assert "smiles" in info, f"Scaffold {name} missing SMILES"
            mol = Chem.MolFromSmiles(info["smiles"])
            assert mol is not None, f"Invalid SMILES for {name}: {info['smiles']}"

    def test_all_scaffolds_have_category(self) -> None:
        """All scaffolds should have a category."""
        for name, info in ALL_SCAFFOLDS.items():
            assert "category" in info, f"Scaffold {name} missing category"

    def test_scaffold_tiers_mapping(self) -> None:
        """SCAFFOLD_TIERS should map all scaffolds to correct tiers."""
        for name in TIER1_SCAFFOLDS:
            assert SCAFFOLD_TIERS[name] == 1

        for name in TIER2_SCAFFOLDS:
            assert SCAFFOLD_TIERS[name] == 2

        for name in TIER3_SCAFFOLDS:
            assert SCAFFOLD_TIERS[name] == 3

    def test_tier1_scaffolds_are_small(self) -> None:
        """Tier 1 scaffolds should be monocyclic (small)."""
        for name, info in TIER1_SCAFFOLDS.items():
            mol = Chem.MolFromSmiles(info["smiles"])
            assert mol.GetNumAtoms() <= 8, f"{name} has too many atoms for Tier 1"

    def test_tier3_scaffolds_are_large(self) -> None:
        """Tier 3 scaffolds should be polycyclic (larger)."""
        for name, info in TIER3_SCAFFOLDS.items():
            mol = Chem.MolFromSmiles(info["smiles"])
            assert mol.GetNumAtoms() >= 12, f"{name} has too few atoms for Tier 3"


class TestScaffold:
    """Tests for Scaffold dataclass."""

    def test_scaffold_creation(self) -> None:
        """Scaffold should be created with correct attributes."""
        scaffold = Scaffold(
            name="test",
            smiles="c1ccccc1",
            tier=1,
            category="aromatic_6",
        )
        assert scaffold.name == "test"
        assert scaffold.smiles == "c1ccccc1"
        assert scaffold.tier == 1
        assert scaffold.category == "aromatic_6"

    def test_scaffold_num_atoms_computed(self) -> None:
        """Scaffold num_atoms should be computed from SMILES."""
        scaffold = Scaffold(
            name="benzene",
            smiles="c1ccccc1",
            tier=1,
            category="aromatic_6",
        )
        assert scaffold.num_atoms == 6

    def test_scaffold_get_graph(self) -> None:
        """Scaffold should produce valid graph."""
        scaffold = Scaffold(
            name="benzene",
            smiles="c1ccccc1",
            tier=1,
            category="aromatic_6",
        )
        graph = scaffold.get_graph(labeled=True)
        assert graph is not None
        assert graph.num_nodes == 6
        assert graph.edge_index.size(1) > 0

    def test_scaffold_to_mol(self) -> None:
        """Scaffold should convert to RDKit Mol."""
        scaffold = Scaffold(
            name="benzene",
            smiles="c1ccccc1",
            tier=1,
            category="aromatic_6",
        )
        mol = scaffold.to_mol()
        assert mol is not None
        assert mol.GetNumAtoms() == 6


class TestScaffoldLibrary:
    """Tests for ScaffoldLibrary class."""

    @pytest.fixture
    def library(self) -> ScaffoldLibrary:
        """Create a scaffold library for testing."""
        return ScaffoldLibrary()

    def test_library_loads_default_scaffolds(self, library: ScaffoldLibrary) -> None:
        """Library should load all default scaffolds."""
        assert len(library) == len(ALL_SCAFFOLDS)

    def test_get_scaffold_by_name(self, library: ScaffoldLibrary) -> None:
        """Library should return scaffold by name."""
        scaffold = library.get_scaffold("benzene")
        assert scaffold.name == "benzene"
        assert scaffold.tier == 1

    def test_get_scaffold_raises_for_unknown(self, library: ScaffoldLibrary) -> None:
        """Library should raise KeyError for unknown scaffold."""
        with pytest.raises(KeyError):
            library.get_scaffold("nonexistent")

    def test_get_scaffolds_by_tier(self, library: ScaffoldLibrary) -> None:
        """Library should return scaffolds by tier."""
        tier1 = library.get_scaffolds_by_tier(1)
        assert len(tier1) == len(TIER1_SCAFFOLDS)
        for scaffold in tier1:
            assert scaffold.tier == 1

    def test_get_scaffolds_by_tier_invalid(self, library: ScaffoldLibrary) -> None:
        """Library should raise for invalid tier."""
        with pytest.raises(ValueError):
            library.get_scaffolds_by_tier(4)

    def test_get_scaffolds_by_category(self, library: ScaffoldLibrary) -> None:
        """Library should return scaffolds by category."""
        aromatic = library.get_scaffolds_by_category("aromatic_6")
        assert len(aromatic) > 0
        for scaffold in aromatic:
            assert scaffold.category == "aromatic_6"

    def test_from_smiles(self, library: ScaffoldLibrary) -> None:
        """Library should create scaffold from custom SMILES."""
        scaffold = library.from_smiles("CC(C)C", name="isobutane")
        assert scaffold.name == "isobutane"
        assert scaffold.num_atoms == 4
        assert scaffold.tier == 0  # Custom scaffolds get tier 0

    def test_from_smiles_invalid(self, library: ScaffoldLibrary) -> None:
        """Library should raise for invalid SMILES."""
        with pytest.raises(ValueError):
            library.from_smiles("invalid_smiles")

    def test_add_scaffold(self, library: ScaffoldLibrary) -> None:
        """Library should allow adding custom scaffolds."""
        custom = Scaffold(
            name="custom",
            smiles="CC",
            tier=0,
            category="custom",
        )
        library.add_scaffold(custom)
        assert "custom" in library

    def test_list_scaffolds(self, library: ScaffoldLibrary) -> None:
        """Library should list all scaffold names."""
        names = library.list_scaffolds()
        assert len(names) == len(library)
        assert "benzene" in names
        assert "naphthalene" in names

    def test_list_categories(self, library: ScaffoldLibrary) -> None:
        """Library should list unique categories."""
        categories = library.list_categories()
        assert "aromatic_6" in categories
        assert "fused_aromatic" in categories

    def test_contains(self, library: ScaffoldLibrary) -> None:
        """Library should support 'in' operator."""
        assert "benzene" in library
        assert "nonexistent" not in library

    def test_getitem(self, library: ScaffoldLibrary) -> None:
        """Library should support indexing syntax."""
        scaffold = library["benzene"]
        assert scaffold.name == "benzene"
