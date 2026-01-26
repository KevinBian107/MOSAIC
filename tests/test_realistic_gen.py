"""Tests for realistic generation analysis module.

Tests substitution pattern analysis, functional group detection,
and distribution comparison metrics.
"""

import pytest
from collections import Counter

from src.realistic_gen import (
    analyze_benzene_substitution,
    analyze_functional_groups,
    compare_distributions,
)
from src.realistic_gen.analysis import (
    classify_disubstitution,
    get_benzene_substitution_count,
)


# ===========================================================================
# Substitution Count Tests
# ===========================================================================


class TestSubstitutionCount:
    """Tests for benzene substitution counting."""

    def test_benzene_unsubstituted(self) -> None:
        """Pure benzene has 0 substituents."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        assert get_benzene_substitution_count(mol) == 0

    def test_toluene_mono(self) -> None:
        """Toluene (methylbenzene) has 1 substituent."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1ccccc1")
        assert get_benzene_substitution_count(mol) == 1

    def test_phenol_mono(self) -> None:
        """Phenol has 1 substituent."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Oc1ccccc1")
        assert get_benzene_substitution_count(mol) == 1

    def test_xylene_di(self) -> None:
        """Xylene (dimethylbenzene) has 2 substituents."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1ccc(C)cc1")  # para-xylene
        assert get_benzene_substitution_count(mol) == 2

    def test_trimethylbenzene_tri(self) -> None:
        """Trimethylbenzene has 3 substituents."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1cc(C)cc(C)c1")
        assert get_benzene_substitution_count(mol) == 3

    def test_no_benzene_returns_zero(self) -> None:
        """Molecule without benzene returns 0."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCCCCC")  # hexane
        assert get_benzene_substitution_count(mol) == 0

    def test_none_mol_returns_zero(self) -> None:
        """None molecule returns 0."""
        assert get_benzene_substitution_count(None) == 0


# ===========================================================================
# Di-substitution Pattern Tests
# ===========================================================================


class TestDisubstitutionPattern:
    """Tests for ortho/meta/para classification."""

    def test_ortho_xylene(self) -> None:
        """ortho-xylene should be classified as ortho."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1ccccc1C")  # ortho
        assert classify_disubstitution(mol) == "ortho"

    def test_meta_xylene(self) -> None:
        """meta-xylene should be classified as meta."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1cccc(C)c1")  # meta
        assert classify_disubstitution(mol) == "meta"

    def test_para_xylene(self) -> None:
        """para-xylene should be classified as para."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1ccc(C)cc1")  # para
        assert classify_disubstitution(mol) == "para"

    def test_mono_returns_none(self) -> None:
        """Mono-substituted benzene returns None."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1ccccc1")  # toluene
        assert classify_disubstitution(mol) is None

    def test_tri_returns_none(self) -> None:
        """Tri-substituted benzene returns None."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("Cc1cc(C)cc(C)c1")
        assert classify_disubstitution(mol) is None


# ===========================================================================
# Full Analysis Tests
# ===========================================================================


class TestAnalyzeBenzeneSubstitution:
    """Tests for full substitution analysis."""

    def test_analyze_mixed_smiles(self) -> None:
        """Test analysis of mixed substitution patterns."""
        smiles_list = [
            "c1ccccc1",  # unsubstituted
            "Cc1ccccc1",  # mono
            "Cc1ccc(C)cc1",  # di (para)
            "Cc1ccccc1C",  # di (ortho)
            "Cc1cccc(C)c1",  # di (meta)
            "Cc1cc(C)cc(C)c1",  # tri
        ]

        results = analyze_benzene_substitution(smiles_list)

        assert results["substitution_count"]["unsubstituted"] == 1
        assert results["substitution_count"]["mono"] == 1
        assert results["substitution_count"]["di"] == 3
        assert results["substitution_count"]["tri"] == 1

        assert results["disubstitution_pattern"]["para"] == 1
        assert results["disubstitution_pattern"]["ortho"] == 1
        assert results["disubstitution_pattern"]["meta"] == 1

    def test_analyze_empty_list(self) -> None:
        """Empty list returns empty counters."""
        results = analyze_benzene_substitution([])
        assert len(results["substitution_count"]) == 0
        assert len(results["disubstitution_pattern"]) == 0

    def test_analyze_invalid_smiles_ignored(self) -> None:
        """Invalid SMILES are ignored."""
        smiles_list = ["invalid", "Cc1ccccc1", None]
        results = analyze_benzene_substitution(smiles_list)
        assert results["substitution_count"]["mono"] == 1


# ===========================================================================
# Functional Group Tests
# ===========================================================================


class TestAnalyzeFunctionalGroups:
    """Tests for functional group analysis."""

    def test_detect_hydroxyl(self) -> None:
        """Detect hydroxyl group on benzene."""
        smiles_list = ["Oc1ccccc1"]  # phenol
        results = analyze_functional_groups(smiles_list)
        assert results["Hydroxyl (-OH)"] >= 1

    def test_detect_amino(self) -> None:
        """Detect amino group on benzene."""
        smiles_list = ["Nc1ccccc1"]  # aniline
        results = analyze_functional_groups(smiles_list)
        assert results["Amino (-NH2)"] >= 1

    def test_detect_halogen(self) -> None:
        """Detect halogen on benzene."""
        smiles_list = ["Clc1ccccc1"]  # chlorobenzene
        results = analyze_functional_groups(smiles_list)
        assert results["Chloro (-Cl)"] >= 1

    def test_detect_methyl(self) -> None:
        """Detect methyl group on benzene."""
        smiles_list = ["Cc1ccccc1"]  # toluene
        results = analyze_functional_groups(smiles_list)
        assert results["Methyl (-CH3)"] >= 1

    def test_empty_list(self) -> None:
        """Empty list returns empty counter."""
        results = analyze_functional_groups([])
        assert len(results) == 0


# ===========================================================================
# Distribution Comparison Tests
# ===========================================================================


class TestCompareDistributions:
    """Tests for distribution comparison metrics."""

    def test_identical_distributions(self) -> None:
        """Identical distributions have 0 distance."""
        counts = Counter({"a": 10, "b": 20, "c": 30})
        metrics = compare_distributions(counts, counts)
        assert metrics["total_variation"] < 1e-6
        assert metrics["kl_divergence"] < 1e-6

    def test_different_distributions(self) -> None:
        """Different distributions have positive distance."""
        train = Counter({"a": 100, "b": 0})
        gen = Counter({"a": 0, "b": 100})
        metrics = compare_distributions(train, gen)
        assert metrics["total_variation"] > 0.9  # Should be close to 1.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping distributions."""
        train = Counter({"a": 50, "b": 50})
        gen = Counter({"a": 70, "b": 30})
        metrics = compare_distributions(train, gen)
        assert 0 < metrics["total_variation"] < 0.5

    def test_empty_counters(self) -> None:
        """Empty counters return 0 distance."""
        metrics = compare_distributions(Counter(), Counter())
        assert metrics["total_variation"] == 0.0
        assert metrics["kl_divergence"] == 0.0

    def test_different_keys(self) -> None:
        """Handle counters with different keys."""
        train = Counter({"a": 50, "b": 50})
        gen = Counter({"b": 50, "c": 50})
        metrics = compare_distributions(train, gen)
        assert metrics["total_variation"] > 0
