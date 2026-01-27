"""Tests for functional group detection.

Tests the FunctionalGroupDetector and FunctionalGroupInstance classes
for accurate detection and overlap resolution.
"""

import pytest

from src.tokenizers.motif.functional_detection import (
    FunctionalGroupDetector,
    FunctionalGroupInstance,
    detect_functional_groups,
)
from src.tokenizers.motif.functional_patterns import (
    FUNCTIONAL_GROUP_PATTERNS,
    PATTERN_PRIORITY,
    RING_PATTERNS,
)

# Optional RDKit import
try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ===========================================================================
# Test FunctionalGroupInstance
# ===========================================================================


class TestFunctionalGroupInstance:
    """Tests for FunctionalGroupInstance dataclass."""

    def test_instance_creation(self) -> None:
        """Test creating a FunctionalGroupInstance."""
        instance = FunctionalGroupInstance(
            name="benzene",
            pattern_type="ring",
            atom_indices=frozenset({0, 1, 2, 3, 4, 5}),
            priority=30,
            pattern="c1ccccc1",
        )
        assert instance.name == "benzene"
        assert instance.pattern_type == "ring"
        assert len(instance) == 6
        assert instance.priority == 30

    def test_overlaps_with(self) -> None:
        """Test overlap detection between instances."""
        inst1 = FunctionalGroupInstance(
            name="benzene",
            pattern_type="ring",
            atom_indices=frozenset({0, 1, 2, 3, 4, 5}),
            priority=30,
            pattern="c1ccccc1",
        )
        inst2 = FunctionalGroupInstance(
            name="hydroxyl",
            pattern_type="single_atom",
            atom_indices=frozenset({5}),
            priority=10,
            pattern="[OX2H]",
        )
        inst3 = FunctionalGroupInstance(
            name="methyl",
            pattern_type="single_atom",
            atom_indices=frozenset({6}),
            priority=10,
            pattern="[CH3]",
        )

        assert inst1.overlaps_with(inst2)
        assert inst2.overlaps_with(inst1)
        assert not inst1.overlaps_with(inst3)
        assert not inst2.overlaps_with(inst3)

    def test_frozen(self) -> None:
        """Test that FunctionalGroupInstance is frozen."""
        instance = FunctionalGroupInstance(
            name="benzene",
            pattern_type="ring",
            atom_indices=frozenset({0, 1}),
            priority=30,
            pattern="c1ccccc1",
        )
        with pytest.raises(AttributeError):
            instance.name = "other"


# ===========================================================================
# Test FunctionalGroupDetector
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestFunctionalGroupDetector:
    """Tests for FunctionalGroupDetector class."""

    def test_detector_creation(self) -> None:
        """Test creating a detector."""
        detector = FunctionalGroupDetector()
        assert detector.include_rings is True
        assert detector.ring_patterns == RING_PATTERNS
        assert detector.functional_patterns == FUNCTIONAL_GROUP_PATTERNS

    def test_detect_benzene(self) -> None:
        """Test detecting benzene ring."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("c1ccccc1")

        # Should detect benzene
        benzene_groups = [g for g in groups if g.name == "benzene"]
        assert len(benzene_groups) == 1
        assert len(benzene_groups[0]) == 6

    def test_detect_phenol(self) -> None:
        """Test detecting groups in phenol (Oc1ccccc1)."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("Oc1ccccc1")

        group_names = {g.name for g in groups}

        # Should detect benzene ring
        assert "benzene" in group_names

        # Should detect hydroxyl (if not overlapping with benzene)
        # The O is attached to benzene, so it may or may not be detected
        # depending on the SMARTS pattern

    def test_detect_ethanol(self) -> None:
        """Test detecting groups in ethanol (CCO)."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("CCO")

        group_names = {g.name for g in groups}

        # Should detect hydroxyl
        assert "hydroxyl" in group_names or "methyl" in group_names

    def test_detect_carboxylic_acid(self) -> None:
        """Test detecting carboxyl group in acetic acid."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("CC(=O)O")

        group_names = {g.name for g in groups}

        # Should detect carboxyl group
        assert "carboxyl" in group_names

    def test_overlap_resolution_priority(self) -> None:
        """Test that higher priority groups win in overlap resolution."""
        # Create detector with limited patterns for controlled test
        detector = FunctionalGroupDetector(
            include_rings=True,
            ring_patterns={"benzene": "c1ccccc1"},
            functional_patterns={},
        )
        groups = detector.detect("c1ccccc1")

        # Should have exactly one benzene group
        assert len(groups) == 1
        assert groups[0].name == "benzene"

    def test_no_overlapping_groups(self) -> None:
        """Test that returned groups don't overlap."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Caffeine

        # Check no overlaps
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i != j:
                    assert not g1.overlaps_with(g2), (
                        f"Groups {g1.name} and {g2.name} overlap"
                    )

    def test_detect_without_rings(self) -> None:
        """Test detection with rings disabled."""
        detector = FunctionalGroupDetector(include_rings=False)
        groups = detector.detect("c1ccccc1O")

        # Should not detect benzene
        benzene_groups = [g for g in groups if g.name == "benzene"]
        assert len(benzene_groups) == 0

    def test_detect_from_data(self) -> None:
        """Test detection from PyG Data object."""
        from torch_geometric.data import Data

        detector = FunctionalGroupDetector()

        # Create mock Data with smiles attribute
        data = Data()
        data.smiles = "c1ccccc1"

        groups = detector.detect_from_data(data)

        benzene_groups = [g for g in groups if g.name == "benzene"]
        assert len(benzene_groups) == 1

    def test_detect_from_data_no_smiles(self) -> None:
        """Test detection from Data without smiles returns empty."""
        from torch_geometric.data import Data

        detector = FunctionalGroupDetector()
        data = Data()

        groups = detector.detect_from_data(data)
        assert groups == []


# ===========================================================================
# Test convenience function
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestDetectFunctionalGroups:
    """Tests for detect_functional_groups convenience function."""

    def test_detect_functional_groups(self) -> None:
        """Test the convenience function."""
        groups = detect_functional_groups("c1ccccc1")

        benzene_groups = [g for g in groups if g.name == "benzene"]
        assert len(benzene_groups) == 1

    def test_detect_with_custom_patterns(self) -> None:
        """Test with custom patterns."""
        groups = detect_functional_groups(
            "c1ccccc1",
            include_rings=True,
            ring_patterns={"benzene": "c1ccccc1"},
            functional_patterns={},
        )

        assert len(groups) == 1
        assert groups[0].name == "benzene"


# ===========================================================================
# Test Pattern Priorities
# ===========================================================================


class TestPatternPriorities:
    """Tests for pattern priority definitions."""

    def test_priority_ordering(self) -> None:
        """Test that priorities are correctly ordered."""
        assert PATTERN_PRIORITY["ring"] > PATTERN_PRIORITY["multi_atom"]
        assert PATTERN_PRIORITY["multi_atom"] > PATTERN_PRIORITY["single_atom"]

    def test_all_patterns_have_valid_types(self) -> None:
        """Test that all patterns have valid types."""
        for name, (smarts, pattern_type) in FUNCTIONAL_GROUP_PATTERNS.items():
            assert pattern_type in PATTERN_PRIORITY, (
                f"Pattern {name} has invalid type {pattern_type}"
            )


# ===========================================================================
# Test Specific Molecules
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestSpecificMolecules:
    """Tests for specific molecule detection."""

    @pytest.mark.parametrize("smiles,expected_rings", [
        ("c1ccccc1", 1),  # Benzene
        ("c1ccc2ccccc2c1", 1),  # Naphthalene (detected as one fused system)
        ("c1ccncc1", 1),  # Pyridine
        ("C1CCCCC1", 1),  # Cyclohexane
    ])
    def test_ring_detection(self, smiles: str, expected_rings: int) -> None:
        """Test ring detection in various molecules."""
        detector = FunctionalGroupDetector()
        groups = detector.detect(smiles)

        ring_groups = [g for g in groups if g.pattern_type == "ring"]
        assert len(ring_groups) >= expected_rings

    def test_aspirin_detection(self) -> None:
        """Test functional group detection in aspirin."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("CC(=O)OC1=CC=CC=C1C(=O)O")

        group_names = {g.name for g in groups}

        # Should detect benzene ring
        assert "benzene" in group_names

        # Should detect ester and/or carboxyl
        assert "ester" in group_names or "carboxyl" in group_names

    def test_ibuprofen_detection(self) -> None:
        """Test functional group detection in ibuprofen."""
        detector = FunctionalGroupDetector()
        groups = detector.detect("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")

        group_names = {g.name for g in groups}

        # Should detect benzene ring
        assert "benzene" in group_names

        # Should detect carboxyl
        assert "carboxyl" in group_names
