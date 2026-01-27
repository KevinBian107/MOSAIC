"""Functional group detection utilities for HDTC tokenization.

This module provides classes and functions for detecting functional groups
in molecular graphs, including overlap resolution based on priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.tokenizers.motif.functional_patterns import (
    FUNCTIONAL_GROUP_PATTERNS,
    PATTERN_PRIORITY,
    RING_PATTERNS,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass(frozen=True)
class FunctionalGroupInstance:
    """A detected functional group instance in a molecule.

    Attributes:
        name: Name of the functional group (e.g., "hydroxyl", "benzene").
        pattern_type: Type of pattern ("ring", "multi_atom", "single_atom").
        atom_indices: Frozenset of atom indices belonging to this group.
        priority: Priority for overlap resolution (higher wins).
        pattern: SMARTS pattern used for detection.
    """

    name: str
    pattern_type: str
    atom_indices: frozenset[int]
    priority: int
    pattern: str

    def __len__(self) -> int:
        """Return number of atoms in functional group."""
        return len(self.atom_indices)

    def overlaps_with(self, other: FunctionalGroupInstance) -> bool:
        """Check if this group shares atoms with another."""
        return bool(self.atom_indices & other.atom_indices)


class FunctionalGroupDetector:
    """Detector for functional groups in molecular graphs.

    Detects functional groups using SMARTS patterns and resolves overlaps
    by priority (ring > multi_atom > single_atom) and then by size.

    Attributes:
        include_rings: Whether to detect ring structures.
        ring_patterns: Custom ring SMARTS patterns (or defaults).
        functional_patterns: Custom functional group patterns (or defaults).
    """

    def __init__(
        self,
        include_rings: bool = True,
        ring_patterns: dict[str, str] | None = None,
        functional_patterns: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the functional group detector.

        Args:
            include_rings: Whether to include ring detection.
            ring_patterns: Custom ring SMARTS patterns. Defaults to RING_PATTERNS.
            functional_patterns: Custom functional group patterns.
                Defaults to FUNCTIONAL_GROUP_PATTERNS.
        """
        self.include_rings = include_rings
        self.ring_patterns = (
            ring_patterns if ring_patterns is not None else RING_PATTERNS
        )
        self.functional_patterns = (
            functional_patterns
            if functional_patterns is not None
            else FUNCTIONAL_GROUP_PATTERNS
        )

    def detect(self, smiles: str) -> list[FunctionalGroupInstance]:
        """Detect functional groups in a molecule from its SMILES.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            List of FunctionalGroupInstance objects for all detected groups,
            with overlaps already resolved.
        """
        try:
            from rdkit import Chem
        except ImportError:
            return []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        groups: list[FunctionalGroupInstance] = []

        # Detect ring patterns
        if self.include_rings:
            for name, smarts in self.ring_patterns.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is None:
                        continue
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        groups.append(
                            FunctionalGroupInstance(
                                name=name,
                                pattern_type="ring",
                                atom_indices=frozenset(match),
                                priority=PATTERN_PRIORITY["ring"],
                                pattern=smarts,
                            )
                        )
                except Exception:
                    continue

        # Detect functional group patterns
        for name, (smarts, pattern_type) in self.functional_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    groups.append(
                        FunctionalGroupInstance(
                            name=name,
                            pattern_type=pattern_type,
                            atom_indices=frozenset(match),
                            priority=PATTERN_PRIORITY.get(pattern_type, 0),
                            pattern=smarts,
                        )
                    )
            except Exception:
                continue

        return self._resolve_overlaps(groups)

    def detect_from_data(self, data: Data) -> list[FunctionalGroupInstance]:
        """Detect functional groups from a PyG Data object.

        Attempts to use the 'smiles' attribute if available.
        Returns empty list if SMILES is not available.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            List of FunctionalGroupInstance objects, or empty list if detection fails.
        """
        smiles = getattr(data, "smiles", None)
        if smiles is None:
            return []
        return self.detect(smiles)

    def _resolve_overlaps(
        self, groups: list[FunctionalGroupInstance]
    ) -> list[FunctionalGroupInstance]:
        """Resolve overlapping functional groups.

        Resolution algorithm:
        1. Sort by priority (descending), then by size (descending)
        2. Greedily select non-overlapping groups
        3. Higher priority wins; larger groups win ties

        Args:
            groups: List of detected functional groups (may overlap).

        Returns:
            List of non-overlapping functional groups.
        """
        if not groups:
            return []

        # Sort by priority (desc), then by size (desc), then by name (for stability)
        sorted_groups = sorted(
            groups,
            key=lambda g: (g.priority, len(g), g.name),
            reverse=True,
        )

        selected: list[FunctionalGroupInstance] = []
        used_atoms: set[int] = set()

        for group in sorted_groups:
            # Check if this group overlaps with already selected atoms
            if not (group.atom_indices & used_atoms):
                selected.append(group)
                used_atoms.update(group.atom_indices)

        return selected


def detect_functional_groups(
    smiles: str,
    include_rings: bool = True,
    ring_patterns: dict[str, str] | None = None,
    functional_patterns: dict[str, tuple[str, str]] | None = None,
) -> list[FunctionalGroupInstance]:
    """Convenience function for detecting functional groups.

    Args:
        smiles: SMILES string of the molecule.
        include_rings: Whether to include ring detection.
        ring_patterns: Custom ring SMARTS patterns.
        functional_patterns: Custom functional group patterns.

    Returns:
        List of non-overlapping FunctionalGroupInstance objects.
    """
    detector = FunctionalGroupDetector(
        include_rings=include_rings,
        ring_patterns=ring_patterns,
        functional_patterns=functional_patterns,
    )
    return detector.detect(smiles)
