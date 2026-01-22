"""Motif detection utilities for molecular graphs.

This module provides functions for detecting structural motifs (e.g., rings,
functional groups) in molecular graphs using SMARTS pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.tokenizers.motif.patterns import CLUSTERING_MOTIFS

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass(frozen=True)
class MotifInstance:
    """A detected motif instance in a molecule.

    Attributes:
        name: Name of the motif type (e.g., "benzene").
        atom_indices: Frozenset of atom indices belonging to this motif.
        pattern: SMARTS pattern used for detection.
    """

    name: str
    atom_indices: frozenset[int]
    pattern: str

    def __len__(self) -> int:
        """Return number of atoms in motif."""
        return len(self.atom_indices)

    def overlaps_with(self, other: MotifInstance) -> bool:
        """Check if this motif shares atoms with another."""
        return bool(self.atom_indices & other.atom_indices)


def detect_motifs_from_smiles(
    smiles: str,
    patterns: dict[str, str] | None = None,
) -> list[MotifInstance]:
    """Detect motif instances in a molecule from its SMILES.

    Args:
        smiles: SMILES string of the molecule.
        patterns: Dictionary mapping motif names to SMARTS patterns.
            Defaults to CLUSTERING_MOTIFS.

    Returns:
        List of MotifInstance objects for all detected motifs.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return []

    if patterns is None:
        patterns = CLUSTERING_MOTIFS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    motifs: list[MotifInstance] = []

    for name, smarts in patterns.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                motifs.append(
                    MotifInstance(
                        name=name,
                        atom_indices=frozenset(match),
                        pattern=smarts,
                    )
                )
        except Exception:
            # Skip invalid patterns
            continue

    return motifs


def detect_motifs_from_data(
    data: Data,
    patterns: dict[str, str] | None = None,
) -> list[MotifInstance]:
    """Detect motif instances from a PyG Data object.

    Attempts to use the 'smiles' attribute if available.
    Returns empty list if SMILES is not available.

    Args:
        data: PyTorch Geometric Data object.
        patterns: Dictionary mapping motif names to SMARTS patterns.

    Returns:
        List of MotifInstance objects, or empty list if detection fails.
    """
    smiles = getattr(data, "smiles", None)
    if smiles is None:
        return []

    return detect_motifs_from_smiles(smiles, patterns)
