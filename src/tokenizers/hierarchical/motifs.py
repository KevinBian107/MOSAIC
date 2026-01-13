"""Motif detection utilities for graph partitioning.

This module provides motif detection for use in motif-aware graph coarsening.
It focuses on ring systems and structural motifs that should be kept together
during graph partitioning.

The primary detection method uses SMARTS patterns via RDKit when SMILES
is available on the Data object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch_geometric.data import Data

# Ring-focused motif patterns for clustering
# These patterns focus on structural motifs that should stay together
CLUSTERING_MOTIFS: dict[str, str] = {
    # Aromatic 6-membered rings
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "pyrazine": "c1cnccn1",
    # Aromatic 5-membered rings
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1cnc[nH]1",
    "oxazole": "c1cocn1",
    "thiazole": "c1cscn1",
    # Fused ring systems
    "naphthalene": "c1ccc2ccccc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",
    # Saturated rings
    "cyclopropane": "C1CC1",
    "cyclobutane": "C1CCC1",
    "cyclopentane": "C1CCCC1",
    "cyclohexane": "C1CCCCC1",
    # Partially unsaturated
    "cyclohexene": "C1=CCCCC1",
    "cyclopentene": "C1=CCCC1",
}


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


def compute_motif_affinity_matrix(
    num_nodes: int,
    motifs: list[MotifInstance],
    normalize_by_size: bool = False,
) -> np.ndarray:
    """Compute motif co-membership affinity matrix.

    Creates a matrix M where M[i,j] = number of motifs containing
    both atoms i and j. This can be added to the adjacency matrix
    to encourage clustering algorithms to keep motif atoms together.

    Args:
        num_nodes: Total number of nodes in the graph.
        motifs: List of detected motif instances.
        normalize_by_size: If True, normalize each motif's contribution
            by 1/motif_size to prevent large motifs from dominating.

    Returns:
        Symmetric numpy array of shape (num_nodes, num_nodes).
    """
    M = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    for motif in motifs:
        atoms = np.array(sorted(motif.atom_indices), dtype=np.int64)

        # Skip motifs with out-of-range indices
        if len(atoms) == 0 or atoms.max() >= num_nodes:
            continue

        weight = 1.0 / len(atoms) if normalize_by_size else 1.0

        # Add weight to all pairs within the motif
        M[np.ix_(atoms, atoms)] += weight

    # Ensure symmetry (should already be symmetric)
    M = (M + M.T) / 2

    return M


def compute_motif_cohesion(
    communities: list[set[int]],
    motifs: list[MotifInstance],
) -> float:
    """Compute the fraction of motifs kept intact by a partition.

    A motif is "intact" if all its atoms are in the same community.

    Args:
        communities: List of sets, each containing node indices for a community.
        motifs: List of detected motif instances.

    Returns:
        Fraction of motifs that are intact (0.0 to 1.0).
        Returns 1.0 if no motifs are present.
    """
    if not motifs:
        return 1.0

    intact_count = 0

    for motif in motifs:
        motif_atoms = motif.atom_indices
        for comm in communities:
            if motif_atoms <= comm:  # Motif is subset of community
                intact_count += 1
                break

    return intact_count / len(motifs)


def get_motif_summary(motifs: list[MotifInstance]) -> dict[str, int]:
    """Get a summary of motif counts by type.

    Args:
        motifs: List of detected motif instances.

    Returns:
        Dictionary mapping motif names to counts.
    """
    summary: dict[str, int] = {}
    for motif in motifs:
        summary[motif.name] = summary.get(motif.name, 0) + 1
    return summary
