"""Motif affinity utilities for graph coarsening.

This module provides functions for computing motif co-membership affinity
matrices and evaluating motif preservation in graph partitions.
"""

from __future__ import annotations

import numpy as np

from src.tokenizers.motif.detection import MotifInstance


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

    # Zero diagonal — self-loops would inflate node degrees in the graph
    # Laplacian when M is added to the adjacency matrix
    np.fill_diagonal(M, 0)

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
