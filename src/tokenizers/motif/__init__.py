"""Motif detection and utilities for graph coarsening.

This module provides motif detection and affinity computation for use in
motif-aware graph coarsening strategies.

Key components:
- CLUSTERING_MOTIFS: Dictionary of SMARTS patterns for ring systems
- MotifInstance: Data class for detected motif instances
- detect_motifs_from_smiles: Detect motifs via SMARTS matching
- detect_motifs_from_data: Detect motifs from PyG Data objects
- compute_motif_affinity_matrix: Compute motif co-membership matrix
- compute_motif_cohesion: Evaluate motif preservation in partitions
"""

from src.tokenizers.motif.affinity import (
    compute_motif_affinity_matrix,
    compute_motif_cohesion,
    get_motif_summary,
)
from src.tokenizers.motif.detection import (
    MotifInstance,
    detect_motifs_from_data,
    detect_motifs_from_smiles,
)
from src.tokenizers.motif.patterns import CLUSTERING_MOTIFS

__all__ = [
    # Patterns
    "CLUSTERING_MOTIFS",
    # Detection
    "MotifInstance",
    "detect_motifs_from_smiles",
    "detect_motifs_from_data",
    # Affinity
    "compute_motif_affinity_matrix",
    "compute_motif_cohesion",
    "get_motif_summary",
]
