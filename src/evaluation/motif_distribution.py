"""Motif distribution comparison metrics for molecular generation.

This module provides metrics for comparing the distribution of molecular
motifs (functional groups, ring systems, BRICS fragments) between
training data and generated molecules.

Includes:
- MotifDistributionMetric: MMD-based comparison of per-molecule motif vectors
- MotifHistogramMetric: KL/Wasserstein comparison of per-motif count distributions
- MotifCooccurrenceMetric: Frobenius norm comparison of motif co-occurrence matrices
"""

from collections import Counter
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Fragments, rdMolDescriptors
from scipy.stats import wasserstein_distance as scipy_wasserstein

from src.evaluation.dist_helper import compute_mmd, gaussian


# Common molecular motif SMARTS patterns
MOLECULAR_MOTIFS = {
    # Aromatic rings
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1cnc[nH]1",
    "pyrimidine": "c1cncnc1",
    "naphthalene": "c1ccc2ccccc2c1",
    # Functional groups
    "hydroxyl": "[OX2H]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "amine_primary": "[NX3;H2;!$(NC=O)]",
    "amine_secondary": "[NX3;H1;!$(NC=O)]",
    "amine_tertiary": "[NX3;H0;!$(NC=O)]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile": "[NX1]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
    "ether": "[OD2]([#6])[#6]",
    "thioether": "[#16X2]",
    "sulfone": "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "sulfonamide": "[#16X4](=[OX1])(=[OX1])([NX3])[#6]",
    "phosphate": "[PX4](=[OX1])([OX2])([OX2])[OX2]",
}


def get_functional_group_counts(smiles: str) -> dict[str, int]:
    """Get counts of all RDKit functional groups in a molecule.

    Uses the rdkit.Chem.Fragments module which contains 82 functional
    group detection patterns.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary mapping functional group names to counts.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    counts = {}
    # Get all fr_* functions from Fragments module
    for name in dir(Fragments):
        if name.startswith("fr_"):
            try:
                func = getattr(Fragments, name)
                count = func(mol)
                if count > 0:
                    counts[name] = count
            except Exception:
                pass

    return counts


def get_motif_counts(smiles: str) -> dict[str, int]:
    """Get counts of common molecular motifs using SMARTS patterns.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary mapping motif names to counts.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    counts = {}
    for name, smarts in MOLECULAR_MOTIFS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    counts[name] = len(matches)
        except Exception:
            pass

    return counts


def get_ring_system_info(smiles: str) -> dict[str, int]:
    """Get ring system information for a molecule.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary with ring system counts.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    try:
        info = {
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "num_saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "num_heterocycles": rdMolDescriptors.CalcNumHeterocycles(mol),
            "num_aromatic_heterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
            "num_spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
            "num_bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        }

        # Add ring size counts
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        for size in [3, 4, 5, 6, 7, 8]:
            info[f"ring_size_{size}"] = ring_sizes.count(size)

        return info
    except Exception:
        return {}


def get_brics_fragments(smiles: str) -> list[str]:
    """Get BRICS fragments from a molecule.

    Args:
        smiles: SMILES string.

    Returns:
        List of BRICS fragment SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        frags = list(BRICS.BRICSDecompose(mol))
        return frags
    except Exception:
        return []


class MotifDistributionMetric:
    """Compare motif distributions between reference and generated molecules.

    This metric extracts molecular motifs (functional groups, ring systems,
    BRICS fragments) from both reference and generated molecules and
    computes the distributional distance using MMD.

    Attributes:
        reference_smiles: List of reference SMILES strings.
        use_functional_groups: Whether to use RDKit functional groups.
        use_smarts_motifs: Whether to use SMARTS-based motifs.
        use_ring_systems: Whether to use ring system information.
        use_brics: Whether to use BRICS fragments.
    """

    def __init__(
        self,
        reference_smiles: list[str],
        use_functional_groups: bool = True,
        use_smarts_motifs: bool = True,
        use_ring_systems: bool = True,
        use_brics: bool = True,
    ) -> None:
        """Initialize the motif distribution metric.

        Args:
            reference_smiles: Reference SMILES strings (typically training set).
            use_functional_groups: Include RDKit functional group counts.
            use_smarts_motifs: Include SMARTS pattern matches.
            use_ring_systems: Include ring system information.
            use_brics: Include BRICS fragment distribution.
        """
        # Filter out invalid SMILES (None, empty, "INVALID" sentinel)
        self.reference_smiles = [
            s for s in reference_smiles
            if s and s not in ["INVALID", ""] and Chem.MolFromSmiles(s) is not None
        ]
        self.use_functional_groups = use_functional_groups
        self.use_smarts_motifs = use_smarts_motifs
        self.use_ring_systems = use_ring_systems
        self.use_brics = use_brics

        # Precompute reference distributions
        self._ref_fg_vectors: Optional[np.ndarray] = None
        self._ref_motif_vectors: Optional[np.ndarray] = None
        self._ref_ring_vectors: Optional[np.ndarray] = None
        self._ref_brics_counts: Optional[Counter] = None

        self._fg_names: list[str] = []
        self._ring_names: list[str] = []

        self._precompute_reference()

    def _precompute_reference(self) -> None:
        """Precompute reference motif statistics."""
        # Functional groups
        if self.use_functional_groups:
            fg_counts_list = []
            for smiles in self.reference_smiles:
                counts = get_functional_group_counts(smiles)
                fg_counts_list.append(counts)

            # Get all functional group names
            all_fg = set()
            for counts in fg_counts_list:
                all_fg.update(counts.keys())
            self._fg_names = sorted(all_fg)

            # Build vectors
            vectors = []
            for counts in fg_counts_list:
                vec = [counts.get(name, 0) for name in self._fg_names]
                vectors.append(vec)
            self._ref_fg_vectors = np.array(vectors, dtype=float)

        # SMARTS motifs
        if self.use_smarts_motifs:
            motif_counts_list = []
            for smiles in self.reference_smiles:
                counts = get_motif_counts(smiles)
                motif_counts_list.append(counts)

            motif_names = sorted(MOLECULAR_MOTIFS.keys())
            vectors = []
            for counts in motif_counts_list:
                vec = [counts.get(name, 0) for name in motif_names]
                vectors.append(vec)
            self._ref_motif_vectors = np.array(vectors, dtype=float)

        # Ring systems
        if self.use_ring_systems:
            ring_info_list = []
            for smiles in self.reference_smiles:
                info = get_ring_system_info(smiles)
                ring_info_list.append(info)

            # Get all ring info keys
            all_keys = set()
            for info in ring_info_list:
                all_keys.update(info.keys())
            self._ring_names = sorted(all_keys)

            vectors = []
            for info in ring_info_list:
                vec = [info.get(name, 0) for name in self._ring_names]
                vectors.append(vec)
            self._ref_ring_vectors = np.array(vectors, dtype=float)

        # BRICS fragments
        if self.use_brics:
            self._ref_brics_counts = Counter()
            for smiles in self.reference_smiles:
                frags = get_brics_fragments(smiles)
                self._ref_brics_counts.update(frags)

    def _get_generated_fg_vectors(self, smiles_list: list[str]) -> np.ndarray:
        """Get functional group vectors for generated molecules."""
        vectors = []
        for smiles in smiles_list:
            counts = get_functional_group_counts(smiles)
            vec = [counts.get(name, 0) for name in self._fg_names]
            vectors.append(vec)
        return np.array(vectors, dtype=float)

    def _get_generated_motif_vectors(self, smiles_list: list[str]) -> np.ndarray:
        """Get SMARTS motif vectors for generated molecules."""
        motif_names = sorted(MOLECULAR_MOTIFS.keys())
        vectors = []
        for smiles in smiles_list:
            counts = get_motif_counts(smiles)
            vec = [counts.get(name, 0) for name in motif_names]
            vectors.append(vec)
        return np.array(vectors, dtype=float)

    def _get_generated_ring_vectors(self, smiles_list: list[str]) -> np.ndarray:
        """Get ring system vectors for generated molecules."""
        vectors = []
        for smiles in smiles_list:
            info = get_ring_system_info(smiles)
            vec = [info.get(name, 0) for name in self._ring_names]
            vectors.append(vec)
        return np.array(vectors, dtype=float)

    def _compute_brics_mmd(self, smiles_list: list[str]) -> float:
        """Compute BRICS fragment distribution MMD."""
        gen_counts = Counter()
        for smiles in smiles_list:
            frags = get_brics_fragments(smiles)
            gen_counts.update(frags)

        if not self._ref_brics_counts or not gen_counts:
            return 0.0

        # Get all unique fragments
        all_frags = set(self._ref_brics_counts.keys()) | set(gen_counts.keys())

        # Build frequency vectors (normalized)
        ref_total = sum(self._ref_brics_counts.values())
        gen_total = sum(gen_counts.values())

        ref_vec = np.array(
            [self._ref_brics_counts.get(f, 0) / ref_total for f in all_frags]
        )
        gen_vec = np.array([gen_counts.get(f, 0) / gen_total for f in all_frags])

        # Simple L2 distance as proxy for MMD
        return float(np.linalg.norm(ref_vec - gen_vec))

    def compute(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute motif distribution metrics.

        Args:
            generated_smiles: List of generated SMILES strings.

        Returns:
            Dictionary of metric names to values.
        """
        # Filter to valid SMILES (skip sentinel values before parsing)
        valid_smiles = []
        for smiles in generated_smiles:
            # Skip invalid sentinel values before calling RDKit
            if not smiles or smiles in ["INVALID", ""]:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)

        if not valid_smiles:
            return {
                "motif_fg_mmd": float("inf"),
                "motif_smarts_mmd": float("inf"),
                "motif_ring_mmd": float("inf"),
                "motif_brics_mmd": float("inf"),
            }

        results = {}

        # Functional group MMD
        if self.use_functional_groups and self._ref_fg_vectors is not None:
            gen_fg = self._get_generated_fg_vectors(valid_smiles)
            if gen_fg.size > 0 and self._ref_fg_vectors.size > 0:
                mmd = compute_mmd(
                    list(self._ref_fg_vectors),
                    list(gen_fg),
                    kernel=gaussian,
                    is_hist=False,
                    sigma=1.0,
                )
                results["motif_fg_mmd"] = mmd

        # SMARTS motif MMD
        if self.use_smarts_motifs and self._ref_motif_vectors is not None:
            gen_motif = self._get_generated_motif_vectors(valid_smiles)
            if gen_motif.size > 0 and self._ref_motif_vectors.size > 0:
                mmd = compute_mmd(
                    list(self._ref_motif_vectors),
                    list(gen_motif),
                    kernel=gaussian,
                    is_hist=False,
                    sigma=1.0,
                )
                results["motif_smarts_mmd"] = mmd

        # Ring system MMD
        if self.use_ring_systems and self._ref_ring_vectors is not None:
            gen_ring = self._get_generated_ring_vectors(valid_smiles)
            if gen_ring.size > 0 and self._ref_ring_vectors.size > 0:
                mmd = compute_mmd(
                    list(self._ref_ring_vectors),
                    list(gen_ring),
                    kernel=gaussian,
                    is_hist=False,
                    sigma=1.0,
                )
                results["motif_ring_mmd"] = mmd

        # BRICS fragment MMD
        if self.use_brics:
            results["motif_brics_mmd"] = self._compute_brics_mmd(valid_smiles)

        return results

    def __call__(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute metrics (callable interface)."""
        return self.compute(generated_smiles)

    def get_motif_summary(self, smiles_list: list[str]) -> dict[str, dict[str, int]]:
        """Get a summary of motifs found in molecules.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Dictionary with motif type -> motif counts.
        """
        fg_total = Counter()
        motif_total = Counter()
        ring_total = Counter()
        brics_total = Counter()

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            if self.use_functional_groups:
                fg_total.update(get_functional_group_counts(smiles))

            if self.use_smarts_motifs:
                motif_total.update(get_motif_counts(smiles))

            if self.use_ring_systems:
                ring_total.update(get_ring_system_info(smiles))

            if self.use_brics:
                brics_total.update(get_brics_fragments(smiles))

        return {
            "functional_groups": dict(fg_total.most_common(20)),
            "smarts_motifs": dict(motif_total.most_common(20)),
            "ring_systems": dict(ring_total),
            "brics_fragments": dict(brics_total.most_common(20)),
        }


# =============================================================================
# Motif Histogram Distribution Metric
# =============================================================================


def compute_motif_histogram(
    smiles_list: list[str],
    motif_name: str,
    max_count: int = 10,
) -> np.ndarray:
    """Compute histogram of motif counts across molecules.

    For a given motif type, builds a probability distribution over the number
    of times that motif appears in each molecule (0, 1, 2, ..., max_count).

    Args:
        smiles_list: List of SMILES strings.
        motif_name: Name of motif (must be in MOLECULAR_MOTIFS).
        max_count: Maximum count to track (counts >= max_count binned together).

    Returns:
        Probability distribution over counts [0, 1, 2, ..., max_count].
    """
    counts = []
    for smiles in smiles_list:
        motif_counts = get_motif_counts(smiles)
        counts.append(motif_counts.get(motif_name, 0))

    # Build histogram
    hist = np.zeros(max_count + 1)
    for c in counts:
        hist[min(c, max_count)] += 1

    # Normalize to probability
    total = hist.sum()
    if total > 0:
        return hist / total
    return hist


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(P || Q) with smoothing.

    Args:
        p: First probability distribution (generated).
        q: Second probability distribution (reference).
        eps: Small constant for numerical stability.

    Returns:
        KL divergence value (non-negative, 0 if distributions are identical).
    """
    # Add smoothing to avoid log(0)
    p_smooth = np.clip(p, eps, 1.0)
    q_smooth = np.clip(q, eps, 1.0)

    # Normalize after clipping
    p_smooth = p_smooth / p_smooth.sum()
    q_smooth = q_smooth / q_smooth.sum()

    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))


def compute_wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute 1-Wasserstein distance between discrete distributions.

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        Wasserstein distance (non-negative, 0 if distributions are identical).
    """
    support = np.arange(len(p))
    return float(scipy_wasserstein(support, support, p, q))


class MotifHistogramMetric:
    """Compare per-motif count distributions between reference and generated.

    For each motif type, computes the distribution of counts (0, 1, 2, ...)
    across molecules and compares using KL divergence or Wasserstein distance.

    This differs from MotifDistributionMetric which compares per-molecule
    feature vectors using MMD. MotifHistogramMetric answers: "Does the model
    generate benzene rings with the same frequency distribution as training?"

    Attributes:
        reference_smiles: List of reference SMILES strings.
        motif_names: List of motif names to track.
        max_count: Maximum count to track in histograms.
        distance_fn: Distance function ('kl' or 'wasserstein').
    """

    def __init__(
        self,
        reference_smiles: list[str],
        motif_names: list[str] | None = None,
        max_count: int = 10,
        distance_fn: str = "kl",
    ) -> None:
        """Initialize the motif histogram metric.

        Args:
            reference_smiles: Reference SMILES strings (typically training set).
            motif_names: List of motif names to track. If None, uses all
                MOLECULAR_MOTIFS.
            max_count: Maximum count to track (counts >= max_count binned).
            distance_fn: Distance function ('kl' or 'wasserstein').
        """
        # Filter to valid SMILES
        self.reference_smiles = [
            s
            for s in reference_smiles
            if s and s not in ["INVALID", ""] and Chem.MolFromSmiles(s) is not None
        ]
        self.motif_names = motif_names or list(MOLECULAR_MOTIFS.keys())
        self.max_count = max_count
        self.distance_fn = distance_fn

        # Precompute reference histograms
        self._ref_histograms: dict[str, np.ndarray] = {}
        for name in self.motif_names:
            self._ref_histograms[name] = compute_motif_histogram(
                self.reference_smiles, name, self.max_count
            )

    def compute(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute per-motif histogram distances.

        Args:
            generated_smiles: List of generated SMILES strings.

        Returns:
            Dictionary with:
            - 'motif_hist_{name}': Distance for each motif
            - 'motif_hist_mean': Average across all motifs
            - 'motif_hist_max': Maximum distance (worst motif)
        """
        # Filter to valid SMILES
        valid_smiles = [
            s
            for s in generated_smiles
            if s and s not in ["INVALID", ""] and Chem.MolFromSmiles(s) is not None
        ]

        if not valid_smiles:
            return {"motif_hist_mean": float("inf"), "motif_hist_max": float("inf")}

        distances = {}
        for name in self.motif_names:
            gen_hist = compute_motif_histogram(valid_smiles, name, self.max_count)
            ref_hist = self._ref_histograms[name]

            if self.distance_fn == "kl":
                dist = kl_divergence(gen_hist, ref_hist)
            else:
                dist = compute_wasserstein_distance(gen_hist, ref_hist)

            distances[f"motif_hist_{name}"] = dist

        all_dists = list(distances.values())
        distances["motif_hist_mean"] = float(np.mean(all_dists))
        distances["motif_hist_max"] = float(np.max(all_dists))

        return distances

    def __call__(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute metrics (callable interface)."""
        return self.compute(generated_smiles)


# =============================================================================
# Motif Co-occurrence Metric
# =============================================================================


def compute_cooccurrence_matrix(
    smiles_list: list[str],
    motif_names: list[str],
) -> np.ndarray:
    """Compute motif co-occurrence matrix.

    C[i,j] = P(motif j present | motif i present)
           = count(both i and j present) / count(i present)

    Args:
        smiles_list: List of SMILES strings.
        motif_names: Ordered list of motif names.

    Returns:
        Co-occurrence matrix of shape (n_motifs, n_motifs).
    """
    n_motifs = len(motif_names)
    n_molecules = len(smiles_list)

    if n_molecules == 0:
        return np.zeros((n_motifs, n_motifs))

    # Build presence matrix: presence_matrix[i, j] = True if motif j in molecule i
    presence_matrix = np.zeros((n_molecules, n_motifs), dtype=bool)

    for i, smiles in enumerate(smiles_list):
        counts = get_motif_counts(smiles)
        for j, name in enumerate(motif_names):
            presence_matrix[i, j] = counts.get(name, 0) > 0

    # Compute co-occurrence: C[i,j] = sum(both present) / sum(i present)
    cooccur = np.zeros((n_motifs, n_motifs))
    for i in range(n_motifs):
        count_i = presence_matrix[:, i].sum()
        if count_i > 0:
            for j in range(n_motifs):
                count_both = (presence_matrix[:, i] & presence_matrix[:, j]).sum()
                cooccur[i, j] = count_both / count_i

    return cooccur


class MotifCooccurrenceMetric:
    """Compare motif co-occurrence patterns between reference and generated.

    Builds conditional probability matrices P(motif_j | motif_i) and compares
    using Frobenius norm. This captures higher-order structural patterns -
    certain motifs naturally co-occur (e.g., benzene + hydroxyl in phenols).

    Attributes:
        reference_smiles: List of reference SMILES strings.
        motif_names: List of motif names to track.
    """

    def __init__(
        self,
        reference_smiles: list[str],
        motif_names: list[str] | None = None,
    ) -> None:
        """Initialize the motif co-occurrence metric.

        Args:
            reference_smiles: Reference SMILES strings (typically training set).
            motif_names: List of motif names to track. If None, uses all
                MOLECULAR_MOTIFS.
        """
        # Filter to valid SMILES
        self.reference_smiles = [
            s
            for s in reference_smiles
            if s and s not in ["INVALID", ""] and Chem.MolFromSmiles(s) is not None
        ]
        self.motif_names = motif_names or list(MOLECULAR_MOTIFS.keys())

        # Precompute reference co-occurrence matrix
        self._ref_cooccur = compute_cooccurrence_matrix(
            self.reference_smiles, self.motif_names
        )

    def compute(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute co-occurrence distance.

        Args:
            generated_smiles: List of generated SMILES strings.

        Returns:
            Dictionary with:
            - 'motif_cooccur_frobenius': Frobenius norm of matrix difference
            - 'motif_cooccur_mean_abs': Mean absolute element-wise difference
        """
        # Filter to valid SMILES
        valid_smiles = [
            s
            for s in generated_smiles
            if s and s not in ["INVALID", ""] and Chem.MolFromSmiles(s) is not None
        ]

        if not valid_smiles:
            return {
                "motif_cooccur_frobenius": float("inf"),
                "motif_cooccur_mean_abs": float("inf"),
            }

        gen_cooccur = compute_cooccurrence_matrix(valid_smiles, self.motif_names)

        diff = gen_cooccur - self._ref_cooccur
        frobenius = np.linalg.norm(diff, "fro")
        mean_abs = np.abs(diff).mean()

        return {
            "motif_cooccur_frobenius": float(frobenius),
            "motif_cooccur_mean_abs": float(mean_abs),
        }

    def __call__(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute metrics (callable interface)."""
        return self.compute(generated_smiles)

    def get_cooccurrence_summary(
        self,
        smiles_list: list[str],
        top_k: int = 10,
    ) -> dict[str, list[tuple[str, str, float]]]:
        """Get top co-occurring motif pairs.

        Args:
            smiles_list: List of SMILES strings.
            top_k: Number of top pairs to return.

        Returns:
            Dictionary with 'top_pairs': List of (motif_i, motif_j, P(j|i)) tuples.
        """
        valid_smiles = [
            s
            for s in smiles_list
            if s and s not in ["INVALID", ""] and Chem.MolFromSmiles(s) is not None
        ]
        cooccur = compute_cooccurrence_matrix(valid_smiles, self.motif_names)

        # Find top off-diagonal pairs
        pairs = []
        n = len(self.motif_names)
        for i in range(n):
            for j in range(n):
                if i != j and cooccur[i, j] > 0:
                    pairs.append(
                        (
                            self.motif_names[i],
                            self.motif_names[j],
                            cooccur[i, j],
                        )
                    )

        pairs.sort(key=lambda x: x[2], reverse=True)
        return {"top_pairs": pairs[:top_k]}
