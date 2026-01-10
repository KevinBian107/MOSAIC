"""Motif distribution comparison metrics for molecular generation.

This module provides metrics for comparing the distribution of molecular
motifs (functional groups, ring systems, BRICS fragments) between
training data and generated molecules.
"""

from collections import Counter
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Fragments, rdMolDescriptors

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
        self.reference_smiles = reference_smiles
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
        # Filter to valid SMILES
        valid_smiles = []
        for smiles in generated_smiles:
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
