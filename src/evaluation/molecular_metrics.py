"""Molecular generation evaluation metrics.

This module provides AutoGraph-style metrics for evaluating molecular graph
generation quality, including validity, uniqueness, novelty, FCD, SNN,
fragment similarity, and scaffold similarity.
"""

from collections import Counter
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold


def compute_validity(smiles_list: list[str]) -> float:
    """Compute the fraction of valid SMILES strings.

    Args:
        smiles_list: List of SMILES strings to validate.

    Returns:
        Fraction of valid molecules (0.0 to 1.0).
    """
    if not smiles_list:
        return 0.0

    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_count += 1
            except Exception:
                pass

    return valid_count / len(smiles_list)


def compute_uniqueness(smiles_list: list[str]) -> float:
    """Compute the fraction of unique valid SMILES strings.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Fraction of unique molecules among valid ones (0.0 to 1.0).
    """
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                # Canonicalize for comparison
                canonical = Chem.MolToSmiles(mol)
                valid_smiles.append(canonical)
            except Exception:
                pass

    if not valid_smiles:
        return 0.0

    unique_smiles = set(valid_smiles)
    return len(unique_smiles) / len(valid_smiles)


def compute_novelty(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> float:
    """Compute the fraction of generated molecules not in reference set.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: List of reference (training) SMILES strings.

    Returns:
        Fraction of novel molecules (0.0 to 1.0).
    """
    # Canonicalize reference set
    reference_canonical = set()
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                canonical = Chem.MolToSmiles(mol)
                reference_canonical.add(canonical)
            except Exception:
                pass

    # Count novel generated molecules
    valid_count = 0
    novel_count = 0
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol)
                valid_count += 1
                if canonical not in reference_canonical:
                    novel_count += 1
            except Exception:
                pass

    if valid_count == 0:
        return 0.0

    return novel_count / valid_count


def compute_snn(
    generated_smiles: list[str],
    reference_smiles: list[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> float:
    """Compute average similarity to nearest neighbor in reference set.

    Uses Morgan fingerprints with Tanimoto similarity.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: List of reference SMILES strings.
        fp_radius: Radius for Morgan fingerprint.
        fp_bits: Number of bits in fingerprint.

    Returns:
        Average similarity to nearest neighbor (0.0 to 1.0).
    """
    # Compute fingerprints for reference set
    ref_fps = []
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                ref_fps.append(fp)
            except Exception:
                pass

    if not ref_fps:
        return 0.0

    # Compute SNN for each generated molecule
    snn_scores = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                # Find nearest neighbor
                sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
                max_sim = max(sims) if sims else 0.0
                snn_scores.append(max_sim)
            except Exception:
                pass

    if not snn_scores:
        return 0.0

    return np.mean(snn_scores)


def get_brics_fragments(smiles: str) -> set[str]:
    """Extract BRICS fragments from a molecule.

    Args:
        smiles: SMILES string.

    Returns:
        Set of BRICS fragment SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    try:
        frags = BRICS.BRICSDecompose(mol)
        return set(frags)
    except Exception:
        return set()


def compute_fragment_similarity(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> float:
    """Compute cosine similarity of BRICS fragment frequency distributions.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: List of reference SMILES strings.

    Returns:
        Cosine similarity of fragment distributions (0.0 to 1.0).
    """
    # Count fragments in reference set
    ref_fragments = Counter()
    for smiles in reference_smiles:
        frags = get_brics_fragments(smiles)
        ref_fragments.update(frags)

    # Count fragments in generated set
    gen_fragments = Counter()
    for smiles in generated_smiles:
        frags = get_brics_fragments(smiles)
        gen_fragments.update(frags)

    if not ref_fragments or not gen_fragments:
        return 0.0

    # Get all unique fragments
    all_fragments = set(ref_fragments.keys()) | set(gen_fragments.keys())

    # Build frequency vectors
    ref_vec = np.array([ref_fragments.get(f, 0) for f in all_fragments], dtype=float)
    gen_vec = np.array([gen_fragments.get(f, 0) for f in all_fragments], dtype=float)

    # Normalize
    ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-10)
    gen_vec = gen_vec / (np.linalg.norm(gen_vec) + 1e-10)

    # Cosine similarity
    return float(np.dot(ref_vec, gen_vec))


def get_scaffold(smiles: str) -> Optional[str]:
    """Extract Bemis-Murcko scaffold from a molecule.

    Args:
        smiles: SMILES string.

    Returns:
        Scaffold SMILES or None if extraction fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def compute_scaffold_similarity(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> float:
    """Compute cosine similarity of Bemis-Murcko scaffold frequency distributions.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: List of reference SMILES strings.

    Returns:
        Cosine similarity of scaffold distributions (0.0 to 1.0).
    """
    # Count scaffolds in reference set
    ref_scaffolds = Counter()
    for smiles in reference_smiles:
        scaffold = get_scaffold(smiles)
        if scaffold:
            ref_scaffolds[scaffold] += 1

    # Count scaffolds in generated set
    gen_scaffolds = Counter()
    for smiles in generated_smiles:
        scaffold = get_scaffold(smiles)
        if scaffold:
            gen_scaffolds[scaffold] += 1

    if not ref_scaffolds or not gen_scaffolds:
        return 0.0

    # Get all unique scaffolds
    all_scaffolds = set(ref_scaffolds.keys()) | set(gen_scaffolds.keys())

    # Build frequency vectors
    ref_vec = np.array([ref_scaffolds.get(s, 0) for s in all_scaffolds], dtype=float)
    gen_vec = np.array([gen_scaffolds.get(s, 0) for s in all_scaffolds], dtype=float)

    # Normalize
    ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-10)
    gen_vec = gen_vec / (np.linalg.norm(gen_vec) + 1e-10)

    # Cosine similarity
    return float(np.dot(ref_vec, gen_vec))


def compute_internal_diversity(
    smiles_list: list[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
    sample_size: int = 1000,
) -> float:
    """Compute internal diversity (average pairwise Tanimoto distance).

    Args:
        smiles_list: List of SMILES strings.
        fp_radius: Radius for Morgan fingerprint.
        fp_bits: Number of bits in fingerprint.
        sample_size: Number of molecules to sample for computation.

    Returns:
        Average pairwise Tanimoto distance (0.0 to 1.0).
    """
    # Compute fingerprints
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                fps.append(fp)
            except Exception:
                pass

    if len(fps) < 2:
        return 0.0

    # Sample if too many
    if len(fps) > sample_size:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(fps), sample_size, replace=False)
        fps = [fps[i] for i in indices]

    # Compute pairwise distances
    distances = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        distances.extend([1.0 - s for s in sims])

    return np.mean(distances) if distances else 0.0


class MolecularMetrics:
    """Evaluator for molecular generation quality.

    Computes AutoGraph-style metrics comparing generated molecules
    to a reference set.

    Attributes:
        reference_smiles: List of reference SMILES strings.
        fp_radius: Radius for Morgan fingerprints.
        fp_bits: Number of bits in fingerprints.
    """

    def __init__(
        self,
        reference_smiles: list[str],
        fp_radius: int = 2,
        fp_bits: int = 2048,
    ) -> None:
        """Initialize the molecular metrics evaluator.

        Args:
            reference_smiles: Reference SMILES strings (typically training set).
            fp_radius: Radius for Morgan fingerprint.
            fp_bits: Number of bits in fingerprint.
        """
        self.reference_smiles = reference_smiles
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

        # Precompute reference fingerprints
        self._ref_fps = []
        self._ref_canonical = set()
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, fp_radius, nBits=fp_bits
                    )
                    self._ref_fps.append(fp)
                    self._ref_canonical.add(Chem.MolToSmiles(mol))
                except Exception:
                    pass

    def compute(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute all molecular metrics.

        Args:
            generated_smiles: List of generated SMILES strings.

        Returns:
            Dictionary of metric names to values.
        """
        return {
            "validity": compute_validity(generated_smiles),
            "uniqueness": compute_uniqueness(generated_smiles),
            "novelty": compute_novelty(generated_smiles, self.reference_smiles),
            "snn": compute_snn(
                generated_smiles,
                self.reference_smiles,
                self.fp_radius,
                self.fp_bits,
            ),
            "frag_similarity": compute_fragment_similarity(
                generated_smiles, self.reference_smiles
            ),
            "scaff_similarity": compute_scaffold_similarity(
                generated_smiles, self.reference_smiles
            ),
            "internal_diversity": compute_internal_diversity(
                generated_smiles, self.fp_radius, self.fp_bits
            ),
        }

    def __call__(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute metrics (callable interface)."""
        return self.compute(generated_smiles)


def compute_fcd(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> float:
    """Compute Frechet ChemNet Distance.

    Note: Requires the 'fcd' package or uses MOSES implementation.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: List of reference SMILES strings.

    Returns:
        FCD score (lower is better).
    """
    try:
        # Try using MOSES package
        import moses
        metrics = moses.get_all_metrics(generated_smiles, test=reference_smiles)
        return metrics.get("FCD/Test", float("inf"))
    except ImportError:
        pass

    try:
        # Try using fcd package directly
        from fcd import get_fcd, load_ref_model, canonical_smiles

        model = load_ref_model()
        gen_canonical = [canonical_smiles(s) for s in generated_smiles if s]
        ref_canonical = [canonical_smiles(s) for s in reference_smiles if s]
        gen_canonical = [s for s in gen_canonical if s]
        ref_canonical = [s for s in ref_canonical if s]

        if not gen_canonical or not ref_canonical:
            return float("inf")

        return get_fcd(gen_canonical, ref_canonical, model)
    except ImportError:
        # FCD not available, return NaN
        return float("nan")
