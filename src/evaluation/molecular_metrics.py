"""Molecular generation evaluation metrics.

This module provides AutoGraph-style metrics for evaluating molecular graph
generation quality, including validity, uniqueness, novelty, FCD, SNN,
fragment similarity, and scaffold similarity.
"""

from collections import Counter
from typing import Optional
import warnings

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit deprecation warnings globally
warnings.filterwarnings('ignore', category=DeprecationWarning)
RDLogger.DisableLog('rdApp.warning')

# Use new MorganGenerator API to avoid deprecation warnings
try:
    from rdkit.Chem import rdMolDescriptors
    MORGAN_GENERATOR = rdMolDescriptors.GetMorganGenerator
    USE_NEW_API = True
except (ImportError, AttributeError):
    # Fall back to old API if new one not available
    USE_NEW_API = False


def get_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint using new or old RDKit API.

    Args:
        mol: RDKit molecule object.
        radius: Fingerprint radius.
        n_bits: Number of bits in fingerprint.

    Returns:
        Morgan fingerprint bit vector.
    """
    if USE_NEW_API:
        gen = MORGAN_GENERATOR(radius=radius, fpSize=n_bits)
        return gen.GetFingerprint(mol)
    else:
        # Use old API with warnings suppressed
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


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
        # Skip invalid sentinel values before calling RDKit
        if not smiles or smiles in ["INVALID", ""]:
            continue
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
        # Skip invalid sentinel values before calling RDKit
        if not smiles or smiles in ["INVALID", ""]:
            continue
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
                fp = get_morgan_fingerprint(mol, fp_radius, fp_bits)
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
                fp = get_morgan_fingerprint(mol, fp_radius, fp_bits)
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
                fp = get_morgan_fingerprint(mol, fp_radius, fp_bits)
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
                    fp = get_morgan_fingerprint(mol, fp_radius, fp_bits)
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
    # Validate inputs - ensure they are lists of strings
    if not isinstance(generated_smiles, list) or not isinstance(reference_smiles, list):
        print(f"ERROR: Invalid input types - gen: {type(generated_smiles)}, ref: {type(reference_smiles)}")
        return float("nan")

    if len(generated_smiles) == 0 or len(reference_smiles) == 0:
        print(f"ERROR: Empty input lists - gen: {len(generated_smiles)}, ref: {len(reference_smiles)}")
        return float("nan")

    # Check that elements are strings, not something else
    if not all(isinstance(s, str) for s in generated_smiles[:5]):
        print(f"ERROR: Generated SMILES contains non-string elements: {[type(s) for s in generated_smiles[:5]]}")
        return float("nan")

    if not all(isinstance(s, str) for s in reference_smiles[:5]):
        print(f"ERROR: Reference SMILES contains non-string elements: {[type(s) for s in reference_smiles[:5]]}")
        return float("nan")

    # Set environment variables to disable multiprocessing in PyTorch (used by FCD)
    import os
    old_num_threads = os.environ.get('OMP_NUM_THREADS')
    old_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    try:
        # Try using fcd package directly first (more reliable than MOSES)
        from fcd import get_fcd, load_ref_model, canonical_smiles
        from rdkit import Chem
        import torch

        # Disable PyTorch multiprocessing
        torch.set_num_threads(1)

        # Use RDKit to canonicalize instead of fcd's function to avoid multiprocessing
        gen_canonical = []
        for s in generated_smiles:
            if s and isinstance(s, str):
                try:
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None:
                        can = Chem.MolToSmiles(mol)
                        gen_canonical.append(can)
                except Exception:
                    pass

        ref_canonical = []
        for s in reference_smiles:
            if s and isinstance(s, str):
                try:
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None:
                        can = Chem.MolToSmiles(mol)
                        ref_canonical.append(can)
                except Exception:
                    pass

        if not gen_canonical or not ref_canonical:
            print(f"ERROR: No valid canonical SMILES - gen: {len(gen_canonical)}, ref: {len(ref_canonical)}")
            return float("inf")

        model = load_ref_model()

        # Wrap in try-except to catch multiprocessing errors
        try:
            fcd_score = get_fcd(gen_canonical, ref_canonical, model)
            # Restore environment variables
            if old_num_threads:
                os.environ['OMP_NUM_THREADS'] = old_num_threads
            else:
                os.environ.pop('OMP_NUM_THREADS', None)
            if old_mkl_threads:
                os.environ['MKL_NUM_THREADS'] = old_mkl_threads
            else:
                os.environ.pop('MKL_NUM_THREADS', None)
            return fcd_score
        except Exception as e:
            print(f"ERROR in get_fcd: {e}")
            raise  # Re-raise to fall through to MOSES
    except ImportError:
        pass
    except Exception as e:
        print(f"ERROR in FCD package: {e}")
        import traceback
        traceback.print_exc()
        pass
    finally:
        # Restore environment variables
        if old_num_threads:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
        else:
            os.environ.pop('OMP_NUM_THREADS', None)
        if old_mkl_threads:
            os.environ['MKL_NUM_THREADS'] = old_mkl_threads
        else:
            os.environ.pop('MKL_NUM_THREADS', None)

    try:
        # Fall back to MOSES package (but it has multiprocessing bugs)
        import moses

        # MOSES expects lists of strings - double check
        gen_list = list(generated_smiles)  # Ensure it's a list
        ref_list = list(reference_smiles)  # Ensure it's a list

        # Call MOSES with explicit parameters to avoid multiprocessing issues
        metrics = moses.get_all_metrics(
            gen=gen_list,
            test=ref_list,
            n_jobs=1,  # Force single-threaded to avoid multiprocessing bugs
        )
        return metrics.get("FCD/Test", float("inf"))
    except ImportError:
        # FCD not available, return NaN
        return float("nan")
    except Exception as e:
        print(f"ERROR in FCD computation: {e}")
        import traceback
        traceback.print_exc()
        return float("nan")
