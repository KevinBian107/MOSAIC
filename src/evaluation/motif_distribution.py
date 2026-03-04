"""Motif distribution comparison metrics for molecular generation.

This module provides metrics for comparing the distribution of molecular
motifs (functional groups, ring systems, BRICS fragments) between
training data and generated molecules.

Includes:
- MotifDistributionMetric: MMD-based comparison of per-molecule motif vectors
- MotifHistogramMetric: KL/Wasserstein comparison of per-motif count distributions
- MotifCooccurrenceMetric: Frobenius norm comparison of motif co-occurrence matrices
"""

import hashlib
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import BRICS, Fragments, rdMolDescriptors
from scipy.stats import wasserstein_distance as scipy_wasserstein

from src.evaluation.dist_helper import compute_mmd, gaussian

log = logging.getLogger(__name__)


def _extract_motif_features_chunk(
    smiles_chunk: list[str],
) -> tuple[
    list[dict[str, int]],
    list[dict[str, int]],
    list[dict[str, int]],
    list[list[str]],
]:
    """Extract FG, motif, ring, and BRICS features for a chunk of SMILES (one parse per molecule)."""
    fg_list = []
    motif_list = []
    ring_list = []
    brics_list = []
    for s in smiles_chunk:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fg_list.append({})
            motif_list.append({})
            ring_list.append({})
            brics_list.append([])
            continue
        fg_list.append(_fg_counts_from_mol(mol))
        motif_list.append(_motif_counts_from_mol(mol))
        ring_list.append(_ring_info_from_mol(mol))
        brics_list.append(_brics_from_mol(mol))
    return (fg_list, motif_list, ring_list, brics_list)


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


def _fg_counts_from_mol(mol: Chem.Mol) -> dict[str, int]:
    """Get functional group counts from an RDKit Mol (no parsing)."""
    counts = {}
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


def _motif_counts_from_mol(mol: Chem.Mol) -> dict[str, int]:
    """Get SMARTS motif counts from an RDKit Mol (no parsing)."""
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


def _ring_info_from_mol(mol: Chem.Mol) -> dict[str, int]:
    """Get ring system info from an RDKit Mol (no parsing)."""
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
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        for size in [3, 4, 5, 6, 7, 8]:
            info[f"ring_size_{size}"] = ring_sizes.count(size)
        return info
    except Exception:
        return {}


def _brics_from_mol(mol: Chem.Mol) -> list[str]:
    """Get BRICS fragments from an RDKit Mol (no parsing)."""
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def get_functional_group_counts(smiles: str) -> dict[str, int]:
    """Get counts of all RDKit functional groups in a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return _fg_counts_from_mol(mol) if mol is not None else {}


def get_motif_counts(smiles: str) -> dict[str, int]:
    """Get counts of common molecular motifs using SMARTS patterns."""
    mol = Chem.MolFromSmiles(smiles)
    return _motif_counts_from_mol(mol) if mol is not None else {}


def get_ring_system_info(smiles: str) -> dict[str, int]:
    """Get ring system information for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return _ring_info_from_mol(mol) if mol is not None else {}


def get_brics_fragments(smiles: str) -> list[str]:
    """Get BRICS fragments from a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return _brics_from_mol(mol) if mol is not None else []


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
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        reference_split: Optional[str] = None,
        actual_ref_size: Optional[int] = None,
        n_workers: Optional[int] = None,
    ) -> None:
        """Initialize the motif distribution metric.

        Args:
            reference_smiles: Reference SMILES strings (typically training set).
            use_functional_groups: Include RDKit functional group counts.
            use_smarts_motifs: Include SMARTS pattern matches.
            use_ring_systems: Include ring system information.
            use_brics: Include BRICS fragment distribution.
            cache_dir: If set with dataset_name/reference_split/actual_ref_size, load/save
                reference precompute to this directory.
            dataset_name: Dataset name for cache key.
            reference_split: Reference split for cache key.
            actual_ref_size: Actual reference size for cache key.
            n_workers: If > 1, use this many processes for feature extraction; None = 1.
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

        # Normalize n_workers (Hydra/CLI may pass it as a string)
        n_workers_normalized: Optional[int]
        if isinstance(n_workers, str):
            try:
                n_workers_normalized = int(n_workers)
            except ValueError:
                log.warning(
                    "MotifDistributionMetric: invalid n_workers=%r; falling back to 1",
                    n_workers,
                )
                n_workers_normalized = 1
        else:
            n_workers_normalized = n_workers

        self._n_workers = (
            n_workers_normalized
            if n_workers_normalized is not None and n_workers_normalized > 1
            else 1
        )

        # Precompute reference distributions
        self._ref_fg_vectors: Optional[np.ndarray] = None
        self._ref_motif_vectors: Optional[np.ndarray] = None
        self._ref_ring_vectors: Optional[np.ndarray] = None
        self._ref_brics_counts: Optional[Counter] = None

        self._fg_names: list[str] = []
        self._ring_names: list[str] = []

        use_cache = (
            cache_dir is not None
            and dataset_name is not None
            and reference_split is not None
            and actual_ref_size is not None
        )
        if use_cache:
            content_hash = hashlib.sha256(
                "".join(sorted(self.reference_smiles)).encode()
            ).hexdigest()[:16]
            cache_path = (
                Path(cache_dir)
                / f"motif_ref_{dataset_name}_{reference_split}_{actual_ref_size}_{content_hash}.npz"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            if cache_path.exists():
                try:
                    loaded = np.load(cache_path, allow_pickle=True)
                    n_ref = int(loaded["n_ref"])
                    if n_ref == len(self.reference_smiles) and bool(
                        loaded["use_functional_groups"]
                    ) == use_functional_groups and bool(
                        loaded["use_smarts_motifs"]
                    ) == use_smarts_motifs and bool(
                        loaded["use_ring_systems"]
                    ) == use_ring_systems and bool(
                        loaded["use_brics"]
                    ) == use_brics:
                        self._fg_names = list(loaded["fg_names"])
                        self._ring_names = list(loaded["ring_names"])
                        if self.use_functional_groups and "ref_fg_vectors" in loaded:
                            self._ref_fg_vectors = loaded["ref_fg_vectors"]
                        if self.use_smarts_motifs and "ref_motif_vectors" in loaded:
                            self._ref_motif_vectors = loaded["ref_motif_vectors"]
                        if self.use_ring_systems and "ref_ring_vectors" in loaded:
                            self._ref_ring_vectors = loaded["ref_ring_vectors"]
                        if self.use_brics:
                            if "ref_brics_counts" in loaded:
                                self._ref_brics_counts = Counter(
                                    loaded["ref_brics_counts"].item()
                                )
                            else:
                                self._ref_brics_counts = Counter()
                        elapsed = time.perf_counter() - t0
                        log.info(
                            "Loading motif reference cache from %s (N_ref=%d)",
                            cache_path,
                            n_ref,
                        )
                        log.info("Motif reference cache load: %.2fs", elapsed)
                        return
                except Exception as e:
                    log.warning("Motif cache load failed (%s), recomputing.", e)

            log.info(
                "Precomputing reference motif features (N_ref=%d)...",
                len(self.reference_smiles),
            )

        t0 = time.perf_counter()
        self._precompute_reference()
        total_elapsed = time.perf_counter() - t0
        log.info("Motif reference precompute total: %.2fs", total_elapsed)

        if use_cache and cache_path is not None:
            try:
                save_kw: dict = {
                    "n_ref": len(self.reference_smiles),
                    "use_functional_groups": np.array(self.use_functional_groups),
                    "use_smarts_motifs": np.array(self.use_smarts_motifs),
                    "use_ring_systems": np.array(self.use_ring_systems),
                    "use_brics": np.array(self.use_brics),
                    "fg_names": np.array(self._fg_names, dtype=object),
                    "ring_names": np.array(self._ring_names, dtype=object),
                    "allow_pickle": True,
                }
                if self._ref_fg_vectors is not None:
                    save_kw["ref_fg_vectors"] = self._ref_fg_vectors
                if self._ref_motif_vectors is not None:
                    save_kw["ref_motif_vectors"] = self._ref_motif_vectors
                if self._ref_ring_vectors is not None:
                    save_kw["ref_ring_vectors"] = self._ref_ring_vectors
                if self.use_brics:
                    brics_arr = np.empty((), dtype=object)
                    brics_arr[()] = dict(self._ref_brics_counts) if self._ref_brics_counts else {}
                    save_kw["ref_brics_counts"] = brics_arr
                np.savez(cache_path, **save_kw)
                log.info("Saved motif reference cache to %s", cache_path)
            except Exception as e:
                log.warning("Motif cache save failed: %s", e)

    def _precompute_reference(self) -> None:
        """Precompute reference motif statistics."""
        n_ref = len(self.reference_smiles)
        if n_ref == 0:
            return

        if self._n_workers > 1:
            self._precompute_reference_parallel()
            return

        # Single pass over reference: parse each molecule once, extract all four feature types
        log.info("  Reference features (one parse per molecule)...")
        t0 = time.perf_counter()
        fg_counts_list = []
        motif_counts_list = []
        ring_info_list = []
        brics_frags_list = []
        for smiles in tqdm(
            self.reference_smiles,
            desc="  Reference features",
            unit="mol",
        ):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fg_counts_list.append({})
                motif_counts_list.append({})
                ring_info_list.append({})
                brics_frags_list.append([])
                continue
            fg_counts_list.append(_fg_counts_from_mol(mol))
            motif_counts_list.append(_motif_counts_from_mol(mol))
            ring_info_list.append(_ring_info_from_mol(mol))
            brics_frags_list.append(_brics_from_mol(mol))
        log.info("  Reference features: done in %.1fs", time.perf_counter() - t0)

        if self.use_functional_groups and fg_counts_list:
            all_fg = set()
            for counts in fg_counts_list:
                all_fg.update(counts.keys())
            self._fg_names = sorted(all_fg)
            vectors = []
            for counts in fg_counts_list:
                vec = [counts.get(name, 0) for name in self._fg_names]
                vectors.append(vec)
            self._ref_fg_vectors = np.array(vectors, dtype=float)

        if self.use_smarts_motifs and motif_counts_list:
            motif_names = sorted(MOLECULAR_MOTIFS.keys())
            vectors = []
            for counts in motif_counts_list:
                vec = [counts.get(name, 0) for name in motif_names]
                vectors.append(vec)
            self._ref_motif_vectors = np.array(vectors, dtype=float)

        if self.use_ring_systems and ring_info_list:
            all_keys = set()
            for info in ring_info_list:
                all_keys.update(info.keys())
            self._ring_names = sorted(all_keys)
            vectors = []
            for info in ring_info_list:
                vec = [info.get(name, 0) for name in self._ring_names]
                vectors.append(vec)
            self._ref_ring_vectors = np.array(vectors, dtype=float)

        if self.use_brics and brics_frags_list:
            self._ref_brics_counts = Counter()
            for frags in brics_frags_list:
                self._ref_brics_counts.update(frags)

    def _precompute_reference_parallel(self) -> None:
        """Precompute reference motif statistics using multiple processes."""
        n_ref = len(self.reference_smiles)
        chunk_size = max(1, (n_ref + self._n_workers - 1) // self._n_workers)
        chunks = [
            self.reference_smiles[i : i + chunk_size]
            for i in range(0, n_ref, chunk_size)
        ]
        log.info("  Extracting reference features (%d chunks, n_workers=%d)...", len(chunks), self._n_workers)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            chunk_results = list(
                tqdm(
                    executor.map(_extract_motif_features_chunk, chunks),
                    total=len(chunks),
                    desc="  Reference features",
                    unit="chunk",
                )
            )
        elapsed = time.perf_counter() - t0
        log.info("  Reference feature extraction: done in %.1fs", elapsed)

        fg_counts_list = []
        motif_counts_list = []
        ring_info_list = []
        brics_frags_list = []
        for fr, mr, rr, br in chunk_results:
            fg_counts_list.extend(fr)
            motif_counts_list.extend(mr)
            ring_info_list.extend(rr)
            brics_frags_list.extend(br)

        if self.use_functional_groups:
            all_fg = set()
            for counts in fg_counts_list:
                all_fg.update(counts.keys())
            self._fg_names = sorted(all_fg)
            vectors = []
            for counts in fg_counts_list:
                vec = [counts.get(name, 0) for name in self._fg_names]
                vectors.append(vec)
            self._ref_fg_vectors = np.array(vectors, dtype=float)

        if self.use_smarts_motifs:
            motif_names = sorted(MOLECULAR_MOTIFS.keys())
            vectors = []
            for counts in motif_counts_list:
                vec = [counts.get(name, 0) for name in motif_names]
                vectors.append(vec)
            self._ref_motif_vectors = np.array(vectors, dtype=float)

        if self.use_ring_systems:
            all_keys = set()
            for info in ring_info_list:
                all_keys.update(info.keys())
            self._ring_names = sorted(all_keys)
            vectors = []
            for info in ring_info_list:
                vec = [info.get(name, 0) for name in self._ring_names]
                vectors.append(vec)
            self._ref_ring_vectors = np.array(vectors, dtype=float)

        if self.use_brics:
            self._ref_brics_counts = Counter()
            for frags in brics_frags_list:
                self._ref_brics_counts.update(frags)

    def _get_generated_fg_vectors(self, smiles_list: list[str]) -> np.ndarray:
        """Get functional group vectors for generated molecules."""
        vectors = []
        for smiles in tqdm(
            smiles_list,
            desc="  Generated FG vectors",
            unit="mol",
        ):
            counts = get_functional_group_counts(smiles)
            vec = [counts.get(name, 0) for name in self._fg_names]
            vectors.append(vec)
        return np.array(vectors, dtype=float)

    def _get_generated_motif_vectors(self, smiles_list: list[str]) -> np.ndarray:
        """Get SMARTS motif vectors for generated molecules."""
        motif_names = sorted(MOLECULAR_MOTIFS.keys())
        vectors = []
        for smiles in tqdm(
            smiles_list,
            desc="  Generated SMARTS vectors",
            unit="mol",
        ):
            counts = get_motif_counts(smiles)
            vec = [counts.get(name, 0) for name in motif_names]
            vectors.append(vec)
        return np.array(vectors, dtype=float)

    def _get_generated_ring_vectors(self, smiles_list: list[str]) -> np.ndarray:
        """Get ring system vectors for generated molecules."""
        vectors = []
        for smiles in tqdm(
            smiles_list,
            desc="  Generated ring vectors",
            unit="mol",
        ):
            info = get_ring_system_info(smiles)
            vec = [info.get(name, 0) for name in self._ring_names]
            vectors.append(vec)
        return np.array(vectors, dtype=float)

    def _get_generated_features_parallel(
        self, smiles_list: list[str]
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Counter,
    ]:
        """Extract all motif features for generated molecules in parallel. Returns (fg_vectors, motif_vectors, ring_vectors, brics_counter)."""
        n = len(smiles_list)
        chunk_size = max(1, (n + self._n_workers - 1) // self._n_workers)
        chunks = [smiles_list[i : i + chunk_size] for i in range(0, n, chunk_size)]
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            chunk_results = list(
                tqdm(
                    executor.map(_extract_motif_features_chunk, chunks),
                    total=len(chunks),
                    desc="  Generated features",
                    unit="chunk",
                )
            )
        fg_list, motif_list, ring_info_list, brics_frags_list = [], [], [], []
        for fr, mr, rr, br in chunk_results:
            fg_list.extend(fr)
            motif_list.extend(mr)
            ring_info_list.extend(rr)
            brics_frags_list.extend(br)

        gen_brics_counts = Counter()
        for frags in brics_frags_list:
            gen_brics_counts.update(frags)

        gen_fg = None
        if self.use_functional_groups and self._fg_names:
            vectors = [
                [counts.get(name, 0) for name in self._fg_names]
                for counts in fg_list
            ]
            gen_fg = np.array(vectors, dtype=float)

        gen_motif = None
        if self.use_smarts_motifs:
            motif_names = sorted(MOLECULAR_MOTIFS.keys())
            vectors = [
                [counts.get(name, 0) for name in motif_names]
                for counts in motif_list
            ]
            gen_motif = np.array(vectors, dtype=float)

        gen_ring = None
        if self.use_ring_systems and self._ring_names:
            vectors = [
                [info.get(name, 0) for name in self._ring_names]
                for info in ring_info_list
            ]
            gen_ring = np.array(vectors, dtype=float)

        return (gen_fg, gen_motif, gen_ring, gen_brics_counts)

    def _compute_brics_mmd(self, smiles_list: list[str]) -> float:
        """Compute BRICS fragment distribution MMD."""
        gen_counts = Counter()
        for smiles in tqdm(
            smiles_list,
            desc="  BRICS",
            unit="mol",
        ):
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

    def _compute_brics_mmd_from_counts(self, gen_counts: Counter) -> float:
        """Compute BRICS MMD from precomputed generated fragment counts."""
        if not self._ref_brics_counts or not gen_counts:
            return 0.0
        all_frags = set(self._ref_brics_counts.keys()) | set(gen_counts.keys())
        ref_total = sum(self._ref_brics_counts.values())
        gen_total = sum(gen_counts.values())
        ref_vec = np.array(
            [self._ref_brics_counts.get(f, 0) / ref_total for f in all_frags]
        )
        gen_vec = np.array([gen_counts.get(f, 0) / gen_total for f in all_frags])
        return float(np.linalg.norm(ref_vec - gen_vec))

    def compute(self, generated_smiles: list[str]) -> dict[str, float]:
        """Compute motif distribution metrics.

        Args:
            generated_smiles: List of generated SMILES strings.

        Returns:
            Dictionary of metric names to values.
        """
        t_total_start = time.perf_counter()
        t_extract_start = time.perf_counter()
        valid_smiles = []
        fg_list = []
        motif_list = []
        ring_list = []
        brics_frags_list = []
        for smiles in tqdm(
            generated_smiles,
            desc="  Generated features",
            unit="mol",
        ):
            if not smiles or smiles in ["INVALID", ""]:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            valid_smiles.append(smiles)
            fg_list.append(_fg_counts_from_mol(mol))
            motif_list.append(_motif_counts_from_mol(mol))
            ring_list.append(_ring_info_from_mol(mol))
            brics_frags_list.append(_brics_from_mol(mol))
        t_filter = t_gen_fg = t_gen_motif = t_gen_ring = t_brics = time.perf_counter() - t_extract_start

        log.info(
            "Motif metrics: %d valid generated molecules (of %d total)",
            len(valid_smiles),
            len(generated_smiles),
        )

        if not valid_smiles:
            return {
                "motif_fg_mmd": float("inf"),
                "motif_smarts_mmd": float("inf"),
                "motif_ring_mmd": float("inf"),
                "motif_brics_mmd": float("inf"),
            }

        results = {}
        t_mmd_fg = t_mmd_smarts = t_mmd_ring = 0.0

        gen_brics_counts = Counter()
        for frags in brics_frags_list:
            gen_brics_counts.update(frags)

        gen_fg = None
        if self.use_functional_groups and self._fg_names and fg_list:
            gen_fg = np.array(
                [[c.get(name, 0) for name in self._fg_names] for c in fg_list],
                dtype=float,
            )
        gen_motif = None
        if self.use_smarts_motifs and motif_list:
            motif_names = sorted(MOLECULAR_MOTIFS.keys())
            gen_motif = np.array(
                [[c.get(name, 0) for name in motif_names] for c in motif_list],
                dtype=float,
            )
        gen_ring = None
        if self.use_ring_systems and self._ring_names and ring_list:
            gen_ring = np.array(
                [[info.get(name, 0) for name in self._ring_names] for info in ring_list],
                dtype=float,
            )

        if self.use_functional_groups and self._ref_fg_vectors is not None and gen_fg is not None and gen_fg.size > 0 and self._ref_fg_vectors.size > 0:
            log.info("  MMD (functional groups)...")
            t0 = time.perf_counter()
            results["motif_fg_mmd"] = compute_mmd(
                list(self._ref_fg_vectors),
                list(gen_fg),
                kernel=gaussian,
                is_hist=False,
                sigma=1.0,
                show_progress=True,
                progress_desc="  MMD FG",
            )
            t_mmd_fg = time.perf_counter() - t0
            log.info("  MMD (FG): %.1fs", t_mmd_fg)

        if self.use_smarts_motifs and self._ref_motif_vectors is not None and gen_motif is not None and gen_motif.size > 0 and self._ref_motif_vectors.size > 0:
            log.info("  MMD (SMARTS motifs)...")
            t0 = time.perf_counter()
            results["motif_smarts_mmd"] = compute_mmd(
                list(self._ref_motif_vectors),
                list(gen_motif),
                kernel=gaussian,
                is_hist=False,
                sigma=1.0,
                show_progress=True,
                progress_desc="  MMD SMARTS",
            )
            t_mmd_smarts = time.perf_counter() - t0
            log.info("  MMD (SMARTS): %.1fs", t_mmd_smarts)

        if self.use_ring_systems and self._ref_ring_vectors is not None and gen_ring is not None and gen_ring.size > 0 and self._ref_ring_vectors.size > 0:
            log.info("  MMD (ring systems)...")
            t0 = time.perf_counter()
            results["motif_ring_mmd"] = compute_mmd(
                list(self._ref_ring_vectors),
                list(gen_ring),
                kernel=gaussian,
                is_hist=False,
                sigma=1.0,
                show_progress=True,
                progress_desc="  MMD ring",
            )
            t_mmd_ring = time.perf_counter() - t0
            log.info("  MMD (ring): %.1fs", t_mmd_ring)

        if self.use_brics:
            results["motif_brics_mmd"] = self._compute_brics_mmd_from_counts(gen_brics_counts)

        total_elapsed = time.perf_counter() - t_total_start
        log.info(
            "Motif benchmark: filter=%.1fs gen_fg=%.1fs gen_motif=%.1fs gen_ring=%.1fs brics=%.1fs mmd_fg=%.1fs mmd_smarts=%.1fs mmd_ring=%.1fs total=%.1fs",
            t_filter,
            t_gen_fg,
            t_gen_motif,
            t_gen_ring,
            t_brics,
            t_mmd_fg,
            t_mmd_smarts,
            t_mmd_ring,
            total_elapsed,
        )
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
                fg_total.update(_fg_counts_from_mol(mol))
            if self.use_smarts_motifs:
                motif_total.update(_motif_counts_from_mol(mol))
            if self.use_ring_systems:
                ring_total.update(_ring_info_from_mol(mol))
            if self.use_brics:
                brics_total.update(_brics_from_mol(mol))

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
