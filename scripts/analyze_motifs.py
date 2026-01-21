#!/usr/bin/env python
"""Analyze motif distributions in molecular datasets.

Generates comprehensive statistics on functional groups, SMARTS patterns,
ring systems, and BRICS fragments for molecular datasets.

Usage:
    python scripts/analyze_motifs.py --dataset moses --num_samples 10000
    python scripts/analyze_motifs.py --dataset moses --num_samples -1  # Full dataset
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, Fragments, rdMolDescriptors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.molecular import MolecularDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# SMARTS patterns for common motifs
SMARTS_MOTIFS = {
    "benzene": "c1ccccc1",
    "pyridine": "n1ccccc1",
    "pyrimidine": "n1cnccc1",
    "thiophene": "c1ccsc1",
    "furan": "c1ccoc1",
    "carbonyl": "[CX3]=[OX1]",
    "ester": "[CX3](=O)[OX2H0]",
    "amide": "[NX3][CX3](=[OX1])",
    "amine_primary": "[NX3;H2;!$(NC=O)]",
    "amine_secondary": "[NX3;H1;!$(NC=O)]",
    "amine_tertiary": "[NX3;H0;!$(NC=O)]",
    "hydroxyl": "[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "thioether": "[SD2]([#6])[#6]",
    "nitrile": "[NX1]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])[#6]",
}

# RDKit functional group descriptors to use
FUNCTIONAL_GROUPS = [
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_COO",
    "fr_COO2",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_Imine",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Nhpyrrole",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]


def count_functional_groups(mol: Chem.Mol) -> dict[str, int]:
    """Count functional groups using RDKit descriptors.

    Args:
        mol: RDKit molecule.

    Returns:
        Dictionary of functional group counts.
    """
    counts = {}
    for fg_name in FUNCTIONAL_GROUPS:
        try:
            count = getattr(Fragments, fg_name)(mol)
            if count > 0:
                counts[fg_name] = count
        except AttributeError:
            continue
    return counts


def count_smarts_motifs(mol: Chem.Mol) -> dict[str, int]:
    """Count SMARTS pattern matches.

    Args:
        mol: RDKit molecule.

    Returns:
        Dictionary of motif counts.
    """
    counts = {}
    for motif_name, smarts in SMARTS_MOTIFS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) > 0:
                counts[motif_name] = len(matches)
    return counts


def analyze_ring_systems(mol: Chem.Mol) -> dict[str, int]:
    """Analyze ring systems in molecule.

    Args:
        mol: RDKit molecule.

    Returns:
        Dictionary of ring statistics.
    """
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()

    # Count heterocycles manually
    num_heterocycles = 0
    num_aromatic_heterocycles = 0

    for ring in rings:
        # Check if ring contains heteroatom
        has_hetero = False
        is_aromatic = True

        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() not in ['C', 'H']:
                has_hetero = True
            if not atom.GetIsAromatic():
                is_aromatic = False

        if has_hetero:
            num_heterocycles += 1
            if is_aromatic:
                num_aromatic_heterocycles += 1

    stats = {
        "num_rings": len(rings),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
        "num_aliphatic_rings": Descriptors.NumAliphaticRings(mol),
        "num_saturated_rings": Descriptors.NumSaturatedRings(mol),
        "num_heterocycles": num_heterocycles,
        "num_aromatic_heterocycles": num_aromatic_heterocycles,
        "num_spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "num_bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
    }

    # Count ring sizes
    for ring in rings:
        ring_size = len(ring)
        key = f"ring_size_{ring_size}"
        stats[key] = stats.get(key, 0) + 1

    return stats


def extract_brics_fragments(mol: Chem.Mol) -> list[str]:
    """Extract BRICS fragments from molecule.

    Args:
        mol: RDKit molecule.

    Returns:
        List of BRICS fragment SMILES.
    """
    try:
        fragments = BRICS.BRICSDecompose(mol)
        return list(fragments)
    except Exception:
        return []


def analyze_dataset(
    dataset_name: str, num_samples: int = -1, split: str = "train"
) -> dict[str, Any]:
    """Analyze motif distribution in dataset.

    Args:
        dataset_name: Name of dataset (moses, qm9).
        num_samples: Number of samples to analyze (-1 for all).
        split: Dataset split to analyze.

    Returns:
        Motif summary dictionary.
    """
    log.info(f"Loading {dataset_name} dataset ({split} split)...")

    # Load dataset
    if dataset_name == "moses":
        dataset = MolecularDataset.from_moses(
            split=split,
            max_molecules=num_samples if num_samples > 0 else None,
            include_hydrogens=False,
            labeled=False,
        )
    elif dataset_name == "qm9":
        dataset = MolecularDataset.from_qm9(
            split=split,
            max_molecules=num_samples if num_samples > 0 else None,
            include_hydrogens=False,
            labeled=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    log.info(f"Analyzing {len(dataset)} molecules...")

    # Aggregate statistics
    fg_counter = Counter()
    smarts_counter = Counter()
    ring_stats_sum = Counter()
    brics_counter = Counter()

    for i in tqdm(range(len(dataset)), desc="Analyzing motifs"):
        smiles = dataset.smiles_list[i]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        # Count functional groups
        fg_counts = count_functional_groups(mol)
        fg_counter.update(fg_counts)

        # Count SMARTS motifs
        smarts_counts = count_smarts_motifs(mol)
        smarts_counter.update(smarts_counts)

        # Analyze ring systems
        ring_stats = analyze_ring_systems(mol)
        ring_stats_sum.update(ring_stats)

        # Extract BRICS fragments
        fragments = extract_brics_fragments(mol)
        brics_counter.update(fragments)

    # Create summary
    summary = {
        "dataset": dataset_name,
        "split": split,
        "num_molecules": len(dataset),
        "motif_summary": {
            "functional_groups": dict(fg_counter.most_common(20)),
            "smarts_motifs": dict(smarts_counter.most_common(20)),
            "ring_systems": dict(ring_stats_sum),
            "brics_fragments": dict(brics_counter.most_common(20)),
        },
    }

    return summary


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze motif distributions in molecular datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="moses",
        choices=["moses", "qm9"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to analyze (-1 for all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: motif_summary_{dataset}_{split}.json)",
    )

    args = parser.parse_args()

    # Analyze dataset
    summary = analyze_dataset(args.dataset, args.num_samples, args.split)

    # Save to file
    if args.output is None:
        output_path = f"motif_summary_{args.dataset}_{args.split}.json"
    else:
        output_path = args.output

    log.info(f"Saving summary to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Done!")
    log.info(f"\nSummary preview:")
    log.info(f"  Molecules analyzed: {summary['num_molecules']}")
    log.info(
        f"  Functional groups: {len(summary['motif_summary']['functional_groups'])}"
    )
    log.info(f"  SMARTS motifs: {len(summary['motif_summary']['smarts_motifs'])}")
    log.info(f"  BRICS fragments: {len(summary['motif_summary']['brics_fragments'])}")


if __name__ == "__main__":
    main()
