#!/usr/bin/env python
"""Visualization script for molecular motifs.

This script visualizes molecules and their detected motifs (functional groups,
ring systems, BRICS fragments) to help understand the molecular structure.

Usage:
    python scripts/visualize_motifs.py --smiles "CCO"
    python scripts/visualize_motifs.py --smiles "c1ccccc1O" --output benzene_motifs.png
    python scripts/visualize_motifs.py --dataset moses --num_molecules 6
    python scripts/visualize_motifs.py --file molecules.txt --output motifs.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, BRICS, Fragments, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from src.evaluation.motif_distribution import (
    MOLECULAR_MOTIFS,
    get_brics_fragments,
    get_functional_group_counts,
    get_motif_counts,
    get_ring_system_info,
)


def highlight_motif(mol, smarts_pattern: str, color: tuple = (1.0, 0.8, 0.8)):
    """Find atoms matching a SMARTS pattern for highlighting.

    Args:
        mol: RDKit molecule.
        smarts_pattern: SMARTS pattern to match.
        color: RGB color tuple for highlighting.

    Returns:
        List of atom indices matching the pattern.
    """
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern is None:
        return []

    matches = mol.GetSubstructMatches(pattern)
    atoms = set()
    for match in matches:
        atoms.update(match)
    return list(atoms)


def visualize_molecule_with_motifs(
    smiles: str,
    output_path: Optional[str] = None,
    figsize: tuple = (16, 12),
    show_plot: bool = True,
) -> Optional[plt.Figure]:
    """Visualize a molecule with its detected motifs.

    Args:
        smiles: SMILES string of the molecule.
        output_path: Optional path to save the figure.
        figsize: Figure size.
        show_plot: Whether to display the plot.

    Returns:
        Matplotlib figure or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Get all motif information
    smarts_motifs = get_motif_counts(smiles)
    functional_groups = get_functional_group_counts(smiles)
    ring_info = get_ring_system_info(smiles)
    brics_frags = get_brics_fragments(smiles)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Main molecule visualization (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    img = Draw.MolToImage(mol, size=(400, 400))
    ax1.imshow(img)
    ax1.set_title(f"Molecule: {smiles[:40]}{'...' if len(smiles) > 40 else ''}")
    ax1.axis("off")

    # Aromatic rings highlighted (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    aromatic_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atoms.append(atom.GetIdx())

    if aromatic_atoms:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        drawer.drawOptions().addAtomIndices = False
        highlight_colors = {i: (1.0, 0.6, 0.6) for i in aromatic_atoms}
        drawer.DrawMolecule(
            mol,
            highlightAtoms=aromatic_atoms,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        # Convert to image
        import io
        from PIL import Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        ax2.imshow(img)
    else:
        img = Draw.MolToImage(mol, size=(400, 400))
        ax2.imshow(img)
    ax2.set_title(f"Aromatic Atoms ({len(aromatic_atoms)} atoms)")
    ax2.axis("off")

    # Functional groups highlighted (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    # Highlight hydroxyl and carbonyl groups
    hydroxyl_atoms = highlight_motif(mol, "[OX2H]")
    carbonyl_atoms = highlight_motif(mol, "[CX3]=[OX1]")
    amine_atoms = highlight_motif(mol, "[NX3;H2,H1,H0;!$(NC=O)]")
    all_fg_atoms = list(set(hydroxyl_atoms + carbonyl_atoms + amine_atoms))

    if all_fg_atoms:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        highlight_colors = {}
        for i in hydroxyl_atoms:
            highlight_colors[i] = (0.6, 0.6, 1.0)  # Blue for OH
        for i in carbonyl_atoms:
            highlight_colors[i] = (1.0, 0.8, 0.6)  # Orange for C=O
        for i in amine_atoms:
            highlight_colors[i] = (0.6, 1.0, 0.6)  # Green for amines
        drawer.DrawMolecule(
            mol,
            highlightAtoms=all_fg_atoms,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        import io
        from PIL import Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        ax3.imshow(img)
    else:
        img = Draw.MolToImage(mol, size=(400, 400))
        ax3.imshow(img)
    ax3.set_title("Functional Groups (OH=blue, C=O=orange, N=green)")
    ax3.axis("off")

    # SMARTS motifs bar chart (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    if smarts_motifs:
        names = list(smarts_motifs.keys())[:10]
        counts = [smarts_motifs[n] for n in names]
        bars = ax4.barh(names, counts, color="steelblue")
        ax4.set_xlabel("Count")
        ax4.set_title("SMARTS Motifs Detected")
        ax4.invert_yaxis()
    else:
        ax4.text(0.5, 0.5, "No SMARTS motifs detected",
                 ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("SMARTS Motifs Detected")

    # Ring system info (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    ring_text = []
    ring_text.append(f"Total Rings: {ring_info.get('num_rings', 0)}")
    ring_text.append(f"Aromatic Rings: {ring_info.get('num_aromatic_rings', 0)}")
    ring_text.append(f"Aliphatic Rings: {ring_info.get('num_aliphatic_rings', 0)}")
    ring_text.append(f"Heterocycles: {ring_info.get('num_heterocycles', 0)}")
    ring_text.append("")
    ring_text.append("Ring Sizes:")
    for size in [3, 4, 5, 6, 7, 8]:
        count = ring_info.get(f"ring_size_{size}", 0)
        if count > 0:
            ring_text.append(f"  {size}-membered: {count}")

    ax5.text(0.1, 0.9, "\n".join(ring_text),
             ha="left", va="top", transform=ax5.transAxes,
             fontsize=10, family="monospace")
    ax5.set_title("Ring System Information")
    ax5.axis("off")

    # BRICS fragments and functional groups (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    info_text = []
    info_text.append("BRICS Fragments:")
    for i, frag in enumerate(brics_frags[:5]):
        info_text.append(f"  {frag[:30]}{'...' if len(frag) > 30 else ''}")
    if len(brics_frags) > 5:
        info_text.append(f"  ... and {len(brics_frags) - 5} more")

    info_text.append("")
    info_text.append("Top Functional Groups:")
    fg_sorted = sorted(functional_groups.items(), key=lambda x: -x[1])[:5]
    for name, count in fg_sorted:
        # Clean up the name
        clean_name = name.replace("fr_", "").replace("_", " ")
        info_text.append(f"  {clean_name}: {count}")

    ax6.text(0.05, 0.95, "\n".join(info_text),
             ha="left", va="top", transform=ax6.transAxes,
             fontsize=9, family="monospace")
    ax6.set_title("Fragments & Functional Groups")
    ax6.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    if show_plot:
        plt.show()

    return fig


def visualize_multiple_molecules(
    smiles_list: list[str],
    output_path: Optional[str] = None,
    figsize: tuple = (16, 10),
    show_plot: bool = True,
) -> Optional[plt.Figure]:
    """Visualize multiple molecules with their motif summaries.

    Args:
        smiles_list: List of SMILES strings.
        output_path: Optional path to save the figure.
        figsize: Figure size.
        show_plot: Whether to display the plot.

    Returns:
        Matplotlib figure or None.
    """
    valid_mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            valid_mols.append(mol)
            valid_smiles.append(smi)

    if not valid_mols:
        print("No valid molecules to visualize")
        return None

    n_mols = len(valid_mols)
    n_cols = min(3, n_mols)
    n_rows = (n_mols + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_mols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (mol, smi) in enumerate(zip(valid_mols, valid_smiles)):
        ax = axes[i]

        # Get motif counts for legend
        smarts_motifs = get_motif_counts(smi)
        ring_info = get_ring_system_info(smi)

        # Create molecule image with aromatic highlighting
        aromatic_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]

        if aromatic_atoms:
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            highlight_colors = {i: (1.0, 0.7, 0.7) for i in aromatic_atoms}
            drawer.DrawMolecule(
                mol,
                highlightAtoms=aromatic_atoms,
                highlightAtomColors=highlight_colors,
            )
            drawer.FinishDrawing()
            import io
            from PIL import Image
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))
            ax.imshow(img)
        else:
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)

        # Build title with motif info
        title_parts = [smi[:25] + ("..." if len(smi) > 25 else "")]
        motif_info = []
        if ring_info.get("num_aromatic_rings", 0) > 0:
            motif_info.append(f"Ar:{ring_info['num_aromatic_rings']}")
        if "benzene" in smarts_motifs:
            motif_info.append(f"Bz:{smarts_motifs['benzene']}")
        if "hydroxyl" in smarts_motifs:
            motif_info.append(f"OH:{smarts_motifs['hydroxyl']}")
        if "carbonyl" in smarts_motifs:
            motif_info.append(f"C=O:{smarts_motifs['carbonyl']}")

        if motif_info:
            title_parts.append(" | ".join(motif_info))

        ax.set_title("\n".join(title_parts), fontsize=9)
        ax.axis("off")

    # Hide empty subplots
    for i in range(n_mols, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    if show_plot:
        plt.show()

    return fig


def load_smiles_from_dataset(dataset_name: str, num_molecules: int = 10) -> list[str]:
    """Load SMILES from a dataset.

    Args:
        dataset_name: Name of dataset ('moses' or 'qm9').
        num_molecules: Number of molecules to load.

    Returns:
        List of SMILES strings.
    """
    if dataset_name == "moses":
        try:
            import moses
            smiles = moses.get_dataset("train")[:num_molecules]
            return smiles
        except ImportError:
            print("MOSES package not installed. Install with: pip install molsets")
            return []
    elif dataset_name == "qm9":
        try:
            from torch_geometric.datasets import QM9
            dataset = QM9(root="data/qm9")
            smiles = []
            for i, data in enumerate(dataset):
                if i >= num_molecules:
                    break
                if hasattr(data, "smiles"):
                    smiles.append(data.smiles)
            return smiles
        except Exception as e:
            print(f"Error loading QM9: {e}")
            return []
    else:
        print(f"Unknown dataset: {dataset_name}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Visualize molecular motifs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES string to visualize",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing SMILES strings (one per line)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["moses", "qm9"],
        help="Load molecules from dataset",
    )
    parser.add_argument(
        "--num_molecules",
        type=int,
        default=6,
        help="Number of molecules to visualize from dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for saving visualization",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)",
    )

    args = parser.parse_args()

    if args.smiles:
        # Single molecule detailed visualization
        visualize_molecule_with_motifs(
            args.smiles,
            output_path=args.output,
            show_plot=not args.no_show,
        )
    elif args.file:
        # Load from file
        with open(args.file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        visualize_multiple_molecules(
            smiles_list[:args.num_molecules],
            output_path=args.output,
            show_plot=not args.no_show,
        )
    elif args.dataset:
        # Load from dataset
        smiles_list = load_smiles_from_dataset(args.dataset, args.num_molecules)
        if smiles_list:
            visualize_multiple_molecules(
                smiles_list,
                output_path=args.output,
                show_plot=not args.no_show,
            )
    else:
        # Demo with example molecules
        print("No input provided. Running demo with example molecules...")
        demo_smiles = [
            "c1ccccc1O",  # Phenol
            "CC(=O)O",  # Acetic acid
            "CCO",  # Ethanol
            "c1ccc2ccccc2c1",  # Naphthalene
            "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]

        print("\nDetailed view of Caffeine:")
        visualize_molecule_with_motifs(
            demo_smiles[-1],
            output_path=args.output.replace(".png", "_detailed.png") if args.output else None,
            show_plot=not args.no_show,
        )

        print("\nMultiple molecule view:")
        visualize_multiple_molecules(
            demo_smiles,
            output_path=args.output,
            show_plot=not args.no_show,
        )


if __name__ == "__main__":
    main()
