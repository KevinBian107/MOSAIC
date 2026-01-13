#!/usr/bin/env python
"""Visualization script for hierarchical graph tokenization.

This script visualizes the hierarchical decomposition and tokenization
of molecular graphs using the H-SENT tokenizer.

Usage:
    python scripts/visualize_htoken.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
    python scripts/visualize_htoken.py --smiles "c1ccccc1" --output benzene.png
    python scripts/visualize_htoken.py --name cholesterol
    python scripts/visualize_htoken.py --demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.tokenizers.hierarchical import (
    HSENTTokenizer,
    visualize_hierarchy,
    quick_visualize,
)


# Common complex molecules for demos
MOLECULES = {
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "sucrose": "OC[C@H]1O[C@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
    "penicillin_g": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "nicotine": "CN1CCCC1C2=CN=CC=C2",
    "dopamine": "NCCC1=CC(O)=C(O)C=C1",
    "serotonin": "NCCC1=CNC2=CC=C(O)C=C12",
    "adrenaline": "CNCC(O)C1=CC(O)=C(O)C=C1",
}


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    if not edges:
        return Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=mol.GetNumAtoms(),
        )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())


def visualize_molecule_hierarchy(
    smiles: str,
    name: str | None = None,
    output: str | None = None,
    show: bool = True,
    seed: int = 42,
) -> plt.Figure | None:
    """Visualize hierarchical tokenization of a molecule.

    Args:
        smiles: SMILES string.
        name: Optional molecule name for title.
        output: Output path to save figure.
        show: Whether to display the plot.
        seed: Random seed for reproducibility.

    Returns:
        Matplotlib figure or None if invalid SMILES.
    """
    from src.tokenizers.hierarchical.visualization import (
        _plot_graph_communities,
        _plot_block_matrix,
        _plot_hierarchy_tree,
        _plot_tokens,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    data = smiles_to_graph(smiles)
    if data is None:
        return None

    # Create tokenizer
    tokenizer = HSENTTokenizer(seed=seed)
    tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    # Build hierarchy
    hg = tokenizer.coarsener.build_hierarchy(data)
    tokens = tokenizer.tokenize_hierarchy(hg)

    # Create title
    title = name or smiles[:50]
    if len(smiles) > 50:
        title += "..."
    title += f"  ({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds, "
    title += f"{hg.num_communities} communities, {len(tokens)} tokens)"

    # Create 2-row visualization: 3 panels on top, tokens on bottom
    fig = plt.figure(figsize=(18, 10))

    # Top row: 3 panels
    ax1 = fig.add_subplot(2, 3, 1)
    _plot_graph_communities(ax1, hg, data)
    ax1.set_title("Graph with Communities")

    ax2 = fig.add_subplot(2, 3, 2)
    _plot_block_matrix(ax2, hg)
    ax2.set_title("Block Matrix")

    ax3 = fig.add_subplot(2, 3, 3)
    _plot_hierarchy_tree(ax3, hg)
    ax3.set_title("Hierarchy Structure")

    # Bottom row: token sequence (spanning all 3 columns)
    ax4 = fig.add_subplot(2, 1, 2)
    _plot_tokens(ax4, tokens.tolist(), tokenizer, max_tokens=150)
    ax4.set_title(f"Token Sequence ({len(tokens)} tokens)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")

    if show:
        plt.show()

    return fig


def visualize_molecule_full(
    smiles: str,
    name: str | None = None,
    output: str | None = None,
    show: bool = True,
    seed: int = 42,
) -> plt.Figure | None:
    """Full 4-panel visualization of a molecule's hierarchical tokenization.

    Args:
        smiles: SMILES string.
        name: Optional molecule name for title.
        output: Output path to save figure.
        show: Whether to display the plot.
        seed: Random seed for reproducibility.

    Returns:
        Matplotlib figure or None if invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    data = smiles_to_graph(smiles)
    if data is None:
        return None

    tokenizer = HSENTTokenizer(seed=seed)
    tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    fig = quick_visualize(data, tokenizer, save_path=output)

    # Add title
    title = name or smiles[:40]
    fig.suptitle(
        f"{title} - Hierarchical Tokenization",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")

    if show:
        plt.show()

    return fig


def run_demo(output_dir: str | None = None, show: bool = True):
    """Run demo with several complex molecules."""
    demo_molecules = [
        ("caffeine", MOLECULES["caffeine"]),
        ("aspirin", MOLECULES["aspirin"]),
        ("cholesterol", MOLECULES["cholesterol"]),
        ("penicillin_g", MOLECULES["penicillin_g"]),
        ("morphine", MOLECULES["morphine"]),
        ("dopamine", MOLECULES["dopamine"]),
    ]

    for name, smiles in demo_molecules:
        print(f"\n{'='*60}")
        print(f"Visualizing: {name}")
        print(f"SMILES: {smiles}")
        print("=" * 60)

        output = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output = f"{output_dir}/{name}_hierarchy.png"

        visualize_molecule_hierarchy(
            smiles,
            name=name.replace("_", " ").title(),
            output=output,
            show=show,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical graph tokenization of molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --smiles "c1ccccc1"
    %(prog)s --name caffeine --output caffeine.png
    %(prog)s --name cholesterol --full
    %(prog)s --demo --output-dir ./figures
    %(prog)s --list

Available molecules: """ + ", ".join(sorted(MOLECULES.keys())),
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES string to visualize",
    )
    parser.add_argument(
        "--name",
        type=str,
        help=f"Molecule name (one of: {', '.join(sorted(MOLECULES.keys()))})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for demo mode",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full 4-panel visualization",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with multiple complex molecules",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available molecule names",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot (only save)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.list:
        print("Available molecules:")
        for name, smiles in sorted(MOLECULES.items()):
            mol = Chem.MolFromSmiles(smiles)
            atoms = mol.GetNumAtoms() if mol else "?"
            print(f"  {name:15} ({atoms:2} atoms): {smiles}")
        return

    if args.demo:
        run_demo(output_dir=args.output_dir, show=not args.no_show)
        return

    # Get SMILES from name or direct input
    smiles = args.smiles
    name = None
    if args.name:
        if args.name.lower() not in MOLECULES:
            print(f"Unknown molecule: {args.name}")
            print(f"Available: {', '.join(sorted(MOLECULES.keys()))}")
            return
        smiles = MOLECULES[args.name.lower()]
        name = args.name.replace("_", " ").title()

    if not smiles:
        print("Please provide --smiles or --name. Use --list to see available molecules.")
        parser.print_help()
        return

    # Visualize
    viz_func = visualize_molecule_full if args.full else visualize_molecule_hierarchy
    viz_func(
        smiles,
        name=name,
        output=args.output,
        show=not args.no_show,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
