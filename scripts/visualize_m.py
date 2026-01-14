#!/usr/bin/env python
"""Visualize the motif co-membership matrix M.

This script shows the motif co-membership matrix M alongside the molecular
graph with detected motifs highlighted. M[i,j] represents the number of
motifs that contain both atoms i and j.

Usage:
    python scripts/visualize_m.py --name cholesterol
    python scripts/visualize_m.py --smiles "c1ccc2ccccc2c1" --name naphthalene
    python scripts/visualize_m.py --demo --output-dir ./figures
    python scripts/visualize_m.py --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from scipy.spatial import ConvexHull
from torch_geometric.data import Data

from src.tokenizers.hierarchical import (
    MotifInstance,
    compute_motif_affinity_matrix,
    detect_motifs_from_smiles,
)

# Colors for motif outlines
MOTIF_COLORS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
]

# Pre-defined complex molecules
MOLECULES = {
    # Fused ring systems
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc3ccccc32",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "fluorene": "c1ccc2c(c1)Cc3ccccc32",
    # Steroids - multiple fused rings
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
    # Alkaloids
    "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "strychnine": "C1CN2CC3=CCO[C@H]4CC(=O)N5[C@H]6[C@H]4[C@H]3C[C@H]2[C@]61C7=CC=CC=C75",
    "quinine": "COC1=CC2=C(C=CN=C2C=C1)[C@@H]([C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "nicotine": "CN1CCC[C@H]1C2=CN=CC=C2",
    # Complex drugs with multiple aromatics
    "atorvastatin": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "diazepam": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "celecoxib": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    # Natural products
    "quercetin": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
    "resveratrol": "C1=CC(=CC=C1C=CC2=CC(=CC(=C2)O)O)O",
    "curcumin": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
    # Antibiotics
    "penicillin_g": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    "ciprofloxacin": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
}


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert SMILES to PyG Data object with smiles attribute."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())
    data.smiles = smiles
    return data


def get_atom_symbols(smiles: str) -> list[str]:
    """Get atom symbols for labeling."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def plot_graph_with_motifs(
    ax: plt.Axes,
    data: Data,
    motifs: list[MotifInstance],
    node_size: int = 500,
    edge_width: float = 1.5,
) -> dict:
    """Plot graph with motif outlines and labels.

    Returns:
        Dictionary mapping motif names to their colors.
    """
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))

    if data.edge_index.numel() > 0:
        ei = data.edge_index.numpy()
        for e in range(ei.shape[1]):
            src, dst = int(ei[0, e]), int(ei[1, e])
            if src < dst:  # Avoid duplicate edges
                G.add_edge(src, dst)

    # Compute layout
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(data.num_nodes))
    else:
        pos = nx.circular_layout(G)

    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, width=edge_width, edge_color="gray")

    # Determine which atoms are in which motifs for coloring
    atom_motif_count = np.zeros(data.num_nodes)
    for motif in motifs:
        for atom in motif.atom_indices:
            atom_motif_count[atom] += 1

    # Color nodes: atoms in motifs get colored, others are light gray
    node_colors = []
    for i in range(data.num_nodes):
        if atom_motif_count[i] > 0:
            # Gradient based on overlap count
            intensity = min(1.0, 0.3 + 0.2 * atom_motif_count[i])
            node_colors.append((0.2, 0.6, 0.9, intensity))
        else:
            node_colors.append((0.85, 0.85, 0.85, 1.0))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    # Draw motif outlines and labels
    motif_colors_used = {}
    motif_type_idx = {}

    for motif in motifs:
        # Assign color based on motif type
        if motif.name not in motif_type_idx:
            motif_type_idx[motif.name] = len(motif_type_idx)
        color_idx = motif_type_idx[motif.name] % len(MOTIF_COLORS)
        color = MOTIF_COLORS[color_idx]
        motif_colors_used[motif.name] = color

        # Get positions of motif atoms
        atoms = sorted(motif.atom_indices)
        if len(atoms) < 3:
            # For small motifs, draw circles
            for atom in atoms:
                circle = plt.Circle(
                    pos[atom],
                    0.15,
                    fill=False,
                    edgecolor=color,
                    linewidth=3,
                    linestyle="--",
                )
                ax.add_patch(circle)
        else:
            # For larger motifs, draw convex hull
            points = np.array([pos[a] for a in atoms])

            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])

                # Add padding
                centroid = points.mean(axis=0)
                padded_points = centroid + 1.15 * (hull_points - centroid)

                ax.plot(
                    padded_points[:, 0],
                    padded_points[:, 1],
                    color=color,
                    linewidth=3,
                    linestyle="--",
                    alpha=0.8,
                )

                # Add motif label at centroid
                ax.annotate(
                    motif.name,
                    centroid,
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
            except Exception:
                # Fallback for degenerate cases
                pass

    ax.set_aspect("equal")
    ax.axis("off")

    return motif_colors_used


def plot_m_matrix(
    ax: plt.Axes,
    M: np.ndarray,
    motifs: list[MotifInstance],
    atom_symbols: list[str] | None = None,
) -> None:
    """Plot the motif co-membership matrix M as a heatmap."""
    n = M.shape[0]

    # Create custom colormap (white to blue)
    cmap = plt.cm.Blues

    # Plot heatmap
    im = ax.imshow(M, cmap=cmap, aspect="equal", vmin=0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-membership count", fontsize=10)

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)

    # Add atom labels
    if n <= 30:
        if atom_symbols and len(atom_symbols) == n:
            labels = [f"{i}\n{atom_symbols[i]}" for i in range(n)]
        else:
            labels = [str(i) for i in range(n)]

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        # For large matrices, show fewer labels
        step = max(1, n // 10)
        ax.set_xticks(range(0, n, step))
        ax.set_yticks(range(0, n, step))

    # Annotate non-zero values for small matrices
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if M[i, j] > 0:
                    text_color = "white" if M[i, j] > M.max() / 2 else "black"
                    ax.text(
                        j, i, f"{int(M[i, j])}",
                        ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold"
                    )

    ax.set_xlabel("Atom index", fontsize=10)
    ax.set_ylabel("Atom index", fontsize=10)


def visualize_m(
    smiles: str,
    name: str | None = None,
    output: str | None = None,
    show: bool = True,
    normalize: bool = False,
) -> plt.Figure | None:
    """Visualize motif co-membership matrix M for a molecule.

    Args:
        smiles: SMILES string.
        name: Optional molecule name for title.
        output: Output path to save figure.
        show: Whether to display the plot.
        normalize: Whether to normalize M by motif size.

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

    # Detect motifs
    motifs = detect_motifs_from_smiles(smiles)
    if not motifs:
        print(f"No motifs detected in {name or smiles}")

    # Compute M matrix
    M = compute_motif_affinity_matrix(data.num_nodes, motifs, normalize_by_size=normalize)

    # Get atom symbols for labeling
    atom_symbols = get_atom_symbols(smiles)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Graph with motifs
    motif_colors = plot_graph_with_motifs(axes[0], data, motifs)

    # Add legend for motifs
    if motif_colors:
        patches = [
            mpatches.Patch(
                facecolor="white",
                edgecolor=color,
                linewidth=2,
                linestyle="--",
                label=name,
            )
            for name, color in motif_colors.items()
        ]
        axes[0].legend(handles=patches, loc="upper left", fontsize=9, title="Motifs")

    # Build motif summary
    motif_summary = {}
    for m in motifs:
        motif_summary[m.name] = motif_summary.get(m.name, 0) + 1
    motif_str = ", ".join(f"{k}: {v}" for k, v in sorted(motif_summary.items()))
    if not motif_str:
        motif_str = "none detected"

    axes[0].set_title(
        f"Molecular Graph with Detected Motifs\n({motif_str})",
        fontsize=11,
        fontweight="bold",
    )

    # Right: M matrix
    plot_m_matrix(axes[1], M, motifs, atom_symbols)

    norm_str = " (normalized)" if normalize else ""
    axes[1].set_title(
        f"Motif Co-membership Matrix M{norm_str}\n"
        f"M[i,j] = # motifs containing both atoms i and j",
        fontsize=11,
        fontweight="bold",
    )

    # Suptitle
    title = name or smiles[:50]
    if len(smiles) > 50:
        title += "..."
    fig.suptitle(
        f"{title}\n({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds, "
        f"{len(motifs)} motif instances)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run_demo(output_dir: str | None = None, show: bool = True, normalize: bool = False):
    """Run demo with complex molecules."""
    demo_molecules = [
        "naphthalene",
        "pyrene",
        "cholesterol",
        "morphine",
        "strychnine",
        "atorvastatin",
        "quercetin",
        "sildenafil",
        "ciprofloxacin",
    ]

    for name in demo_molecules:
        if name not in MOLECULES:
            continue

        smiles = MOLECULES[name]
        print(f"\n{'='*60}")
        print(f"Visualizing M matrix: {name}")
        print(f"SMILES: {smiles}")
        print("=" * 60)

        output = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output = f"{output_dir}/{name}_m_matrix.png"

        visualize_m(
            smiles,
            name=name.replace("_", " ").title(),
            output=output,
            show=show,
            normalize=normalize,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the motif co-membership matrix M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    %(prog)s --name cholesterol
    %(prog)s --smiles "c1ccc2ccccc2c1" --name naphthalene
    %(prog)s --name morphine --normalize
    %(prog)s --demo --output-dir ./figures
    %(prog)s --list

The M matrix shows M[i,j] = number of detected motifs containing both atoms
i and j. This matrix is added to the adjacency matrix (scaled by alpha) to
bias spectral clustering toward keeping motif atoms together.

Available molecules: {', '.join(sorted(MOLECULES.keys()))}
""",
    )
    parser.add_argument("--smiles", type=str, help="SMILES string to visualize")
    parser.add_argument("--name", type=str, help="Molecule name from predefined list")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--output-dir", type=str, help="Output directory for demo")
    parser.add_argument("--demo", action="store_true", help="Run demo with complex molecules")
    parser.add_argument("--list", action="store_true", help="List available molecules with motif counts")
    parser.add_argument("--no-show", action="store_true", help="Don't display (only save)")
    parser.add_argument("--normalize", action="store_true", help="Normalize M by motif size")

    args = parser.parse_args()

    if args.list:
        print("Available molecules with detected motifs:\n")
        print(f"{'Name':<20} {'Atoms':<6} {'Motifs':<8} {'Types'}")
        print("-" * 70)
        for name, smiles in sorted(MOLECULES.items()):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            motifs = detect_motifs_from_smiles(smiles)
            motif_types = {}
            for m in motifs:
                motif_types[m.name] = motif_types.get(m.name, 0) + 1
            types_str = ", ".join(f"{k}:{v}" for k, v in sorted(motif_types.items()))
            if not types_str:
                types_str = "-"
            print(f"{name:<20} {mol.GetNumAtoms():<6} {len(motifs):<8} {types_str}")
        return

    if args.demo:
        run_demo(
            output_dir=args.output_dir,
            show=not args.no_show,
            normalize=args.normalize,
        )
        return

    # Get SMILES
    smiles = args.smiles
    name = args.name

    if name and not smiles:
        if name.lower() not in MOLECULES:
            print(f"Unknown molecule: {name}")
            print(f"Available: {', '.join(sorted(MOLECULES.keys()))}")
            return
        smiles = MOLECULES[name.lower()]
        name = name.replace("_", " ").title()

    if not smiles:
        print("Please provide --smiles or --name")
        parser.print_help()
        return

    visualize_m(
        smiles,
        name=name,
        output=args.output,
        show=not args.no_show,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
