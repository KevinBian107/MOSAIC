#!/usr/bin/env python
"""Compare standard vs motif-aware coarsening visualization.

This script creates a side-by-side comparison of hierarchical tokenization
with and without motif-aware coarsening, showing how motif affinity helps
preserve ring structures during graph partitioning.

Usage:
    python scripts/visualize_motif_coarsening.py --smiles "c1ccccc1" --name benzene
    python scripts/visualize_motif_coarsening.py --name cholesterol --alpha 5.0
    python scripts/visualize_motif_coarsening.py --demo --output-dir ./figures
    python scripts/visualize_motif_coarsening.py --list
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
    HSENTTokenizer,
    MotifInstance,
    compute_motif_cohesion,
    detect_motifs_from_smiles,
)
from src.tokenizers.hierarchical.structures import HierarchicalGraph
from src.tokenizers.hierarchical.visualization import (
    COMMUNITY_COLORS,
    _plot_block_matrix,
)

# Colors for motif outlines (distinct from community colors)
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


# Pre-defined molecules for demos
MOLECULES = {
    # Simple aromatics
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc3ccccc32",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    # Common drugs
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "diazepam": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "naproxen": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
    "omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC",
    "sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    # Steroids - multiple fused rings
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
    "cortisol": "CC12CCC(=O)C=C1CCC3C2C(O)CC4(C3CCC4(O)C(=O)CO)C",
    # Alkaloids - complex ring systems
    "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "codeine": "COC1=C2C3=C(C[C@@H]4[C@@]5([C@H]3OC2=C(C=C1)O)CCN([C@@H]4C=C5)C)O",
    "strychnine": "C1CN2CC3=CCO[C@H]4CC(=O)N5[C@H]6[C@H]4[C@H]3C[C@H]2[C@]61C7=CC=CC=C75",
    "quinine": "COC1=CC2=C(C=CN=C2C=C1)[C@@H]([C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)O",
    "nicotine": "CN1CCC[C@H]1C2=CN=CC=C2",
    # Complex drugs
    "atorvastatin": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "celecoxib": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "losartan": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
    # Antibiotics
    "penicillin_g": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    "amoxicillin": "CC1(C)SC2C(NC(=O)C(N)C3=CC=C(O)C=C3)C(=O)N2C1C(=O)O",
    "ciprofloxacin": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    # Natural products
    "quercetin": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
    "resveratrol": "C1=CC(=CC=C1C=CC2=CC(=CC(=C2)O)O)O",
    "curcumin": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
    "capsaicin": "COC1=C(C=CC(=C1)CNC(=O)CCCC=CC(C)C)O",
    # Neurotransmitters
    "dopamine": "NCCC1=CC(O)=C(O)C=C1",
    "serotonin": "NCCC1=CNC2=CC=C(O)C=C12",
    "adrenaline": "CNCC(C1=CC(=C(C=C1)O)O)O",
    "melatonin": "CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC",
    # Vitamins
    "vitamin_e": "CC1=C(C(=C(C2=C1OC(CC2)(C)CCCC(C)CCCC(C)CCCC(C)C)C)O)C",
    "folic_acid": "C1=CC(=CC=C1C(=O)NC(CCC(=O)O)C(=O)O)NCC2=CN=C3C(=N2)C(=O)NC(=N3)N",
}


def _plot_graph_with_motifs(
    ax: plt.Axes,
    hg: HierarchicalGraph,
    data: Data,
    motifs: list[MotifInstance],
    node_size: int = 400,
    edge_width: float = 1.5,
) -> dict:
    """Plot graph with community coloring and motif outlines.

    Args:
        ax: Matplotlib axes.
        hg: Hierarchical graph with community assignments.
        data: Original graph data.
        motifs: List of detected motifs to highlight.
        node_size: Size of nodes.
        edge_width: Width of edges.

    Returns:
        Dictionary mapping motif names to their colors for legend.
    """
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(hg.num_nodes))

    # Add edges from reconstructed graph
    reconstructed = hg.reconstruct()
    if reconstructed.edge_index.numel() > 0:
        ei = reconstructed.edge_index.numpy()
        for e in range(ei.shape[1]):
            G.add_edge(int(ei[0, e]), int(ei[1, e]))

    # Node colors based on community assignment
    node_colors = [
        COMMUNITY_COLORS[hg.community_assignment[n] % len(COMMUNITY_COLORS)]
        for n in range(hg.num_nodes)
    ]

    # Compute layout
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(hg.num_nodes))
    else:
        pos = nx.circular_layout(G)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=edge_width)

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
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    # Draw motif outlines and labels
    motif_colors_used = {}
    motif_type_idx = {}  # Track color index per motif type

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
            # For small motifs, just draw a circle around them
            for atom in atoms:
                circle = plt.Circle(
                    pos[atom],
                    0.12,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                    linestyle="--",
                )
                ax.add_patch(circle)
        else:
            # For larger motifs, draw convex hull
            points = np.array([pos[a] for a in atoms])

            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                # Close the hull
                hull_points = np.vstack([hull_points, hull_points[0]])

                # Add padding to hull
                center = points.mean(axis=0)
                padded_points = center + 1.15 * (hull_points - center)

                ax.plot(
                    padded_points[:, 0],
                    padded_points[:, 1],
                    color=color,
                    linewidth=2.5,
                    linestyle="--",
                    alpha=0.8,
                )
                ax.fill(
                    padded_points[:, 0],
                    padded_points[:, 1],
                    color=color,
                    alpha=0.1,
                )
            except Exception:
                # Fallback if hull fails (collinear points)
                for atom in atoms:
                    circle = plt.Circle(
                        pos[atom],
                        0.1,
                        fill=False,
                        edgecolor=color,
                        linewidth=2,
                        linestyle="--",
                    )
                    ax.add_patch(circle)

        # Add motif label at centroid
        centroid = np.mean([pos[a] for a in atoms], axis=0)
        # Offset label slightly
        offset = np.array([0.0, 0.15])
        label_pos = centroid + offset

        ax.annotate(
            motif.name,
            xy=centroid,
            xytext=label_pos,
            fontsize=7,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=color),
        )

    # Add community legend
    unique_communities = sorted(set(hg.community_assignment))
    comm_handles = [
        mpatches.Patch(
            color=COMMUNITY_COLORS[c % len(COMMUNITY_COLORS)],
            label=f"Community {c}",
        )
        for c in unique_communities
    ]

    # Add motif legend
    motif_handles = [
        mpatches.Patch(
            facecolor=color,
            edgecolor=color,
            alpha=0.3,
            label=name,
            linestyle="--",
        )
        for name, color in motif_colors_used.items()
    ]

    all_handles = comm_handles + motif_handles
    if len(all_handles) <= 8:
        ax.legend(
            handles=all_handles,
            loc="upper left",
            fontsize=7,
            framealpha=0.9,
        )

    ax.set_aspect("equal")
    ax.axis("off")

    return motif_colors_used


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


def compare_coarsening(
    smiles: str,
    name: str,
    alpha: float = 10.0,
    min_community_size: int = 2,
    output_path: str | None = None,
    show: bool = True,
    seed: int = 42,
) -> dict:
    """Compare standard vs motif-aware coarsening on a molecule.

    Args:
        smiles: SMILES string of the molecule.
        name: Display name for the molecule.
        alpha: Motif affinity weight (higher = stronger motif preference).
        min_community_size: Minimum nodes per community.
        output_path: Path to save the figure.
        show: Whether to display the plot.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with comparison metrics.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return {}

    data = smiles_to_graph(smiles)
    if data is None:
        return {}

    # Detect motifs for cohesion comparison
    motifs = detect_motifs_from_smiles(smiles)
    motif_summary = {}
    for m in motifs:
        motif_summary[m.name] = motif_summary.get(m.name, 0) + 1

    # Create tokenizers
    standard_tokenizer = HSENTTokenizer(
        seed=seed,
        motif_aware=False,
        min_community_size=min_community_size,
    )
    standard_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    motif_tokenizer = HSENTTokenizer(
        seed=seed,
        motif_aware=True,
        motif_alpha=alpha,
        min_community_size=min_community_size,
    )
    motif_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    # Build hierarchies
    std_hg = standard_tokenizer.coarsener.build_hierarchy(data)
    motif_hg = motif_tokenizer.coarsener.build_hierarchy(data)

    # Compute cohesion - extract communities from partitions
    std_communities = [set(p.global_node_indices) for p in std_hg.partitions]
    motif_communities = [set(p.global_node_indices) for p in motif_hg.partitions]

    std_cohesion = compute_motif_cohesion(std_communities, motifs)
    motif_cohesion = compute_motif_cohesion(motif_communities, motifs)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Standard coarsening with motif labels
    _plot_graph_with_motifs(axes[0, 0], std_hg, data, motifs)
    axes[0, 0].set_title(
        f"Standard Spectral Coarsening\n"
        f"{std_hg.num_communities} communities, Cohesion: {std_cohesion:.0%}",
        fontsize=11,
    )

    _plot_block_matrix(axes[0, 1], std_hg)
    axes[0, 1].set_title("Standard: Block Matrix", fontsize=11)

    # Motif-aware coarsening with motif labels
    _plot_graph_with_motifs(axes[1, 0], motif_hg, data, motifs)
    axes[1, 0].set_title(
        f"Motif-Aware Coarsening (\u03b1={alpha})\n"
        f"{motif_hg.num_communities} communities, Cohesion: {motif_cohesion:.0%}",
        fontsize=11,
    )

    _plot_block_matrix(axes[1, 1], motif_hg)
    axes[1, 1].set_title("Motif-Aware: Block Matrix", fontsize=11)

    # Add suptitle with molecule info
    motif_str = ", ".join(f"{k}:{v}" for k, v in sorted(motif_summary.items()))
    if not motif_str:
        motif_str = "none detected"

    title = f"{name}: {smiles[:60]}{'...' if len(smiles) > 60 else ''}\n"
    title += f"Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}, "
    title += f"Motifs: {motif_str}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "name": name,
        "smiles": smiles,
        "num_atoms": mol.GetNumAtoms(),
        "num_motifs": len(motifs),
        "motif_summary": motif_summary,
        "std_communities": std_hg.num_communities,
        "motif_communities": motif_hg.num_communities,
        "std_cohesion": std_cohesion,
        "motif_cohesion": motif_cohesion,
        "cohesion_improvement": motif_cohesion - std_cohesion,
    }


def run_demo(
    output_dir: str | None = None,
    alpha: float = 10.0,
    show: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run comparison on multiple complex molecules.

    Args:
        output_dir: Directory to save figures.
        alpha: Motif affinity weight.
        show: Whether to display plots.
        verbose: Whether to print detailed output.

    Returns:
        List of result dictionaries.
    """
    # Select interesting molecules for demo
    demo_molecules = [
        # Steroids with fused rings
        "cholesterol",
        "testosterone",
        # Alkaloids
        "morphine",
        "strychnine",
        "quinine",
        # Complex drugs with multiple benzene rings
        "atorvastatin",
        "sildenafil",
        "diazepam",
        # Natural products
        "resveratrol",
        "quercetin",
        "curcumin",
        # Polycyclic aromatics
        "pyrene",
        "phenanthrene",
    ]

    results = []

    if verbose:
        print("Comparing standard vs motif-aware coarsening")
        print("=" * 70)

    for name in demo_molecules:
        smiles = MOLECULES[name]
        if verbose:
            print(f"\nProcessing: {name}")

        output_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = f"{output_dir}/compare_{name}.png"

        result = compare_coarsening(
            smiles,
            name,
            alpha=alpha,
            output_path=output_path,
            show=show,
        )

        if result:
            results.append(result)
            if verbose:
                print(f"  Motifs: {result['motif_summary']}")
                print(f"  Standard cohesion:    {result['std_cohesion']:.0%}")
                print(f"  Motif-aware cohesion: {result['motif_cohesion']:.0%}")
                print(f"  Improvement:          {result['cohesion_improvement']:+.0%}")

    # Print summary
    if verbose and results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            f"\n{'Molecule':<15} {'Atoms':<6} {'Motifs':<7} "
            f"{'Std':<8} {'MA':<8} {'Improvement':<12}"
        )
        print("-" * 70)

        total_improvement = 0
        for r in results:
            print(
                f"{r['name']:<15} {r['num_atoms']:<6} {r['num_motifs']:<7} "
                f"{r['std_cohesion']:<8.0%} {r['motif_cohesion']:<8.0%} "
                f"{r['cohesion_improvement']:+.0%}"
            )
            total_improvement += r["cohesion_improvement"]

        avg_improvement = total_improvement / len(results)
        print("-" * 70)
        print(f"Average improvement: {avg_improvement:+.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard vs motif-aware graph coarsening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --smiles "c1ccccc1" --name benzene
    %(prog)s --name cholesterol --alpha 5.0
    %(prog)s --name morphine --output morphine_compare.png
    %(prog)s --demo --output-dir ./figures --no-show
    %(prog)s --list

The motif-aware coarsening uses the formula A' = A + alpha * M where:
  - A is the original adjacency matrix
  - M is the motif co-membership matrix (M[i,j] = shared motif count)
  - alpha controls the strength of motif preservation

Higher alpha values more strongly encourage keeping ring structures together.
""",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES string to analyze",
    )
    parser.add_argument(
        "--name",
        type=str,
        help=f"Molecule name (one of: {', '.join(sorted(MOLECULES.keys())[:8])}...)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Motif affinity weight (default: 10.0). Higher = stronger motif preference",
    )
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=2,
        help="Minimum nodes per community (default: 2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for single molecule",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for demo mode",
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
        help="Don't display plots (only save)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available molecules:")
        for name, smiles in sorted(MOLECULES.items()):
            mol = Chem.MolFromSmiles(smiles)
            atoms = mol.GetNumAtoms() if mol else "?"
            motifs = detect_motifs_from_smiles(smiles)
            print(f"  {name:15} ({atoms:2} atoms, {len(motifs):2} motifs): {smiles}")
        return

    # Demo mode
    if args.demo:
        run_demo(
            output_dir=args.output_dir,
            alpha=args.alpha,
            show=not args.no_show,
        )
        return

    # Single molecule mode
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

    if not name:
        name = "Custom"

    result = compare_coarsening(
        smiles,
        name,
        alpha=args.alpha,
        min_community_size=args.min_community_size,
        output_path=args.output,
        show=not args.no_show,
        seed=args.seed,
    )

    if result:
        print(f"\nResults for {name}:")
        print(f"  SMILES: {smiles}")
        print(f"  Atoms: {result['num_atoms']}, Motifs: {result['num_motifs']}")
        print(f"  Motif types: {result['motif_summary']}")
        print(f"  Standard cohesion:    {result['std_cohesion']:.0%}")
        print(f"  Motif-aware cohesion: {result['motif_cohesion']:.0%}")
        print(f"  Improvement:          {result['cohesion_improvement']:+.0%}")


if __name__ == "__main__":
    main()
