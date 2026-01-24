"""Visualization utilities for generated molecules in test.py.

This module provides side-by-side visualization of generated molecules:
- Left panel: Molecule structure with color-coded motif highlighting
- Right panel: H-graph showing community structure with labels

For hierarchical tokenizers (H-SENT, HDT), shows the graph partitioning.
For flat SENT tokenizer, shows molecule only (no H-graph panel).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from torch_geometric.data import Data

from src.evaluation.motif_distribution import MOLECULAR_MOTIFS

if TYPE_CHECKING:
    from src.tokenizers import HDTTokenizer, HSENTTokenizer, SENTTokenizer

log = logging.getLogger(__name__)


# Categorical colormap for motif types (organized by category)
# Same palette as train.py for consistency
MOTIF_COLORS = {
    # Aromatic rings - blues/purples
    "benzene": (0.2, 0.4, 0.8),
    "pyridine": (0.3, 0.3, 0.9),
    "pyrrole": (0.4, 0.2, 0.8),
    "furan": (0.5, 0.3, 0.7),
    "thiophene": (0.3, 0.5, 0.8),
    "imidazole": (0.4, 0.4, 0.9),
    "pyrimidine": (0.2, 0.5, 0.7),
    "naphthalene": (0.1, 0.3, 0.9),
    # Functional groups - greens/yellows
    "hydroxyl": (0.2, 0.8, 0.3),
    "carboxyl": (0.8, 0.6, 0.2),
    "carbonyl": (0.9, 0.7, 0.1),
    "aldehyde": (0.7, 0.8, 0.2),
    "ester": (0.6, 0.7, 0.3),
    "amide": (0.5, 0.8, 0.4),
    "amine_primary": (0.3, 0.9, 0.5),
    "amine_secondary": (0.4, 0.85, 0.5),
    "amine_tertiary": (0.5, 0.8, 0.5),
    "nitro": (0.9, 0.2, 0.2),
    "nitrile": (0.7, 0.3, 0.5),
    # Halogens - oranges/reds
    "halogen": (1.0, 0.5, 0.0),
    "fluorine": (0.9, 0.6, 0.1),
    "chlorine": (0.8, 0.5, 0.2),
    "bromine": (0.7, 0.4, 0.1),
    "iodine": (0.6, 0.3, 0.2),
    # Others - teals/cyans
    "ether": (0.2, 0.7, 0.7),
    "thioether": (0.3, 0.6, 0.6),
    "sulfone": (0.4, 0.5, 0.7),
    "sulfonamide": (0.5, 0.6, 0.8),
    "phosphate": (0.6, 0.4, 0.6),
}

# Community colors for H-graph visualization (colorblind-friendly)
COMMUNITY_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-green
    "#17becf",  # Cyan
]


def visualize_generated_molecules(
    generated_graphs: list[Data],
    generated_smiles: list[str],
    tokenizer: "HDTTokenizer | HSENTTokenizer | SENTTokenizer",
    output_dir: Path,
    max_molecules: int = 12,
    dpi: int = 150,
) -> None:
    """Generate side-by-side visualizations for generated molecules.

    Creates PNG files with:
    - Left panel: Molecule structure with color-coded motif highlighting
    - Right panel: H-graph showing community structure (hierarchical only)

    Args:
        generated_graphs: List of generated PyG Data objects.
        generated_smiles: List of generated SMILES strings.
        tokenizer: Tokenizer instance (SENT, H-SENT, or HDT).
        output_dir: Directory to save visualization files.
        max_molecules: Maximum number of molecules to visualize.
        dpi: DPI for saved images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if tokenizer is hierarchical
    tokenizer_type = type(tokenizer).__name__
    is_hierarchical = tokenizer_type in ("HSENTTokenizer", "HDTTokenizer")

    visualized_count = 0
    skipped_invalid = 0
    skipped_large = 0

    for idx, (graph, smiles) in enumerate(zip(generated_graphs, generated_smiles)):
        if visualized_count >= max_molecules:
            break

        # Skip invalid SMILES
        if not smiles or smiles == "INVALID":
            skipped_invalid += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped_invalid += 1
            continue

        # Skip large molecules (>100 nodes)
        if graph.num_nodes > 100:
            skipped_large += 1
            log.debug(f"Skipping molecule {idx}: too large ({graph.num_nodes} nodes)")
            continue

        # Skip empty graphs
        if graph.num_nodes == 0:
            skipped_invalid += 1
            continue

        try:
            if is_hierarchical:
                fig = _create_hierarchical_visualization(
                    graph, smiles, mol, tokenizer, tokenizer_type
                )
            else:
                fig = _create_flat_visualization(graph, smiles, mol)

            # Save figure
            output_path = output_dir / f"molecule_{visualized_count + 1:04d}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            visualized_count += 1

        except Exception as e:
            log.warning(f"Failed to visualize molecule {idx}: {e}")
            continue

    log.info(f"Visualized {visualized_count} molecules")
    if skipped_invalid > 0:
        log.info(f"  Skipped {skipped_invalid} invalid molecules")
    if skipped_large > 0:
        log.info(f"  Skipped {skipped_large} large molecules (>100 nodes)")


def _create_hierarchical_visualization(
    graph: Data,
    smiles: str,
    mol: Chem.Mol,
    tokenizer: "HDTTokenizer | HSENTTokenizer",
    tokenizer_type: str,
) -> plt.Figure:
    """Create 2-panel visualization for hierarchical tokenizers.

    Left: Molecule graph with motif coloring
    Right: Same graph with community coloring
    Both use identical layout for easy comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Build shared graph and layout for both panels
    G, pos = _build_graph_and_layout(graph)

    # Build hierarchy for community info
    hg = tokenizer.coarsener.build_hierarchy(graph)

    # Left panel: Graph with motif coloring
    _plot_graph_with_motifs(axes[0], G, pos, smiles, mol)

    # Right panel: Same graph with community coloring
    _plot_graph_with_communities(axes[1], G, pos, hg, tokenizer_type)

    # Title
    fig.suptitle(
        f"Generated Molecule ({mol.GetNumAtoms()} atoms, "
        f"{hg.num_communities} communities)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def _create_flat_visualization(
    graph: Data,
    smiles: str,
    mol: Chem.Mol,
) -> plt.Figure:
    """Create single-panel visualization for flat SENT tokenizer.

    Shows molecule graph with motif coloring.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Build graph and layout
    G, pos = _build_graph_and_layout(graph)

    # Draw graph with motif coloring
    _plot_graph_with_motifs(ax, G, pos, smiles, mol)

    fig.suptitle(
        f"Generated Molecule ({mol.GetNumAtoms()} atoms)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def _build_graph_and_layout(graph: Data) -> tuple[nx.Graph, dict]:
    """Build NetworkX graph and compute layout from PyG Data.

    Args:
        graph: PyG Data object.

    Returns:
        Tuple of (NetworkX graph, position dict).
    """
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))

    # Add edges from graph
    if graph.edge_index is not None and graph.edge_index.numel() > 0:
        edge_index = graph.edge_index.numpy()
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u < v:  # Avoid duplicate edges
                G.add_edge(u, v)

    # Compute layout - use Kamada-Kawai for better molecular layouts
    if G.number_of_edges() > 0:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
    else:
        pos = nx.circular_layout(G)

    return G, pos


def _plot_graph_with_motifs(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    smiles: str,
    mol: Chem.Mol,
) -> None:
    """Plot molecule graph with color-coded motif highlighting.

    Uses same graph layout as community panel for easy comparison.
    """
    # Shared visual parameters
    node_size = 400
    edge_width = 1.5
    edge_color = "#888888"
    edge_alpha = 0.6
    font_size = 9
    default_node_color = "#E0E0E0"

    # Detect motifs and assign colors to atoms
    atom_colors = {}
    motifs_found = {}

    for motif_name, smarts in MOLECULAR_MOTIFS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                continue

            color = MOTIF_COLORS.get(motif_name, (0.5, 0.5, 0.5))
            motifs_found[motif_name] = len(matches)

            for match in matches:
                for atom_idx in match:
                    if atom_idx not in atom_colors:
                        atom_colors[atom_idx] = color

        except Exception:
            continue

    # Build node color list
    node_colors = []
    for node in range(len(G.nodes())):
        if node in atom_colors:
            node_colors.append(atom_colors[node])
        else:
            node_colors.append(default_node_color)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=edge_alpha, width=edge_width, edge_color=edge_color
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.0,
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight="bold")

    # Add legend for motifs found
    if motifs_found:
        legend_patches = []
        for motif_name in sorted(motifs_found.keys()):
            if motif_name in MOTIF_COLORS:
                color = MOTIF_COLORS[motif_name]
                count = motifs_found[motif_name]
                patch = mpatches.Patch(color=color, label=f"{motif_name} ({count})")
                legend_patches.append(patch)

        if legend_patches:
            ax.legend(
                handles=legend_patches[:8],  # Limit to 8 for space
                loc="upper left",
                fontsize=7,
                framealpha=0.9,
            )

    ax.set_title("Motif Coloring", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_graph_with_communities(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    hg: "HierarchicalGraph",
    tokenizer_type: str,
) -> None:
    """Plot graph with community coloring.

    Uses same layout and styling as motif panel for easy comparison.
    """
    # Shared visual parameters (same as _plot_graph_with_motifs)
    node_size = 400
    edge_width = 1.5
    edge_color = "#888888"
    edge_alpha = 0.6
    font_size = 9

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=edge_alpha, width=edge_width, edge_color=edge_color
    )

    # Color nodes by community
    node_colors = []
    for node in range(len(G.nodes())):
        comm_id = hg.community_assignment[node]
        color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        node_colors.append(color)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.0,
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight="bold")

    # Create legend for communities
    unique_communities = sorted(set(hg.community_assignment))
    legend_patches = []
    for comm_id in unique_communities:
        color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        # Count nodes in this community
        count = sum(1 for c in hg.community_assignment if c == comm_id)
        patch = mpatches.Patch(color=color, label=f"C{comm_id} ({count})")
        legend_patches.append(patch)

    ax.legend(
        handles=legend_patches[:8],  # Limit to 8 for space
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
    )

    ax.set_title(
        f"Community Coloring ({hg.num_communities} communities)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.axis("off")


# Import for type checking
if TYPE_CHECKING:
    from src.tokenizers.structures import HierarchicalGraph
