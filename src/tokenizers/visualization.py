"""Visualization utilities for hierarchical graph tokenization.

This module provides visualization functions for inspecting hierarchical
graph decompositions, inspired by HiGen's Figure 1-style visualizations.

Key visualizations:
- Graph with community coloring
- Block matrix representation (partitions + bipartites)
- Token sequence visualization
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from torch_geometric.data import Data

from src.tokenizers.structures import (
    Bipartite,
    HierarchicalGraph,
    Partition,
)


# Color palette for communities (colorblind-friendly)
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


def visualize_hierarchy(
    hg: HierarchicalGraph,
    original_data: Optional[Data] = None,
    figsize: tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create HiGen Figure 1-style visualization of hierarchical decomposition.

    Creates a three-panel figure showing:
    1. Original graph with nodes colored by community
    2. Block matrix representation
    3. Hierarchy structure

    Args:
        hg: HierarchicalGraph to visualize.
        original_data: Original PyG Data for graph layout (optional).
        figsize: Figure size as (width, height).
        save_path: Path to save figure (optional).
        title: Figure title (optional).

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Graph with community coloring
    _plot_graph_communities(axes[0], hg, original_data)
    axes[0].set_title("Graph with Communities")

    # Panel 2: Block matrix
    _plot_block_matrix(axes[1], hg)
    axes[1].set_title("Block Matrix")

    # Panel 3: Hierarchy structure
    _plot_hierarchy_tree(axes[2], hg)
    axes[2].set_title("Hierarchy Structure")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_graph_communities(
    hg: HierarchicalGraph,
    original_data: Optional[Data] = None,
    ax: Optional[plt.Axes] = None,
    node_size: int = 300,
    edge_width: float = 1.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize graph with nodes colored by community assignment.

    Args:
        hg: HierarchicalGraph with community information.
        original_data: Original graph data for edge information.
        ax: Matplotlib axes to plot on (creates new figure if None).
        node_size: Size of nodes in the plot.
        edge_width: Width of edges.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    _plot_graph_communities(ax, hg, original_data, node_size, edge_width)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_block_matrix(
    hg: HierarchicalGraph,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize the block matrix representation of the hierarchy.

    Shows diagonal blocks (partitions) and off-diagonal blocks (bipartites).

    Args:
        hg: HierarchicalGraph to visualize.
        ax: Matplotlib axes to plot on (creates new figure if None).
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    _plot_block_matrix(ax, hg)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_tokens(
    tokens: Sequence[int],
    tokenizer,
    ax: Optional[plt.Axes] = None,
    max_tokens: int = 100,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize token sequence with color-coded tokens.

    Args:
        tokens: Token sequence.
        tokenizer: HSENTTokenizer for token interpretation.
        ax: Matplotlib axes to plot on (creates new figure if None).
        max_tokens: Maximum number of tokens to display.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 2))
    else:
        fig = ax.get_figure()

    _plot_tokens(ax, tokens, tokenizer, max_tokens)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ===========================================================================
# Internal plotting functions
# ===========================================================================


def _plot_graph_communities(
    ax: plt.Axes,
    hg: HierarchicalGraph,
    original_data: Optional[Data] = None,
    node_size: int = 300,
    edge_width: float = 1.0,
) -> None:
    """Plot graph with community coloring."""
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
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.circular_layout(G)

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, width=edge_width)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_size, edgecolors="black"
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Legend
    unique_communities = sorted(set(hg.community_assignment))
    legend_handles = [
        mpatches.Patch(
            color=COMMUNITY_COLORS[c % len(COMMUNITY_COLORS)],
            label=f"Community {c}",
        )
        for c in unique_communities
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_aspect("equal")
    ax.axis("off")


def _plot_block_matrix(ax: plt.Axes, hg: HierarchicalGraph) -> None:
    """Plot block matrix representation."""
    n = hg.num_nodes
    if n == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        ax.axis("off")
        return

    # Build full adjacency matrix
    adj = np.zeros((n, n))
    reconstructed = hg.reconstruct()
    if reconstructed.edge_index.numel() > 0:
        ei = reconstructed.edge_index.numpy()
        for e in range(ei.shape[1]):
            adj[int(ei[0, e]), int(ei[1, e])] = 1

    # Reorder by community
    community_order = sorted(range(n), key=lambda x: hg.community_assignment[x])
    adj_reordered = adj[np.ix_(community_order, community_order)]

    # Create color matrix for block visualization
    block_colors = np.zeros((n, n, 3))

    # Color diagonal blocks (partitions)
    idx = 0
    for part in hg.partitions:
        part_size = part.num_nodes
        color = np.array(
            [int(c * 255) for c in plt.cm.tab10(part.part_id % 10)[:3]]
        ) / 255
        for i in range(part_size):
            for j in range(part_size):
                if adj_reordered[idx + i, idx + j] > 0:
                    block_colors[idx + i, idx + j] = color
        idx += part_size

    # Draw matrix
    ax.imshow(block_colors, aspect="equal", origin="upper")

    # Draw partition boundaries
    idx = 0
    for i, part in enumerate(hg.partitions):
        part_size = part.num_nodes
        ax.axhline(y=idx - 0.5, color="black", linewidth=1)
        ax.axvline(x=idx - 0.5, color="black", linewidth=1)
        idx += part_size
    ax.axhline(y=idx - 0.5, color="black", linewidth=1)
    ax.axvline(x=idx - 0.5, color="black", linewidth=1)

    # Labels
    ax.set_xlabel("Node Index (reordered)")
    ax.set_ylabel("Node Index (reordered)")


def _plot_hierarchy_tree(ax: plt.Axes, hg: HierarchicalGraph) -> None:
    """Plot hierarchy structure as a tree."""
    # Create hierarchy visualization
    levels_info = []

    # Level 0: Partitions
    partition_info = [f"P{p.part_id}: {p.num_nodes} nodes" for p in hg.partitions]
    bipartite_info = [
        f"B({b.left_part_id},{b.right_part_id}): {b.num_edges} edges"
        for b in hg.bipartites
    ]

    # Draw as text-based tree
    y_pos = 0.9
    ax.text(
        0.5,
        y_pos,
        f"HierarchicalGraph ({hg.num_nodes} nodes, {hg.num_communities} communities)",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    y_pos -= 0.1

    ax.text(0.1, y_pos, "Partitions:", ha="left", va="top", fontsize=9, fontweight="bold")
    y_pos -= 0.05
    for info in partition_info:
        ax.text(0.15, y_pos, f"- {info}", ha="left", va="top", fontsize=8)
        y_pos -= 0.05

    y_pos -= 0.05
    ax.text(0.1, y_pos, "Bipartites:", ha="left", va="top", fontsize=9, fontweight="bold")
    y_pos -= 0.05
    for info in bipartite_info:
        ax.text(0.15, y_pos, f"- {info}", ha="left", va="top", fontsize=8)
        y_pos -= 0.05

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _plot_tokens(
    ax: plt.Axes,
    tokens: Sequence[int],
    tokenizer,
    max_tokens: int = 100,
) -> None:
    """Plot token sequence with color coding."""
    tokens = list(tokens)[:max_tokens]
    n_tokens = len(tokens)

    # Token colors
    colors = []
    labels = []
    for tok in tokens:
        if tok in tokenizer.SPECIAL_TOKEN_NAMES:
            name = tokenizer.SPECIAL_TOKEN_NAMES[tok]
            colors.append("lightgray")
            labels.append(name)
        else:
            val = tok - tokenizer.IDX_OFFSET
            colors.append("lightblue")
            labels.append(str(val))

    # Draw as colored boxes
    box_width = 0.8 / n_tokens
    for i, (color, label) in enumerate(zip(colors, labels)):
        x = i / n_tokens
        rect = mpatches.Rectangle(
            (x, 0.2), box_width, 0.6, facecolor=color, edgecolor="black", linewidth=0.5
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2, 0.5, label, ha="center", va="center", fontsize=6, rotation=45
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Token Sequence ({n_tokens} tokens)")


def quick_visualize(
    data: Data,
    tokenizer,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Quick visualization of a graph's hierarchical tokenization.

    Convenience function that builds the hierarchy and creates a
    comprehensive visualization.

    Args:
        data: PyTorch Geometric Data object.
        tokenizer: HSENTTokenizer instance.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib Figure object.
    """
    hg = tokenizer.coarsener.build_hierarchy(data)
    tokens = tokenizer.tokenize_hierarchy(hg)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Graph with communities
    _plot_graph_communities(axes[0, 0], hg, data)
    axes[0, 0].set_title("Graph with Communities")

    # Block matrix
    _plot_block_matrix(axes[0, 1], hg)
    axes[0, 1].set_title("Block Matrix (Reordered Adjacency)")

    # Hierarchy info
    _plot_hierarchy_tree(axes[1, 0], hg)
    axes[1, 0].set_title("Hierarchy Structure")

    # Token sequence
    _plot_tokens(axes[1, 1], tokens.tolist(), tokenizer)
    axes[1, 1].set_title(f"Token Sequence ({len(tokens)} tokens)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
