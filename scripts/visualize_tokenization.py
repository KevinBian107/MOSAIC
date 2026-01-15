#!/usr/bin/env python
"""Visualization script comparing SENT, H-SENT, and HDT tokenization.

This script provides side-by-side comparison of:
- Molecule structure with motif highlighting
- SENT: Flat random walk tokenization (different colors per walk segment)
- H-SENT: Hierarchical tokenization with community structure
- HDT: Hierarchical DFS tokenization with tree structure

Usage:
    python scripts/visualize_tokenization.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
    python scripts/visualize_tokenization.py --name caffeine --output caffeine_compare.png
    python scripts/visualize_tokenization.py --demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, ConnectionPatch
from matplotlib.path import Path as MPath
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.tokenizers.sent import SENTTokenizer
from src.tokenizers.hierarchical import HSENTTokenizer
from src.tokenizers.hierarchical.hdt import HDTTokenizer


# Common molecules for demos
MOLECULES = {
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "phenol": "Oc1ccccc1",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "dopamine": "NCCC1=CC(O)=C(O)C=C1",
    "penicillin_g": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    # Complex molecules with multiple motifs
    "quercetin": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C(O)=C2C3=CC(O)=C(O)C=C3",
    "resveratrol": "OC1=CC=C(C=C1)/C=C/C2=CC(O)=CC(O)=C2",
    "estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
    "testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "indole_benzene": "c1ccc2[nH]ccc2c1-c3ccccc3",  # Indole linked to benzene
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
}

# SMARTS patterns for common motifs
MOTIF_PATTERNS = {
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "imidazole": "c1cnc[nH]1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "purine": "c1nc2c(n1)nc[nH]2",
    "pyrimidine": "c1cncnc1",
}

# Colors for motifs
MOTIF_COLORS = {
    "benzene": "#FF6B6B",      # Red
    "pyridine": "#4ECDC4",     # Teal
    "pyrrole": "#45B7D1",      # Blue
    "imidazole": "#96CEB4",    # Green
    "furan": "#FFEAA7",        # Yellow
    "thiophene": "#DDA0DD",    # Plum
    "naphthalene": "#FF8C00",  # Orange
    "indole": "#9B59B6",       # Purple
    "purine": "#E74C3C",       # Dark Red
    "pyrimidine": "#3498DB",   # Blue
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
    data = Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())
    data.smiles = smiles
    return data


def detect_motifs(smiles: str) -> dict[str, list[tuple[int, ...]]]:
    """Detect motifs in a molecule and return atom indices for each."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    motifs = {}
    for motif_name, pattern in MOTIF_PATTERNS.items():
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            continue
        matches = mol.GetSubstructMatches(query)
        if matches:
            motifs[motif_name] = list(matches)
    return motifs


def draw_curved_edge(
    ax: plt.Axes,
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    color: str = "gray",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    linestyle: str = "-",
    curve_amount: float = 0.15,
    arrow: bool = False,
    zorder: int = 1,
) -> None:
    """Draw a curved edge between two points using a quadratic bezier curve."""
    x1, y1 = pos1
    x2, y2 = pos2

    # Calculate midpoint and perpendicular offset for curve
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 0.01:
        return

    # Perpendicular direction
    px, py = -dy / dist, dx / dist

    # Control point offset
    cx = mx + px * curve_amount * dist
    cy = my + py * curve_amount * dist

    # Create bezier curve
    verts = [(x1, y1), (cx, cy), (x2, y2)]
    codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
    path = MPath(verts, codes)

    if arrow:
        # Draw with arrow
        patch = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=12,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
        )
        ax.add_patch(patch)
    else:
        # Draw as path
        from matplotlib.patches import PathPatch
        patch = PathPatch(
            path, facecolor="none", edgecolor=color,
            linewidth=linewidth, alpha=alpha, linestyle=linestyle,
            zorder=zorder,
        )
        ax.add_patch(patch)


def get_consistent_layout(data: Data, seed: int = 42) -> dict[int, tuple[float, float]]:
    """Get consistent node positions for a graph."""
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=seed, k=1.5, iterations=100)
    return pos


# ===========================================================================
# Molecule Structure with Motifs
# ===========================================================================


def plot_molecule_with_motifs(
    ax: plt.Axes,
    data: Data,
    smiles: str,
    pos: dict[int, tuple[float, float]],
) -> None:
    """Plot molecule structure with motif highlighting."""
    G = to_networkx(data, to_undirected=True)

    # Detect motifs
    motifs = detect_motifs(smiles)

    # Create node colors based on motifs
    node_colors = ["#E8E8E8"] * data.num_nodes  # Default light gray
    node_motifs = [None] * data.num_nodes

    # Assign colors based on motifs (later motifs override earlier)
    for motif_name, matches in motifs.items():
        color = MOTIF_COLORS.get(motif_name, "#CCCCCC")
        for match in matches:
            for atom_idx in match:
                if atom_idx < data.num_nodes:
                    node_colors[atom_idx] = color
                    node_motifs[atom_idx] = motif_name

    # Draw edges with curved lines
    drawn_edges = set()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_key = (min(u, v), max(u, v))
        if edge_key not in drawn_edges:
            drawn_edges.add(edge_key)
            draw_curved_edge(
                ax, pos[u], pos[v],
                color="#555555", linewidth=2, curve_amount=0.1
            )

    # Draw nodes
    for node in range(data.num_nodes):
        x, y = pos[node]
        circle = plt.Circle(
            (x, y), 0.08, facecolor=node_colors[node],
            edgecolor="black", linewidth=1.5, zorder=3
        )
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4)

    # Create legend for motifs found
    legend_patches = []
    for motif_name in motifs.keys():
        if motif_name in MOTIF_COLORS:
            patch = mpatches.Patch(
                color=MOTIF_COLORS[motif_name],
                label=motif_name.capitalize()
            )
            legend_patches.append(patch)

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper left", fontsize=7,
                  framealpha=0.9)

    ax.set_title("Molecule Structure with Motifs", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


# ===========================================================================
# SENT Visualization - Random Walk with Segment Colors
# ===========================================================================


def plot_sent_walks(
    ax: plt.Axes,
    data: Data,
    tokens: list[int],
    tokenizer: SENTTokenizer,
    pos: dict[int, tuple[float, float]],
) -> None:
    """Plot SENT showing walk on original molecule graph.

    Uses graph isomorphism to map decoded nodes back to original positions.
    Shows visit order numbers along the walk paths.
    """
    # Decode the tokens to get the reconstructed graph
    decoded = tokenizer.decode(torch.tensor(tokens))
    decoded_G = to_networkx(decoded, to_undirected=True)
    original_G = to_networkx(data, to_undirected=True)

    # Find isomorphism mapping from decoded to original graph
    from networkx.algorithms import isomorphism
    GM = isomorphism.GraphMatcher(decoded_G, original_G)

    # Get mapping: decoded_node -> original_node
    decoded_to_original = {}
    if GM.is_isomorphic():
        decoded_to_original = GM.mapping
    else:
        # Fallback: try to match by degree sequence or use identity
        for i in range(min(decoded.num_nodes, data.num_nodes)):
            decoded_to_original[i] = i

    # Parse tokens into walk segments (visit order indices)
    segments = []
    current_segment = []
    inside_adj = False

    for tok in tokens:
        if tok == tokenizer.ladj:
            inside_adj = True
        elif tok == tokenizer.radj:
            inside_adj = False
        elif tok == tokenizer.reset:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        elif tok >= tokenizer.idx_offset and not inside_adj:
            visit_idx = tok - tokenizer.idx_offset
            if visit_idx < decoded.num_nodes:
                current_segment.append(visit_idx)

    if current_segment:
        segments.append(current_segment)

    # Draw base graph edges (light gray)
    drawn_edges = set()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_key = (min(u, v), max(u, v))
        if edge_key not in drawn_edges:
            drawn_edges.add(edge_key)
            draw_curved_edge(
                ax, pos[u], pos[v],
                color="#DDDDDD", linewidth=1.5, curve_amount=0.1
            )

    # Color palette for segments
    segment_colors = ["#E74C3C", "#00CED1", "#32CD32", "#FF8C00", "#9B59B6", "#FF69B4"]

    # Track visit order for each original node
    node_visit_info = {}  # original_node -> (segment_idx, visit_order_in_walk)
    global_visit_order = 0

    for seg_idx, segment in enumerate(segments):
        for local_idx, decoded_node in enumerate(segment):
            orig_node = decoded_to_original.get(decoded_node, decoded_node)
            if orig_node < data.num_nodes:
                node_visit_info[orig_node] = (seg_idx, global_visit_order)
                global_visit_order += 1

    # Draw walk paths with curved arrows
    for seg_idx, segment in enumerate(segments):
        color = segment_colors[seg_idx % len(segment_colors)]

        if len(segment) > 1:
            for i in range(len(segment) - 1):
                decoded_u, decoded_v = segment[i], segment[i + 1]
                u = decoded_to_original.get(decoded_u, decoded_u)
                v = decoded_to_original.get(decoded_v, decoded_v)

                if u != v and u in pos and v in pos:
                    curve = 0.25 if i % 2 == 0 else -0.25
                    draw_curved_edge(
                        ax, pos[u], pos[v],
                        color=color, linewidth=3,
                        curve_amount=curve, arrow=True, zorder=10
                    )

    # Draw nodes with visit order numbers on top
    for node in range(data.num_nodes):
        x, y = pos[node]

        # Get segment color and visit order if visited
        visit_info = node_visit_info.get(node)
        if visit_info is not None:
            seg_idx, visit_order = visit_info
            node_color = segment_colors[seg_idx % len(segment_colors)]
        else:
            node_color = "#E8E8E8"
            visit_order = None

        circle = plt.Circle(
            (x, y), 0.08, facecolor=node_color,
            edgecolor="black", linewidth=2, zorder=5
        )
        ax.add_patch(circle)

        # Show visit order number on the node (not original index)
        if visit_order is not None:
            ax.text(x, y, str(visit_order), ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=7)
        else:
            # Show original node index for unvisited nodes
            ax.text(x, y, str(node), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#666666", zorder=7)

    # Legend
    patches = []
    for i in range(min(len(segments), 6)):
        patches.append(mpatches.Patch(color=segment_colors[i], label=f"Walk {i+1}"))

    if patches:
        ax.legend(handles=patches, loc="upper left", fontsize=7)

    ax.set_title(
        f"SENT: {len(segments)} Walk{'s' if len(segments) != 1 else ''}",
        fontsize=11, fontweight="bold"
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits (same as molecule)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


# ===========================================================================
# H-SENT Visualization - Community Structure
# ===========================================================================


def plot_hsent_structure(
    ax: plt.Axes,
    data: Data,
    hg,
    tokens: list[int],
    tokenizer: HSENTTokenizer,
    pos: dict[int, tuple[float, float]],
) -> None:
    """Plot H-SENT with community structure using curved edges."""
    G = to_networkx(data, to_undirected=True)

    # Community colors
    comm_colors = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"]
    num_comm = len(hg.partitions)

    # Draw inter-community edges (dashed, curved)
    for bipart in hg.bipartites:
        left_part = hg.get_partition(bipart.left_part_id)
        right_part = hg.get_partition(bipart.right_part_id)
        if bipart.edge_index.numel() > 0:
            for i in range(bipart.edge_index.shape[1]):
                left_local = bipart.edge_index[0, i].item()
                right_local = bipart.edge_index[1, i].item()
                u = left_part.global_node_indices[left_local]
                v = right_part.global_node_indices[right_local]
                draw_curved_edge(
                    ax, pos[u], pos[v],
                    color="#888888", linewidth=1.5,
                    linestyle="--", curve_amount=0.2, zorder=1
                )

    # Draw each community
    for part_idx, part in enumerate(hg.partitions):
        comm_color = comm_colors[part_idx % len(comm_colors)]
        nodes = part.global_node_indices

        # Draw intra-community edges (solid, curved)
        if part.edge_index.numel() > 0:
            drawn = set()
            for i in range(part.edge_index.shape[1]):
                src_local = part.edge_index[0, i].item()
                dst_local = part.edge_index[1, i].item()
                edge_key = (min(src_local, dst_local), max(src_local, dst_local))
                if edge_key not in drawn:
                    drawn.add(edge_key)
                    u = nodes[src_local]
                    v = nodes[dst_local]
                    draw_curved_edge(
                        ax, pos[u], pos[v],
                        color=comm_color, linewidth=2.5,
                        curve_amount=0.12, zorder=2
                    )

        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            circle = plt.Circle(
                (x, y), 0.08, facecolor=comm_color,
                edgecolor="black", linewidth=2, zorder=3
            )
            ax.add_patch(circle)

    # Add labels
    for node in range(data.num_nodes):
        x, y = pos[node]
        ax.text(x, y, str(node), ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=4)

    # Legend
    patches = [
        mpatches.Patch(color=comm_colors[i % len(comm_colors)], label=f"Comm. {i}")
        for i in range(min(num_comm, 6))
    ]
    patches.append(mpatches.Patch(color="#888888", label="Cross-comm."))
    ax.legend(handles=patches, loc="upper left", fontsize=7)

    ax.set_title(f"H-SENT: {num_comm} Communities", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


# ===========================================================================
# HDT Visualization - Tree Structure with Traversal
# ===========================================================================


def draw_bidirectional_arrow(
    ax: plt.Axes,
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    color: str = "black",
    linewidth: float = 1.5,
    curve_amount: float = 0.1,
    zorder: int = 1,
) -> None:
    """Draw bidirectional curved arrow with arrows in the middle of each direction."""
    x1, y1 = pos1
    x2, y2 = pos2

    # Calculate midpoint and perpendicular offset
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 0.01:
        return

    # Perpendicular direction
    px, py = -dy / dist, dx / dist

    # Draw forward arrow (pos1 -> pos2) with positive curve
    cx1 = mx + px * curve_amount * dist
    cy1 = my + py * curve_amount * dist

    # Points along the curve for forward direction
    t_vals = np.linspace(0, 1, 20)
    forward_points = []
    for t in t_vals:
        # Quadratic bezier
        bx = (1-t)**2 * x1 + 2*(1-t)*t * cx1 + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * cy1 + t**2 * y2
        forward_points.append((bx, by))

    # Draw backward arrow (pos2 -> pos1) with negative curve
    cx2 = mx - px * curve_amount * dist
    cy2 = my - py * curve_amount * dist

    backward_points = []
    for t in t_vals:
        bx = (1-t)**2 * x2 + 2*(1-t)*t * cx2 + t**2 * x1
        by = (1-t)**2 * y2 + 2*(1-t)*t * cy2 + t**2 * y1
        backward_points.append((bx, by))

    # Draw curves as lines
    forward_x = [p[0] for p in forward_points]
    forward_y = [p[1] for p in forward_points]
    ax.plot(forward_x, forward_y, color=color, linewidth=linewidth, zorder=zorder)

    backward_x = [p[0] for p in backward_points]
    backward_y = [p[1] for p in backward_points]
    ax.plot(backward_x, backward_y, color=color, linewidth=linewidth, zorder=zorder)

    # Add arrow markers at middle of each curve
    mid_idx = len(forward_points) // 2

    # Forward arrow at middle
    fx1, fy1 = forward_points[mid_idx - 1]
    fx2, fy2 = forward_points[mid_idx + 1]
    ax.annotate("", xy=(fx2, fy2), xytext=(fx1, fy1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth),
                zorder=zorder + 1)

    # Backward arrow at middle
    bx1, by1 = backward_points[mid_idx - 1]
    bx2, by2 = backward_points[mid_idx + 1]
    ax.annotate("", xy=(bx2, by2), xytext=(bx1, by1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth),
                zorder=zorder + 1)


def plot_hdt_tree(
    ax: plt.Axes,
    data: Data,
    hg,
    tokens: list[int],
    tokenizer: HDTTokenizer,
) -> None:
    """Plot HDT with tree-like hierarchical structure and bidirectional arrows."""
    # Calculate hierarchy depth
    depth = hg.depth + 1
    num_partitions = len(hg.partitions)

    # Layout parameters
    level_height = 1.4  # Increased for better visibility

    positions = {}
    node_to_partition = {}

    # Root position (level 0)
    root_pos = (0, depth * level_height)
    positions["root"] = root_pos

    # Calculate total width based on nodes
    node_spacing = 0.45  # Increased spacing

    # Partition positions (level 1)
    partition_positions = {}
    partition_widths = []
    for part in hg.partitions:
        partition_widths.append(len(part.global_node_indices) * node_spacing)

    total_width = sum(partition_widths) + (num_partitions - 1) * 0.6
    current_x = -total_width / 2

    for part_idx, part in enumerate(hg.partitions):
        part_width = partition_widths[part_idx]
        part_center_x = current_x + part_width / 2
        partition_positions[part_idx] = (part_center_x, (depth - 1) * level_height)
        current_x += part_width + 0.6

        # Node positions within partition (level 2)
        nodes = part.global_node_indices
        num_nodes_in_part = len(nodes)
        start_x = part_center_x - (num_nodes_in_part - 1) * node_spacing / 2

        for local_idx, global_idx in enumerate(nodes):
            node_x = start_x + local_idx * node_spacing
            node_y = (depth - 2) * level_height
            positions[global_idx] = (node_x, node_y)
            node_to_partition[global_idx] = part_idx

    # Draw level labels
    for level in range(depth):
        y = (depth - 1 - level) * level_height
        ax.text(
            -total_width / 2 - 0.8, y,
            f"l = {level}", fontsize=9, ha="right", va="center",
            style="italic", color="#666666"
        )

    # Community colors
    comm_colors = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"]

    # Draw bidirectional arrows from root to partitions (hierarchy traversal)
    for part_idx, part_pos in partition_positions.items():
        draw_bidirectional_arrow(
            ax, root_pos, part_pos,
            color="#333333", linewidth=1.2,
            curve_amount=0.12, zorder=2
        )

    # Draw bidirectional arrows from partitions to atoms (hierarchy traversal)
    for part_idx, part in enumerate(hg.partitions):
        part_pos = partition_positions[part_idx]
        for global_idx in part.global_node_indices:
            node_pos = positions[global_idx]
            draw_bidirectional_arrow(
                ax, part_pos, node_pos,
                color="#333333", linewidth=1.0,
                curve_amount=0.15, zorder=2
            )

    # Draw intra-community edges (ORANGE, curved)
    for part in hg.partitions:
        if part.edge_index.numel() > 0:
            nodes = part.global_node_indices
            drawn = set()
            for i in range(part.edge_index.shape[1]):
                src_local = part.edge_index[0, i].item()
                dst_local = part.edge_index[1, i].item()
                edge_key = (min(src_local, dst_local), max(src_local, dst_local))
                if edge_key not in drawn:
                    drawn.add(edge_key)
                    u = nodes[src_local]
                    v = nodes[dst_local]
                    draw_curved_edge(
                        ax, positions[u], positions[v],
                        color="#FF8C00", linewidth=3,
                        curve_amount=0.2, zorder=3
                    )

    # Draw inter-community edges (ORANGE, dashed, curved)
    for bipart in hg.bipartites:
        left_part = hg.get_partition(bipart.left_part_id)
        right_part = hg.get_partition(bipart.right_part_id)
        if bipart.edge_index.numel() > 0:
            for i in range(bipart.edge_index.shape[1]):
                left_local = bipart.edge_index[0, i].item()
                right_local = bipart.edge_index[1, i].item()
                u = left_part.global_node_indices[left_local]
                v = right_part.global_node_indices[right_local]
                draw_curved_edge(
                    ax, positions[u], positions[v],
                    color="#FF8C00", linewidth=2.5, linestyle="--",
                    curve_amount=0.3, zorder=3
                )

    # Draw partition nodes
    for part_idx, part in enumerate(hg.partitions):
        part_pos = partition_positions[part_idx]
        comm_color = comm_colors[part_idx % len(comm_colors)]

        circle = plt.Circle(
            part_pos, 0.2, facecolor=comm_color,
            edgecolor="black", linewidth=2, zorder=5
        )
        ax.add_patch(circle)
        ax.text(part_pos[0], part_pos[1], f"P{part_idx}", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=6)

    # Draw root node
    circle = plt.Circle(
        root_pos, 0.25, facecolor="white",
        edgecolor="black", linewidth=2, zorder=5
    )
    ax.add_patch(circle)
    ax.text(root_pos[0], root_pos[1], "R", ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=6)

    # Draw atom nodes (level 2)
    for part_idx, part in enumerate(hg.partitions):
        comm_color = comm_colors[part_idx % len(comm_colors)]
        for global_idx in part.global_node_indices:
            node_pos = positions[global_idx]
            circle = plt.Circle(
                node_pos, 0.14, facecolor=comm_color,
                edgecolor="black", linewidth=1.5, zorder=5
            )
            ax.add_patch(circle)
            ax.text(node_pos[0], node_pos[1], str(global_idx), ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=6)

    # Set axis limits
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - 1.0, max(all_x) + 1.0)
    ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#FF8C00", linewidth=3, label="Graph edges"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.5, marker=">",
                   markersize=6, label="DFS traversal\n(↓down ↑up)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    ax.set_title("HDT: Hierarchical Tree", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


# ===========================================================================
# Token Visualization
# ===========================================================================


def plot_tokens(
    ax: plt.Axes,
    tokens: list[int],
    tokenizer,
    title: str,
    max_tokens: int = 40,
) -> None:
    """Plot token sequence as colored blocks."""
    if hasattr(tokenizer, "ENTER"):
        token_colors = {
            tokenizer.SOS: "#2ecc71", tokenizer.EOS: "#e74c3c",
            tokenizer.PAD: "#bdc3c7", tokenizer.ENTER: "#9b59b6",
            tokenizer.EXIT: "#9b59b6", tokenizer.LEDGE: "#3498db",
            tokenizer.REDGE: "#3498db",
        }
        token_names = {
            tokenizer.SOS: "S", tokenizer.EOS: "E", tokenizer.PAD: "P",
            tokenizer.ENTER: "v", tokenizer.EXIT: "^",
            tokenizer.LEDGE: "[", tokenizer.REDGE: "]",
        }
        idx_offset = tokenizer.IDX_OFFSET
    elif hasattr(tokenizer, "SOS"):
        token_colors = {
            tokenizer.SOS: "#2ecc71", tokenizer.EOS: "#e74c3c",
            tokenizer.PAD: "#bdc3c7", tokenizer.RESET: "#9b59b6",
            tokenizer.LADJ: "#3498db", tokenizer.RADJ: "#3498db",
            tokenizer.LCOM: "#f39c12", tokenizer.RCOM: "#f39c12",
            tokenizer.LBIP: "#1abc9c", tokenizer.RBIP: "#1abc9c",
            tokenizer.SEP: "#95a5a6",
        }
        token_names = {
            tokenizer.SOS: "S", tokenizer.EOS: "E", tokenizer.PAD: "P",
            tokenizer.RESET: "R", tokenizer.LADJ: "[", tokenizer.RADJ: "]",
            tokenizer.LCOM: "{", tokenizer.RCOM: "}",
            tokenizer.LBIP: "<", tokenizer.RBIP: ">", tokenizer.SEP: "|",
        }
        idx_offset = tokenizer.IDX_OFFSET
    else:
        token_colors = {
            tokenizer.sos: "#2ecc71", tokenizer.eos: "#e74c3c",
            tokenizer.pad: "#bdc3c7", tokenizer.reset: "#9b59b6",
            tokenizer.ladj: "#3498db", tokenizer.radj: "#3498db",
        }
        token_names = {
            tokenizer.sos: "S", tokenizer.eos: "E", tokenizer.pad: "P",
            tokenizer.reset: "R", tokenizer.ladj: "[", tokenizer.radj: "]",
        }
        idx_offset = tokenizer.idx_offset

    display_tokens = tokens[:max_tokens]
    truncated = len(tokens) > max_tokens

    cols = min(12, len(display_tokens))
    rows = (len(display_tokens) + cols - 1) // cols

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)

    for idx, tok in enumerate(display_tokens):
        row = idx // cols
        col = idx % cols
        y = rows - 1 - row

        if tok in token_colors:
            color = token_colors[tok]
        elif tok >= idx_offset:
            color = plt.cm.Pastel1((tok - idx_offset) % 9)
        else:
            color = "#ecf0f1"

        rect = plt.Rectangle(
            (col - 0.45, y - 0.45), 0.9, 0.9,
            facecolor=color, edgecolor="black", linewidth=0.5,
            joinstyle="round"
        )
        ax.add_patch(rect)

        label = str(tok - idx_offset) if tok >= idx_offset else token_names.get(tok, "?")
        ax.text(col, y, label, ha="center", va="center", fontsize=6, fontweight="bold")

    title_text = f"{title}" + (f" ({max_tokens}/{len(tokens)})" if truncated else "")
    ax.set_title(title_text, fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


# ===========================================================================
# Main Comparison Function
# ===========================================================================


def compare_all_tokenizations(
    smiles: str,
    name: str | None = None,
    output: str | None = None,
    show: bool = True,
    seed: int = 42,
) -> plt.Figure | None:
    """Create comparison of molecule, SENT, H-SENT, and HDT tokenization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    data = smiles_to_graph(smiles)
    if data is None:
        return None

    # Get consistent layout for molecule views
    pos = get_consistent_layout(data, seed)

    # Create tokenizers
    sent_tokenizer = SENTTokenizer(seed=seed)
    sent_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    hsent_tokenizer = HSENTTokenizer(seed=seed)
    hsent_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    hdt_tokenizer = HDTTokenizer(seed=seed, min_community_size=2)
    hdt_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    # Tokenize
    sent_tokens = sent_tokenizer.tokenize(data).tolist()
    hsent_tokens = hsent_tokenizer.tokenize(data).tolist()
    hdt_tokens = hdt_tokenizer.tokenize(data).tolist()

    # Get hierarchies
    hsent_hg = hsent_tokenizer.coarsener.build_hierarchy(data)
    hdt_hg = hdt_tokenizer.coarsener.build_hierarchy(data)

    # Create figure: 4 columns x 2 rows
    fig = plt.figure(figsize=(20, 10))

    # Title
    title = name or smiles[:35]
    if len(smiles) > 35:
        title += "..."
    fig.suptitle(
        f"{title}  ({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)",
        fontsize=14, fontweight="bold"
    )

    # Row 1: Graph visualizations
    ax1 = fig.add_subplot(2, 4, 1)
    plot_molecule_with_motifs(ax1, data, smiles, pos)

    ax2 = fig.add_subplot(2, 4, 2)
    plot_sent_walks(ax2, data, sent_tokens, sent_tokenizer, pos)

    ax3 = fig.add_subplot(2, 4, 3)
    plot_hsent_structure(ax3, data, hsent_hg, hsent_tokens, hsent_tokenizer, pos)

    ax4 = fig.add_subplot(2, 4, 4)
    plot_hdt_tree(ax4, data, hdt_hg, hdt_tokens, hdt_tokenizer)

    # Row 2: Token sequences
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.axis("off")
    ax5.text(0.5, 0.5, "Motif patterns detected\nin the molecule above",
             ha="center", va="center", fontsize=10, style="italic", color="gray")

    ax6 = fig.add_subplot(2, 4, 6)
    plot_tokens(ax6, sent_tokens, sent_tokenizer, f"SENT ({len(sent_tokens)} tok)")

    ax7 = fig.add_subplot(2, 4, 7)
    plot_tokens(ax7, hsent_tokens, hsent_tokenizer, f"H-SENT ({len(hsent_tokens)} tok)")

    ax8 = fig.add_subplot(2, 4, 8)
    plot_tokens(ax8, hdt_tokens, hdt_tokenizer, f"HDT ({len(hdt_tokens)} tok)")

    plt.tight_layout()

    # Stats
    stats = (
        f"SENT: {len(sent_tokens)} | "
        f"H-SENT: {len(hsent_tokens)} ({hsent_hg.num_communities} comm) | "
        f"HDT: {len(hdt_tokens)} ({hdt_hg.num_communities} comm)"
    )
    fig.text(0.5, 0.01, stats, ha="center", fontsize=10, style="italic", color="gray")

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")

    if show:
        plt.show()

    return fig


def run_demo(output_dir: str | None = None, show: bool = True):
    """Run demo with complex molecules containing multiple motifs."""
    demo_molecules = [
        ("cholesterol", MOLECULES["cholesterol"]),
        ("morphine", MOLECULES["morphine"]),
        ("caffeine", MOLECULES["caffeine"]),
        ("penicillin_g", MOLECULES["penicillin_g"]),
    ]

    for name, smiles in demo_molecules:
        print(f"\n{'='*60}")
        print(f"Comparing tokenization: {name}")
        print(f"SMILES: {smiles}")
        print("=" * 60)

        output = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output = f"{output_dir}/{name}_comparison.png"

        compare_all_tokenizations(
            smiles,
            name=name.replace("_", " ").title(),
            output=output,
            show=show,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare SENT, H-SENT, and HDT tokenization of molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    %(prog)s --smiles "c1ccccc1"
    %(prog)s --name caffeine --output caffeine_compare.png
    %(prog)s --demo --output-dir ./figures

Available molecules: {', '.join(sorted(MOLECULES.keys()))}
""",
    )
    parser.add_argument("--smiles", type=str, help="SMILES string to visualize")
    parser.add_argument("--name", type=str, help="Molecule name from predefined list")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--output-dir", type=str, help="Output directory for demo")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--list", action="store_true", help="List available molecules")
    parser.add_argument("--no-show", action="store_true", help="Don't display")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
        print("Please provide --smiles or --name")
        parser.print_help()
        return

    compare_all_tokenizations(
        smiles, name=name, output=args.output,
        show=not args.no_show, seed=args.seed,
    )


if __name__ == "__main__":
    main()
