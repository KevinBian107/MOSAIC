#!/usr/bin/env python
"""Visualization script comparing SENT, H-SENT, HDT, and HDTC tokenization.

This script provides side-by-side comparison of:
- Molecule structure with motif highlighting
- SENT: Flat random walk tokenization (different colors per walk segment)
- H-SENT: Hierarchical tokenization with community structure
- HDT: Hierarchical DFS tokenization with tree structure (motif-community)
- HDTC: Compositional tokenization with functional hierarchy

Usage:
    python scripts/visualize_tokenization.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
    python scripts/visualize_tokenization.py --name caffeine --output caffeine_compare.png
    python scripts/visualize_tokenization.py --demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer


# Common molecules for demos
MOLECULES = {
    # Simple / drug-like (MOSES-scale)
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
    "quercetin": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C(O)=C2C3=CC(O)=C(O)C=C3",
    "resveratrol": "OC1=CC=C(C=C1)/C=C/C2=CC(O)=CC(O)=C2",
    "estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
    "testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "indole_benzene": "c1ccc2[nH]ccc2c1-c3ccccc3",
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    # Natural products (COCONUT-scale)
    "strychnine": "O=C1C[C@H]2OCC=C3CN4CC[C@@]56[C@@H]4C[C@H]3[C@H]2[C@H]5N1c1ccccc16",
    "camptothecin": "CCC1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3ccccc3nc2-1",
    "vinblastine": "CCC1(O)[C@H]2CC3(CC)c4c(cc5c(c4OC)N(C=O)c4cc6c(cc4[C@H]5[C@@H]3[C@H](OC(C)=O)[C@@]1(O2)C(=O)OC)OCO6)C",
    "reserpine": "CO[C@H]1[C@@H](CC2CN3CCC4=C([C@H]3C[C@@H]2[C@@H]1C(=O)OC)NC5=CC(OC)=C(OC)C(OC)=C45)OC(=O)C6=CC(OC)=C(OC)C(OC)=C6",
    "taxol": "CC1=C2C(C(=O)C3(C)C(CC4OC(=O)C(C(c5ccccc5)NC(=O)c5ccccc5)O4)C3C2(C)C)C(OC(=O)C)C1OC(=O)c1ccccc1",
    "erythromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O",
    "artemisinin": "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@@]23OO4",
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

    # Set axis limits
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

    Shows visit order numbers on nodes based on walk sequence.
    """
    # Parse tokens into walk segments (visit order)
    # Use identity mapping since SENT preserves node count
    decoded_to_original = {i: i for i in range(data.num_nodes)}

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
            if visit_idx < data.num_nodes:
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
                        curve_amount=curve, arrow=True, zorder=2
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
    num_atoms = data.num_nodes

    # Fixed level height - taller for square aspect
    level_height = 3.5

    # Adaptive sizing based on complexity - use larger nodes
    if num_atoms > 20 or num_partitions > 6:
        node_spacing = 0.35
        part_gap = 0.5
        root_radius = 0.26
        part_radius = 0.20
        atom_radius = 0.15
        font_size_root = 9
        font_size_part = 7
        font_size_atom = 6
        font_size_level = 7
    else:
        node_spacing = 0.45
        part_gap = 0.6
        root_radius = 0.30
        part_radius = 0.24
        atom_radius = 0.18
        font_size_root = 10
        font_size_part = 8
        font_size_atom = 7
        font_size_level = 8

    positions = {}
    node_to_partition = {}

    # Root position (level 0)
    root_pos = (0, depth * level_height)
    positions["root"] = root_pos

    # Partition positions (level 1)
    partition_positions = {}
    partition_widths = []
    for part in hg.partitions:
        partition_widths.append(len(part.global_node_indices) * node_spacing)

    total_width = sum(partition_widths) + (num_partitions - 1) * part_gap
    current_x = -total_width / 2

    for part_idx, part in enumerate(hg.partitions):
        part_width = partition_widths[part_idx]
        part_center_x = current_x + part_width / 2
        partition_positions[part_idx] = (part_center_x, (depth - 1) * level_height)
        current_x += part_width + part_gap

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
            -total_width / 2 - 0.6, y,
            f"l={level}", fontsize=font_size_level, ha="right", va="center",
            style="italic", color="#666666"
        )

    # Community colors
    comm_colors = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"]

    # Draw bidirectional arrows from root to partitions (hierarchy traversal)
    for part_idx, part_pos in partition_positions.items():
        draw_bidirectional_arrow(
            ax, root_pos, part_pos,
            color="#333333", linewidth=1.5,
            curve_amount=0.12, zorder=2
        )

    # Draw bidirectional arrows from partitions to atoms (hierarchy traversal)
    for part_idx, part in enumerate(hg.partitions):
        part_pos = partition_positions[part_idx]
        for global_idx in part.global_node_indices:
            node_pos = positions[global_idx]
            draw_bidirectional_arrow(
                ax, part_pos, node_pos,
                color="#333333", linewidth=1.2,
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
                        color="#FF8C00", linewidth=2,
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
                    color="#FF8C00", linewidth=1.5, linestyle="--",
                    curve_amount=0.3, zorder=3
                )

    # Draw partition nodes (adaptive size)
    for part_idx, part in enumerate(hg.partitions):
        part_pos = partition_positions[part_idx]
        comm_color = comm_colors[part_idx % len(comm_colors)]

        circle = plt.Circle(
            part_pos, part_radius, facecolor=comm_color,
            edgecolor="black", linewidth=2, zorder=5
        )
        ax.add_patch(circle)
        ax.text(part_pos[0], part_pos[1], f"P{part_idx}", ha="center", va="center",
                fontsize=font_size_part, fontweight="bold", zorder=6)

    # Draw root node (adaptive size)
    circle = plt.Circle(
        root_pos, root_radius, facecolor="white",
        edgecolor="black", linewidth=2, zorder=5
    )
    ax.add_patch(circle)
    ax.text(root_pos[0], root_pos[1], "R", ha="center", va="center",
            fontsize=font_size_root, fontweight="bold", zorder=6)

    # Draw atom nodes (level 2) - adaptive size
    for part_idx, part in enumerate(hg.partitions):
        comm_color = comm_colors[part_idx % len(comm_colors)]
        for global_idx in part.global_node_indices:
            node_pos = positions[global_idx]
            circle = plt.Circle(
                node_pos, atom_radius, facecolor=comm_color,
                edgecolor="black", linewidth=1.5, zorder=5
            )
            ax.add_patch(circle)
            ax.text(node_pos[0], node_pos[1], str(global_idx), ha="center", va="center",
                    fontsize=font_size_atom, fontweight="bold", zorder=6)

    # Set axis limits with padding
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    x_margin = max(0.5, (max(all_x) - min(all_x)) * 0.05)
    y_margin = max(0.5, (max(all_y) - min(all_y)) * 0.1)
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#FF8C00", linewidth=2, label="Intra-partition"),
        plt.Line2D([0], [0], color="#FF8C00", linewidth=1.5, linestyle="--", label="Cross-partition"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.5, marker=">",
                   markersize=5, label="DFS traversal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6)

    ax.set_title("HDT: Hierarchical DFS Tree", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_hdtc_structure(
    ax: plt.Axes,
    data: Data,
    hierarchy,  # TwoLevelHierarchy
    tokens: list[int],
    tokenizer,
) -> None:
    """Plot HDTC as a full tree structure like HDT.

    Shows:
    - Root node at top (level 0)
    - Community nodes in middle (level 1)
    - Atom nodes at bottom (level 2)
    - Bidirectional arrows for hierarchy traversal
    - Graph edges (internal and cross-community)
    """
    ax.set_title("HDTC: Functional Hierarchy", fontsize=12, fontweight="bold")

    if hierarchy.num_communities == 0:
        ax.text(0.5, 0.5, "Empty hierarchy", ha="center", va="center")
        ax.axis("off")
        return

    num_communities = len(hierarchy.communities)
    num_atoms = data.num_nodes

    # Adaptive sizing based on complexity
    if num_atoms > 20 or num_communities > 6:
        node_spacing = 0.35
        comm_gap = 0.5
        root_radius = 0.26
        comm_radius = 0.20
        atom_radius = 0.15
        font_size_root = 9
        font_size_comm = 7
        font_size_atom = 6
    else:
        node_spacing = 0.45
        comm_gap = 0.6
        root_radius = 0.30
        comm_radius = 0.24
        atom_radius = 0.18
        font_size_root = 10
        font_size_comm = 8
        font_size_atom = 7

    # Fixed level height - taller for square aspect
    level_height = 3.5

    # Community type colors
    type_colors = {
        "ring": "#FF6B6B",       # Red for rings
        "functional": "#4ECDC4", # Teal for functional groups
        "singleton": "#95A5A6",  # Gray for singletons
    }

    positions = {}  # atom positions
    comm_positions = {}  # community positions

    # Calculate community widths
    comm_widths = [
        max(1, len(comm.atom_indices)) * node_spacing for comm in hierarchy.communities
    ]
    total_width = sum(comm_widths) + (num_communities - 1) * comm_gap

    # Root position (level 0)
    root_pos = (0, 2 * level_height)

    # Community positions (level 1)
    current_x = -total_width / 2
    for comm_idx, comm in enumerate(hierarchy.communities):
        comm_width = comm_widths[comm_idx]
        comm_center_x = current_x + comm_width / 2
        comm_positions[comm.community_id] = (comm_center_x, level_height)
        current_x += comm_width + comm_gap

        # Atom positions within community (level 2)
        num_atoms_in_comm = len(comm.atom_indices)
        start_x = comm_center_x - (num_atoms_in_comm - 1) * node_spacing / 2
        for local_idx, global_idx in enumerate(comm.atom_indices):
            node_x = start_x + local_idx * node_spacing
            positions[global_idx] = (node_x, 0)

    # Draw level labels
    ax.text(-total_width / 2 - 0.6, 2 * level_height, "l=0",
            fontsize=8, ha="right", va="center", style="italic", color="#666666")
    ax.text(-total_width / 2 - 0.6, level_height, "l=1",
            fontsize=8, ha="right", va="center", style="italic", color="#666666")
    ax.text(-total_width / 2 - 0.6, 0, "l=2",
            fontsize=8, ha="right", va="center", style="italic", color="#666666")

    # Draw bidirectional arrows from root to communities
    for comm in hierarchy.communities:
        comm_pos = comm_positions[comm.community_id]
        draw_bidirectional_arrow(
            ax, root_pos, comm_pos,
            color="#333333", linewidth=1.5,
            curve_amount=0.12, zorder=2
        )

    # Draw bidirectional arrows from communities to atoms
    for comm in hierarchy.communities:
        comm_pos = comm_positions[comm.community_id]
        for global_idx in comm.atom_indices:
            node_pos = positions[global_idx]
            draw_bidirectional_arrow(
                ax, comm_pos, node_pos,
                color="#333333", linewidth=1.2,
                curve_amount=0.15, zorder=2
            )

    # Draw community-to-community connections (super-edges at community level)
    comm_pairs_drawn = set()
    for se in hierarchy.super_edges:
        src_comm = se.source_community
        dst_comm = se.target_community
        pair_key = (min(src_comm, dst_comm), max(src_comm, dst_comm))

        if pair_key not in comm_pairs_drawn:
            comm_pairs_drawn.add(pair_key)
            if src_comm in comm_positions and dst_comm in comm_positions:
                src_pos = comm_positions[src_comm]
                dst_pos = comm_positions[dst_comm]
                draw_curved_edge(
                    ax, src_pos, dst_pos,
                    color="#2E86AB", linewidth=2, linestyle="-",
                    curve_amount=0.3, zorder=4
                )

    # Draw internal edges within communities - solid orange
    for comm in hierarchy.communities:
        drawn = set()
        for src, dst in comm.internal_edges:
            edge_key = (min(src, dst), max(src, dst))
            if edge_key not in drawn and src in positions and dst in positions:
                drawn.add(edge_key)
                draw_curved_edge(
                    ax, positions[src], positions[dst],
                    color="#FF8C00", linewidth=2,
                    curve_amount=0.2, zorder=3
                )

    # Draw cross-community edges (from super_edges) - dashed orange
    for se in hierarchy.super_edges:
        if se.source_atom in positions and se.target_atom in positions:
            src_pos = positions[se.source_atom]
            dst_pos = positions[se.target_atom]
            draw_curved_edge(
                ax, src_pos, dst_pos,
                color="#FF8C00", linewidth=1.5, linestyle="--",
                curve_amount=0.3, zorder=3
            )

    # Draw root node
    circle = plt.Circle(
        root_pos, root_radius, facecolor="white",
        edgecolor="black", linewidth=2, zorder=5
    )
    ax.add_patch(circle)
    ax.text(root_pos[0], root_pos[1], "R", ha="center", va="center",
            fontsize=font_size_root, fontweight="bold", zorder=6)

    # Draw community nodes
    for comm in hierarchy.communities:
        comm_pos = comm_positions[comm.community_id]
        comm_color = type_colors.get(comm.community_type, "#95A5A6")

        circle = plt.Circle(
            comm_pos, comm_radius, facecolor=comm_color,
            edgecolor="black", linewidth=2, zorder=5
        )
        ax.add_patch(circle)

        # Label: type initial + id
        type_initial = comm.community_type[0].upper()
        ax.text(
            comm_pos[0], comm_pos[1], f"{type_initial}{comm.community_id}",
            ha="center", va="center", fontsize=font_size_comm, fontweight="bold", zorder=6
        )

    # Draw atom nodes
    for comm in hierarchy.communities:
        comm_color = type_colors.get(comm.community_type, "#95A5A6")
        for global_idx in comm.atom_indices:
            if global_idx in positions:
                node_pos = positions[global_idx]
                circle = plt.Circle(
                    node_pos, atom_radius, facecolor=comm_color,
                    edgecolor="black", linewidth=1.5, zorder=5
                )
                ax.add_patch(circle)
                ax.text(
                    node_pos[0], node_pos[1], str(global_idx),
                    ha="center", va="center", fontsize=font_size_atom, fontweight="bold", zorder=6
                )

    # Set axis limits
    all_x = [p[0] for p in positions.values()] + [p[0] for p in comm_positions.values()] + [root_pos[0]]
    all_y = [p[1] for p in positions.values()] + [p[1] for p in comm_positions.values()] + [root_pos[1]]
    x_margin = max(0.5, (max(all_x) - min(all_x)) * 0.05)
    y_margin = max(0.5, (max(all_y) - min(all_y)) * 0.1)
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#FF6B6B", edgecolor="black", label="Ring"),
        mpatches.Patch(facecolor="#4ECDC4", edgecolor="black", label="Functional"),
        mpatches.Patch(facecolor="#95A5A6", edgecolor="black", label="Singleton"),
        plt.Line2D([0], [0], color="#2E86AB", linewidth=2, label="Super-edges"),
        plt.Line2D([0], [0], color="#FF8C00", linewidth=2, label="Intra-community"),
        plt.Line2D([0], [0], color="#FF8C00", linewidth=1.5, linestyle="--", label="Cross-community"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.5, marker=">",
                   markersize=5, label="DFS traversal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=5)

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
    max_tokens: int = 80,
) -> None:
    """Plot token sequence as flattened text with actual token symbols."""
    # Build token name mappings
    if hasattr(tokenizer, "COMM_START"):
        # HDTC tokenizer
        token_names = {
            tokenizer.SOS: "[SOS]", tokenizer.EOS: "[EOS]", tokenizer.PAD: "[PAD]",
            tokenizer.COMM_START: "{", tokenizer.COMM_END: "}",
            tokenizer.LEDGE: "[", tokenizer.REDGE: "]",
            tokenizer.SUPER_START: "<S", tokenizer.SUPER_END: "S>",
            tokenizer.TYPE_RING: "R", tokenizer.TYPE_FUNC: "F", tokenizer.TYPE_SINGLETON: "·",
        }
        idx_offset = tokenizer.IDX_OFFSET
    elif hasattr(tokenizer, "ENTER"):
        # HDT tokenizer
        token_names = {
            tokenizer.SOS: "[SOS]", tokenizer.EOS: "[EOS]", tokenizer.PAD: "[PAD]",
            tokenizer.ENTER: "↓", tokenizer.EXIT: "↑",
            tokenizer.LEDGE: "[", tokenizer.REDGE: "]",
        }
        idx_offset = tokenizer.IDX_OFFSET
    elif hasattr(tokenizer, "SOS"):
        # H-SENT tokenizer
        token_names = {
            tokenizer.SOS: "[SOS]", tokenizer.EOS: "[EOS]", tokenizer.PAD: "[PAD]",
            tokenizer.RESET: "[R]", tokenizer.LADJ: "[", tokenizer.RADJ: "]",
            tokenizer.LCOM: "{", tokenizer.RCOM: "}",
            tokenizer.LBIP: "<", tokenizer.RBIP: ">", tokenizer.SEP: "|",
        }
        idx_offset = tokenizer.IDX_OFFSET
    else:
        # SENT tokenizer
        token_names = {
            tokenizer.sos: "[SOS]", tokenizer.eos: "[EOS]", tokenizer.pad: "[PAD]",
            tokenizer.reset: "[R]", tokenizer.ladj: "[", tokenizer.radj: "]",
        }
        idx_offset = tokenizer.idx_offset

    # Convert tokens to string symbols
    display_tokens = tokens[:max_tokens]
    truncated = len(tokens) > max_tokens

    token_strs = []
    for tok in display_tokens:
        if tok in token_names:
            token_strs.append(token_names[tok])
        elif tok >= idx_offset:
            token_strs.append(str(tok - idx_offset))
        else:
            token_strs.append(f"?{tok}")

    # Join tokens with line breaks every N tokens
    tokens_per_line = 20
    lines = []
    for i in range(0, len(token_strs), tokens_per_line):
        lines.append(" ".join(token_strs[i:i + tokens_per_line]))
    token_text = "\n".join(lines)
    if truncated:
        token_text += " ..."

    # Display as multi-line text
    ax.text(
        0.5, 0.5, token_text,
        ha="center", va="center",
        fontsize=6,
        fontfamily="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dee2e6"),
    )

    title_text = title + (f" ({max_tokens}/{len(tokens)})" if truncated else "")
    ax.set_title(title_text, fontsize=10, fontweight="bold")
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
    """Create comparison of H-SENT, SENT, HDT, and HDTC tokenization.

    Layout (2x2 grid):
        (A) H-SENT community graph  |  (B) SENT walks
        (C) HDT tree                 |  (D) HDTC functional hierarchy
    """
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

    # HDT uses spectral coarsening (same as HSENT) for consistency
    hdt_tokenizer = HDTTokenizer(seed=seed, min_community_size=2)
    hdt_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    hdtc_tokenizer = HDTCTokenizer(seed=seed)
    hdtc_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    # Build shared spectral hierarchy for HSENT and HDT
    shared_hg = hsent_tokenizer.coarsener.build_hierarchy(data)

    # Tokenize
    sent_tokens = sent_tokenizer.tokenize(data).tolist()

    hsent_tokens = None
    hsent_hg = None
    try:
        hsent_tokens = hsent_tokenizer.tokenize(data).tolist()
        hsent_hg = shared_hg
    except (IndexError, ValueError) as e:
        print(f"  Warning: H-SENT failed ({e}), skipping H-SENT visualization")

    hdt_tokens = hdt_tokenizer.tokenize(data).tolist()
    hdtc_tokens = hdtc_tokenizer.tokenize(data).tolist()

    # Use shared hierarchy for HDT tree rendering (same communities as HSENT)
    hdt_hg = shared_hg
    hdtc_hierarchy = hdtc_tokenizer.hierarchy_builder.build(data)

    # Create figure: graphs on top row, legends beneath
    fig = plt.figure(figsize=(40, 12))

    title = name or smiles[:35]
    if len(smiles) > 35:
        title += "..."
    fig.suptitle(
        f"{title}  ({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)",
        fontsize=14, fontweight="bold"
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1.5, 1.5],
        height_ratios=[4, 1],
        hspace=-0.15, wspace=0.12
    )

    # (A) SENT walks (swapped: was B)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_sent_walks(ax_a, data, sent_tokens, sent_tokenizer, pos)
    ax_a.text(
        -0.02, 1.05, "(A)", transform=ax_a.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right"
    )

    # (B) H-SENT community graph (swapped: was A)
    ax_b = fig.add_subplot(gs[0, 1])
    if hsent_hg is not None and hsent_tokens is not None:
        plot_hsent_structure(ax_b, data, hsent_hg, hsent_tokens, hsent_tokenizer, pos)
    else:
        ax_b.text(
            0.5, 0.5, "H-SENT failed\n(spectral coarsening error)",
            ha="center", va="center", fontsize=10, color="gray", style="italic",
            transform=ax_b.transAxes
        )
        ax_b.set_title("H-SENT: Error", fontsize=11, fontweight="bold")
        ax_b.axis("off")
    ax_b.text(
        -0.02, 1.05, "(B)", transform=ax_b.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right"
    )

    # (C) HDT tree
    ax_c = fig.add_subplot(gs[0, 2])
    plot_hdt_tree(ax_c, data, hdt_hg, hdt_tokens, hdt_tokenizer)
    ax_c.text(
        -0.02, 1.03, "(C)", transform=ax_c.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right"
    )

    # (D) HDTC functional hierarchy
    ax_d = fig.add_subplot(gs[0, 3])
    plot_hdtc_structure(ax_d, data, hdtc_hierarchy, hdtc_tokens, hdtc_tokenizer)
    ax_d.text(
        -0.02, 1.03, "(D)", transform=ax_d.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right"
    )

    # Move legends from inside axes to dedicated legend row beneath
    for col, ax in enumerate([ax_a, ax_b, ax_c, ax_d]):
        leg = ax.get_legend()
        if leg is not None:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_leg = fig.add_subplot(gs[1, col])
            ax_leg.axis("off")
            # Use 2 rows for legends with many items
            ncols = (len(handles) + 1) // 2
            ax_leg.legend(
                handles, labels, loc="upper center",
                fontsize=10, ncol=ncols,
                frameon=False, handlelength=1.5
            )

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
        description="Compare SENT, H-SENT, HDT, and HDTC tokenization of molecules",
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
