#!/usr/bin/env python
"""Compare HAC vs Spectral community structures on MOSES vs COCONUT.

Produces two types of figures:
1. Example progression figures (4 molecules, small→large, 2x2 grid each)
   showing molecule+community overlay and hierarchy tree for both HAC and Spectral.
2. Aggregate statistics figure (2x3 grid) comparing distributions across
   MOSES-HAC, MOSES-Spectral, COCONUT-HAC, COCONUT-Spectral.

Usage:
    python scripts/visualization/compare_community_structure.py
    python scripts/visualization/compare_community_structure.py --output-dir figures/
    python scripts/visualization/compare_community_structure.py --num-stats-samples 500
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MPath
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.molecular import smiles_to_graph  # noqa: E402
from src.tokenizers.coarsening.hac import AffinityCoarsening  # noqa: E402
from src.tokenizers.coarsening.spectral import SpectralCoarsening  # noqa: E402
from src.tokenizers.structures import HierarchicalGraph, Partition  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Color palette
# ============================================================================

COMMUNITY_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#bcbd22",  # Yellow-green
    "#17becf",  # Cyan
    "#7f7f7f",  # Gray
    "#aec7e8",  # Light blue
    "#ffbb78",  # Light orange
    "#98df8a",  # Light green
    "#ff9896",  # Light red
    "#c5b0d5",  # Light purple
]

GROUP_COLORS = {
    "MOSES-HAC": "#1f77b4",
    "MOSES-Spectral": "#aec7e8",
    "COCONUT-HAC": "#d62728",
    "COCONUT-Spectral": "#ff9896",
}

SINGLETON_COLOR = "#D3D3D3"

# ============================================================================
# Section 1: Data Loading
# ============================================================================


def load_smiles_from_cache(cache_path: str) -> list[str]:
    """Load SMILES strings from a .pt cache file.

    Args:
        cache_path: Path to the .pt cache file.

    Returns:
        List of SMILES strings.
    """
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    return data["smiles"]


def analyze_molecule(
    smiles: str,
    coarsener: AffinityCoarsening | SpectralCoarsening,
) -> dict[str, Any] | None:
    """Convert SMILES to graph, build hierarchy, and compute stats.

    Args:
        smiles: SMILES string.
        coarsener: Coarsening strategy (HAC or Spectral).

    Returns:
        Dictionary with analysis results, or None if conversion fails.
    """
    data = smiles_to_graph(smiles, labeled=True)
    if data is None or data.num_nodes < 4:
        return None

    try:
        hg = coarsener.build_hierarchy(data)
    except Exception:
        return None

    stats = compute_hierarchy_stats(hg)
    return {
        "smiles": smiles,
        "data": data,
        "hg": hg,
        "num_atoms": data.num_nodes,
        **stats,
    }


def analyze_dataset(
    smiles_list: list[str],
    coarsener: AffinityCoarsening | SpectralCoarsening,
    max_samples: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Analyze a batch of molecules with a given coarsener.

    Args:
        smiles_list: List of SMILES strings.
        coarsener: Coarsening strategy.
        max_samples: Maximum number of molecules to analyze.
        seed: Random seed for sampling.

    Returns:
        List of analysis result dictionaries.
    """
    rng = random.Random(seed)
    sample = (
        rng.sample(smiles_list, min(max_samples, len(smiles_list)))
        if len(smiles_list) > max_samples
        else list(smiles_list)
    )

    results = []
    for smiles in sample:
        result = analyze_molecule(smiles, coarsener)
        if result is not None:
            results.append(result)

    return results


# ============================================================================
# Section 2: Hierarchy Utilities
# ============================================================================


def compute_hierarchy_stats(hg: HierarchicalGraph) -> dict[str, Any]:
    """Compute recursive statistics for a hierarchical graph.

    Args:
        hg: HierarchicalGraph object.

    Returns:
        Dictionary with depth, num_communities, singleton_fraction,
        largest_community_size, community_sizes, non_singleton_sizes.
    """
    depth = hg.depth + 1  # +1 because depth=0 means single-level
    num_communities = hg.num_communities
    sizes = [p.num_nodes for p in hg.partitions]
    singletons = sum(1 for s in sizes if s == 1)
    singleton_fraction = singletons / max(num_communities, 1)
    largest = max(sizes) if sizes else 0
    non_singleton_sizes = [s for s in sizes if s > 1]

    return {
        "depth": depth,
        "num_communities": num_communities,
        "singleton_fraction": singleton_fraction,
        "largest_community_size": largest,
        "community_sizes": sizes,
        "non_singleton_sizes": non_singleton_sizes,
    }


def flatten_hierarchy_tree(
    hg: HierarchicalGraph, max_depth: int = 4
) -> list[dict[str, Any]]:
    """Flatten nested hierarchy into renderable tree nodes.

    Args:
        hg: HierarchicalGraph object.
        max_depth: Maximum depth to render.

    Returns:
        List of node dicts with keys: id, label, size, children,
        level, x, y, color, is_singleton_group.
    """
    nodes: list[dict[str, Any]] = []
    node_id_counter = [0]

    def _recurse(partition: Partition, level: int, parent_id: int | None) -> int:
        nid = node_id_counter[0]
        node_id_counter[0] += 1

        is_leaf = partition.child_hierarchy is None
        children_ids = []

        if not is_leaf and level < max_depth:
            child_hg = partition.child_hierarchy
            # Group singletons
            singleton_nodes = []
            for cp in child_hg.partitions:
                if cp.num_nodes == 1:
                    singleton_nodes.append(cp)
                else:
                    cid = _recurse(cp, level + 1, nid)
                    children_ids.append(cid)

            if singleton_nodes:
                # Create singleton group node
                sg_id = node_id_counter[0]
                node_id_counter[0] += 1
                nodes.append(
                    {
                        "id": sg_id,
                        "label": f"S x{len(singleton_nodes)}",
                        "size": len(singleton_nodes),
                        "level": level + 1,
                        "parent_id": nid,
                        "children": [],
                        "is_singleton_group": True,
                        "color": SINGLETON_COLOR,
                    }
                )
                children_ids.append(sg_id)

        nodes.append(
            {
                "id": nid,
                "label": f"C{nid}" if partition.num_nodes > 1 else "s",
                "size": partition.num_nodes,
                "level": level,
                "parent_id": parent_id,
                "children": children_ids,
                "is_singleton_group": False,
                "color": COMMUNITY_COLORS[nid % len(COMMUNITY_COLORS)],
            }
        )
        return nid

    # Root node
    root_id = node_id_counter[0]
    node_id_counter[0] += 1
    root_children = []

    # Group top-level singletons
    top_singletons = []
    for part in hg.partitions:
        if part.num_nodes == 1:
            top_singletons.append(part)
        else:
            cid = _recurse(part, 1, root_id)
            root_children.append(cid)

    if top_singletons:
        sg_id = node_id_counter[0]
        node_id_counter[0] += 1
        nodes.append(
            {
                "id": sg_id,
                "label": f"S x{len(top_singletons)}",
                "size": len(top_singletons),
                "level": 1,
                "parent_id": root_id,
                "children": [],
                "is_singleton_group": True,
                "color": SINGLETON_COLOR,
            }
        )
        root_children.append(sg_id)

    nodes.append(
        {
            "id": root_id,
            "label": "Root",
            "size": hg.num_nodes,
            "level": 0,
            "parent_id": None,
            "children": root_children,
            "is_singleton_group": False,
            "color": "white",
        }
    )

    return nodes


# ============================================================================
# Section 3: Layout
# ============================================================================


def compute_rdkit_2d_layout(smiles: str) -> dict[int, tuple[float, float]] | None:
    """Compute 2D molecular layout using RDKit.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary mapping atom index to (x, y) coordinates, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer()

        pos = {}
        for i in range(mol.GetNumAtoms()):
            atom_pos = conformer.GetAtomPosition(i)
            pos[i] = (atom_pos.x, atom_pos.y)

        if not pos:
            return None

        # Normalize to [-1, 1] range
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
        y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1
        scale = max(x_range, y_range, 1e-6)
        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2

        for node_id in pos:
            x, y = pos[node_id]
            pos[node_id] = ((x - x_center) / scale, (y - y_center) / scale)

        return pos

    except Exception:
        return None


def compute_spring_layout(data: Data, seed: int = 42) -> dict[int, tuple[float, float]]:
    """Fallback spring layout using networkx.

    Args:
        data: PyG Data object.
        seed: Random seed.

    Returns:
        Dictionary mapping node index to (x, y) coordinates.
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    return nx.spring_layout(G, seed=seed, k=1.5, iterations=100)


# ============================================================================
# Section 4: Drawing Utilities
# ============================================================================


def draw_curved_edge(
    ax: plt.Axes,
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    color: str = "gray",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    linestyle: str = "-",
    curve_amount: float = 0.15,
    zorder: int = 1,
) -> None:
    """Draw a curved edge between two points using a quadratic bezier curve.

    Args:
        ax: Matplotlib axes.
        pos1: Start position (x, y).
        pos2: End position (x, y).
        color: Edge color.
        linewidth: Line width.
        alpha: Transparency.
        linestyle: Line style ("-", "--", etc.).
        curve_amount: Curvature factor.
        zorder: Z-order for rendering.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001:
        return

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    px, py = -dy / dist, dx / dist
    cx = mx + px * curve_amount * dist
    cy = my + py * curve_amount * dist

    verts = [(x1, y1), (cx, cy), (x2, y2)]
    codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
    path = MPath(verts, codes)

    from matplotlib.patches import PathPatch

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)


def draw_community_hull(
    ax: plt.Axes,
    positions: dict[int, tuple[float, float]],
    node_indices: list[int],
    color: str,
    alpha: float = 0.15,
    pad: float = 0.06,
    linestyle: str = "-",
    linewidth: float = 1.5,
    zorder: int = 0,
) -> None:
    """Draw a convex hull or ellipse around community nodes.

    Args:
        ax: Matplotlib axes.
        positions: Node positions.
        node_indices: Indices of nodes in the community.
        color: Hull color.
        alpha: Fill alpha.
        pad: Padding around hull.
        linestyle: Border line style.
        linewidth: Border line width.
        zorder: Z-order for rendering.
    """
    valid = [i for i in node_indices if i in positions]
    if len(valid) < 2:
        return  # Skip singletons and missing nodes

    pts = np.array([positions[i] for i in valid])

    if len(valid) == 2:
        # Draw an ellipse between 2 points
        cx, cy = pts.mean(axis=0)
        dx, dy = pts[1] - pts[0]
        width = np.sqrt(dx**2 + dy**2) + pad * 4
        height = pad * 4
        angle = np.degrees(np.arctan2(dy, dx))
        ellipse = mpatches.Ellipse(
            (cx, cy),
            width,
            height,
            angle=angle,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        ax.add_patch(ellipse)
        return

    # 3+ points: convex hull with padding
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]

        # Pad outward
        centroid = hull_pts.mean(axis=0)
        padded = []
        for pt in hull_pts:
            direction = pt - centroid
            norm = np.linalg.norm(direction)
            if norm > 0:
                padded.append(pt + direction / norm * pad)
            else:
                padded.append(pt)
        padded = np.array(padded)

        polygon = mpatches.Polygon(
            padded,
            closed=True,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        ax.add_patch(polygon)
    except Exception:
        pass  # Degenerate hull (collinear points)


# ============================================================================
# Section 5: Molecule + Community Overlay
# ============================================================================


def plot_molecule_with_communities(
    ax: plt.Axes,
    data: Data,
    smiles: str,
    hg: HierarchicalGraph,
    pos: dict[int, tuple[float, float]],
    title: str,
    max_hull_depth: int = 2,
) -> None:
    """Plot molecule structure with community overlay hulls.

    Level-1 hulls: solid border, alpha=0.15
    Level-2 sub-hulls: dashed border, alpha=0.08
    Intra-community edges: colored. Inter-community edges: gray dashed.
    Singletons: gray nodes without hull.

    Args:
        ax: Matplotlib axes.
        data: PyG Data object.
        smiles: SMILES string.
        hg: HierarchicalGraph object.
        pos: Node positions.
        title: Subplot title.
        max_hull_depth: Maximum hull nesting depth to display.
    """
    # Layer 1: Level-1 hulls
    for part_idx, part in enumerate(hg.partitions):
        if part.num_nodes >= 2:
            color = COMMUNITY_COLORS[part_idx % len(COMMUNITY_COLORS)]
            draw_community_hull(
                ax,
                pos,
                part.global_node_indices,
                color=color,
                alpha=0.15,
                pad=0.08,
                linestyle="-",
                linewidth=1.5,
                zorder=0,
            )

            # Layer 2: Level-2 sub-hulls if child hierarchy exists
            if (
                max_hull_depth >= 2
                and part.child_hierarchy is not None
                and part.child_hierarchy.num_communities > 1
            ):
                for sub_part in part.child_hierarchy.partitions:
                    if sub_part.num_nodes >= 2:
                        draw_community_hull(
                            ax,
                            pos,
                            sub_part.global_node_indices,
                            color=color,
                            alpha=0.08,
                            pad=0.04,
                            linestyle="--",
                            linewidth=1.0,
                            zorder=0,
                        )

    # Build node-to-community mapping
    node_to_comm = {}
    for part_idx, part in enumerate(hg.partitions):
        for node in part.global_node_indices:
            node_to_comm[node] = part_idx

    # Layer 3: Edges
    drawn_edges = set()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_key = (min(u, v), max(u, v))
        if edge_key in drawn_edges:
            continue
        drawn_edges.add(edge_key)

        if u not in pos or v not in pos:
            continue

        comm_u = node_to_comm.get(u, -1)
        comm_v = node_to_comm.get(v, -1)

        if comm_u == comm_v and comm_u >= 0:
            color = COMMUNITY_COLORS[comm_u % len(COMMUNITY_COLORS)]
            draw_curved_edge(
                ax,
                pos[u],
                pos[v],
                color=color,
                linewidth=2.0,
                curve_amount=0.08,
                alpha=0.8,
                zorder=1,
            )
        else:
            draw_curved_edge(
                ax,
                pos[u],
                pos[v],
                color="#999999",
                linewidth=1.2,
                linestyle="--",
                curve_amount=0.12,
                alpha=0.5,
                zorder=1,
            )

    # Layer 4: Nodes
    for node in range(data.num_nodes):
        if node not in pos:
            continue
        x, y = pos[node]
        comm = node_to_comm.get(node, -1)
        part_size = (
            hg.partitions[comm].num_nodes if 0 <= comm < len(hg.partitions) else 1
        )

        if part_size == 1:
            node_color = SINGLETON_COLOR
        else:
            node_color = COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)]

        circle = plt.Circle(
            (x, y),
            0.04,
            facecolor=node_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(circle)

    # Layer 5: Labels for smaller molecules
    if data.num_nodes <= 40:
        for node in range(data.num_nodes):
            if node not in pos:
                continue
            x, y = pos[node]
            ax.text(
                x,
                y,
                str(node),
                ha="center",
                va="center",
                fontsize=5,
                fontweight="bold",
                zorder=4,
            )

    # Stats annotation
    stats = compute_hierarchy_stats(hg)
    stats_text = (
        f"comms={stats['num_communities']}, "
        f"depth={stats['depth']}, "
        f"sing={stats['singleton_fraction']:.0%}"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=7,
        color="gray",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.15
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


# ============================================================================
# Section 6: Hierarchy Tree
# ============================================================================


def plot_hierarchy_tree(
    ax: plt.Axes,
    hg: HierarchicalGraph,
    title: str,
    max_depth: int = 4,
    max_singletons_shown: int = 3,
) -> None:
    """Plot hierarchical community tree with proportional-width layout.

    Root at top, proportional-width layout, singletons collapsed,
    node radius proportional to sqrt(size), color-coded, level labels.

    Args:
        ax: Matplotlib axes.
        hg: HierarchicalGraph object.
        title: Subplot title.
        max_depth: Maximum tree depth to display.
        max_singletons_shown: Maximum individual singletons before grouping.
    """
    tree_nodes = flatten_hierarchy_tree(hg, max_depth=max_depth)
    if not tree_nodes:
        ax.text(
            0.5,
            0.5,
            "Empty hierarchy",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    # Build lookup
    node_map = {n["id"]: n for n in tree_nodes}
    root = None
    for n in tree_nodes:
        if n["parent_id"] is None:
            root = n
            break

    if root is None:
        ax.text(
            0.5, 0.5, "No root found", ha="center", va="center", transform=ax.transAxes
        )
        ax.axis("off")
        return

    # Compute leaf counts for proportional layout
    def count_leaves(nid: int) -> int:
        node = node_map[nid]
        if not node["children"]:
            return 1
        return sum(count_leaves(cid) for cid in node["children"])

    # Layout: proportional width allocation
    positions: dict[int, tuple[float, float]] = {}
    level_height = 1.0

    def layout(nid: int, x_start: float, x_end: float, level: int) -> None:
        node = node_map[nid]
        x_mid = (x_start + x_end) / 2
        y = -level * level_height
        positions[nid] = (x_mid, y)

        children = node["children"]
        if not children:
            return

        # Proportional width allocation
        total_leaves = sum(count_leaves(cid) for cid in children)
        if total_leaves == 0:
            total_leaves = len(children)

        current_x = x_start
        for cid in children:
            child_leaves = count_leaves(cid)
            fraction = child_leaves / total_leaves
            child_width = (x_end - x_start) * fraction
            layout(cid, current_x, current_x + child_width, level + 1)
            current_x += child_width

    layout(root["id"], 0, 10, 0)

    # Determine max level rendered
    max_level = max(node_map[nid]["level"] for nid in positions)

    # Draw edges
    for n in tree_nodes:
        if n["id"] not in positions:
            continue
        for cid in n["children"]:
            if cid not in positions:
                continue
            px, py = positions[n["id"]]
            cx, cy = positions[cid]
            ax.plot(
                [px, cx],
                [py, cy],
                color="#666666",
                linewidth=1.0,
                zorder=1,
            )

    # Draw nodes
    max_size = max((n["size"] for n in tree_nodes), default=1)
    for n in tree_nodes:
        if n["id"] not in positions:
            continue
        x, y = positions[n["id"]]

        # Radius proportional to sqrt(size)
        radius = 0.1 + 0.25 * np.sqrt(n["size"] / max(max_size, 1))
        radius = min(radius, 0.4)

        color = n["color"]
        if n["is_singleton_group"]:
            color = SINGLETON_COLOR

        circle = plt.Circle(
            (x, y),
            radius,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(circle)

        # Label
        label = n["label"]
        fontsize = 6 if len(label) > 4 else 7
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            zorder=4,
        )

    # Level labels
    for lvl in range(max_level + 1):
        y = -lvl * level_height
        ax.text(
            -0.3,
            y,
            f"L{lvl}",
            fontsize=7,
            ha="right",
            va="center",
            style="italic",
            color="#666666",
        )

    # Depth overflow indicator
    actual_depth = hg.depth + 1
    if actual_depth > max_depth:
        ax.text(
            0.98,
            0.02,
            f"... (+{actual_depth - max_depth} levels)",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="gray",
            style="italic",
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    x_margin = max(0.5, (max(all_x) - min(all_x)) * 0.08)
    y_margin = max(0.5, (max(all_y) - min(all_y)) * 0.15)
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)


# ============================================================================
# Section 7: Example Figure Generation
# ============================================================================


def select_example_molecules(
    moses_analyses: list[dict[str, Any]],
    coconut_analyses: list[dict[str, Any]],
    num_examples: int = 4,
) -> list[dict[str, Any]]:
    """Select molecules for example figures spanning atom count range.

    Computes evenly-spaced atom count targets across the combined range
    of both datasets, then picks the closest molecule from either dataset
    for each target.

    Args:
        moses_analyses: MOSES analysis results.
        coconut_analyses: COCONUT analysis results.
        num_examples: Number of examples to select.

    Returns:
        List of analysis dicts with 'dataset' field added, sorted by
        increasing atom count.
    """
    # Pool all analyses with dataset labels
    all_analyses = []
    for a in moses_analyses:
        entry = dict(a)
        entry["dataset"] = "MOSES"
        all_analyses.append(entry)
    for a in coconut_analyses:
        entry = dict(a)
        entry["dataset"] = "COCONUT"
        all_analyses.append(entry)

    if not all_analyses:
        return []

    # Compute atom count range and evenly-spaced targets
    atom_counts = [a["num_atoms"] for a in all_analyses]
    min_atoms = min(atom_counts)
    max_atoms = max(atom_counts)

    if num_examples == 1:
        targets = [(min_atoms + max_atoms) / 2]
    else:
        targets = np.linspace(min_atoms, max_atoms, num_examples).tolist()

    # Greedily pick closest molecule to each target
    selected = []
    used_smiles: set[str] = set()

    for target in targets:
        best = None
        best_dist = float("inf")
        for a in all_analyses:
            if a["smiles"] in used_smiles:
                continue
            dist = abs(a["num_atoms"] - target)
            if dist < best_dist:
                best_dist = dist
                best = a
        if best is not None:
            selected.append(best)
            used_smiles.add(best["smiles"])

    # Sort by atom count for clean progression
    selected.sort(key=lambda a: a["num_atoms"])
    return selected


def create_example_figure(
    smiles: str,
    data: Data,
    hac_hg: HierarchicalGraph,
    spectral_hg: HierarchicalGraph,
    idx: int,
    dataset: str,
    output_path: str,
    dpi: int = 150,
) -> plt.Figure:
    """Create a 2x2 example figure for a single molecule.

    Layout:
        [HAC Communities]     [Spectral Communities]
        [HAC Hierarchy Tree]  [Spectral Hierarchy Tree]

    Args:
        smiles: SMILES string.
        data: PyG Data object.
        hac_hg: HAC HierarchicalGraph.
        spectral_hg: Spectral HierarchicalGraph.
        idx: Example index.
        dataset: Dataset name ("MOSES" or "COCONUT").
        output_path: Path to save the figure.
        dpi: Output DPI.

    Returns:
        Matplotlib Figure.
    """
    pos = compute_rdkit_2d_layout(smiles)
    if pos is None:
        pos = compute_spring_layout(data)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[3, 3, 0.3],
        hspace=0.25,
        wspace=0.2,
    )

    # Row 1: Molecule + communities
    ax_hac_mol = fig.add_subplot(gs[0, 0])
    plot_molecule_with_communities(
        ax_hac_mol,
        data,
        smiles,
        hac_hg,
        pos,
        "HAC Communities",
    )

    ax_spec_mol = fig.add_subplot(gs[0, 1])
    plot_molecule_with_communities(
        ax_spec_mol,
        data,
        smiles,
        spectral_hg,
        pos,
        "Spectral Communities",
    )

    # Row 2: Hierarchy trees
    ax_hac_tree = fig.add_subplot(gs[1, 0])
    plot_hierarchy_tree(ax_hac_tree, hac_hg, "HAC Hierarchy Tree")

    ax_spec_tree = fig.add_subplot(gs[1, 1])
    plot_hierarchy_tree(ax_spec_tree, spectral_hg, "Spectral Hierarchy Tree")

    # Row 3: Stats bar
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")

    hac_stats = compute_hierarchy_stats(hac_hg)
    spec_stats = compute_hierarchy_stats(spectral_hg)
    stats_text = (
        f"atoms={data.num_nodes}  |  "
        f"HAC: comms={hac_stats['num_communities']}, "
        f"depth={hac_stats['depth']}, "
        f"singleton={hac_stats['singleton_fraction']:.0%}  |  "
        f"Spectral: comms={spec_stats['num_communities']}, "
        f"depth={spec_stats['depth']}, "
        f"singleton={spec_stats['singleton_fraction']:.0%}"
    )
    ax_stats.text(
        0.5,
        0.5,
        stats_text,
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dee2e6"),
        transform=ax_stats.transAxes,
    )

    # Truncate SMILES for title
    smiles_display = smiles[:50] + "..." if len(smiles) > 50 else smiles
    fig.suptitle(
        f"Example {idx + 1}: {dataset} ({data.num_nodes} atoms)\n{smiles_display}",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return fig


# ============================================================================
# Section 8: Aggregate Statistics
# ============================================================================


def create_aggregate_figure(
    moses_hac: list[dict[str, Any]],
    moses_spectral: list[dict[str, Any]],
    coconut_hac: list[dict[str, Any]],
    coconut_spectral: list[dict[str, Any]],
    output_path: str,
    dpi: int = 150,
) -> plt.Figure:
    """Create aggregate statistics comparison figure.

    2x3 grid:
        Row 1: Hierarchy Depth | Num Communities | Largest Community Size
        Row 2: Singleton Fraction | Non-singleton Sizes | Depth vs Atom Count

    Args:
        moses_hac: MOSES HAC analysis results.
        moses_spectral: MOSES Spectral analysis results.
        coconut_hac: COCONUT HAC analysis results.
        coconut_spectral: COCONUT Spectral analysis results.
        output_path: Path to save the figure.
        dpi: Output DPI.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    groups = {
        "MOSES-HAC": moses_hac,
        "MOSES-Spectral": moses_spectral,
        "COCONUT-HAC": coconut_hac,
        "COCONUT-Spectral": coconut_spectral,
    }

    # Row 1, Col 1: Hierarchy Depth
    _plot_comparison_histogram(
        axes[0, 0],
        {k: [a["depth"] for a in v] for k, v in groups.items()},
        "Hierarchy Depth",
        "Depth",
        bins=range(0, 8),
    )

    # Row 1, Col 2: Num Communities
    _plot_comparison_histogram(
        axes[0, 1],
        {k: [a["num_communities"] for a in v] for k, v in groups.items()},
        "Number of Communities",
        "Communities",
        bins=20,
    )

    # Row 1, Col 3: Largest Community Size
    _plot_comparison_histogram(
        axes[0, 2],
        {k: [a["largest_community_size"] for a in v] for k, v in groups.items()},
        "Largest Community Size",
        "Nodes",
        bins=20,
    )

    # Row 2, Col 1: Singleton Fraction
    _plot_comparison_histogram(
        axes[1, 0],
        {k: [a["singleton_fraction"] for a in v] for k, v in groups.items()},
        "Singleton Fraction",
        "Fraction",
        bins=np.linspace(0, 1, 21),
    )

    # Row 2, Col 2: Non-singleton Sizes box plot
    _plot_non_singleton_boxplot(
        axes[1, 1],
        {
            k: [s for a in v for s in a["non_singleton_sizes"]]
            for k, v in groups.items()
        },
    )

    # Row 2, Col 3: Depth vs Atom Count scatter
    _plot_depth_vs_atoms_scatter(axes[1, 2], groups)

    fig.suptitle(
        "Community Structure: HAC vs Spectral on MOSES vs COCONUT",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return fig


def _plot_comparison_histogram(
    ax: plt.Axes,
    values_dict: dict[str, list[float]],
    title: str,
    xlabel: str,
    bins: Any = 20,
) -> None:
    """Plot overlapping histograms for 4 groups.

    Args:
        ax: Matplotlib axes.
        values_dict: Dict mapping group name to list of values.
        title: Plot title.
        xlabel: X-axis label.
        bins: Histogram bin specification.
    """
    for group_name, values in values_dict.items():
        if not values:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.4,
            label=f"{group_name} (n={len(values)})",
            color=GROUP_COLORS[group_name],
            edgecolor=GROUP_COLORS[group_name],
            linewidth=1.0,
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def _plot_non_singleton_boxplot(
    ax: plt.Axes,
    sizes_dict: dict[str, list[int]],
) -> None:
    """Box plot of non-singleton community sizes.

    Args:
        ax: Matplotlib axes.
        sizes_dict: Dict mapping group name to list of community sizes.
    """
    data_list = []
    labels = []
    colors = []
    for group_name, sizes in sizes_dict.items():
        if sizes:
            data_list.append(sizes)
            labels.append(group_name.replace("-", "\n"))
            colors.append(GROUP_COLORS[group_name])

    if not data_list:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    bp = ax.boxplot(
        data_list,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("Non-singleton Community Sizes", fontsize=10, fontweight="bold")
    ax.set_ylabel("Nodes per Community", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=7)


def _plot_depth_vs_atoms_scatter(
    ax: plt.Axes,
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    """Scatter plot of hierarchy depth vs atom count with trend lines.

    Args:
        ax: Matplotlib axes.
        groups: Dict mapping group name to analysis results.
    """
    for group_name, analyses in groups.items():
        if not analyses:
            continue
        atoms = [a["num_atoms"] for a in analyses]
        depths = [a["depth"] for a in analyses]

        ax.scatter(
            atoms,
            depths,
            alpha=0.3,
            s=15,
            color=GROUP_COLORS[group_name],
            label=group_name,
        )

        # Trend line
        if len(atoms) > 5:
            z = np.polyfit(atoms, depths, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(atoms), max(atoms), 50)
            ax.plot(
                x_range,
                p(x_range),
                color=GROUP_COLORS[group_name],
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

    ax.set_title("Depth vs Atom Count", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Atoms", fontsize=9)
    ax.set_ylabel("Hierarchy Depth", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


# ============================================================================
# Section 9: CLI
# ============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare HAC vs Spectral community structures on MOSES vs COCONUT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp/feature/hac-improvement/community_comparison",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="Number of example molecules to visualize",
    )
    parser.add_argument(
        "--num-stats-samples",
        type=int,
        default=200,
        help="Number of molecules per dataset for aggregate statistics",
    )
    parser.add_argument(
        "--moses-cache",
        type=str,
        default="data/cache/moses_train_hdt_1000_d111408d.pt",
        help="Path to MOSES cache .pt file",
    )
    parser.add_argument(
        "--coconut-cache",
        type=str,
        default="data/cache/coconut_train_hdt_5000_d111408d.pt",
        help="Path to COCONUT cache .pt file",
    )
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=4,
        help="Minimum community size for coarsening",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-show", action="store_true", help="Don't display figures")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SMILES
    print("Loading SMILES from cache files...")
    moses_smiles = load_smiles_from_cache(args.moses_cache)
    coconut_smiles = load_smiles_from_cache(args.coconut_cache)
    print(f"  MOSES: {len(moses_smiles)} molecules")
    print(f"  COCONUT: {len(coconut_smiles)} molecules")

    # Create coarseners
    hac = AffinityCoarsening(
        min_community_size=args.min_community_size,
        seed=args.seed,
    )
    spectral = SpectralCoarsening(
        min_community_size=args.min_community_size,
        seed=args.seed,
    )

    # Analyze datasets for aggregate statistics
    print(f"\nAnalyzing {args.num_stats_samples} molecules per dataset...")

    print("  MOSES + HAC...")
    moses_hac = analyze_dataset(moses_smiles, hac, args.num_stats_samples, args.seed)
    print(f"    {len(moses_hac)} successful")

    print("  MOSES + Spectral...")
    moses_spectral = analyze_dataset(
        moses_smiles,
        spectral,
        args.num_stats_samples,
        args.seed,
    )
    print(f"    {len(moses_spectral)} successful")

    print("  COCONUT + HAC...")
    coconut_hac = analyze_dataset(
        coconut_smiles,
        hac,
        args.num_stats_samples,
        args.seed,
    )
    print(f"    {len(coconut_hac)} successful")

    print("  COCONUT + Spectral...")
    coconut_spectral = analyze_dataset(
        coconut_smiles,
        spectral,
        args.num_stats_samples,
        args.seed,
    )
    print(f"    {len(coconut_spectral)} successful")

    # Generate example figures
    print("\nSelecting example molecules...")
    examples = select_example_molecules(
        moses_hac,
        coconut_hac,
        num_examples=args.num_examples,
    )

    print(f"\nGenerating {len(examples)} example figures...")
    for idx, example in enumerate(examples):
        smiles = example["smiles"]
        dataset = example["dataset"]
        num_atoms = example["num_atoms"]
        data = example["data"]

        print(f"  Example {idx + 1}: {dataset}, {num_atoms} atoms")

        # Build hierarchies with both coarseners
        hac_hg = hac.build_hierarchy(data)
        try:
            spectral_hg = spectral.build_hierarchy(data)
        except Exception:
            print("    Spectral failed, skipping")
            continue

        filename = f"example_{idx + 1}_{dataset.lower()}_{num_atoms}atoms.png"
        create_example_figure(
            smiles,
            data,
            hac_hg,
            spectral_hg,
            idx,
            dataset,
            str(output_dir / filename),
            dpi=args.dpi,
        )

    # Generate aggregate statistics figure
    print("\nGenerating aggregate statistics figure...")
    create_aggregate_figure(
        moses_hac,
        moses_spectral,
        coconut_hac,
        coconut_spectral,
        str(output_dir / "aggregate_stats.png"),
        dpi=args.dpi,
    )

    print(f"\nAll figures saved to {output_dir}/")

    if not args.no_show:
        print("(Use --no-show to suppress display)")


if __name__ == "__main__":
    main()
