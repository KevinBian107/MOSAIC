#!/usr/bin/env python
"""Compare 4 coarsening paradigms on the same molecule.

Produces a 4-row x 2-col figure:
  Row A: Spectral Coarsening (data-driven, no chemical knowledge)
  Row B: HAC / Affinity Coarsening (hierarchical agglomerative)
  Row C: Direct Motif Identification (MotifCommunityCoarsening)
  Row D: Motif + Functional Group (FunctionalHierarchyBuilder)

Each row: [molecule graph with communities | hierarchy tree]

By default, runs on a list of COCONUT-scale natural products.

Usage:
    python scripts/visualization/compare_community_structure.py
    python scripts/visualization/compare_community_structure.py --name vinblastine
    python scripts/visualization/compare_community_structure.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MPath
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.molecular import smiles_to_graph  # noqa: E402
from src.tokenizers.coarsening.functional_hierarchy import (  # noqa: E402
    FunctionalHierarchyBuilder,
)
from src.tokenizers.coarsening.hac import AffinityCoarsening  # noqa: E402
from src.tokenizers.coarsening.motif_community import (  # noqa: E402
    MotifCommunityCoarsening,
)
from src.tokenizers.coarsening.spectral import SpectralCoarsening  # noqa: E402
from src.tokenizers.structures import (  # noqa: E402
    Bipartite,
    HierarchicalGraph,
    Partition,
    TwoLevelHierarchy,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Predefined molecules
# ============================================================================

MOLECULES = {
    # Drug-like (MOSES-scale)
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "dopamine": "NCCC1=CC(O)=C(O)C=C1",
    "penicillin_g": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    "estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
    "quercetin": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C(O)=C2C3=CC(O)=C(O)C=C3",
    # Natural products (COCONUT-scale)
    "strychnine": "O=C1C[C@H]2OCC=C3CN4CC[C@@]56[C@@H]4C[C@H]3[C@H]2[C@H]5N1c1ccccc16",
    "camptothecin": "CCC1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3ccccc3nc2-1",
    "vinblastine": "CCC1(O)[C@H]2CC3(CC)c4c(cc5c(c4OC)N(C=O)c4cc6c(cc4[C@H]5[C@@H]3[C@H](OC(C)=O)[C@@]1(O2)C(=O)OC)OCO6)C",
    "reserpine": "CO[C@H]1[C@@H](CC2CN3CCC4=C([C@H]3C[C@@H]2[C@@H]1C(=O)OC)NC5=CC(OC)=C(OC)C(OC)=C45)OC(=O)C6=CC(OC)=C(OC)C(OC)=C6",
    "taxol": "CC1=C2C(C(=O)C3(C)C(CC4OC(=O)C(C(c5ccccc5)NC(=O)c5ccccc5)O4)C3C2(C)C)C(OC(=O)C)C1OC(=O)c1ccccc1",
    "erythromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O",
    "artemisinin": "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@@]23OO4",
    # COCONUT diverse (small / medium / large)
    "coconut_furanone": "COc1cc2[nH]c(=O)c(C(=O)NCc3ccco3)c(O)c2cc1OC",
    "coconut_flavone": "COc1ccc2c(=O)c(O)c(-c3cc(OC)c(O)c(OC)c3)oc2c1",
    "coconut_isoflavone": "O=C1C[C@@H](c2ccc(O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)cc2)Oc2cc(O)ccc21",
    "coconut_chromene": "CC1(C)C=CC2=C(C=C[C@@]3(O)[C@@H]2O[C@@H]2c4ccc(O)cc4OC[C@@H]23)O1",
    "coconut_sesquiterpene": "CC(=O)O[C@H]1C[C@H]2OC[C@@]2(OC(C)=O)[C@H]2[C@H](OC(=O)c3ccccc3)[C@]3(C)[C@@H](OC(C)=O)C(=O)[C@@H](OC(C)=O)[C@@]3(C)[C@H](OC(C)=O)[C@H]12",
    "coconut_glycoside": "O=C(O)CC(=O)OCC1OC(Oc2cc(O)c3c(=O)c(-c4ccc(OC5OC(CO)C(O)C(O)C5O)cc4)coc3c2)C(O)C(O)C1O",
}

# Default molecules to run when no --name or --smiles is given
DEFAULT_MOLECULES = [
    "vinblastine",
    "reserpine",
    "strychnine",
    "coconut_sesquiterpene",
]

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

# Type-based colors for functional hierarchy
TYPE_COLORS = {
    "ring": "#FF6B6B",
    "functional": "#4ECDC4",
    "singleton": "#B0B0B0",
}

SINGLETON_COLOR = "#D3D3D3"


# ============================================================================
# TwoLevelHierarchy -> HierarchicalGraph adapter
# ============================================================================


def functional_hierarchy_to_hg(
    hierarchy: TwoLevelHierarchy,
    data: Data,
) -> HierarchicalGraph:
    """Convert a TwoLevelHierarchy to a HierarchicalGraph for rendering.

    Each FunctionalCommunity becomes a Partition. Super-edges become Bipartite
    objects.

    Args:
        hierarchy: TwoLevelHierarchy from FunctionalHierarchyBuilder.
        data: Original PyG Data object.

    Returns:
        HierarchicalGraph with one level of partitions.
    """
    import torch

    partitions: list[Partition] = []
    for comm in hierarchy.communities:
        if comm.internal_edges:
            global_to_local = {g: i for i, g in enumerate(comm.atom_indices)}
            edges_local = []
            for src, dst in comm.internal_edges:
                if src in global_to_local and dst in global_to_local:
                    edges_local.append(
                        (global_to_local[src], global_to_local[dst])
                    )
                    edges_local.append(
                        (global_to_local[dst], global_to_local[src])
                    )
            if edges_local:
                edge_index = torch.tensor(edges_local, dtype=torch.long).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        partitions.append(
            Partition(
                part_id=comm.community_id,
                global_node_indices=list(comm.atom_indices),
                edge_index=edge_index,
            )
        )

    bipartites: list[Bipartite] = []
    pair_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for se in hierarchy.super_edges:
        key = (
            min(se.source_community, se.target_community),
            max(se.source_community, se.target_community),
        )
        if key not in pair_edges:
            pair_edges[key] = []
        pair_edges[key].append((se.source_atom, se.target_atom))

    part_map = {p.part_id: p for p in partitions}
    for (left_id, right_id), edges in pair_edges.items():
        left_part = part_map[left_id]
        right_part = part_map[right_id]
        left_g2l = {
            g: i for i, g in enumerate(left_part.global_node_indices)
        }
        right_g2l = {
            g: i for i, g in enumerate(right_part.global_node_indices)
        }

        local_edges = []
        for src_atom, dst_atom in edges:
            if src_atom in left_g2l and dst_atom in right_g2l:
                local_edges.append(
                    (left_g2l[src_atom], right_g2l[dst_atom])
                )
            elif src_atom in right_g2l and dst_atom in left_g2l:
                local_edges.append(
                    (left_g2l[dst_atom], right_g2l[src_atom])
                )

        if local_edges:
            edge_index = torch.tensor(local_edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        bipartites.append(
            Bipartite(
                left_part_id=left_id,
                right_part_id=right_id,
                edge_index=edge_index,
            )
        )

    community_assignment = list(hierarchy.atom_to_community)

    return HierarchicalGraph(
        partitions=partitions,
        bipartites=bipartites,
        community_assignment=community_assignment,
    )


# ============================================================================
# Layout
# ============================================================================


def compute_rdkit_2d_layout(
    smiles: str,
) -> dict[int, tuple[float, float]] | None:
    """Compute 2D molecular layout using RDKit.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary mapping atom index to (x, y) coordinates, or None.
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


def compute_spring_layout(
    data: Data, seed: int = 42
) -> dict[int, tuple[float, float]]:
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
# Drawing Utilities
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
        linestyle: Line style.
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
        return

    pts = np.array([positions[i] for i in valid])

    if len(valid) == 2:
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

    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
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
        pass


# ============================================================================
# Molecule + Community Overlay
# ============================================================================


def plot_molecule_with_communities(
    ax: plt.Axes,
    data: Data,
    smiles: str,
    hg: HierarchicalGraph,
    pos: dict[int, tuple[float, float]],
    part_colors: list[str] | None = None,
) -> None:
    """Plot molecule structure with community overlay hulls.

    Args:
        ax: Matplotlib axes.
        data: PyG Data object.
        smiles: SMILES string.
        hg: HierarchicalGraph object.
        pos: Node positions.
        part_colors: Optional explicit color per partition index.
    """
    if part_colors is None:
        part_colors = [
            COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
            for i in range(len(hg.partitions))
        ]

    # Layer 1: Hulls
    for part_idx, part in enumerate(hg.partitions):
        if part.num_nodes >= 2:
            color = part_colors[part_idx]
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

    # Build node-to-community mapping
    node_to_comm = {}
    for part_idx, part in enumerate(hg.partitions):
        for node in part.global_node_indices:
            node_to_comm[node] = part_idx

    # Layer 2: Edges
    drawn_edges: set[tuple[int, int]] = set()
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
            color = part_colors[comm_u]
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

    # Layer 3: Nodes
    for node in range(data.num_nodes):
        if node not in pos:
            continue
        x, y = pos[node]
        comm = node_to_comm.get(node, -1)
        part_size = (
            hg.partitions[comm].num_nodes
            if 0 <= comm < len(hg.partitions)
            else 1
        )

        if part_size == 1:
            node_color = SINGLETON_COLOR
        else:
            node_color = part_colors[comm]

        circle = plt.Circle(
            (x, y),
            0.04,
            facecolor=node_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(circle)

    # Layer 4: Labels
    if data.num_nodes <= 60:
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

    ax.set_aspect("equal")
    ax.axis("off")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.15
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


# ============================================================================
# Hierarchy Tree
# ============================================================================


def flatten_hierarchy_tree(
    hg: HierarchicalGraph,
    part_colors: list[str] | None = None,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    """Flatten nested hierarchy into renderable tree nodes.

    Args:
        hg: HierarchicalGraph object.
        part_colors: Explicit color per top-level partition.
        max_depth: Maximum depth to render.

    Returns:
        List of node dicts.
    """
    nodes: list[dict[str, Any]] = []
    node_id_counter = [0]

    if part_colors is None:
        part_colors = [
            COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
            for i in range(len(hg.partitions))
        ]

    def _recurse(
        partition: Partition,
        level: int,
        parent_id: int | None,
        color: str,
    ) -> int:
        nid = node_id_counter[0]
        node_id_counter[0] += 1

        is_leaf = partition.child_hierarchy is None
        children_ids = []

        if not is_leaf and level < max_depth:
            child_hg = partition.child_hierarchy
            singleton_nodes = []
            for cp in child_hg.partitions:
                if cp.num_nodes == 1:
                    singleton_nodes.append(cp)
                else:
                    cid = _recurse(cp, level + 1, nid, color)
                    children_ids.append(cid)

            if singleton_nodes:
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
                "color": color,
            }
        )
        return nid

    root_id = node_id_counter[0]
    node_id_counter[0] += 1
    root_children = []

    top_singletons = []
    for part_idx, part in enumerate(hg.partitions):
        if part.num_nodes == 1:
            top_singletons.append(part)
        else:
            color = part_colors[part_idx]
            cid = _recurse(part, 1, root_id, color)
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


def plot_hierarchy_tree(
    ax: plt.Axes,
    hg: HierarchicalGraph,
    part_colors: list[str] | None = None,
    max_depth: int = 4,
) -> None:
    """Plot hierarchical community tree with proportional-width layout.

    Args:
        ax: Matplotlib axes.
        hg: HierarchicalGraph object.
        part_colors: Explicit color list aligned with graph panel.
        max_depth: Maximum tree depth to display.
    """
    tree_nodes = flatten_hierarchy_tree(
        hg, part_colors=part_colors, max_depth=max_depth
    )
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

    node_map = {n["id"]: n for n in tree_nodes}
    root = None
    for n in tree_nodes:
        if n["parent_id"] is None:
            root = n
            break

    if root is None:
        ax.text(
            0.5,
            0.5,
            "No root found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    def count_leaves(nid: int) -> int:
        node = node_map[nid]
        if not node["children"]:
            return 1
        return sum(count_leaves(cid) for cid in node["children"])

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

    ax.set_aspect("equal")
    ax.axis("off")

    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    x_margin = max(0.5, (max(all_x) - min(all_x)) * 0.08)
    y_margin = max(0.5, (max(all_y) - min(all_y)) * 0.15)
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)


# ============================================================================
# Color helpers for functional hierarchy
# ============================================================================


def get_functional_partition_colors(
    hierarchy: TwoLevelHierarchy,
) -> list[str]:
    """Get colors for each partition based on community type.

    Args:
        hierarchy: TwoLevelHierarchy.

    Returns:
        List of color strings, one per community.
    """
    return [
        TYPE_COLORS.get(comm.community_type, SINGLETON_COLOR)
        for comm in hierarchy.communities
    ]


# ============================================================================
# Figure generation
# ============================================================================


def create_comparison_figure(
    smiles: str,
    data: Data,
    spectral_hg: HierarchicalGraph,
    hac_hg: HierarchicalGraph,
    motif_hg: HierarchicalGraph,
    functional_hierarchy: TwoLevelHierarchy,
    functional_hg: HierarchicalGraph,
    pos: dict[int, tuple[float, float]],
    title: str,
    output_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Create a 4-row x 2-col comparison figure.

    Args:
        smiles: SMILES string.
        data: PyG Data object.
        spectral_hg: HierarchicalGraph from SpectralCoarsening.
        hac_hg: HierarchicalGraph from AffinityCoarsening.
        motif_hg: HierarchicalGraph from MotifCommunityCoarsening.
        functional_hierarchy: TwoLevelHierarchy from FunctionalHierarchyBuilder.
        functional_hg: Adapted HierarchicalGraph from functional_hierarchy.
        pos: Node positions.
        title: Figure title.
        output_path: Path to save the figure, or None.
        dpi: Output DPI.

    Returns:
        Matplotlib Figure.
    """
    fig = plt.figure(figsize=(16, 22))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[1, 1, 1, 1],
        hspace=0.30,
        wspace=0.15,
    )

    row_configs = [
        {
            "label": "(A)",
            "title_graph": "Spectral Coarsening",
            "title_tree": "Hierarchy Tree",
            "hg": spectral_hg,
            "colors": None,
        },
        {
            "label": "(B)",
            "title_graph": "HAC (Affinity Coarsening)",
            "title_tree": "Hierarchy Tree",
            "hg": hac_hg,
            "colors": None,
        },
        {
            "label": "(C)",
            "title_graph": "Direct Motif Identification",
            "title_tree": "Hierarchy Tree",
            "hg": motif_hg,
            "colors": None,
        },
        {
            "label": "(D)",
            "title_graph": "Motif + Functional Group",
            "title_tree": "Hierarchy Tree",
            "hg": functional_hg,
            "colors": get_functional_partition_colors(functional_hierarchy),
        },
    ]

    for row_idx, cfg in enumerate(row_configs):
        hg = cfg["hg"]
        colors = cfg["colors"]

        # Left: molecule graph with communities
        ax_graph = fig.add_subplot(gs[row_idx, 0])
        plot_molecule_with_communities(
            ax_graph, data, smiles, hg, pos, part_colors=colors
        )
        ax_graph.set_title(
            cfg["title_graph"], fontsize=10, fontweight="bold"
        )

        # Row label
        ax_graph.text(
            -0.05,
            1.05,
            cfg["label"],
            transform=ax_graph.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

        # Stats annotation
        n_comm = len(hg.partitions)
        n_singleton = sum(1 for p in hg.partitions if p.num_nodes == 1)
        stats_text = f"comms={n_comm}, singletons={n_singleton}"
        ax_graph.text(
            0.02,
            0.02,
            stats_text,
            transform=ax_graph.transAxes,
            fontsize=7,
            color="gray",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.7
            ),
        )

        # Right: hierarchy tree
        ax_tree = fig.add_subplot(gs[row_idx, 1])
        plot_hierarchy_tree(ax_tree, hg, part_colors=colors)
        ax_tree.set_title(
            cfg["title_tree"], fontsize=10, fontweight="bold"
        )

    # Add type legend for row D
    legend_patches = [
        mpatches.Patch(color=TYPE_COLORS["ring"], label="Ring"),
        mpatches.Patch(color=TYPE_COLORS["functional"], label="Functional"),
        mpatches.Patch(color=TYPE_COLORS["singleton"], label="Singleton"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=9,
        title="Community types (D)",
        title_fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.98, 0.01),
    )

    fig.suptitle(title, fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


# ============================================================================
# Single-molecule pipeline
# ============================================================================


def process_molecule(
    name: str,
    smiles: str,
    output_dir: Path,
    seed: int = 42,
    dpi: int = 150,
    show: bool = False,
) -> None:
    """Build all 4 hierarchies and generate comparison figure for one molecule.

    Args:
        name: Display name.
        smiles: SMILES string.
        output_dir: Output directory.
        seed: Random seed.
        dpi: Output DPI.
        show: Whether to show the figure.
    """
    data = smiles_to_graph(smiles, labeled=True)
    if data is None:
        print(f"  Failed to convert SMILES: {smiles}")
        return

    pos = compute_rdkit_2d_layout(smiles)
    if pos is None:
        pos = compute_spring_layout(data, seed=seed)

    print(f"Analyzing: {name} ({data.num_nodes} atoms)")
    print(f"  SMILES: {smiles}")

    print("  Building Spectral hierarchy...")
    spectral = SpectralCoarsening(min_community_size=4, seed=seed)
    spectral_hg = spectral.build_hierarchy(data)

    print("  Building HAC hierarchy...")
    hac = AffinityCoarsening(min_community_size=4, seed=seed)
    hac_hg = hac.build_hierarchy(data)

    print("  Building Motif Community hierarchy...")
    motif = MotifCommunityCoarsening(min_community_size=4, seed=seed)
    motif_hg = motif.build_hierarchy(data)

    print("  Building Functional hierarchy...")
    functional = FunctionalHierarchyBuilder()
    functional_hierarchy = functional.build(data)
    functional_hg = functional_hierarchy_to_hg(functional_hierarchy, data)

    safe_name = name.lower().replace(" ", "_")
    output_path = str(output_dir / f"{safe_name}_coarsening_comparison.png")

    smiles_short = smiles[:50] + "..." if len(smiles) > 50 else smiles
    title = f"{name} ({data.num_nodes} atoms): {smiles_short}"

    fig = create_comparison_figure(
        smiles=smiles,
        data=data,
        spectral_hg=spectral_hg,
        hac_hg=hac_hg,
        motif_hg=motif_hg,
        functional_hierarchy=functional_hierarchy,
        functional_hg=functional_hg,
        pos=pos,
        title=title,
        output_path=output_path,
        dpi=dpi,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare 4 coarsening paradigms on the same molecule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available molecules: {', '.join(sorted(MOLECULES.keys()))}",
    )
    parser.add_argument("--smiles", type=str, help="SMILES string")
    parser.add_argument(
        "--name",
        type=str,
        help="Predefined molecule name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp/community_comparison",
        help="Output directory for figures",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display figures"
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single molecule mode
    if args.smiles or args.name:
        smiles = args.smiles
        display_name = "Custom"
        if args.name:
            key = args.name.lower()
            if key not in MOLECULES:
                print(f"Unknown molecule: {args.name}")
                print(
                    f"Available: {', '.join(sorted(MOLECULES.keys()))}"
                )
                return
            smiles = MOLECULES[key]
            display_name = args.name.replace("_", " ").title()

        process_molecule(
            name=display_name,
            smiles=smiles,
            output_dir=output_dir,
            seed=args.seed,
            dpi=args.dpi,
            show=not args.no_show,
        )
        return

    # Default: run on list of COCONUT molecules
    print(
        f"No molecule specified, running on {len(DEFAULT_MOLECULES)} "
        "default COCONUT molecules...\n"
    )
    for mol_name in DEFAULT_MOLECULES:
        smiles = MOLECULES[mol_name]
        display_name = mol_name.replace("_", " ").title()
        process_molecule(
            name=display_name,
            smiles=smiles,
            output_dir=output_dir,
            seed=args.seed,
            dpi=args.dpi,
            show=not args.no_show,
        )
        print()

    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
