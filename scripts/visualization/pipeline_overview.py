#!/usr/bin/env python
"""Pipeline overview visualization for the MOSAIC HDTC pipeline.

Produces a single static image showing the full end-to-end pipeline:
  Input Molecule -> Decomposed Motifs -> MOSAIC -> Generated Decomposed Motifs -> Generated Molecule

Usage:
    python scripts/visualization/pipeline_overview.py --no-generate --name camptothecin
    python scripts/visualization/pipeline_overview.py --name camptothecin --num-generate 50
    python scripts/visualization/pipeline_overview.py --smiles "..." --output-dir ./figures
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MPath
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.molecular import (  # noqa: E402
    ATOM_TYPES,
    graph_to_smiles,
    smiles_to_graph,
)
from src.tokenizers.coarsening.functional_hierarchy import (  # noqa: E402
    FunctionalHierarchyBuilder,
)
from src.tokenizers.structures import TwoLevelHierarchy  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Predefined molecules (reused from compare_community_structure.py)
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
    "strychnine": (
        "O=C1C[C@H]2OCC=C3CN4CC[C@@]56[C@@H]4C[C@H]3[C@H]2[C@H]5N1c1ccccc16"
    ),
    "camptothecin": "CCC1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3ccccc3nc2-1",
    "vinblastine": (
        "CCC1(O)[C@H]2CC3(CC)c4c(cc5c(c4OC)N(C=O)"
        "c4cc6c(cc4[C@H]5[C@@H]3[C@H](OC(C)=O)"
        "[C@@]1(O2)C(=O)OC)OCO6)C"
    ),
    "reserpine": (
        "CO[C@H]1[C@@H](CC2CN3CCC4=C([C@H]3C[C@@H]2"
        "[C@@H]1C(=O)OC)NC5=CC(OC)=C(OC)C(OC)=C45)"
        "OC(=O)C6=CC(OC)=C(OC)C(OC)=C6"
    ),
    "artemisinin": (
        "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@@]23OO4"
    ),
}

# ============================================================================
# Color scheme
# ============================================================================

TYPE_COLORS = {
    "ring": "#FF6B6B",
    "functional": "#4ECDC4",
    "singleton": "#95A5A6",
}

ELEMENT_COLORS = {
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
}
DEFAULT_ELEMENT_COLOR = "#808080"

ARROW_COLOR = "#333333"
SUPEREDGE_COLOR = "#2E86AB"


# ============================================================================
# Layout utilities
# ============================================================================


def compute_rdkit_2d_layout(
    smiles: str,
    coord_scale: float = 1.8,
) -> dict[int, tuple[float, float]] | None:
    """Compute 2D molecular layout using RDKit.

    Args:
        smiles: SMILES string.
        coord_scale: Scale factor for coordinates. Values > 1 spread atoms
            further apart (default 1.8 gives good spacing for 3D atoms).

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
            pos[node_id] = (
                (x - x_center) / scale * coord_scale,
                (y - y_center) / scale * coord_scale,
            )

        return pos

    except Exception:
        return None


def compute_exploded_layout(
    hierarchy: TwoLevelHierarchy,
    base_positions: dict[int, tuple[float, float]],
    spread_factor: float = 2.5,
) -> dict[int, tuple[float, float]]:
    """Explode communities apart while preserving internal geometry.

    Takes the full-molecule RDKit 2D layout and shifts each community's
    atoms outward from the global centroid, proportional to the distance
    between the community centroid and the global centroid.

    Args:
        hierarchy: TwoLevelHierarchy from FunctionalHierarchyBuilder.
        base_positions: Original atom positions from RDKit layout.
        spread_factor: How much to spread communities apart.

    Returns:
        New positions with communities visually separated.
    """
    # Global centroid of all atoms
    all_pts = np.array(list(base_positions.values()))
    global_centroid = all_pts.mean(axis=0)

    new_positions: dict[int, tuple[float, float]] = {}

    for comm in hierarchy.communities:
        # Community centroid
        comm_pts = np.array(
            [base_positions[i] for i in comm.atom_indices if i in base_positions]
        )
        if len(comm_pts) == 0:
            continue
        comm_centroid = comm_pts.mean(axis=0)

        # Shift vector: push community away from global centroid
        shift = (comm_centroid - global_centroid) * (spread_factor - 1)

        for atom_idx in comm.atom_indices:
            if atom_idx in base_positions:
                bx, by = base_positions[atom_idx]
                new_positions[atom_idx] = (bx + shift[0], by + shift[1])

    # Renormalize to [-coord_scale, coord_scale]
    coord_scale = 1.8
    if new_positions:
        xs = [p[0] for p in new_positions.values()]
        ys = [p[1] for p in new_positions.values()]
        x_range = max(xs) - min(xs) if len(xs) > 1 else 1
        y_range = max(ys) - min(ys) if len(ys) > 1 else 1
        scale = max(x_range, y_range, 1e-6)
        x_center = (max(xs) + min(xs)) / 2
        y_center = (max(ys) + min(ys)) / 2

        for node_id in new_positions:
            x, y = new_positions[node_id]
            new_positions[node_id] = (
                (x - x_center) / scale * coord_scale,
                (y - y_center) / scale * coord_scale,
            )

    return new_positions


# ============================================================================
# Drawing utilities
# ============================================================================


def _get_element_symbol(data, atom_idx: int) -> str:
    """Get element symbol for an atom from graph data.

    Args:
        data: PyG Data object.
        atom_idx: Atom index.

    Returns:
        Element symbol string.
    """
    import torch

    if data.x is None:
        return "?"
    if data.x.dtype in (torch.long, torch.int64):
        # Integer label format
        idx = int(data.x[atom_idx])
        if idx < len(ATOM_TYPES):
            return ATOM_TYPES[idx]
        return "?"
    else:
        # One-hot format
        feat = data.x[atom_idx].numpy()
        atom_idx_val = int(np.argmax(feat[: len(ATOM_TYPES) + 1]))
        if atom_idx_val < len(ATOM_TYPES):
            return ATOM_TYPES[atom_idx_val]
        return "?"


def _get_bond_type(data, src: int, dst: int) -> int:
    """Get bond type index for an edge.

    Args:
        data: PyG Data object.
        src: Source atom index.
        dst: Destination atom index.

    Returns:
        Bond type index (0=single, 1=double, 2=triple, 3=aromatic).
    """
    import torch

    if data.edge_index is None or data.edge_attr is None:
        return 0
    ei = data.edge_index
    for i in range(ei.shape[1]):
        if int(ei[0, i]) == src and int(ei[1, i]) == dst:
            ea = data.edge_attr
            if ea.dtype in (torch.long, torch.int64):
                return int(ea[i])
            else:
                # One-hot: argmax of first NUM_BOND_TYPES elements
                return int(np.argmax(ea[i].numpy()[:5]))
    return 0


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
    from matplotlib.patches import PathPatch

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
    if len(valid) == 0:
        return

    pts = np.array([positions[i] for i in valid])

    if len(valid) == 1:
        # Single node: draw a circle around it
        cx, cy = pts[0]
        circle = mpatches.Circle(
            (cx, cy),
            pad * 3,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        ax.add_patch(circle)
        return

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


def _darken_color(hex_color: str, factor: float = 0.6) -> str:
    """Darken a hex color by a factor.

    Args:
        hex_color: Hex color string (e.g. "#FF6B6B").
        factor: Darkening factor (0=black, 1=unchanged).

    Returns:
        Darkened hex color string.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten_color(hex_color: str, factor: float = 0.4) -> str:
    """Lighten a hex color by blending toward white.

    Args:
        hex_color: Hex color string.
        factor: Blend factor (0=unchanged, 1=white).

    Returns:
        Lightened hex color string.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_3d_atom(
    ax: plt.Axes,
    x: float,
    y: float,
    radius: float,
    color: str,
    label: str,
    label_fontsize: float = 5,
    zorder_base: int = 3,
) -> None:
    """Draw a 3D-looking atom sphere with shadow and specular highlight.

    Args:
        ax: Matplotlib axes.
        x: X position.
        y: Y position.
        radius: Sphere radius.
        color: Base element color.
        label: Element symbol label.
        label_fontsize: Font size for the label.
        zorder_base: Base z-order (shadow is -1, highlight is +1).
    """
    # Drop shadow (offset down-right, dark, blurred)
    shadow = plt.Circle(
        (x + radius * 0.12, y - radius * 0.12),
        radius * 1.08,
        facecolor="#000000",
        alpha=0.18,
        edgecolor="none",
        zorder=zorder_base - 1,
    )
    ax.add_patch(shadow)

    # Main sphere body with dark edge for depth
    dark_edge = _darken_color(color, 0.5)
    sphere = plt.Circle(
        (x, y),
        radius,
        facecolor=color,
        edgecolor=dark_edge,
        linewidth=0.6,
        zorder=zorder_base,
    )
    ax.add_patch(sphere)

    # Gradient ring: slightly lighter inner area
    inner = plt.Circle(
        (x - radius * 0.1, y + radius * 0.1),
        radius * 0.75,
        facecolor=_lighten_color(color, 0.2),
        edgecolor="none",
        alpha=0.5,
        zorder=zorder_base,
    )
    ax.add_patch(inner)

    # Specular highlight (upper-left, white)
    highlight = plt.Circle(
        (x - radius * 0.28, y + radius * 0.28),
        radius * 0.35,
        facecolor="white",
        alpha=0.55,
        edgecolor="none",
        zorder=zorder_base + 1,
    )
    ax.add_patch(highlight)

    # Element label
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=label_fontsize,
        fontweight="bold",
        color="white",
        zorder=zorder_base + 2,
    )


def _draw_bond(
    ax: plt.Axes,
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    bond_type: int,
    color: str = "#333333",
    zorder: int = 1,
) -> None:
    """Draw a 3D tube-like bond between two atoms.

    Renders a dark base line with a lighter highlight stripe on top
    to simulate cylindrical depth.

    Args:
        ax: Matplotlib axes.
        pos1: Start position (x, y).
        pos2: End position (x, y).
        bond_type: 0=single, 1=double, 2=triple, 3=aromatic.
        color: Bond color.
        zorder: Z-order.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001:
        return

    highlight = _lighten_color(color, 0.5)

    # Perpendicular offset for double/triple bonds
    px, py = -dy / dist, dx / dist
    offset = 0.025

    def _tube(x1: float, y1: float, x2: float, y2: float, lw: float) -> None:
        """Draw a single tube-like bond line."""
        # Dark base
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
            zorder=zorder,
        )
        # Light highlight stripe
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=highlight,
            linewidth=max(lw * 0.35, 0.5),
            solid_capstyle="round",
            alpha=0.6,
            zorder=zorder,
        )

    if bond_type == 0:
        _tube(x1, y1, x2, y2, 2.2)
    elif bond_type == 1:
        for sign in (-1, 1):
            ox, oy = sign * px * offset, sign * py * offset
            _tube(x1 + ox, y1 + oy, x2 + ox, y2 + oy, 1.8)
    elif bond_type == 2:
        for sign in (-1, 0, 1):
            ox, oy = sign * px * offset, sign * py * offset
            _tube(x1 + ox, y1 + oy, x2 + ox, y2 + oy, 1.4)
    elif bond_type == 3:
        # Aromatic: dashed base + highlight
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=2.2,
            linestyle="--",
            solid_capstyle="round",
            zorder=zorder,
        )
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=highlight,
            linewidth=0.8,
            linestyle="--",
            solid_capstyle="round",
            alpha=0.5,
            zorder=zorder,
        )
    else:
        _tube(x1, y1, x2, y2, 2.2)


# ============================================================================
# Panel drawing functions
# ============================================================================


def draw_molecule(
    ax: plt.Axes,
    smiles: str,
    data,
    positions: dict[int, tuple[float, float]],
    title: str = "",
) -> None:
    """Draw a full molecule using RDKit 2D layout.

    Atoms are colored circles with element symbols. Bonds are styled
    by type (single/double/triple/aromatic).

    Args:
        ax: Matplotlib axes.
        smiles: SMILES string.
        data: PyG Data object.
        positions: Atom positions from compute_rdkit_2d_layout().
        title: Panel title.
    """
    # Draw bonds
    drawn_edges: set[tuple[int, int]] = set()
    if data.edge_index is not None:
        ei = data.edge_index.numpy()
        for i in range(ei.shape[1]):
            u, v = int(ei[0, i]), int(ei[1, i])
            edge_key = (min(u, v), max(u, v))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            if u not in positions or v not in positions:
                continue
            bt = _get_bond_type(data, u, v)
            _draw_bond(ax, positions[u], positions[v], bt, zorder=1)

    # Draw 3D atoms
    node_radius = 0.042
    for node in range(data.num_nodes):
        if node not in positions:
            continue
        x, y = positions[node]
        elem = _get_element_symbol(data, node)
        color = ELEMENT_COLORS.get(elem, DEFAULT_ELEMENT_COLOR)
        _draw_3d_atom(ax, x, y, node_radius, color, elem, label_fontsize=5)

    ax.set_aspect("equal")
    ax.axis("off")

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin = 0.2
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)


def draw_decomposed_communities(
    ax: plt.Axes,
    hierarchy: TwoLevelHierarchy,
    positions: dict[int, tuple[float, float]],
    data,
    title: str = "",
) -> None:
    """Draw communities as separated sub-graphs with hulls and super-edges.

    Each community gets a light-colored background hull, colored by type
    (ring/functional/singleton). Super-edges are shown as gray dotted lines.

    Args:
        ax: Matplotlib axes.
        hierarchy: TwoLevelHierarchy from FunctionalHierarchyBuilder.
        positions: Atom positions (exploded layout).
        data: PyG Data object.
        title: Panel title.
    """
    # Layer 1: Community hulls
    for comm in hierarchy.communities:
        color = TYPE_COLORS.get(comm.community_type, "#B0B0B0")
        draw_community_hull(
            ax,
            positions,
            list(comm.atom_indices),
            color=color,
            alpha=0.20,
            pad=0.12,
            linewidth=1.5,
            zorder=0,
        )

    # Layer 2: Internal edges within communities
    drawn_edges: set[tuple[int, int]] = set()
    for comm in hierarchy.communities:
        color = TYPE_COLORS.get(comm.community_type, "#B0B0B0")
        atom_set = set(comm.atom_indices)
        if data.edge_index is not None:
            ei = data.edge_index.numpy()
            for i in range(ei.shape[1]):
                u, v = int(ei[0, i]), int(ei[1, i])
                edge_key = (min(u, v), max(u, v))
                if edge_key in drawn_edges:
                    continue
                if u in atom_set and v in atom_set:
                    drawn_edges.add(edge_key)
                    if u in positions and v in positions:
                        bt = _get_bond_type(data, u, v)
                        _draw_bond(
                            ax,
                            positions[u],
                            positions[v],
                            bt,
                            color=color,
                            zorder=1,
                        )

    # Layer 3: Super-edges (dotted gray lines between communities)
    for se in hierarchy.super_edges:
        src, dst = se.source_atom, se.target_atom
        if src in positions and dst in positions:
            draw_curved_edge(
                ax,
                positions[src],
                positions[dst],
                color=SUPEREDGE_COLOR,
                linewidth=1.0,
                linestyle=":",
                alpha=0.6,
                curve_amount=0.1,
                zorder=1,
            )

    # Layer 4: 3D Atoms
    node_radius = 0.038
    for comm in hierarchy.communities:
        for atom_idx in comm.atom_indices:
            if atom_idx not in positions:
                continue
            x, y = positions[atom_idx]
            elem = _get_element_symbol(data, atom_idx)
            color = ELEMENT_COLORS.get(elem, DEFAULT_ELEMENT_COLOR)
            _draw_3d_atom(ax, x, y, node_radius, color, elem, label_fontsize=4.5)

    # Layer 5: Community type labels
    for comm in hierarchy.communities:
        valid = [i for i in comm.atom_indices if i in positions]
        if not valid:
            continue
        pts = np.array([positions[i] for i in valid])
        cx, cy = pts.mean(axis=0)

        # Build label from group name and type
        if comm.community_type == "singleton":
            elem = _get_element_symbol(data, comm.atom_indices[0])
            label = elem
        else:
            label = comm.group_name
            type_suffix = f" ({comm.community_type[0].upper()})"
            label = label + type_suffix

        # Place label slightly below the community centroid
        offset_y = -0.08 if len(valid) > 1 else -0.06
        ax.text(
            cx,
            cy + offset_y,
            label,
            ha="center",
            va="top",
            fontsize=4,
            fontstyle="italic",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.axis("off")

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin = 0.25
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)


def draw_pipeline_arrow(
    ax: plt.Axes,
    label: str = "",
) -> None:
    """Draw a horizontal pipeline arrow with optional label.

    Args:
        ax: Matplotlib axes (thin column).
        label: Text label above the arrow.
    """
    arrow = mpatches.FancyArrowPatch(
        (0.1, 0.5),
        (0.9, 0.5),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=3,
        color=ARROW_COLOR,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(arrow)

    if label:
        ax.text(
            0.5,
            0.68,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontstyle="italic",
            color=ARROW_COLOR,
            transform=ax.transAxes,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_hdtc_tokenization(ax: plt.Axes) -> None:
    """Draw HDTC hierarchy tree feeding into a transformer block.

    Upper region: mini tree (root -> communities -> atoms) with edges.
    Lower region: transformer block diagram.
    Arrow connecting them: tree serializes into tokens fed to transformer.

    Args:
        ax: Matplotlib axes.
    """
    # -- Colors ---------------------------------------------------------------
    dark = "#16213e"
    ring_color = TYPE_COLORS["ring"]       # #FF6B6B
    func_color = TYPE_COLORS["functional"] # #4ECDC4
    sing_color = TYPE_COLORS["singleton"]  # #95A5A6
    edge_color = "#FF8C00"                 # orange for intra-community edges
    super_color = "#2E86AB"                # blue for super-edges

    # Compute aspect ratio correction so circles aren't squished.
    # The axes bbox gives actual display size; we want r_x/r_y to
    # compensate so a circle in axes coords looks round on screen.
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    aspect = bbox.height / max(bbox.width, 1)  # >1 means taller than wide

    def _circle(center, r, **kwargs):
        """Draw a circle that appears round regardless of axes aspect."""
        return mpatches.Ellipse(
            center, width=2 * r * aspect, height=2 * r,
            transform=ax.transAxes, **kwargs,
        )

    # -- Outer frame ----------------------------------------------------------
    outer = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.94,
        boxstyle="round,pad=0.02",
        facecolor="#f8f9fa", edgecolor=super_color,
        linewidth=2, transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(outer)

    # Title
    ax.text(
        0.5, 0.96, "MOSAIC",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=dark,
        transform=ax.transAxes, zorder=5,
    )

    # =====================================================================
    # UPPER: Mini hierarchy tree  (y ~ 0.55 .. 0.90)
    # =====================================================================
    tree_cx = 0.50
    node_r = 0.022  # base radius in y-coords (aspect-corrected for x)

    # --- Root node ---
    root_xy = (tree_cx, 0.88)
    ax.add_patch(_circle(root_xy, node_r * 1.15,
                         facecolor=dark, edgecolor="white",
                         linewidth=1.5, zorder=5))
    ax.text(*root_xy, "Root", ha="center", va="center",
            fontsize=7, fontweight="bold", color="white",
            transform=ax.transAxes, zorder=6)

    # --- Community nodes ---
    comm_y = 0.71
    comm_spread = 0.16  # half-spread in visual units, scaled by aspect for x
    communities = [
        {"label": "Ring", "x": tree_cx - comm_spread * aspect,
         "color": ring_color},
        {"label": "Func", "x": tree_cx,
         "color": func_color},
        {"label": "Sing", "x": tree_cx + comm_spread * aspect,
         "color": sing_color},
    ]
    comm_xys = []
    for comm in communities:
        cx, cy = comm["x"], comm_y
        comm_xys.append((cx, cy))
        ax.add_patch(_circle((cx, cy), node_r,
                             facecolor=comm["color"], edgecolor="#333333",
                             linewidth=1.2, zorder=5))
        ax.text(cx, cy, comm["label"],
                ha="center", va="center", fontsize=5.5, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=6)
        # Edge: root -> community
        ax.annotate(
            "", xy=(cx, cy + node_r), xytext=(root_xy[0], root_xy[1] - node_r * 1.15),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0),
            transform=ax.transAxes, zorder=3,
        )

    # --- Atom nodes ---
    atom_y = 0.53
    atom_r = node_r * 0.70
    atom_sp = 0.035 * aspect  # horizontal spacing between sibling atoms
    atom_groups = [
        [{"label": "a0", "x": communities[0]["x"] - atom_sp, "parent": 0},
         {"label": "a1", "x": communities[0]["x"], "parent": 0},
         {"label": "a2", "x": communities[0]["x"] + atom_sp, "parent": 0}],
        [{"label": "a3", "x": communities[1]["x"] - atom_sp * 0.7, "parent": 1},
         {"label": "a4", "x": communities[1]["x"] + atom_sp * 0.7, "parent": 1}],
        [{"label": "a5", "x": communities[2]["x"], "parent": 2}],
    ]
    atom_xys = {}
    for group_idx, group in enumerate(atom_groups):
        parent_color = communities[group_idx]["color"]
        for atom in group:
            ax_pos, ay_pos = atom["x"], atom_y
            atom_xys[atom["label"]] = (ax_pos, ay_pos)
            ax.add_patch(_circle((ax_pos, ay_pos), atom_r,
                                 facecolor="white", edgecolor=parent_color,
                                 linewidth=1.5, zorder=5))
            ax.text(ax_pos, ay_pos, atom["label"],
                    ha="center", va="center", fontsize=5, fontweight="bold",
                    color="#333333", transform=ax.transAxes, zorder=6)
            parent_xy = comm_xys[atom["parent"]]
            ax.annotate(
                "", xy=(ax_pos, ay_pos + atom_r),
                xytext=(parent_xy[0], parent_xy[1] - node_r),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8),
                transform=ax.transAxes, zorder=3,
            )

    # --- Intra-community edges (dashed orange) ---
    for (l1, l2) in [("a0", "a1"), ("a1", "a2"), ("a0", "a2")]:
        x1, y1 = atom_xys[l1]
        x2, y2 = atom_xys[l2]
        ax.plot([x1, x2], [y1 - atom_r - 0.003, y2 - atom_r - 0.003],
                color=edge_color, linewidth=1.0, linestyle="--",
                transform=ax.transAxes, zorder=2)
    if "a3" in atom_xys and "a4" in atom_xys:
        x1, y1 = atom_xys["a3"]
        x2, y2 = atom_xys["a4"]
        ax.plot([x1, x2], [y1 - atom_r - 0.003, y2 - atom_r - 0.003],
                color=edge_color, linewidth=1.0, linestyle="--",
                transform=ax.transAxes, zorder=2)

    # --- Super-edge between C0 and C1 (dashed blue, above the nodes) ---
    se_y_offset = node_r + 0.012
    ax.annotate(
        "", xy=(comm_xys[1][0], comm_xys[1][1] + se_y_offset),
        xytext=(comm_xys[0][0], comm_xys[0][1] + se_y_offset),
        arrowprops=dict(arrowstyle="<->", color=super_color, lw=1.2,
                        linestyle="dashed",
                        connectionstyle="arc3,rad=-0.25"),
        transform=ax.transAxes, zorder=3,
    )

    # =====================================================================
    # ARROW: tree  -->  transformer  ("tokenize")
    # =====================================================================
    arrow_top = atom_y - atom_r - 0.025
    arrow_bot = 0.42
    ax.annotate(
        "", xy=(tree_cx, arrow_bot), xytext=(tree_cx, arrow_top),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.0),
        transform=ax.transAxes, zorder=4,
    )
    ax.text(tree_cx + 0.02, (arrow_top + arrow_bot) / 2, "tokenize",
            ha="left", va="center", fontsize=7, fontstyle="italic",
            color="#555555", transform=ax.transAxes, zorder=5)

    # =====================================================================
    # LOWER: Transformer block  (y ~ 0.06 .. 0.40)
    # =====================================================================
    tf_x, tf_y = 0.10, 0.06
    tf_w, tf_h = 0.80, 0.34
    tf_box = mpatches.FancyBboxPatch(
        (tf_x, tf_y), tf_w, tf_h,
        boxstyle="round,pad=0.018",
        facecolor="#e8edf2", edgecolor=dark,
        linewidth=1.5, transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(tf_box)

    # Stacked sub-blocks inside the transformer
    bw = 0.62
    bh = 0.050
    bcx = 0.50
    gap = 0.015
    blocks = [
        ("Linear + Softmax",  dark),
        ("Feed-Forward",      "#FF8C00"),
        ("Multi-Head Attn",   "#FF6B6B"),
        ("Pos. Encoding",     "#4ECDC4"),
    ]
    n = len(blocks)
    total = n * bh + (n - 1) * gap
    by_start = tf_y + (tf_h - total) / 2

    for i, (label, color) in enumerate(blocks):
        by = by_start + i * (bh + gap)
        sub = mpatches.FancyBboxPatch(
            (bcx - bw / 2, by), bw, bh,
            boxstyle="round,pad=0.010",
            facecolor=color, edgecolor="#333333",
            linewidth=0.8, transform=ax.transAxes, zorder=3,
        )
        ax.add_patch(sub)
        ax.text(bcx, by + bh / 2, label,
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=4)
        if i < n - 1:
            ax.annotate(
                "", xy=(bcx, by + bh + 0.002),
                xytext=(bcx, by + bh + gap - 0.002),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8),
                transform=ax.transAxes, zorder=4,
            )

    # "GPT-2" label on the side
    ax.text(tf_x + tf_w - 0.02, tf_y + 0.02, "GPT-2",
            ha="right", va="bottom", fontsize=7.5, fontweight="bold",
            fontstyle="italic", color="#777777",
            transform=ax.transAxes, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ============================================================================
# Model loading and generation
# ============================================================================


def _infer_tokenizer_type(checkpoint_path: str) -> str:
    """Infer tokenizer type from checkpoint path name."""
    name = Path(checkpoint_path).parent.name.lower()
    if "hdtc" in name:
        return "hdtc"
    if "hsent" in name:
        return "hsent"
    if "hdt" in name:
        return "hdt"
    return "sent"


def load_model(
    checkpoint_path: str,
    tokenizer_type: str | None = None,
    labeled_graph: bool = True,
    coarsening_strategy: str = "spectral",
):
    """Load model and create appropriate tokenizer from checkpoint.

    Supports all tokenizer types: sent, hsent, hdt, hdtc.

    Args:
        checkpoint_path: Path to the model checkpoint.
        tokenizer_type: One of "sent", "hsent", "hdt", "hdtc".
            If None, inferred from checkpoint path.
        labeled_graph: Whether the model uses labeled graphs.
        coarsening_strategy: Coarsening strategy for hierarchical tokenizers.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch

    from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES
    from src.models.transformer import GraphGeneratorModule
    from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer

    if tokenizer_type is None:
        tokenizer_type = _infer_tokenizer_type(checkpoint_path)
    print(f"  Tokenizer type: {tokenizer_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    wte_key = "model.model.transformer.wte.weight"
    if "state_dict" in checkpoint and wte_key in checkpoint["state_dict"]:
        checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
    else:
        raise ValueError(
            f"Cannot determine vocab size from checkpoint: {checkpoint_path}"
        )

    if tokenizer_type == "hdtc":
        tokenizer = HDTCTokenizer(max_length=2048, labeled_graph=labeled_graph)
    elif tokenizer_type == "hdt":
        tokenizer = HDTTokenizer(
            max_length=2048, labeled_graph=labeled_graph,
            coarsening_strategy=coarsening_strategy,
        )
    elif tokenizer_type == "hsent":
        tokenizer = HSENTTokenizer(
            max_length=2048, labeled_graph=labeled_graph,
            coarsening_strategy=coarsening_strategy,
        )
    else:
        tokenizer = SENTTokenizer(max_length=2048, labeled_graph=labeled_graph)

    idx_offset = (
        tokenizer.IDX_OFFSET if hasattr(tokenizer, "IDX_OFFSET")
        else tokenizer.idx_offset
    )

    if labeled_graph:
        checkpoint_max_num_nodes = (
            checkpoint_vocab_size - idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES
        )
        if checkpoint_max_num_nodes <= 0:
            print(
                f"  Warning: labeled formula gives non-positive max_num_nodes "
                f"({checkpoint_max_num_nodes}), falling back to unlabeled"
            )
            tokenizer.labeled_graph = False
            checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset
    else:
        checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset

    # Force-set before set_num_node_and_edge_types (it reads max_num_nodes)
    tokenizer.max_num_nodes = checkpoint_max_num_nodes

    if tokenizer.labeled_graph:
        tokenizer.set_num_node_and_edge_types(
            num_node_types=NUM_ATOM_TYPES,
            num_edge_types=NUM_BOND_TYPES,
        )

    assert tokenizer.vocab_size == checkpoint_vocab_size, (
        f"Vocab mismatch: tokenizer={tokenizer.vocab_size}, "
        f"checkpoint={checkpoint_vocab_size} "
        f"(type={tokenizer_type}, max_num_nodes={checkpoint_max_num_nodes}, "
        f"labeled={labeled_graph})"
    )

    # Extract max position embeddings from checkpoint
    wpe_key = "model.model.transformer.wpe.weight"
    load_kwargs: dict = {"tokenizer": tokenizer, "weights_only": False}
    if "state_dict" in checkpoint and wpe_key in checkpoint["state_dict"]:
        checkpoint_max_length = checkpoint["state_dict"][wpe_key].shape[0]
        load_kwargs["sampling_max_length"] = checkpoint_max_length

    model = GraphGeneratorModule.load_from_checkpoint(checkpoint_path, **load_kwargs)
    model.eval()

    return model, tokenizer


def generate_and_select(
    model,
    tokenizer,
    target_num_nodes: int,
    num_generate: int = 50,
    top_k: int = 10,
    temperature: float = 1.0,
) -> tuple | None:
    """Generate molecules and select the one closest to target size.

    Args:
        model: GraphGeneratorModule.
        tokenizer: HDTCTokenizer.
        target_num_nodes: Desired number of atoms.
        num_generate: Number of molecules to generate.
        top_k: Top-k sampling parameter.
        temperature: Sampling temperature.

    Returns:
        Tuple of (data, smiles) for the best match, or None if no valid.
    """
    print(f"  Generating {num_generate} molecules...")
    graphs, avg_time, token_lengths = model.generate(
        num_samples=num_generate, show_progress=True
    )

    # Filter valid molecules
    valid_pairs: list[tuple] = []
    for g in graphs:
        smi = graph_to_smiles(g)
        if smi is not None:
            valid_pairs.append((g, smi))

    print(
        f"  Valid: {len(valid_pairs)}/{len(graphs)} "
        f"({100 * len(valid_pairs) / max(len(graphs), 1):.0f}%)"
    )

    if not valid_pairs:
        print("  Warning: No valid molecules generated!")
        return None

    # Select closest by num_nodes
    best = min(
        valid_pairs,
        key=lambda pair: abs(pair[0].num_nodes - target_num_nodes),
    )
    print(
        f"  Selected: {best[1]} ({best[0].num_nodes} atoms, "
        f"target was {target_num_nodes})"
    )
    return best


# ============================================================================
# Figure assembly
# ============================================================================


def create_pipeline_figure(
    input_smiles: str,
    input_data,
    input_hierarchy: TwoLevelHierarchy,
    input_positions: dict[int, tuple[float, float]],
    gen_data=None,
    gen_smiles: str | None = None,
    gen_hierarchy: TwoLevelHierarchy | None = None,
    gen_positions: dict[int, tuple[float, float]] | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """Create the full pipeline overview figure.

    Args:
        input_smiles: Input molecule SMILES.
        input_data: Input PyG Data.
        input_hierarchy: Input TwoLevelHierarchy.
        input_positions: Input atom positions.
        gen_data: Generated PyG Data (None if --no-generate).
        gen_smiles: Generated SMILES.
        gen_hierarchy: Generated TwoLevelHierarchy.
        gen_positions: Generated atom positions.
        dpi: Output DPI.

    Returns:
        Matplotlib Figure.
    """
    has_generation = gen_data is not None

    fig = plt.figure(figsize=(32, 8), dpi=dpi)

    # Always show full pipeline (9 columns)
    gs = GridSpec(
        1, 9, figure=fig,
        width_ratios=[2, 0.5, 3, 0.5, 1.8, 0.5, 3, 0.5, 2],
        wspace=0.02,
    )

    # Col 0: Input molecule
    ax_input = fig.add_subplot(gs[0, 0])
    draw_molecule(
        ax_input, input_smiles, input_data, input_positions,
        title="Input Molecule",
    )

    # Col 1: Arrow "Decompose"
    ax_arrow1 = fig.add_subplot(gs[0, 1])
    draw_pipeline_arrow(ax_arrow1, label="Decompose")

    # Col 2: Input decomposed
    ax_decomp = fig.add_subplot(gs[0, 2])
    exploded_input = compute_exploded_layout(
        input_hierarchy, input_positions, spread_factor=2.5
    )
    draw_decomposed_communities(
        ax_decomp, input_hierarchy, exploded_input, input_data,
        title="Functional Decomposition",
    )

    # Col 3: Arrow "Tokenize"
    ax_arrow2 = fig.add_subplot(gs[0, 3])
    draw_pipeline_arrow(ax_arrow2, label="Tokenize")

    # Col 4: HDTC tokenization diagram (hierarchy -> token sequence)
    ax_mosaic = fig.add_subplot(gs[0, 4])
    draw_hdtc_tokenization(ax_mosaic)

    # Col 5: Arrow "Decode"
    ax_arrow3 = fig.add_subplot(gs[0, 5])
    draw_pipeline_arrow(ax_arrow3, label="Decode")

    # Col 6: Generated decomposed
    ax_gen_decomp = fig.add_subplot(gs[0, 6])
    if has_generation:
        exploded_gen = compute_exploded_layout(
            gen_hierarchy, gen_positions, spread_factor=2.5
        )
        draw_decomposed_communities(
            ax_gen_decomp, gen_hierarchy, exploded_gen, gen_data,
            title="Generated Decomposition",
        )
    else:
        # Mirror the input decomposition as illustrative placeholder
        draw_decomposed_communities(
            ax_gen_decomp, input_hierarchy, exploded_input, input_data,
            title="Generated Decomposition",
        )

    # Col 7: Arrow "Reconstruct"
    ax_arrow4 = fig.add_subplot(gs[0, 7])
    draw_pipeline_arrow(ax_arrow4, label="Reconstruct")

    # Col 8: Generated molecule
    ax_gen = fig.add_subplot(gs[0, 8])
    if has_generation:
        draw_molecule(
            ax_gen, gen_smiles, gen_data, gen_positions,
            title="Generated Molecule",
        )
    else:
        # Mirror the input molecule as illustrative placeholder
        draw_molecule(
            ax_gen, input_smiles, input_data, input_positions,
            title="Generated Molecule",
        )

    # Type legend at bottom - bigger
    legend_patches = [
        mpatches.Patch(color=TYPE_COLORS["ring"], label="Ring"),
        mpatches.Patch(color=TYPE_COLORS["functional"], label="Functional Group"),
        mpatches.Patch(color=TYPE_COLORS["singleton"], label="Singleton"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=12,
        title="Community Types",
        title_fontsize=13,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "MOSAIC Pipeline Overview",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    return fig


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a static pipeline overview figure for the MOSAIC HDTC pipeline"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available molecules: {', '.join(sorted(MOLECULES.keys()))}",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="camptothecin",
        help="Predefined molecule name (default: camptothecin)",
    )
    parser.add_argument("--smiles", type=str, help="Custom SMILES string")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default=None,
        choices=["sent", "hsent", "hdt", "hdtc"],
        help="Tokenizer type (default: inferred from checkpoint path)",
    )
    parser.add_argument(
        "--coarsening-strategy",
        type=str,
        default="hac",
        choices=["spectral", "hac"],
        help="Coarsening strategy for hierarchical tokenizers (default: hac)",
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Use unlabeled graph mode (default: labeled)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures",
        help="Output directory (default: ./figures)",
    )
    parser.add_argument(
        "--num-generate",
        type=int,
        default=50,
        help="Number of molecules to generate (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress plt.show()",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation (debug mode for layout iteration)",
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    # Resolve molecule
    if args.smiles:
        smiles = args.smiles
        display_name = "custom"
    else:
        key = args.name.lower()
        if key not in MOLECULES:
            print(f"Unknown molecule: {args.name}")
            print(f"Available: {', '.join(sorted(MOLECULES.keys()))}")
            return
        smiles = MOLECULES[key]
        display_name = args.name.lower()

    print(f"Input molecule: {display_name}")
    print(f"  SMILES: {smiles}")

    # Convert to graph
    data = smiles_to_graph(smiles, labeled=True)
    if data is None:
        print(f"  Failed to convert SMILES: {smiles}")
        return
    data.smiles = smiles
    print(f"  Atoms: {data.num_nodes}")

    # Compute layout
    positions = compute_rdkit_2d_layout(smiles)
    if positions is None:
        print("  Failed to compute 2D layout")
        return

    # Build functional hierarchy for input
    print("  Building functional hierarchy...")
    builder = FunctionalHierarchyBuilder()
    hierarchy = builder.build(data)
    print(
        f"  Communities: {len(hierarchy.communities)} "
        f"({sum(1 for c in hierarchy.communities if c.community_type == 'ring')} ring, "
        f"{sum(1 for c in hierarchy.communities if c.community_type == 'functional')} func, "
        f"{sum(1 for c in hierarchy.communities if c.community_type == 'singleton')} singleton)"
    )

    # Generation (optional)
    gen_data = None
    gen_smiles = None
    gen_hierarchy = None
    gen_positions = None

    if not args.no_generate:
        import torch

        torch.manual_seed(args.seed)

        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            # Auto-find a checkpoint from outputs/
            import glob as _glob
            candidates = sorted(_glob.glob("outputs/**/best.ckpt", recursive=True))
            if not candidates:
                candidates = sorted(_glob.glob("outputs/**/last.ckpt", recursive=True))
            if candidates:
                checkpoint_path = candidates[0]
                print(f"  Auto-detected checkpoint: {checkpoint_path}")
            else:
                print("  No checkpoint found in outputs/")
                print(
                    "  Use --no-generate to skip generation, or provide --checkpoint"
                )
                return
        if not Path(checkpoint_path).exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            print(
                "  Use --no-generate to skip generation, or provide a valid --checkpoint"
            )
            return

        print(f"  Loading model from: {checkpoint_path}")
        model, tokenizer = load_model(
            checkpoint_path,
            tokenizer_type=args.tokenizer_type,
            labeled_graph=not args.unlabeled,
            coarsening_strategy=args.coarsening_strategy,
        )

        result = generate_and_select(
            model,
            tokenizer,
            target_num_nodes=data.num_nodes,
            num_generate=args.num_generate,
        )

        if result is not None:
            gen_data, gen_smiles = result

            # Ensure smiles attribute is set for functional group detection
            gen_data.smiles = gen_smiles

            # Compute layout for generated molecule
            gen_positions = compute_rdkit_2d_layout(gen_smiles)
            if gen_positions is None:
                print("  Warning: Failed to compute layout for generated molecule")
                gen_data = None
            else:
                # Build functional hierarchy for generated molecule
                gen_hierarchy = builder.build(gen_data)
                print(
                    f"  Generated communities: {len(gen_hierarchy.communities)} "
                    f"({sum(1 for c in gen_hierarchy.communities if c.community_type == 'ring')} ring, "
                    f"{sum(1 for c in gen_hierarchy.communities if c.community_type == 'functional')} func, "
                    f"{sum(1 for c in gen_hierarchy.communities if c.community_type == 'singleton')} singleton)"
                )

    # Create figure
    print("  Creating pipeline figure...")
    fig = create_pipeline_figure(
        input_smiles=smiles,
        input_data=data,
        input_hierarchy=hierarchy,
        input_positions=positions,
        gen_data=gen_data,
        gen_smiles=gen_smiles,
        gen_hierarchy=gen_hierarchy,
        gen_positions=gen_positions,
        dpi=args.dpi,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pipeline_overview_{display_name}.png"
    fig.savefig(str(output_path), dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
