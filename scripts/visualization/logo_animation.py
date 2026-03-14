#!/usr/bin/env python
"""Animated MOSAIC logo: multiple MOSES molecules → coarsened → MOSAIC letters + rim.

Produces a looping GIF showing:
  Phase 1: Multiple MOSES drug molecules (3D atoms + bonds)
  Phase 2: Communities highlighted with colored halos (coarsening)
  Phase 3: Community atoms form M-O-S-A-I-C letters,
           singleton atoms form a decorative rim around the word
  Phase 4: Hold the MOSAIC text with rim
  Phase 5: Reverse back to molecules (seamless loop)

Uses the same 3D atom/bond rendering style as pipeline_overview.py.

Usage:
    python scripts/visualization/logo_animation.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from src.data.molecular import ATOM_TYPES, smiles_to_graph  # noqa: E402
from src.tokenizers.coarsening.functional_hierarchy import (  # noqa: E402
    FunctionalHierarchyBuilder,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Constants
# ============================================================================

# Top MOSES molecules by heavy atom count — diverse elements, good community structure
MOSES_SMILES = [
    # 27at: 4 pyridine/imidazole rings, 5 N atoms
    "c1ccc(-c2nccc(-c3ccc(-n4cnc5ccccc54)cc3)n2)nc1",
    # 26at: 3 benzene rings + 2 amide groups, N/O diversity
    "O=C(NCc1ccccc1)c1ccccc1C(=O)NCc1ccccc1",
    # 26at: 3 pyridine rings + 2 amide groups
    "O=C(NCc1ccccn1)c1cccc(C(=O)NCc2ccccn2)c1",
    # 26at: fused ring system with 6 N atoms
    "Cc1nn(-c2ccccc2)c2nc3c(nc12)c(C)nn3-c1ccccc1",
]

# Community type colors
TYPE_COLORS = {
    "ring": "#FF6B6B",
    "functional": "#4ECDC4",
    "singleton": "#95A5A6",
}

# Element colors (matching pipeline_overview.py)
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

BOND_COLOR = "#333333"
BG_COLOR = "white"

# Desired atom count per letter (guides community assignment)
LETTER_TARGETS = {"M": 12, "O": 12, "S": 12, "A": 12, "I": 12, "C": 12}

# Per-letter color palette (distinct, vibrant)
LETTER_COLORS = {
    "M": "#FF6B6B",  # coral red
    "O": "#4ECDC4",  # teal
    "S": "#FFD93D",  # warm yellow
    "A": "#6C5CE7",  # purple
    "I": "#A8E6CF",  # mint green
    "C": "#FF8A5C",  # orange
}

# Letter stroke paths (coordinates in [0,1]x[0,1])
LETTER_STROKES: dict[str, list[list[tuple[float, float]]]] = {
    "M": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(0.0, 1.0), (0.5, 0.4)],
        [(0.5, 0.4), (1.0, 1.0)],
        [(1.0, 1.0), (1.0, 0.0)],
    ],
    "O": [
        [(0.5, 1.0), (0.15, 0.85), (0.0, 0.5), (0.15, 0.15), (0.5, 0.0),
         (0.85, 0.15), (1.0, 0.5), (0.85, 0.85), (0.5, 1.0)],
    ],
    "S": [
        [(0.9, 0.9), (0.5, 1.0), (0.1, 0.85), (0.1, 0.65),
         (0.5, 0.5),
         (0.9, 0.35), (0.9, 0.15), (0.5, 0.0), (0.1, 0.1)],
    ],
    "A": [
        [(0.0, 0.0), (0.5, 1.0)],
        [(0.5, 1.0), (1.0, 0.0)],
        [(0.18, 0.38), (0.82, 0.38)],
    ],
    "I": [
        [(0.2, 1.0), (0.8, 1.0)],
        [(0.5, 1.0), (0.5, 0.0)],
        [(0.2, 0.0), (0.8, 0.0)],
    ],
    "C": [
        [(0.9, 0.85), (0.5, 1.0), (0.15, 0.85), (0.0, 0.5),
         (0.15, 0.15), (0.5, 0.0), (0.9, 0.15)],
    ],
}


def _sample_stroke_points(
    strokes: list[list[tuple[float, float]]], n_points: int,
) -> list[tuple[float, float]]:
    """Sample n_points evenly along the combined stroke paths."""
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total_length = 0.0
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            p1, p2 = stroke[i], stroke[i + 1]
            d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            segments.append((p1, p2, d))
            total_length += d

    if total_length < 1e-8 or n_points < 1:
        return [(0.5, 0.5)] * n_points

    points: list[tuple[float, float]] = []
    for i in range(n_points):
        target_dist = (i / max(n_points - 1, 1)) * total_length
        cumulative = 0.0
        for p1, p2, d in segments:
            if cumulative + d >= target_dist - 1e-8:
                if d < 1e-8:
                    t = 0.0
                else:
                    t = (target_dist - cumulative) / d
                t = np.clip(t, 0.0, 1.0)
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                points.append((x, y))
                break
            cumulative += d
        else:
            last_stroke = strokes[-1]
            points.append(last_stroke[-1])

    return points


# ============================================================================
# Color helpers
# ============================================================================


def _darken_color(hex_color: str, factor: float = 0.6) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten_color(hex_color: str, factor: float = 0.4) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    return mcolors.to_rgb(hex_color)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _blend_colors(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(
        r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t,
    )


# ============================================================================
# 3D atom rendering
# ============================================================================


def _draw_3d_atom(
    ax: plt.Axes, x: float, y: float, radius: float, color: str,
    label: str = "", alpha: float = 1.0,
    label_fontsize: float = 4, zorder_base: int = 3,
) -> None:
    if alpha < 0.01:
        return

    shadow = plt.Circle(
        (x + radius * 0.12, y - radius * 0.12), radius * 1.08,
        facecolor="#000000", alpha=0.18 * alpha, edgecolor="none",
        zorder=zorder_base - 1,
    )
    ax.add_patch(shadow)

    dark_edge = _darken_color(color, 0.5)
    sphere = plt.Circle(
        (x, y), radius, facecolor=color, edgecolor=dark_edge,
        linewidth=0.6, alpha=alpha, zorder=zorder_base,
    )
    ax.add_patch(sphere)

    inner = plt.Circle(
        (x - radius * 0.1, y + radius * 0.1), radius * 0.75,
        facecolor=_lighten_color(color, 0.2), edgecolor="none",
        alpha=0.5 * alpha, zorder=zorder_base,
    )
    ax.add_patch(inner)

    highlight = plt.Circle(
        (x - radius * 0.28, y + radius * 0.28), radius * 0.35,
        facecolor="white", alpha=0.55 * alpha, edgecolor="none",
        zorder=zorder_base + 1,
    )
    ax.add_patch(highlight)

    if label and alpha > 0.3:
        ax.text(
            x, y, label, ha="center", va="center",
            fontsize=label_fontsize, fontweight="bold",
            color="white", alpha=alpha, zorder=zorder_base + 2,
        )


def _draw_bond(
    ax: plt.Axes, pos1: tuple[float, float], pos2: tuple[float, float],
    bond_type: int = 0, color: str = "#333333",
    alpha: float = 1.0, zorder: int = 1,
) -> None:
    x1, y1 = pos1
    x2, y2 = pos2
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001 or alpha < 0.01:
        return

    highlight = _lighten_color(color, 0.5)
    px, py = -dy / dist, dx / dist
    offset = 0.018

    def _tube(x1, y1, x2, y2, lw):
        ax.plot(
            [x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder,
        )
        ax.plot(
            [x1, x2], [y1, y2], color=highlight,
            linewidth=max(lw * 0.35, 0.3),
            solid_capstyle="round", alpha=0.6 * alpha, zorder=zorder,
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
            _tube(x1 + ox, y1 + oy, x2 + ox, y2 + oy, 1.2)
    elif bond_type == 3:
        ax.plot(
            [x1, x2], [y1, y2], color=color, linewidth=2.2,
            linestyle="--", solid_capstyle="round", alpha=alpha, zorder=zorder,
        )
    else:
        _tube(x1, y1, x2, y2, 2.2)


# ============================================================================
# Multi-molecule data preparation
# ============================================================================


def _get_atom_elements(data, num_atoms: int) -> dict[int, str]:
    """Extract element symbol for each atom."""
    import torch

    elements: dict[int, str] = {}
    for i in range(num_atoms):
        if data.x is not None:
            if data.x.dtype in (torch.long, torch.int64):
                idx = int(data.x[i])
            else:
                idx = int(np.argmax(data.x[i].numpy()[: len(ATOM_TYPES) + 1]))
            elements[i] = ATOM_TYPES[idx] if idx < len(ATOM_TYPES) else "?"
        else:
            elements[i] = "?"
    return elements


def _get_bond_types(data) -> dict[tuple[int, int], int]:
    """Extract bond type for each edge."""
    import torch

    bond_types: dict[tuple[int, int], int] = {}
    if data.edge_index is None or data.edge_attr is None:
        return bond_types
    ei = data.edge_index.numpy()
    ea = data.edge_attr
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        key = (min(u, v), max(u, v))
        if key not in bond_types:
            if ea.dtype in (torch.long, torch.int64):
                bond_types[key] = int(ea[i])
            else:
                bond_types[key] = int(np.argmax(ea[i].numpy()[:5]))
    return bond_types


def _compute_rdkit_layout(smiles: str) -> dict[int, tuple[float, float]]:
    """Compute normalized 2D layout (centered at origin, fits in [-1,1])."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {}
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pos[i] = (p.x, p.y)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    for k in pos:
        x, y = pos[k]
        pos[k] = ((x - cx) / scale, (y - cy) / scale)
    return pos


def prepare_multi_molecule_data(
    smiles_list: list[str],
    mol_scale: float = 1.2,
) -> dict:
    """Process multiple MOSES molecules and prepare all animation data.

    Returns dict with keys:
        mol_pos: global atom positions (molecule phase)
        letter_pos: global atom positions (MOSAIC phase)
        elements: global_id -> element symbol
        bond_type_map: (u,v) -> bond type
        edges: list of (u,v) edges
        letter_communities: list of (global_ids, comm_type) for letter atoms
        letter_assignment: list of (letter_char, global_ids_list)
        rim_atoms: list of global_ids for rim atoms
        atom_to_comm_color: global_id -> community hex color
        all_atom_ids: set of all global ids
    """
    builder = FunctionalHierarchyBuilder()

    # Molecule placement: spread in a row with slight y-stagger
    n_mols = len(smiles_list)
    mol_spread = 8.0
    mol_centers_x = np.linspace(-mol_spread / 2.5, mol_spread / 2.5, n_mols)
    mol_centers_y = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3][:n_mols]

    global_pos: dict[int, tuple[float, float]] = {}
    global_elements: dict[int, str] = {}
    global_bond_types: dict[tuple[int, int], int] = {}
    global_edges: list[tuple[int, int]] = []

    # Communities: list of (global_ids, comm_type, size)
    letter_pool: list[tuple[list[int], str]] = []  # size >= 4: for letters
    rim_pool: list[int] = []  # singletons + tiny groups: for rim

    offset = 0  # global atom ID offset

    for mi, smi in enumerate(smiles_list):
        data = smiles_to_graph(smi, labeled=True)
        local_pos = _compute_rdkit_layout(smi)
        local_elems = _get_atom_elements(data, data.num_nodes)
        local_bonds = _get_bond_types(data)

        # Build local adjacency and edges
        ei = data.edge_index.numpy()
        local_edges: set[tuple[int, int]] = set()
        for i in range(ei.shape[1]):
            u, v = int(ei[0, i]), int(ei[1, i])
            local_edges.add((min(u, v), max(u, v)))

        # Community detection
        hierarchy = builder.build(data)

        # Map local -> global and place molecule
        cx, cy = mol_centers_x[mi], mol_centers_y[mi]
        for local_id in range(data.num_nodes):
            gid = offset + local_id
            lx, ly = local_pos[local_id]
            global_pos[gid] = (cx + lx * mol_scale, cy + ly * mol_scale)
            global_elements[gid] = local_elems[local_id]

        for (lu, lv), bt in local_bonds.items():
            global_bond_types[(offset + lu, offset + lv)] = bt

        for lu, lv in local_edges:
            global_edges.append((offset + lu, offset + lv))

        # Classify communities (size >= 3 for letters, rest for rim)
        for comm in hierarchy.communities:
            gids = [offset + a for a in comm.atom_indices]
            if comm.community_type == "singleton" or len(gids) < 3:
                rim_pool.extend(gids)
            else:
                letter_pool.append((gids, comm.community_type))

        offset += data.num_nodes

    # --- Assign communities to letters ---
    letter_assignment = _assign_communities_to_letters(letter_pool)

    # Build community color map (per-letter colors for variety)
    atom_to_comm_color: dict[int, str] = {}
    for letter, comm_groups in letter_assignment:
        color = LETTER_COLORS.get(letter, "#808080")
        for gids, ctype in comm_groups:
            for gid in gids:
                atom_to_comm_color[gid] = color
    for gid in rim_pool:
        atom_to_comm_color[gid] = TYPE_COLORS["singleton"]

    # --- Compute letter target positions ---
    letter_pos = _compute_letter_target_positions(letter_assignment, global_pos)

    # --- Compute rim target positions ---
    rim_pos = _compute_rim_positions(rim_pool, global_pos, letter_pos)
    letter_pos.update(rim_pos)

    # Build flat community list for hull drawing
    all_communities: list[tuple[list[int], str]] = []
    for letter, comm_groups in letter_assignment:
        for gids, ctype in comm_groups:
            all_communities.append((gids, ctype))

    return {
        "mol_pos": global_pos,
        "letter_pos": letter_pos,
        "elements": global_elements,
        "bond_type_map": global_bond_types,
        "edges": global_edges,
        "communities": all_communities,
        "letter_assignment": letter_assignment,
        "rim_atoms": rim_pool,
        "atom_to_comm_color": atom_to_comm_color,
        "all_atom_ids": set(global_pos.keys()),
    }


def _assign_communities_to_letters(
    letter_pool: list[tuple[list[int], str]],
) -> list[tuple[str, list[tuple[list[int], str]]]]:
    """Greedily assign communities to letters for balanced distribution.

    Each community goes to the letter with the largest remaining deficit.
    Returns list of (letter_char, [(global_ids, comm_type), ...]).
    """
    letters = "MOSAIC"
    remaining = {ch: LETTER_TARGETS[ch] for ch in letters}
    assignment: dict[str, list[tuple[list[int], str]]] = {ch: [] for ch in letters}

    # Sort communities by size descending (assign big ones first)
    pool = sorted(letter_pool, key=lambda x: -len(x[0]))

    for gids, ctype in pool:
        size = len(gids)
        # Find letter with largest remaining deficit (skip if already oversaturated)
        candidates = [(remaining[ch], ch) for ch in letters if remaining[ch] > -3]
        if not candidates:
            break
        candidates.sort(key=lambda x: -x[0])
        best_letter = candidates[0][1]
        assignment[best_letter].append((gids, ctype))
        remaining[best_letter] -= size

    return [(ch, assignment[ch]) for ch in letters]


def _compute_letter_target_positions(
    letter_assignment: list[tuple[str, list[tuple[list[int], str]]]],
    mol_pos: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Compute target positions for letter atoms using stroke-based rendering."""
    total_width = 7.0
    n_letters = 6
    letter_spacing = total_width / n_letters
    letter_width = letter_spacing * 0.55
    letter_height = letter_width * 1.6

    letter_pos: dict[int, tuple[float, float]] = {}

    for order, (letter, comm_groups) in enumerate(letter_assignment):
        # Gather all atoms for this letter
        all_atoms: list[int] = []
        for gids, _ in comm_groups:
            all_atoms.extend(gids)

        if not all_atoms:
            continue

        strokes = LETTER_STROKES[letter]
        n_atoms = len(all_atoms)

        # Letter center
        lx_center = -total_width / 2 + letter_spacing * (order + 0.5)
        ly_center = 0.0

        # Sample points along strokes
        raw_pts = _sample_stroke_points(strokes, n_atoms)
        scaled_dots = [
            (lx_center + (px - 0.5) * letter_width,
             ly_center + (py - 0.5) * letter_height)
            for px, py in raw_pts
        ]

        # Hungarian matching to minimize travel distance
        cost = np.zeros((n_atoms, n_atoms))
        for ai, atom in enumerate(all_atoms):
            ax_v, ay_v = mol_pos[atom]
            for di, (dx, dy) in enumerate(scaled_dots):
                cost[ai, di] = (ax_v - dx) ** 2 + (ay_v - dy) ** 2
        r_ind, c_ind = linear_sum_assignment(cost)
        for ai, di in zip(r_ind, c_ind):
            letter_pos[all_atoms[ai]] = scaled_dots[di]

    return letter_pos


def _compute_rim_positions(
    rim_atoms: list[int],
    mol_pos: dict[int, tuple[float, float]],
    letter_pos: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Place rim atoms in an evenly-spaced elliptical border around MOSAIC."""
    if not rim_atoms:
        return {}

    # Compute letter bounding box
    if letter_pos:
        lx = [p[0] for p in letter_pos.values()]
        ly = [p[1] for p in letter_pos.values()]
        cx = (max(lx) + min(lx)) / 2
        cy = (max(ly) + min(ly)) / 2
        rx = (max(lx) - min(lx)) / 2 + 1.6  # wide padding for clear separation
        ry = (max(ly) - min(ly)) / 2 + 1.2
    else:
        cx, cy, rx, ry = 0.0, 0.0, 3.0, 1.5

    n = len(rim_atoms)
    rim_pos: dict[int, tuple[float, float]] = {}

    # Place evenly around ellipse
    rim_targets = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)
        rim_targets.append((x, y))

    # Hungarian matching
    cost = np.zeros((n, n))
    for ai, atom in enumerate(rim_atoms):
        ax_v, ay_v = mol_pos[atom]
        for di, (dx, dy) in enumerate(rim_targets):
            cost[ai, di] = (ax_v - dx) ** 2 + (ay_v - dy) ** 2
    r_ind, c_ind = linear_sum_assignment(cost)
    for ai, di in zip(r_ind, c_ind):
        rim_pos[rim_atoms[ai]] = rim_targets[di]

    return rim_pos


# ============================================================================
# Animation helpers
# ============================================================================


def smoothstep(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def lerp_pos(
    pos_a: dict[int, tuple[float, float]],
    pos_b: dict[int, tuple[float, float]],
    t: float,
) -> dict[int, tuple[float, float]]:
    result = {}
    for k in pos_a:
        if k in pos_b:
            ax_v, ay_v = pos_a[k]
            bx, by = pos_b[k]
            result[k] = (ax_v + (bx - ax_v) * t, ay_v + (by - ay_v) * t)
        else:
            result[k] = pos_a[k]
    return result


# ============================================================================
# Frame state
# ============================================================================

FRAMES_HOLD_MOL = 20
FRAMES_COARSEN = 30
FRAMES_HOLD_COARSENED = 15
FRAMES_LETTER_FORM = 40
FRAMES_HOLD_MOSAIC = 30

F0 = 0
F1 = F0 + FRAMES_HOLD_MOL
F2 = F1 + FRAMES_COARSEN
F3 = F2 + FRAMES_HOLD_COARSENED
F4 = F3 + FRAMES_LETTER_FORM
F5 = F4 + FRAMES_HOLD_MOSAIC

FORWARD_FRAMES = F5
TOTAL_FRAMES = FORWARD_FRAMES * 2


def get_frame_state(
    frame: int,
    data: dict,
) -> dict:
    """Compute animation state for a given frame."""
    mol_pos = data["mol_pos"]
    letter_pos = data["letter_pos"]
    elements = data["elements"]
    atom_to_comm_color = data["atom_to_comm_color"]

    # Palindrome
    if frame >= FORWARD_FRAMES:
        frame = TOTAL_FRAMES - 1 - frame

    # Element colors
    elem_colors = {
        a: ELEMENT_COLORS.get(elements.get(a, "?"), DEFAULT_ELEMENT_COLOR)
        for a in mol_pos
    }

    if frame < F1:
        return {
            "positions": mol_pos,
            "atom_colors": elem_colors,
            "hull_alpha": 0.0,
            "bond_alpha": 1.0,
            "global_alpha": 1.0,
            "show_bonds": True,
            "show_labels": True,
            "show_mosaic_text": False,
        }
    elif frame < F2:
        t = smoothstep((frame - F1) / max(FRAMES_COARSEN - 1, 1))
        blended = {}
        for a in mol_pos:
            blended[a] = _blend_colors(
                elem_colors[a],
                atom_to_comm_color.get(a, "#808080"),
                t,
            )
        return {
            "positions": mol_pos,
            "atom_colors": blended,
            "hull_alpha": t * 0.3,
            "bond_alpha": 1.0 - t * 0.4,
            "global_alpha": 1.0,
            "show_bonds": True,
            "show_labels": t < 0.5,
            "show_mosaic_text": False,
        }
    elif frame < F3:
        return {
            "positions": mol_pos,
            "atom_colors": {
                a: atom_to_comm_color.get(a, "#808080") for a in mol_pos
            },
            "hull_alpha": 0.3,
            "bond_alpha": 0.6,
            "global_alpha": 1.0,
            "show_bonds": True,
            "show_labels": False,
            "show_mosaic_text": False,
        }
    elif frame < F4:
        t = smoothstep((frame - F3) / max(FRAMES_LETTER_FORM - 1, 1))
        positions = lerp_pos(mol_pos, letter_pos, t)
        return {
            "positions": positions,
            "atom_colors": {
                a: atom_to_comm_color.get(a, "#808080") for a in mol_pos
            },
            "hull_alpha": max(0.0, 0.3 * (1.0 - t)),
            "bond_alpha": max(0.0, 0.6 * (1.0 - t * 1.5)),
            "global_alpha": 1.0,
            "show_bonds": t < 0.7,
            "show_labels": False,
            "show_mosaic_text": t > 0.8,
        }
    else:
        return {
            "positions": letter_pos,
            "atom_colors": {
                a: atom_to_comm_color.get(a, "#808080") for a in mol_pos
            },
            "hull_alpha": 0.0,
            "bond_alpha": 0.0,
            "global_alpha": 1.0,
            "show_bonds": False,
            "show_labels": False,
            "show_mosaic_text": True,
        }


# ============================================================================
# Rendering
# ============================================================================


def draw_community_hull(
    ax: plt.Axes,
    positions: dict[int, tuple[float, float]],
    node_indices: list[int],
    color: str,
    alpha: float = 0.15,
    pad: float = 0.12,
) -> None:
    valid = [i for i in node_indices if i in positions]
    if len(valid) < 1:
        return

    pts = np.array([positions[i] for i in valid])

    if len(valid) == 1:
        cx, cy = pts[0]
        ax.add_patch(mpatches.Circle(
            (cx, cy), pad * 3, facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.5, zorder=0,
        ))
        return

    if len(valid) == 2:
        cx, cy = pts.mean(axis=0)
        dx, dy = pts[1] - pts[0]
        width = np.sqrt(dx**2 + dy**2) + pad * 4
        angle = np.degrees(np.arctan2(dy, dx))
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), width, pad * 4, angle=angle,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=1.5, zorder=0,
        ))
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
        ax.add_patch(mpatches.Polygon(
            np.array(padded), closed=True, facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.5, zorder=0,
        ))
    except Exception:
        pass


def render_frame(
    ax: plt.Axes,
    state: dict,
    data: dict,
    fixed_xlim: tuple[float, float],
    fixed_ylim: tuple[float, float],
) -> None:
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    positions = state["positions"]
    atom_colors = state["atom_colors"]
    hull_alpha = state["hull_alpha"]
    bond_alpha = state["bond_alpha"]
    global_alpha = state["global_alpha"]

    communities = data["communities"]
    edges = data["edges"]
    elements = data["elements"]
    bond_type_map = data["bond_type_map"]

    # Build atom-to-community for coloring bonds
    atom_to_comm: dict[int, int] = {}
    comm_types: list[str] = []
    for ci, (gids, ctype) in enumerate(communities):
        comm_types.append(ctype)
        for a in gids:
            atom_to_comm[a] = ci

    # Layer 1: Community hulls
    if hull_alpha > 0.01:
        for gids, ctype in communities:
            color = TYPE_COLORS.get(ctype, TYPE_COLORS["singleton"])
            draw_community_hull(ax, positions, gids, color, alpha=hull_alpha)

    # Layer 2: Bonds
    if state["show_bonds"] and bond_alpha > 0.01:
        for u, v in edges:
            if u in positions and v in positions:
                comm_u = atom_to_comm.get(u, -1)
                comm_v = atom_to_comm.get(v, -1)
                bt = bond_type_map.get((min(u, v), max(u, v)), 0)
                if comm_u == comm_v and comm_u >= 0 and hull_alpha > 0.01:
                    ctype = comm_types[comm_u]
                    color = TYPE_COLORS.get(ctype, TYPE_COLORS["singleton"])
                    ba = bond_alpha * global_alpha * 0.8
                else:
                    color = BOND_COLOR
                    ba = bond_alpha * global_alpha * 0.6
                _draw_bond(
                    ax, positions[u], positions[v],
                    bond_type=bt, color=color, alpha=ba, zorder=1,
                )

    # Layer 3: Atoms
    atom_radius = 0.06
    for atom_id in sorted(positions.keys()):
        x, y = positions[atom_id]
        color = atom_colors.get(atom_id, DEFAULT_ELEMENT_COLOR)
        elem = elements.get(atom_id, "")
        label = elem if state["show_labels"] else ""
        _draw_3d_atom(
            ax, x, y, atom_radius, color, label=label,
            alpha=global_alpha, label_fontsize=4, zorder_base=3,
        )

    # Layer 4: "MOSAIC" text
    if state["show_mosaic_text"]:
        ax.text(
            0.0, fixed_ylim[0] + 0.25,
            "M   O   S   A   I   C",
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            color="#2B5EA7", alpha=0.9,
            fontfamily="sans-serif", zorder=10,
        )

    ax.set_xlim(*fixed_xlim)
    ax.set_ylim(*fixed_ylim)


# ============================================================================
# Main
# ============================================================================


def create_logo_animation(
    output_path: str = "figures/mosaic_logo.gif",
    fps: int = 30,
    dpi: int = 150,
) -> None:
    print("Preparing molecules and communities...")
    data = prepare_multi_molecule_data(MOSES_SMILES)

    n_total = len(data["all_atom_ids"])
    n_rim = len(data["rim_atoms"])
    print(f"  {len(MOSES_SMILES)} molecules, {n_total} total atoms")
    print(f"  {n_total - n_rim} letter atoms, {n_rim} rim atoms")
    for letter, comm_groups in data["letter_assignment"]:
        n = sum(len(g) for g, _ in comm_groups)
        n_comms = len(comm_groups)
        print(f"  {letter}: {n} atoms from {n_comms} communities")

    # Compute fixed axis limits
    all_x = ([p[0] for p in data["mol_pos"].values()]
             + [p[0] for p in data["letter_pos"].values()])
    all_y = ([p[1] for p in data["mol_pos"].values()]
             + [p[1] for p in data["letter_pos"].values()])
    margin = 0.5
    fixed_xlim = (min(all_x) - margin, max(all_x) + margin)
    fixed_ylim = (min(all_y) - margin - 0.3, max(all_y) + margin)

    print(f"Creating animation ({TOTAL_FRAMES} frames at {fps} fps)...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    def update(frame: int) -> list:
        state = get_frame_state(frame, data)
        render_frame(ax, state, data, fixed_xlim, fixed_ylim)
        return []

    anim = FuncAnimation(
        fig, update, frames=TOTAL_FRAMES, interval=1000 // fps, blit=False,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MOSAIC logo animation")
    parser.add_argument(
        "--output", type=str, default="figures/mosaic_logo.gif",
        help="Output GIF path",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()
    create_logo_animation(output_path=args.output, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
