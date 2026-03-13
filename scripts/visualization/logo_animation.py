#!/usr/bin/env python
"""Animated MOSAIC logo: molecule → coarsened communities → letter mosaic → loop.

Produces a looping GIF showing:
  Phase 1: Molecular graph fades in (3D atoms + styled bonds)
  Phase 2: Communities highlighted with colored halos (coarsening)
  Phase 3: Atoms rearrange locally within each community to form M-O-S-A-I-C
  Phase 4: Hold the MOSAIC text with label
  Phase 5: Reverse back to molecule
  → seamless loop

Uses the same 3D atom/bond rendering style as pipeline_overview.py.

Usage:
    python scripts/visualization/logo_animation.py
    python scripts/visualization/logo_animation.py --fps 24 --dpi 150
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

# Vancomycin: 101 heavy atoms, 3 benzene rings + many amide/ether groups.
# ~17 atoms per community after merging → clear letter shapes.
MOLECULE_SMILES = (
    "CC1C(O)CC(NC(=O)C(CC(=O)N)NC(=O)C2CC(O)C(OC3OC(CO)C(O)C(O)"
    "C3NC(C)=O)C(Oc4cc5cc(Oc6ccc(cc6Cl)C(O)C6NC(=O)C(NC(=O)"
    "C(CC(N)=O)NC1=O)C(O)C1=CC(O)=CC(O)=C1C1CC(C(NC6=O)C(=O)O)"
    "NC1=O)c5cc4Cl)O2)C(O)=O"
)

# Community type colors (matching pipeline_overview.py)
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

# 5x7 dot grid letter patterns: list of (col, row) positions
# col in [0..4], row in [0..6]
LETTER_PATTERNS: dict[str, list[tuple[int, int]]] = {
    "M": [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 1),
        (2, 2),
        (3, 1),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    ],
    "O": [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (4, 1),
        (0, 2), (4, 2),
        (0, 3), (4, 3),
        (0, 4), (4, 4),
        (0, 5), (4, 5),
        (1, 6), (2, 6), (3, 6),
    ],
    "S": [
        (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1),
        (1, 2), (2, 2), (3, 2),
        (4, 3),
        (4, 4),
        (0, 5), (1, 5),
        (0, 6), (1, 6), (2, 6), (3, 6),
    ],
    "A": [
        (2, 0),
        (1, 1), (3, 1),
        (0, 2), (4, 2),
        (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
        (0, 4), (4, 4),
        (0, 5), (4, 5),
        (0, 6), (4, 6),
    ],
    "I": [
        (1, 0), (2, 0), (3, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (1, 6), (2, 6), (3, 6),
    ],
    "C": [
        (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 6), (2, 6), (3, 6), (4, 6),
    ],
}


# ============================================================================
# Color helpers (from pipeline_overview.py)
# ============================================================================


def _darken_color(hex_color: str, factor: float = 0.6) -> str:
    """Darken a hex color by a factor."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten_color(hex_color: str, factor: float = 0.4) -> str:
    """Lighten a hex color by blending toward white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex to [0,1] RGB tuple."""
    return mcolors.to_rgb(hex_color)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert [0,1] RGB to hex string."""
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _blend_colors(c1: str, c2: str, t: float) -> str:
    """Blend two hex colors by factor t (0=c1, 1=c2)."""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(
        r1 + (r2 - r1) * t,
        g1 + (g2 - g1) * t,
        b1 + (b2 - b1) * t,
    )


# ============================================================================
# 3D atom rendering (adapted from pipeline_overview.py)
# ============================================================================


def _draw_3d_atom(
    ax: plt.Axes,
    x: float,
    y: float,
    radius: float,
    color: str,
    label: str = "",
    alpha: float = 1.0,
    label_fontsize: float = 4,
    zorder_base: int = 3,
) -> None:
    """Draw a 3D-looking atom sphere with shadow and specular highlight."""
    if alpha < 0.01:
        return

    # Drop shadow
    shadow = plt.Circle(
        (x + radius * 0.12, y - radius * 0.12),
        radius * 1.08,
        facecolor="#000000",
        alpha=0.18 * alpha,
        edgecolor="none",
        zorder=zorder_base - 1,
    )
    ax.add_patch(shadow)

    # Main sphere body
    dark_edge = _darken_color(color, 0.5)
    sphere = plt.Circle(
        (x, y),
        radius,
        facecolor=color,
        edgecolor=dark_edge,
        linewidth=0.6,
        alpha=alpha,
        zorder=zorder_base,
    )
    ax.add_patch(sphere)

    # Lighter inner area
    inner = plt.Circle(
        (x - radius * 0.1, y + radius * 0.1),
        radius * 0.75,
        facecolor=_lighten_color(color, 0.2),
        edgecolor="none",
        alpha=0.5 * alpha,
        zorder=zorder_base,
    )
    ax.add_patch(inner)

    # Specular highlight
    highlight = plt.Circle(
        (x - radius * 0.28, y + radius * 0.28),
        radius * 0.35,
        facecolor="white",
        alpha=0.55 * alpha,
        edgecolor="none",
        zorder=zorder_base + 1,
    )
    ax.add_patch(highlight)

    # Element label
    if label and alpha > 0.3:
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=label_fontsize, fontweight="bold",
            color="white", alpha=alpha,
            zorder=zorder_base + 2,
        )


def _draw_bond(
    ax: plt.Axes,
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    bond_type: int = 0,
    color: str = "#333333",
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    """Draw a styled bond between two atoms (from pipeline_overview.py)."""
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
            [x1, x2], [y1, y2],
            color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder,
        )
        ax.plot(
            [x1, x2], [y1, y2],
            color=highlight, linewidth=max(lw * 0.35, 0.3),
            solid_capstyle="round", alpha=0.6 * alpha, zorder=zorder,
        )

    if bond_type == 0:
        _tube(x1, y1, x2, y2, 1.8)
    elif bond_type == 1:
        for sign in (-1, 1):
            ox, oy = sign * px * offset, sign * py * offset
            _tube(x1 + ox, y1 + oy, x2 + ox, y2 + oy, 1.4)
    elif bond_type == 2:
        for sign in (-1, 0, 1):
            ox, oy = sign * px * offset, sign * py * offset
            _tube(x1 + ox, y1 + oy, x2 + ox, y2 + oy, 1.0)
    elif bond_type == 3:
        ax.plot(
            [x1, x2], [y1, y2],
            color=color, linewidth=1.8, linestyle="--",
            solid_capstyle="round", alpha=alpha, zorder=zorder,
        )
    else:
        _tube(x1, y1, x2, y2, 1.8)


# ============================================================================
# Layout
# ============================================================================


def compute_rdkit_2d_layout(
    smiles: str, coord_scale: float = 1.8,
) -> dict[int, tuple[float, float]]:
    """Compute 2D molecular layout using RDKit, normalized and scaled."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformer()

    pos = {}
    for i in range(mol.GetNumAtoms()):
        p = conformer.GetAtomPosition(i)
        pos[i] = (p.x, p.y)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_range = max(xs) - min(xs) if len(xs) > 1 else 1
    y_range = max(ys) - min(ys) if len(ys) > 1 else 1
    scale = max(x_range, y_range, 1e-6)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    for k in pos:
        x, y = pos[k]
        pos[k] = ((x - cx) / scale * coord_scale, (y - cy) / scale * coord_scale)

    return pos


# ============================================================================
# Atom element info
# ============================================================================


def get_atom_elements(data, num_atoms: int) -> dict[int, str]:
    """Extract element symbol for each atom."""
    import torch

    elements: dict[int, str] = {}
    for i in range(num_atoms):
        if data.x is not None:
            if data.x.dtype in (torch.long, torch.int64):
                idx = int(data.x[i])
            else:
                idx = int(np.argmax(data.x[i].numpy()[: len(ATOM_TYPES) + 1]))
            if idx < len(ATOM_TYPES):
                elements[i] = ATOM_TYPES[idx]
            else:
                elements[i] = "?"
        else:
            elements[i] = "?"
    return elements


def get_bond_types(data) -> dict[tuple[int, int], int]:
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


# ============================================================================
# Community preparation
# ============================================================================


def prepare_six_communities(
    smiles: str,
) -> tuple[
    dict[int, tuple[float, float]],
    list[list[int]],
    list[str],
    list[tuple[int, int]],
    int,
    dict[int, str],
    dict[tuple[int, int], int],
]:
    """Build functional hierarchy and post-process to exactly 6 communities.

    Returns:
        (pos, communities, community_types, edges, num_atoms, elements, bond_types)
    """
    data = smiles_to_graph(smiles, labeled=True)
    pos = compute_rdkit_2d_layout(smiles)
    num_atoms = data.num_nodes

    # Get element info and bond types
    elements = get_atom_elements(data, num_atoms)
    bond_type_map = get_bond_types(data)

    # Get edges
    ei = data.edge_index.numpy()
    edges_set: set[tuple[int, int]] = set()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        edges_set.add((min(u, v), max(u, v)))
    edges = list(edges_set)

    # Build adjacency
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Build functional hierarchy
    builder = FunctionalHierarchyBuilder()
    hierarchy = builder.build(data)

    # Extract communities
    communities: list[list[int]] = []
    community_types: list[str] = []
    for comm in hierarchy.communities:
        communities.append(list(comm.atom_indices))
        community_types.append(comm.community_type)

    # Merge singletons into nearest neighbor community
    communities, community_types = _merge_singletons(
        communities, community_types, adj
    )

    # Adjust to exactly 6 communities
    while len(communities) > 6:
        communities, community_types = _merge_smallest_adjacent(
            communities, community_types, adj
        )

    while len(communities) < 6:
        communities, community_types = _split_largest(
            communities, community_types, adj, pos
        )

    return pos, communities, community_types, edges, num_atoms, elements, bond_type_map


def _merge_singletons(
    communities: list[list[int]],
    types: list[str],
    adj: dict[int, set[int]],
) -> tuple[list[list[int]], list[str]]:
    """Merge singleton communities into their nearest non-singleton neighbor."""
    non_singleton_ids = [i for i, c in enumerate(communities) if len(c) > 1]
    if not non_singleton_ids:
        return communities, types

    atom_to_comm: dict[int, int] = {}
    for ci, atoms in enumerate(communities):
        for a in atoms:
            atom_to_comm[a] = ci

    merged = set()
    for ci, atoms in enumerate(communities):
        if len(atoms) == 1:
            atom = atoms[0]
            best_target = None
            for neighbor in adj.get(atom, set()):
                nc = atom_to_comm.get(neighbor, -1)
                if nc >= 0 and nc != ci and nc not in merged:
                    if len(communities[nc]) > 1 or nc in non_singleton_ids:
                        best_target = nc
                        break
            if best_target is None:
                for neighbor in adj.get(atom, set()):
                    nc = atom_to_comm.get(neighbor, -1)
                    if nc >= 0 and nc != ci:
                        best_target = nc
                        break
            if best_target is not None:
                communities[best_target].append(atom)
                atom_to_comm[atom] = best_target
                merged.add(ci)

    new_comms = []
    new_types = []
    for ci in range(len(communities)):
        if ci not in merged:
            new_comms.append(communities[ci])
            new_types.append(types[ci])

    return new_comms, new_types


def _merge_smallest_adjacent(
    communities: list[list[int]],
    types: list[str],
    adj: dict[int, set[int]],
) -> tuple[list[list[int]], list[str]]:
    """Merge the two smallest adjacent communities."""
    sizes = [len(c) for c in communities]
    atom_to_comm: dict[int, int] = {}
    for ci, atoms in enumerate(communities):
        for a in atoms:
            atom_to_comm[a] = ci

    best_pair = None
    best_size = float("inf")
    for ci in range(len(communities)):
        neighbors_ci = set()
        for atom in communities[ci]:
            for nb in adj.get(atom, set()):
                nc = atom_to_comm.get(nb, -1)
                if nc >= 0 and nc != ci:
                    neighbors_ci.add(nc)
        for cj in neighbors_ci:
            combined = sizes[ci] + sizes[cj]
            if combined < best_size:
                best_size = combined
                best_pair = (min(ci, cj), max(ci, cj))

    if best_pair is None:
        best_pair = (len(communities) - 2, len(communities) - 1)

    i, j = best_pair
    communities[i] = communities[i] + communities[j]
    dominant = types[i] if len(communities[i]) >= len(communities[j]) else types[j]
    types[i] = dominant

    del communities[j]
    del types[j]

    return communities, types


def _split_largest(
    communities: list[list[int]],
    types: list[str],
    adj: dict[int, set[int]],
    pos: dict[int, tuple[float, float]],
) -> tuple[list[list[int]], list[str]]:
    """Split the largest community into two halves using spatial bisection."""
    sizes = [len(c) for c in communities]
    idx = int(np.argmax(sizes))
    atoms = communities[idx]

    if len(atoms) < 2:
        communities.append([atoms[0]])
        types.append(types[idx])
        return communities, types

    # Spatial split: sort by x-coordinate, split in half
    atoms_sorted = sorted(atoms, key=lambda a: pos.get(a, (0, 0))[0])
    half = len(atoms_sorted) // 2
    part_a = atoms_sorted[:half]
    part_b = atoms_sorted[half:]

    communities[idx] = part_a
    communities.append(part_b)
    types.append(types[idx])

    return communities, types


# ============================================================================
# Letter position computation
# ============================================================================


def compute_letter_positions(
    communities: list[list[int]],
    mol_pos: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Compute target letter positions for each atom.

    Uses a two-step assignment:
    1. Match communities to letters by size similarity (Hungarian), constrained
       so that spatial order is roughly preserved (penalizes large reorderings).
    2. Within each community, match atoms to letter dot positions (Hungarian).
    """
    letters = "MOSAIC"

    # Sort communities by centroid x-coordinate
    centroids = []
    for ci, atoms in enumerate(communities):
        cx = np.mean([mol_pos[a][0] for a in atoms])
        cy = np.mean([mol_pos[a][1] for a in atoms])
        centroids.append((cx, cy, ci))
    centroids.sort(key=lambda t: t[0])

    # Compute letter dot counts
    letter_dots = [len(LETTER_PATTERNS[ch]) for ch in letters]
    comm_sizes = [len(communities[centroids[i][2]]) for i in range(len(centroids))]

    # Match communities (in spatial order) to letter slots using Hungarian
    # Cost = |size_diff| + penalty for spatial reordering
    n = len(letters)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            size_diff = abs(comm_sizes[i] - letter_dots[j])
            order_penalty = abs(i - j) * 3  # keep spatial order mostly intact
            cost_matrix[i, j] = size_diff + order_penalty
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # community_order[spatial_rank] = letter_index
    comm_to_letter = {row_ind[k]: col_ind[k] for k in range(n)}

    n_letters = len(letters)
    total_width = 5.5
    letter_spacing = total_width / n_letters
    letter_width = letter_spacing * 0.65
    letter_height = letter_width * 1.4  # 5x7 aspect ratio

    letter_pos: dict[int, tuple[float, float]] = {}

    for spatial_rank, (_, _, ci) in enumerate(centroids):
        atoms = communities[ci]
        letter_idx = comm_to_letter[spatial_rank]
        letter = letters[letter_idx]
        pattern = LETTER_PATTERNS[letter]

        # Letter position is determined by its index in "MOSAIC" (not spatial rank)
        lx_center = -total_width / 2 + letter_spacing * (letter_idx + 0.5)
        ly_center = 0.0

        # Scale pattern dots to fit
        scaled_dots = []
        for col, row in pattern:
            x = lx_center + (col / 4.0 - 0.5) * letter_width
            y = ly_center + (0.5 - row / 6.0) * letter_height
            scaled_dots.append((x, y))

        n_atoms = len(atoms)
        n_dots = len(scaled_dots)

        if n_atoms == 0:
            continue

        if n_atoms <= n_dots:
            # More dots than atoms: pick best n_atoms dots
            cost = np.zeros((n_atoms, n_dots))
            for ai, atom in enumerate(atoms):
                ax, ay = mol_pos[atom]
                for di, (dx, dy) in enumerate(scaled_dots):
                    cost[ai, di] = (ax - dx) ** 2 + (ay - dy) ** 2
            row_ind, col_ind = linear_sum_assignment(cost)
            for ai, di in zip(row_ind, col_ind):
                letter_pos[atoms[ai]] = scaled_dots[di]
        else:
            # More atoms than dots: assign all dots, extras on outline
            cost = np.zeros((n_dots, n_atoms))
            for di, (dx, dy) in enumerate(scaled_dots):
                for ai, atom in enumerate(atoms):
                    ax, ay = mol_pos[atom]
                    cost[di, ai] = (ax - dx) ** 2 + (ay - dy) ** 2
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_atoms = set()
            for di, ai in zip(row_ind, col_ind):
                letter_pos[atoms[ai]] = scaled_dots[di]
                assigned_atoms.add(ai)

            # Place remaining atoms along the letter outline with jitter
            rng = np.random.RandomState(42)
            unassigned = [ai for ai in range(n_atoms) if ai not in assigned_atoms]
            for ui, ai in enumerate(unassigned):
                dot_idx = ui % n_dots
                dx, dy = scaled_dots[dot_idx]
                jx = rng.uniform(-0.03, 0.03)
                jy = rng.uniform(-0.03, 0.03)
                letter_pos[atoms[ai]] = (dx + jx, dy + jy)

    return letter_pos


# ============================================================================
# Animation helpers
# ============================================================================


def smoothstep(t: float) -> float:
    """Smoothstep easing: 3t^2 - 2t^3."""
    t = np.clip(t, 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def lerp_pos(
    pos_a: dict[int, tuple[float, float]],
    pos_b: dict[int, tuple[float, float]],
    t: float,
) -> dict[int, tuple[float, float]]:
    """Linearly interpolate between two position dicts."""
    result = {}
    for k in pos_a:
        if k in pos_b:
            ax, ay = pos_a[k]
            bx, by = pos_b[k]
            result[k] = (ax + (bx - ax) * t, ay + (by - ay) * t)
        else:
            result[k] = pos_a[k]
    return result


# ============================================================================
# Frame state computation
# ============================================================================

# Frame layout (palindrome: forward + reverse for seamless loop)
FRAMES_FADE_IN = 15
FRAMES_HOLD_MOL = 15
FRAMES_COARSEN = 30
FRAMES_HOLD_COARSENED = 15
FRAMES_LETTER_FORM = 45
FRAMES_HOLD_MOSAIC = 30

# Cumulative boundaries (forward half only)
F0 = 0
F1 = F0 + FRAMES_FADE_IN         # 15
F2 = F1 + FRAMES_HOLD_MOL        # 30
F3 = F2 + FRAMES_COARSEN         # 60
F4 = F3 + FRAMES_HOLD_COARSENED  # 75
F5 = F4 + FRAMES_LETTER_FORM     # 120
F6 = F5 + FRAMES_HOLD_MOSAIC     # 150

FORWARD_FRAMES = F6
TOTAL_FRAMES = FORWARD_FRAMES * 2  # palindrome


def get_frame_state(
    frame: int,
    mol_pos: dict[int, tuple[float, float]],
    letter_pos: dict[int, tuple[float, float]],
    communities: list[list[int]],
    community_types: list[str],
    elements: dict[int, str],
) -> dict:
    """Compute animation state for a given frame.

    Returns dict with: positions, atom_colors (hex), hull_alpha, bond_alpha,
    global_alpha, show_bonds, show_labels, color_blend (0=element, 1=community),
    show_mosaic_text.
    """
    # Palindrome: reverse second half
    if frame >= FORWARD_FRAMES:
        frame = TOTAL_FRAMES - 1 - frame

    # Build element color map
    elem_colors = {
        a: ELEMENT_COLORS.get(elements.get(a, "?"), DEFAULT_ELEMENT_COLOR)
        for a in mol_pos
    }

    # Build community color map
    comm_colors: dict[int, str] = {}
    for ci, atoms in enumerate(communities):
        ctype = community_types[ci]
        cc = TYPE_COLORS.get(ctype, TYPE_COLORS["singleton"])
        for a in atoms:
            comm_colors[a] = cc

    if frame < F1:
        # Fade in molecule
        t = smoothstep(frame / max(FRAMES_FADE_IN - 1, 1))
        return {
            "positions": mol_pos,
            "atom_colors": elem_colors,
            "hull_alpha": 0.0,
            "bond_alpha": t,
            "global_alpha": t,
            "show_bonds": True,
            "show_labels": True,
            "show_mosaic_text": False,
        }
    elif frame < F2:
        # Hold molecule
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
    elif frame < F3:
        # Coarsening: hulls fade in, atoms blend from element -> community color
        t = smoothstep((frame - F2) / max(FRAMES_COARSEN - 1, 1))
        blended = {}
        for a in mol_pos:
            blended[a] = _blend_colors(elem_colors[a], comm_colors.get(a, "#808080"), t)
        return {
            "positions": mol_pos,
            "atom_colors": blended,
            "hull_alpha": t * 0.3,
            "bond_alpha": 1.0 - t * 0.4,
            "global_alpha": 1.0,
            "show_bonds": True,
            "show_labels": t < 0.5,  # labels fade out during coarsening
            "show_mosaic_text": False,
        }
    elif frame < F4:
        # Hold coarsened
        return {
            "positions": mol_pos,
            "atom_colors": {a: comm_colors.get(a, "#808080") for a in mol_pos},
            "hull_alpha": 0.3,
            "bond_alpha": 0.6,
            "global_alpha": 1.0,
            "show_bonds": True,
            "show_labels": False,
            "show_mosaic_text": False,
        }
    elif frame < F5:
        # Letter formation: atoms interpolate, bonds fade out
        t = smoothstep((frame - F4) / max(FRAMES_LETTER_FORM - 1, 1))
        positions = lerp_pos(mol_pos, letter_pos, t)
        return {
            "positions": positions,
            "atom_colors": {a: comm_colors.get(a, "#808080") for a in mol_pos},
            "hull_alpha": max(0.0, 0.3 * (1.0 - t)),
            "bond_alpha": max(0.0, 0.6 * (1.0 - t * 1.5)),
            "global_alpha": 1.0,
            "show_bonds": t < 0.7,
            "show_labels": False,
            "show_mosaic_text": t > 0.8,
        }
    else:
        # Hold MOSAIC
        return {
            "positions": letter_pos,
            "atom_colors": {a: comm_colors.get(a, "#808080") for a in mol_pos},
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
    """Draw a convex hull or ellipse around community nodes."""
    valid = [i for i in node_indices if i in positions]
    if len(valid) < 1:
        return

    pts = np.array([positions[i] for i in valid])

    if len(valid) == 1:
        cx, cy = pts[0]
        circle = mpatches.Circle(
            (cx, cy), pad * 3,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=1.5, zorder=0,
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
            (cx, cy), width, height, angle=angle,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=1.5, zorder=0,
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
            padded, closed=True, facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.5, linestyle="-", zorder=0,
        )
        ax.add_patch(polygon)
    except Exception:
        pass


def render_frame(
    ax: plt.Axes,
    state: dict,
    communities: list[list[int]],
    community_types: list[str],
    edges: list[tuple[int, int]],
    mol_pos: dict[int, tuple[float, float]],
    letter_pos: dict[int, tuple[float, float]],
    elements: dict[int, str],
    bond_type_map: dict[tuple[int, int], int],
    fixed_xlim: tuple[float, float],
    fixed_ylim: tuple[float, float],
) -> None:
    """Render a single animation frame with 3D atoms and styled bonds."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    positions = state["positions"]
    atom_colors = state["atom_colors"]
    hull_alpha = state["hull_alpha"]
    bond_alpha = state["bond_alpha"]
    global_alpha = state["global_alpha"]

    # Build atom-to-community map
    atom_to_comm: dict[int, int] = {}
    for ci, atoms in enumerate(communities):
        for a in atoms:
            atom_to_comm[a] = ci

    # Layer 1: Community hulls
    if hull_alpha > 0.01:
        for ci, atoms in enumerate(communities):
            ctype = community_types[ci]
            color = TYPE_COLORS.get(ctype, TYPE_COLORS["singleton"])
            draw_community_hull(ax, positions, atoms, color, alpha=hull_alpha)

    # Layer 2: Bonds (3D style)
    if state["show_bonds"] and bond_alpha > 0.01:
        for u, v in edges:
            if u in positions and v in positions:
                comm_u = atom_to_comm.get(u, -1)
                comm_v = atom_to_comm.get(v, -1)
                bt = bond_type_map.get((min(u, v), max(u, v)), 0)
                if comm_u == comm_v and hull_alpha > 0.01:
                    ctype = community_types[comm_u]
                    color = TYPE_COLORS.get(ctype, TYPE_COLORS["singleton"])
                    ba = bond_alpha * global_alpha * 0.8
                else:
                    color = BOND_COLOR
                    ba = bond_alpha * global_alpha * 0.5
                _draw_bond(
                    ax, positions[u], positions[v],
                    bond_type=bt, color=color, alpha=ba, zorder=1,
                )

    # Layer 3: 3D Atoms
    atom_radius = 0.055
    for atom_id in sorted(positions.keys()):
        if atom_id not in positions:
            continue
        x, y = positions[atom_id]
        color = atom_colors.get(atom_id, DEFAULT_ELEMENT_COLOR)
        elem = elements.get(atom_id, "")
        label = elem if state["show_labels"] else ""

        _draw_3d_atom(
            ax, x, y, atom_radius, color,
            label=label, alpha=global_alpha,
            label_fontsize=4, zorder_base=3,
        )

    # Layer 4: "MOSAIC" text label below the letters
    if state["show_mosaic_text"]:
        ax.text(
            0.0, fixed_ylim[0] + 0.25,
            "M   O   S   A   I   C",
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            color="#333333", alpha=0.9,
            fontfamily="sans-serif",
            zorder=10,
        )

    # Fixed axis limits for stable framing
    ax.set_xlim(*fixed_xlim)
    ax.set_ylim(*fixed_ylim)


# ============================================================================
# Main animation
# ============================================================================


def create_logo_animation(
    output_path: str = "figures/mosaic_logo.gif",
    fps: int = 30,
    dpi: int = 150,
) -> None:
    """Create the MOSAIC logo animation GIF."""
    print("Preparing molecule and communities...")
    result = prepare_six_communities(MOLECULE_SMILES)
    pos, communities, community_types, edges, num_atoms, elements, bond_type_map = result
    print(f"  {num_atoms} atoms, {len(communities)} communities")
    for ci, atoms in enumerate(communities):
        print(f"  Community {ci} ({community_types[ci]}): {len(atoms)} atoms")

    print("Computing letter positions...")
    letter_pos = compute_letter_positions(communities, pos)
    print(f"  Assigned {len(letter_pos)} atoms to letter positions")

    # Ensure all atoms have a letter position
    for atom_id in pos:
        if atom_id not in letter_pos:
            letter_pos[atom_id] = pos[atom_id]

    # Compute fixed axis limits encompassing both molecule and letter positions
    all_x = [p[0] for p in pos.values()] + [p[0] for p in letter_pos.values()]
    all_y = [p[1] for p in pos.values()] + [p[1] for p in letter_pos.values()]
    margin = 0.4
    fixed_xlim = (min(all_x) - margin, max(all_x) + margin)
    fixed_ylim = (min(all_y) - margin - 0.3, max(all_y) + margin)  # extra room for text

    print(f"Creating animation ({TOTAL_FRAMES} frames at {fps} fps)...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    def update(frame: int) -> list:
        state = get_frame_state(
            frame, pos, letter_pos, communities, community_types, elements
        )
        render_frame(
            ax, state, communities, community_types, edges,
            pos, letter_pos, elements, bond_type_map,
            fixed_xlim, fixed_ylim,
        )
        return []

    anim = FuncAnimation(
        fig, update, frames=TOTAL_FRAMES, interval=1000 // fps, blit=False,
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    # Report file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    """CLI entry point."""
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
