#!/usr/bin/env python
"""Generation demo script for visualizing autoregressive molecule generation.

This script generates molecules using trained models with different tokenization
schemes (HDT, HSENT, SENT, HDTC) and creates animated GIFs showing the
step-by-step generation process.

Usage:
    python scripts/visualization/generation_demo.py
    python scripts/visualization/generation_demo.py generation.num_samples=5 animation.fps=3
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from src.data.molecular import (  # noqa: E402
    ATOM_TYPES,
    BOND_TYPES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    graph_to_smiles,
)
from src.models.transformer import GraphGeneratorModule  # noqa: E402
from src.tokenizers import (  # noqa: E402
    HDTCTokenizer,
    HDTTokenizer,
    HSENTTokenizer,
    SENTTokenizer,
)
from src.tokenizers.motif.functional_detection import (  # noqa: E402
    FunctionalGroupDetector,
    FunctionalGroupInstance,
)
from src.tokenizers.motif.functional_patterns import PATTERN_PRIORITY  # noqa: E402

# Color scheme for visualization
ATOM_COLORS = {
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
DEFAULT_ATOM_COLOR = "#808080"

BOND_STYLES = {
    0: {"color": "#000000", "width": 2.0, "style": "-"},
    1: {"color": "#000000", "width": 3.0, "style": "-"},
    2: {"color": "#000000", "width": 4.0, "style": "-"},
    3: {"color": "#666666", "width": 2.0, "style": "--"},
}

# Community colors for visualization
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

# HDTC community type colors
HDTC_TYPE_COLORS = {
    "ring": "#4169E1",
    "functional": "#228B22",
    "singleton": "#808080",
}

# Motif highlighting colors
MOTIF_COLORS = {
    "ring": "#4169E1",
    "multi_atom": "#228B22",
    "single_atom": "#FF8C00",
}
MOTIF_SHADING_ALPHA = 0.20


# =============================================================================
# Base Visualizer (ABC)
# =============================================================================


class BaseGenerationVisualizer(ABC):
    """Abstract base class for step-by-step molecule generation visualization.

    Subclasses implement token parsing and optional side panels for each
    tokenization scheme.
    """

    def __init__(self, tokenizer: object, tokenizer_type: str) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type

    # ----- Abstract methods -----

    @abstractmethod
    def _parse_token_state(self, tokens: list[int]) -> dict:
        """Parse tokens to extract visualization state.

        Returns dict with at minimum:
            - phase: str describing current generation phase
            - communities: dict mapping node_id -> community_id
            - visible_nodes: set of node ids that should be visible
            - bipartite_edges: set of (src, dst) tuples for cross-community edges
            - current_community: int, ID of currently-active community (-1 if none)
        """

    @abstractmethod
    def _draw_side_panel(self, ax: plt.Axes, token_state: dict) -> None:
        """Draw an optional side panel (tree, block diagram, etc.).

        Called only when _has_side_panel() returns True.
        """

    @abstractmethod
    def _has_side_panel(self) -> bool:
        """Whether this visualizer needs a side panel."""

    # ----- Shared methods -----

    def generate_with_history(
        self,
        model: GraphGeneratorModule,
        max_length: int = 512,
        top_k: int = 10,
        temperature: float = 1.0,
    ) -> tuple[list[int], list[Data]]:
        """Generate tokens and create partial graphs at each step."""
        device = model.device
        sos = getattr(self.tokenizer, "sos", getattr(self.tokenizer, "SOS", 0))
        eos = getattr(self.tokenizer, "eos", getattr(self.tokenizer, "EOS", 1))

        tokens = [sos]
        graphs_history = []

        with torch.inference_mode():
            for _ in range(max_length - 1):
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :]

                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                top_k_probs = torch.softmax(top_k_logits / temperature, dim=-1)
                next_token_idx = torch.multinomial(top_k_probs, 1).item()
                next_token = top_k_indices[next_token_idx].item()

                tokens.append(next_token)

                try:
                    partial_graph = self._decode_partial(tokens)
                    if partial_graph is not None and partial_graph.num_nodes > 0:
                        graphs_history.append(partial_graph)
                    elif graphs_history:
                        graphs_history.append(graphs_history[-1])
                    else:
                        graphs_history.append(self._empty_graph())
                except Exception:
                    if graphs_history:
                        graphs_history.append(graphs_history[-1])
                    else:
                        graphs_history.append(self._empty_graph())

                if next_token == eos:
                    break

        return tokens, graphs_history

    def _decode_partial(self, tokens: list[int]) -> Optional[Data]:
        """Decode partial token sequence to graph."""
        eos = getattr(self.tokenizer, "eos", getattr(self.tokenizer, "EOS", 1))

        tokens_with_eos = list(tokens)
        if tokens_with_eos[-1] != eos:
            tokens_with_eos.append(eos)

        tokens_tensor = torch.tensor(tokens_with_eos, dtype=torch.long)

        try:
            graph = self.tokenizer.decode(tokens_tensor)
            return graph
        except Exception:
            return None

    def _empty_graph(self) -> Data:
        """Create an empty graph."""
        return Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=0,
        )

    def _detect_motifs_with_edges(
        self,
        graph: Data,
        motif_cfg: Optional[DictConfig] = None,
    ) -> list[dict]:
        """Detect motifs directly from a PyG graph.

        Builds an RDKit mol from the graph (preserving atom index mapping)
        and runs SMARTS matching on it, so motif atom indices correspond
        exactly to graph node indices.

        Returns list of dicts with:
            - name: motif name
            - pattern_type: "ring", "multi_atom", or "single_atom"
            - atoms: frozenset of atom indices
            - edges: set of (i, j) tuples (sorted, i < j) for required bonds
        """
        try:
            from rdkit import Chem
        except ImportError:
            return []

        include_rings = True
        if motif_cfg is not None:
            if not motif_cfg.get("enabled", True):
                return []
            include_rings = motif_cfg.get("include_rings", True)

        mol = self._graph_to_mol(graph)
        if mol is None:
            return []

        # Run SMARTS matching directly on the graph-built mol so that
        # atom indices are in graph-node space.
        detector = FunctionalGroupDetector(include_rings=include_rings)
        groups: list[FunctionalGroupInstance] = []

        if include_rings:
            for name, smarts in detector.ring_patterns.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is None:
                        continue
                    for match in mol.GetSubstructMatches(pattern):
                        groups.append(
                            FunctionalGroupInstance(
                                name=name,
                                pattern_type="ring",
                                atom_indices=frozenset(match),
                                priority=PATTERN_PRIORITY["ring"],
                                pattern=smarts,
                            )
                        )
                except Exception:
                    continue

        for name, (smarts, pattern_type) in detector.functional_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue
                for match in mol.GetSubstructMatches(pattern):
                    groups.append(
                        FunctionalGroupInstance(
                            name=name,
                            pattern_type=pattern_type,
                            atom_indices=frozenset(match),
                            priority=PATTERN_PRIORITY.get(pattern_type, 0),
                            pattern=smarts,
                        )
                    )
            except Exception:
                continue

        resolved = detector._resolve_overlaps(groups)
        if not resolved:
            return []

        result = []
        for group in resolved:
            atoms = group.atom_indices
            edges = set()
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in atoms and a2 in atoms:
                    edges.add((min(a1, a2), max(a1, a2)))

            result.append(
                {
                    "name": group.name,
                    "pattern_type": group.pattern_type,
                    "atoms": atoms,
                    "edges": edges,
                }
            )

        return result

    def create_animation(
        self,
        tokens: list[int],
        graphs_history: list[Data],
        output_path: Path,
        fps: int = 2,
        max_frames: int = 150,
        figsize: tuple[int, int] = (10, 12),
        show_tokens: bool = True,
        side_panel_width: int = 4,
        motif_cfg: Optional[DictConfig] = None,
    ) -> None:
        """Create animated GIF of generation process."""
        if not graphs_history:
            return

        # Sample frames if too many
        if len(graphs_history) > max_frames:
            indices = np.linspace(0, len(graphs_history) - 1, max_frames, dtype=int)
            graphs_sampled = [graphs_history[i] for i in indices]
            token_indices = indices.tolist()
        else:
            graphs_sampled = graphs_history
            token_indices = list(range(len(graphs_history)))

        # Pre-compute layout from final graph (used for all frames)
        final_graph = graphs_sampled[-1]
        positions = self._compute_layout(final_graph)

        # Detect motifs from final graph (indices match graph nodes directly)
        detected_motifs = self._detect_motifs_with_edges(final_graph, motif_cfg)

        # Pre-compute token states for each frame.
        # tokens_so_far includes SOS (index 0) + generated tokens up through
        # the token at token_idx, hence the +2 offset.
        token_states = []
        for token_idx in token_indices:
            tokens_so_far = tokens[: token_idx + 2]
            state = self._parse_token_state(tokens_so_far)
            token_states.append(state)

        # Build figure layout based on subclass configuration
        has_panel = self._has_side_panel()
        graph_width = figsize[0]
        tree_width = side_panel_width

        if has_panel:
            if show_tokens:
                fig = plt.figure(figsize=(tree_width + graph_width, figsize[1]))
                gs = fig.add_gridspec(
                    2,
                    2,
                    width_ratios=[tree_width, graph_width],
                    height_ratios=[4, 1],
                )
                ax_panel = fig.add_subplot(gs[0, 0])
                ax_graph = fig.add_subplot(gs[0, 1])
                ax_tokens = fig.add_subplot(gs[1, :])
            else:
                fig = plt.figure(figsize=(tree_width + graph_width, figsize[1] - 2))
                gs = fig.add_gridspec(1, 2, width_ratios=[tree_width, graph_width])
                ax_panel = fig.add_subplot(gs[0, 0])
                ax_graph = fig.add_subplot(gs[0, 1])
                ax_tokens = None
        else:
            ax_panel = None
            if show_tokens:
                fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[4, 1])
                ax_graph, ax_tokens = axes
            else:
                fig, ax_graph = plt.subplots(figsize=figsize)
                ax_tokens = None

        # Motif config for drawing
        show_motif_labels = True
        show_motif_gallery = True
        gallery_max_items = 8
        if motif_cfg is not None:
            show_motif_labels = motif_cfg.get("show_labels", True)
            show_motif_gallery = motif_cfg.get("show_gallery", True)
            gallery_max_items = motif_cfg.get("gallery_max_items", 8)

        def update(frame_idx: int) -> list:
            ax_graph.clear()
            graph = graphs_sampled[frame_idx]
            token_up_to = token_indices[frame_idx] + 1
            state = token_states[frame_idx]

            # Draw side panel if applicable
            if ax_panel is not None:
                ax_panel.clear()
                self._draw_side_panel(ax_panel, state)

            # Draw graph with community colors, edge styles, and motif shading
            self._draw_graph(
                ax_graph,
                graph,
                frame_idx,
                len(graphs_sampled),
                positions,
                token_state=state,
                motifs=detected_motifs,
                show_motif_labels=show_motif_labels,
            )

            # Draw motif gallery
            if show_motif_gallery and detected_motifs:
                self._draw_motif_gallery(
                    ax_graph, detected_motifs, graph, state, gallery_max_items
                )

            # Title with phase annotation
            phase = state.get("phase", "Generating")
            title = (
                f"{self.tokenizer_type.upper()}: {phase}\n"
                f"Step {frame_idx + 1}/{len(graphs_sampled)}"
            )
            ax_graph.set_title(title, fontsize=14, fontweight="bold")

            if ax_tokens is not None:
                ax_tokens.clear()
                self._draw_token_sequence(ax_tokens, tokens[:token_up_to])

            fig.tight_layout()
            return []

        anim = FuncAnimation(
            fig, update, frames=len(graphs_sampled), interval=1000 // fps, blit=True
        )

        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
        plt.close(fig)

    def _compute_layout(self, graph: Data) -> dict[int, tuple[float, float]]:
        """Compute node positions using RDKit 2D coordinates with spring fallback."""
        if graph.num_nodes == 0:
            return {}

        pos = self._compute_rdkit_layout(graph)
        if pos is not None:
            return pos

        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))

        if graph.edge_index.numel() > 0:
            edge_index = graph.edge_index.numpy()
            for k in range(edge_index.shape[1]):
                i, j = int(edge_index[0, k]), int(edge_index[1, k])
                if i < j and i < graph.num_nodes and j < graph.num_nodes:
                    G.add_edge(i, j)

        layout_scale = 1.5 * max(1.0, np.sqrt(graph.num_nodes / 15))
        if G.number_of_edges() > 0:
            pos = nx.spring_layout(
                G, seed=42, k=2.5, iterations=100, scale=layout_scale
            )
        else:
            pos = nx.circular_layout(G, scale=layout_scale)

        return pos

    def _graph_to_mol(self, graph: Data) -> Optional[object]:
        """Build an RDKit mol from a PyG graph with atom indices preserved.

        Node i in the graph becomes atom i in the mol, so substructure
        match indices can be used directly as graph node indices.

        Args:
            graph: PyG Data object with node/edge features.

        Returns:
            RDKit Mol or None if construction fails.
        """
        try:
            from rdkit import Chem
        except ImportError:
            return None

        try:
            num_nodes = graph.num_nodes
            edge_index = graph.edge_index

            mol = Chem.RWMol()

            labeled = graph.x.dtype == torch.long or graph.x.dtype == torch.int64

            for i in range(num_nodes):
                if labeled:
                    atom_type_idx = int(graph.x[i])
                    if atom_type_idx < len(ATOM_TYPES):
                        atom_symbol = ATOM_TYPES[atom_type_idx]
                    else:
                        atom_symbol = "C"
                elif graph.x is not None and graph.x.dim() >= 2:
                    atom_type_idx = int(torch.argmax(graph.x[i, :NUM_ATOM_TYPES]))
                    if atom_type_idx < len(ATOM_TYPES):
                        atom_symbol = ATOM_TYPES[atom_type_idx]
                    else:
                        atom_symbol = "C"
                else:
                    atom_symbol = "C"

                mol.AddAtom(Chem.Atom(atom_symbol))

            added_bonds = set()
            for k in range(edge_index.size(1)):
                i = int(edge_index[0, k])
                j = int(edge_index[1, k])
                if i < j and (i, j) not in added_bonds:
                    added_bonds.add((i, j))

                    bond_type = Chem.rdchem.BondType.SINGLE
                    if graph.edge_attr is not None and graph.edge_attr.size(0) > k:
                        if labeled:
                            bond_type_idx = int(graph.edge_attr[k])
                        else:
                            bond_type_idx = int(
                                torch.argmax(graph.edge_attr[k, :NUM_BOND_TYPES])
                            )
                        if bond_type_idx < len(BOND_TYPES):
                            bond_type = BOND_TYPES[bond_type_idx]

                    mol.AddBond(i, j, bond_type)

            return mol.GetMol()

        except Exception:
            return None

    def _compute_rdkit_layout(
        self, graph: Data
    ) -> Optional[dict[int, tuple[float, float]]]:
        """Compute 2D molecular layout using RDKit."""
        try:
            from rdkit.Chem import AllChem
        except ImportError:
            return None

        mol = self._graph_to_mol(graph)
        if mol is None:
            return None

        try:
            num_nodes = graph.num_nodes

            AllChem.Compute2DCoords(mol)
            conformer = mol.GetConformer()

            pos = {}
            for i in range(num_nodes):
                atom_pos = conformer.GetAtomPosition(i)
                pos[i] = (atom_pos.x, atom_pos.y)

            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
                y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1
                scale = max(x_range, y_range) if max(x_range, y_range) > 0 else 1
                x_center = (max(x_coords) + min(x_coords)) / 2
                y_center = (max(y_coords) + min(y_coords)) / 2

                # Scale output size with sqrt(num_nodes) so larger molecules
                # get more room for motif shading and readable labels.
                output_scale = 3.0 * max(1.0, np.sqrt(num_nodes / 15))

                for node_id in pos:
                    x, y = pos[node_id]
                    pos[node_id] = (
                        (x - x_center) / scale * output_scale,
                        (y - y_center) / scale * output_scale,
                    )

            return pos

        except Exception:
            return None

    def _draw_graph(
        self,
        ax: plt.Axes,
        graph: Data,
        step: int,
        total_steps: int,
        fixed_positions: Optional[dict] = None,
        token_state: Optional[dict] = None,
        motifs: Optional[list[dict]] = None,
        show_motif_labels: bool = True,
    ) -> None:
        """Draw a molecular graph with community colors, edge styles, and motif shading."""
        ax.set_aspect("equal")
        ax.axis("off")

        if graph.num_nodes == 0:
            ax.text(
                0.5,
                0.5,
                "Generating...",
                ha="center",
                va="center",
                fontsize=16,
                transform=ax.transAxes,
            )
            return

        visible_nodes = None
        communities = {}
        cross_community_edges = set()
        if token_state:
            visible_nodes = token_state.get("visible_nodes")
            communities = token_state.get("communities", {})
            cross_community_edges = token_state.get("bipartite_edges", set())
            cross_community_edges = cross_community_edges.union(
                token_state.get("cross_community_edges", set())
            )

        if visible_nodes is not None and len(visible_nodes) > 0:
            nodes_to_show = visible_nodes & set(range(graph.num_nodes))
        else:
            nodes_to_show = set(range(graph.num_nodes))

        G = nx.Graph()
        for i in nodes_to_show:
            atom_label = "C"
            if hasattr(graph, "x") and graph.x is not None and i < graph.x.size(0):
                if graph.x.dtype == torch.long:
                    atom_idx = int(graph.x[i])
                    if atom_idx < len(ATOM_TYPES):
                        atom_label = ATOM_TYPES[atom_idx]
                elif graph.x.dim() >= 2 and graph.x.size(1) >= NUM_ATOM_TYPES:
                    atom_idx = int(torch.argmax(graph.x[i, :NUM_ATOM_TYPES]))
                    if atom_idx < len(ATOM_TYPES):
                        atom_label = ATOM_TYPES[atom_idx]
            community_id = communities.get(i, -1)
            G.add_node(i, atom=atom_label, community=community_id)

        if graph.edge_index.numel() > 0:
            edge_index = graph.edge_index.numpy()
            for k in range(edge_index.shape[1]):
                i, j = int(edge_index[0, k]), int(edge_index[1, k])
                if i < j and i in nodes_to_show and j in nodes_to_show:
                    bond_type = 0
                    if (
                        hasattr(graph, "edge_attr")
                        and graph.edge_attr is not None
                        and k < graph.edge_attr.size(0)
                    ):
                        if graph.edge_attr.dtype == torch.long:
                            bond_type = int(graph.edge_attr[k])
                        elif (
                            graph.edge_attr.dim() >= 2
                            and graph.edge_attr.size(1) >= NUM_BOND_TYPES
                        ):
                            bond_type = int(
                                torch.argmax(graph.edge_attr[k, :NUM_BOND_TYPES])
                            )

                    is_cross_community = (i, j) in cross_community_edges or (
                        j,
                        i,
                    ) in cross_community_edges
                    G.add_edge(
                        i,
                        j,
                        bond_type=bond_type,
                        is_cross_community=is_cross_community,
                    )

        if G.number_of_nodes() == 0:
            ax.text(
                0.5,
                0.5,
                "Building structure...",
                ha="center",
                va="center",
                fontsize=16,
                transform=ax.transAxes,
            )
            return

        # Use fixed positions
        if fixed_positions:
            pos = {n: fixed_positions[n] for n in G.nodes() if n in fixed_positions}
            for n in G.nodes():
                if n not in pos:
                    pos[n] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        else:
            if G.number_of_edges() > 0:
                pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)
            else:
                pos = nx.circular_layout(G)

        # Draw motif shading for complete motifs
        if motifs and pos:
            current_edges = set()
            for u, v in G.edges():
                current_edges.add((min(u, v), max(u, v)))

            for motif in motifs:
                motif_atoms = motif["atoms"]
                motif_edges = motif["edges"]
                pattern_type = motif.get("pattern_type", "ring")

                all_atoms_present = all(atom in G.nodes() for atom in motif_atoms)
                if not all_atoms_present:
                    continue

                all_edges_present = all(edge in current_edges for edge in motif_edges)
                if not all_edges_present:
                    continue

                # Color by pattern type
                shade_color = MOTIF_COLORS.get(pattern_type, "#4169E1")

                motif_positions = [pos[atom] for atom in motif_atoms if atom in pos]
                if len(motif_positions) >= 3:
                    try:
                        from scipy.spatial import ConvexHull

                        points = np.array(motif_positions)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.fill(
                            hull_points[:, 0],
                            hull_points[:, 1],
                            color=shade_color,
                            alpha=MOTIF_SHADING_ALPHA,
                            zorder=0,
                        )
                    except Exception:
                        pass

                    # Draw motif label at centroid
                    if show_motif_labels:
                        centroid = np.mean(points, axis=0)
                        # Scale label offset with hull size
                        hull_span = np.ptp(points, axis=0).max()
                        label_offset = max(0.25, hull_span * 0.3)
                        ax.text(
                            centroid[0],
                            centroid[1] + label_offset,
                            motif["name"],
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontstyle="italic",
                            color=shade_color,
                            zorder=5,
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                facecolor="white",
                                edgecolor=shade_color,
                                alpha=0.7,
                            ),
                        )

                elif len(motif_positions) == 2:
                    from matplotlib.patches import Ellipse

                    p1, p2 = motif_positions
                    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                    width = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) + 0.3
                    ellipse = Ellipse(
                        center,
                        width=width,
                        height=0.4,
                        angle=np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])),
                        color=shade_color,
                        alpha=MOTIF_SHADING_ALPHA,
                        zorder=0,
                    )
                    ax.add_patch(ellipse)

                    if show_motif_labels:
                        ax.text(
                            center[0],
                            center[1] + 0.25,
                            motif["name"],
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontstyle="italic",
                            color=shade_color,
                            zorder=5,
                        )

                elif len(motif_positions) == 1:
                    from matplotlib.patches import Circle

                    cx, cy = motif_positions[0]
                    circle = Circle(
                        (cx, cy),
                        radius=0.35,
                        color=shade_color,
                        alpha=MOTIF_SHADING_ALPHA,
                        zorder=0,
                    )
                    ax.add_patch(circle)

                    if show_motif_labels:
                        ax.text(
                            cx,
                            cy + 0.45,
                            motif["name"],
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontstyle="italic",
                            color=shade_color,
                            zorder=5,
                        )

        # Draw edges
        from matplotlib.patches import FancyArrowPatch

        for u, v, data in G.edges(data=True):
            bond_type = data.get("bond_type", 0)
            is_cross_community = data.get("is_cross_community", False)
            style = BOND_STYLES.get(bond_type, BOND_STYLES[0])

            if is_cross_community:
                posA = pos[u]
                posB = pos[v]
                arrow = FancyArrowPatch(
                    posA,
                    posB,
                    connectionstyle="arc3,rad=0.3",
                    color="#9932CC",
                    linewidth=2.5,
                    linestyle="--",
                    zorder=1,
                    arrowstyle="-",
                )
                ax.add_patch(arrow)
            else:
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    linewidth=style["width"],
                    linestyle="-",
                    zorder=1,
                )

        # Draw nodes with community colors.
        # Scale node size and font so larger graphs stay readable.
        n_shown = G.number_of_nodes()
        if n_shown <= 15:
            node_size = 1200
            node_lw = 4
            label_fontsize = 9
        elif n_shown <= 30:
            node_size = 800
            node_lw = 3
            label_fontsize = 7
        else:
            node_size = 500
            node_lw = 2
            label_fontsize = 6

        node_colors = []
        node_edge_colors = []
        node_labels = {}
        for node in G.nodes():
            atom = G.nodes[node].get("atom", "C")
            community = G.nodes[node].get("community", -1)

            node_colors.append(ATOM_COLORS.get(atom, DEFAULT_ATOM_COLOR))

            if community >= 0:
                edge_color = COMMUNITY_COLORS[community % len(COMMUNITY_COLORS)]
            else:
                edge_color = "black"
            node_edge_colors.append(edge_color)

            if community >= 0:
                node_labels[node] = f"{atom}\n(C{community})"
            else:
                node_labels[node] = atom

        node_list = list(G.nodes())
        node_positions = np.array([pos[n] for n in node_list])

        for idx, node in enumerate(node_list):
            ax.scatter(
                node_positions[idx, 0],
                node_positions[idx, 1],
                s=node_size,
                c=[node_colors[idx]],
                edgecolors=[node_edge_colors[idx]],
                linewidths=node_lw,
                zorder=2,
            )

        for node, (x, y) in pos.items():
            label = node_labels.get(node, "")
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        # Add padding scaled to layout extent
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            x_span = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 2.0
            y_span = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 2.0
            margin = max(0.5, 0.1 * max(x_span, y_span))
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

        # Info box
        total_graph_nodes = graph.num_nodes
        shown_nodes = G.number_of_nodes()
        info_text = (
            f"Nodes: {shown_nodes}/{total_graph_nodes}, Edges: {G.number_of_edges()}"
        )
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Legend for communities and edge types
        legend_elements = []
        max_legend_entries = 10

        unique_communities = sorted(
            {
                G.nodes[n].get("community", -1)
                for n in G.nodes()
                if G.nodes[n].get("community", -1) >= 0
            }
        )
        for comm_id in unique_communities[:max_legend_entries]:
            color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
            legend_elements.append(
                mpatches.Patch(
                    facecolor=color, edgecolor=color, label=f"Community {comm_id}"
                )
            )
        if len(unique_communities) > max_legend_entries:
            remaining = len(unique_communities) - max_legend_entries
            legend_elements.append(
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="gray",
                    label=f"... +{remaining} more",
                )
            )

        if cross_community_edges:
            legend_elements.append(
                mpatches.Patch(
                    facecolor="white", edgecolor="black", label="Intra-community"
                )
            )
            legend_elements.append(
                mpatches.Patch(
                    facecolor="white", edgecolor="#9932CC", label="Cross-community"
                )
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=7,
                framealpha=0.8,
                ncol=1,
            )

    def _draw_motif_gallery(
        self,
        ax: plt.Axes,
        motifs: list[dict],
        graph: Data,
        token_state: dict,
        max_items: int = 8,
    ) -> None:
        """Draw a compact motif gallery listing completed motifs in the graph panel."""
        if not motifs:
            return

        # Determine which motifs are complete in this frame
        visible_nodes = token_state.get("visible_nodes", set())
        current_edges = set()
        if graph.edge_index.numel() > 0:
            edge_index = graph.edge_index.numpy()
            for k in range(edge_index.shape[1]):
                i, j = int(edge_index[0, k]), int(edge_index[1, k])
                if i in visible_nodes and j in visible_nodes:
                    current_edges.add((min(i, j), max(i, j)))

        completed = []
        for motif in motifs:
            all_atoms = all(a in visible_nodes for a in motif["atoms"])
            all_edges = all(e in current_edges for e in motif["edges"])
            if all_atoms and all_edges:
                completed.append(motif)

        if not completed:
            return

        # Deduplicate by name
        seen = set()
        unique = []
        for m in completed:
            if m["name"] not in seen:
                seen.add(m["name"])
                unique.append(m)

        display = unique[:max_items]

        # Draw a small text box listing completed motifs
        lines = [f"  {m['name']} ({m['pattern_type'][0].upper()})" for m in display]
        gallery_text = "Motifs:\n" + "\n".join(lines)

        ax.text(
            0.98,
            0.02,
            gallery_text,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="right",
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightyellow",
                edgecolor="gray",
                alpha=0.85,
            ),
            zorder=10,
        )

    def _draw_token_sequence(
        self,
        ax: plt.Axes,
        tokens: list[int],
        max_display: int = 50,
    ) -> None:
        """Draw the token sequence."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        if len(tokens) > max_display:
            display_tokens = tokens[-max_display:]
        else:
            display_tokens = tokens

        n_tokens = len(display_tokens)
        if n_tokens == 0:
            return

        idx_offset = getattr(
            self.tokenizer, "IDX_OFFSET", getattr(self.tokenizer, "idx_offset", 6)
        )

        box_width = 0.95 / n_tokens
        box_height = 0.5

        for i, tok in enumerate(display_tokens):
            x = 0.025 + i * (0.95 / n_tokens)
            y = 0.25

            if tok < idx_offset:
                color = "lightcoral"
                label = f"T{tok}"
            else:
                color = "lightblue"
                label = str(tok - idx_offset)

            edge_color = "red" if i == n_tokens - 1 else "black"
            edge_width = 3 if i == n_tokens - 1 else 1

            rect = mpatches.Rectangle(
                (x, y),
                box_width * 0.9,
                box_height,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
            )
            ax.add_patch(rect)
            ax.text(
                x + box_width * 0.45,
                y + box_height / 2,
                label,
                ha="center",
                va="center",
                fontsize=6,
            )

        ax.text(0.5, 0.05, f"Tokens: {len(tokens)}", ha="center", fontsize=9)


# =============================================================================
# SENT Visualizer
# =============================================================================


class SENTVisualizer(BaseGenerationVisualizer):
    """Visualizer for SENT (flat random walk) tokenization."""

    def _has_side_panel(self) -> bool:
        return False

    def _draw_side_panel(self, ax: plt.Axes, token_state: dict) -> None:
        pass

    def _parse_token_state(self, tokens: list[int]) -> dict:
        state = {
            "phase": "Initializing",
            "communities": {},
            "visible_nodes": set(),
            "bipartite_edges": set(),
            "current_community": -1,
        }

        idx_offset = self.tokenizer.idx_offset
        RESET = self.tokenizer.reset
        LADJ = self.tokenizer.ladj
        RADJ = self.tokenizer.radj

        is_labeled = getattr(self.tokenizer, "labeled_graph", False)
        node_idx_offset = getattr(self.tokenizer, "node_idx_offset", None)
        max_num_nodes = getattr(self.tokenizer, "max_num_nodes", 100)

        in_bracket = False
        node_count = 0
        trail_count = 1

        for tok in tokens:
            if tok == RESET:
                trail_count += 1
            elif tok == LADJ:
                in_bracket = True
            elif tok == RADJ:
                in_bracket = False
            elif tok >= idx_offset and not in_bracket:
                if is_labeled and node_idx_offset is not None:
                    if tok < node_idx_offset:
                        state["visible_nodes"].add(node_count)
                        node_count += 1
                else:
                    node_id = tok - idx_offset
                    if node_id < max_num_nodes:
                        state["visible_nodes"].add(node_count)
                        node_count += 1

        state["phase"] = f"Random Walk (Trail {trail_count})"
        return state


# =============================================================================
# HDT Visualizer
# =============================================================================


class HDTVisualizer(BaseGenerationVisualizer):
    """Visualizer for HDT (hierarchical DFS) tokenization with abstract tree panel."""

    def _has_side_panel(self) -> bool:
        return True

    def _draw_side_panel(self, ax: plt.Axes, token_state: dict) -> None:
        self._draw_abstract_tree(ax, token_state)

    def _parse_token_state(self, tokens: list[int]) -> dict:
        state = {
            "phase": "Initializing",
            "communities": {},
            "visible_nodes": set(),
            "bipartite_edges": set(),
            "current_community": -1,
        }

        eos = self.tokenizer.EOS
        tokens_with_eos = list(tokens)
        if tokens_with_eos[-1] != eos:
            tokens_with_eos.append(eos)

        tokens_tensor = torch.tensor(tokens_with_eos, dtype=torch.long)

        try:
            hg = self.tokenizer.parse_tokens(tokens_tensor)
        except Exception:
            state["phase"] = "Parsing..."
            return state

        state["abstract_tree"] = {}
        state["community_order"] = []
        state["cross_community_edges"] = set()

        for part in hg.partitions:
            part_id = part.part_id
            nodes = part.global_node_indices
            if nodes:
                state["abstract_tree"][part_id] = list(nodes)
                state["community_order"].append(part_id)
                for node_idx in nodes:
                    state["visible_nodes"].add(node_idx)
                    state["communities"][node_idx] = part_id

        for bipart in hg.bipartites:
            left_part = hg.get_partition(bipart.left_part_id)
            right_part = hg.get_partition(bipart.right_part_id)
            if bipart.edge_index.numel() > 0:
                ei = bipart.edge_index.numpy()
                for e in range(ei.shape[1]):
                    left_local = int(ei[0, e])
                    right_local = int(ei[1, e])
                    left_global = left_part.local_to_global(left_local)
                    right_global = right_part.local_to_global(right_local)
                    state["cross_community_edges"].add((left_global, right_global))
                    state["cross_community_edges"].add((right_global, left_global))

        idx_offset = self.tokenizer.IDX_OFFSET
        ENTER = self.tokenizer.ENTER

        current_community = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == ENTER and i + 2 < len(tokens):
                level = tokens[i + 1] - idx_offset if tokens[i + 1] >= idx_offset else 0
                part_id = (
                    tokens[i + 2] - idx_offset if tokens[i + 2] >= idx_offset else 0
                )
                if level >= 1:
                    current_community = part_id
                    break

        if current_community >= 0:
            state["current_community"] = current_community
            state["phase"] = f"Community {current_community}"
        else:
            state["phase"] = "Building hierarchy"

        if not state["visible_nodes"]:
            state["phase"] = "Building Structure"

        return state

    def _draw_abstract_tree(self, ax: plt.Axes, token_state: dict) -> None:
        """Draw abstract tree showing communities and leaf nodes."""
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Abstract Tree", fontsize=12, fontweight="bold")

        abstract_tree = token_state.get("abstract_tree", {})
        community_order = token_state.get("community_order", [])
        current_community = token_state.get("current_community", -1)

        if not abstract_tree:
            ax.text(
                0.5,
                0.5,
                "Building hierarchy...",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            return

        G = nx.DiGraph()
        G.add_node("root", label="Root", node_type="root")

        pos = {"root": (0.5, 0.9)}
        num_communities = len(community_order)

        if num_communities > 0:
            comm_spacing = 0.8 / max(num_communities, 1)
            for idx, comm_id in enumerate(community_order):
                comm_node = f"C{comm_id}"
                G.add_node(comm_node, label=f"C{comm_id}", node_type="community")
                G.add_edge("root", comm_node)
                comm_x = 0.1 + idx * comm_spacing + comm_spacing / 2
                pos[comm_node] = (comm_x, 0.6)

                leaf_nodes = abstract_tree.get(comm_id, [])
                if leaf_nodes:
                    leaf_spacing = comm_spacing / max(len(leaf_nodes) + 1, 1)
                    for leaf_idx, leaf_id in enumerate(leaf_nodes):
                        leaf_node = f"L{leaf_id}"
                        G.add_node(leaf_node, label=str(leaf_id), node_type="leaf")
                        G.add_edge(comm_node, leaf_node)
                        leaf_x = (
                            comm_x - comm_spacing / 2 + (leaf_idx + 1) * leaf_spacing
                        )
                        pos[leaf_node] = (leaf_x, 0.2)

        for u, v in G.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color="gray", linewidth=1.5, zorder=1)

        for node in G.nodes():
            node_type = G.nodes[node].get("node_type", "")
            label = G.nodes[node].get("label", "")
            x, y = pos[node]

            if node_type == "root":
                color = "#808080"
                size = 400
            elif node_type == "community":
                comm_id = int(label[1:])
                color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
                size = 500
                if comm_id == current_community:
                    ax.scatter(x, y, s=800, c="yellow", alpha=0.5, zorder=0)
            else:
                color = "#404040"
                size = 300

            ax.scatter(
                x, y, s=size, c=color, edgecolors="black", linewidths=1.5, zorder=2
            )
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.0)


# =============================================================================
# HDTC Visualizer
# =============================================================================


class HDTCVisualizer(BaseGenerationVisualizer):
    """Visualizer for HDTC (compositional) tokenization with typed abstract tree."""

    def _has_side_panel(self) -> bool:
        return True

    def _draw_side_panel(self, ax: plt.Axes, token_state: dict) -> None:
        self._draw_typed_abstract_tree(ax, token_state)

    def _parse_token_state(self, tokens: list[int]) -> dict:
        state = {
            "phase": "Initializing",
            "communities": {},
            "visible_nodes": set(),
            "bipartite_edges": set(),
            "cross_community_edges": set(),
            "current_community": -1,
            "community_types": {},
            "abstract_tree": {},
            "community_order": [],
        }

        eos = self.tokenizer.EOS
        tokens_with_eos = list(tokens)
        if tokens_with_eos[-1] != eos:
            tokens_with_eos.append(eos)

        tokens_tensor = torch.tensor(tokens_with_eos, dtype=torch.long)

        try:
            hierarchy = self.tokenizer.parse_tokens(tokens_tensor)
        except Exception:
            state["phase"] = "Parsing..."
            return state

        # Extract community info from TwoLevelHierarchy
        for comm in hierarchy.communities:
            cid = comm.community_id
            state["abstract_tree"][cid] = list(comm.atom_indices)
            state["community_order"].append(cid)
            state["community_types"][cid] = comm.community_type

            for atom_idx in comm.atom_indices:
                state["visible_nodes"].add(atom_idx)
                state["communities"][atom_idx] = cid

        # Extract super-graph edges (atom-level for molecule graph,
        # community-level for tree panel)
        state["super_edge_community_pairs"] = set()
        for se in hierarchy.super_edges:
            state["cross_community_edges"].add((se.source_atom, se.target_atom))
            state["cross_community_edges"].add((se.target_atom, se.source_atom))
            pair = (
                min(se.source_community, se.target_community),
                max(se.source_community, se.target_community),
            )
            state["super_edge_community_pairs"].add(pair)

        # Determine current phase by scanning raw tokens backward
        COMM_START = self.tokenizer.COMM_START
        SUPER_START = self.tokenizer.SUPER_START

        current_community = -1
        phase = "Building hierarchy"

        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == SUPER_START:
                phase = "Super-graph connections"
                break
            if tokens[i] == COMM_START:
                # Find the community ID from next tokens
                idx_offset = self.tokenizer.IDX_OFFSET
                # COMM_START TYPE_X <comm_id> ...
                if i + 2 < len(tokens) and tokens[i + 2] >= idx_offset:
                    current_community = tokens[i + 2] - idx_offset
                    comm_type = state["community_types"].get(
                        current_community, "unknown"
                    )
                    phase = f"Community {current_community} ({comm_type})"
                break

        state["current_community"] = current_community
        state["phase"] = phase

        if not state["visible_nodes"]:
            state["phase"] = "Building Structure"

        return state

    def _draw_typed_abstract_tree(self, ax: plt.Axes, token_state: dict) -> None:
        """Draw abstract tree with community type labels (R/F/S)."""
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Compositional Tree", fontsize=12, fontweight="bold")

        abstract_tree = token_state.get("abstract_tree", {})
        community_order = token_state.get("community_order", [])
        community_types = token_state.get("community_types", {})
        current_community = token_state.get("current_community", -1)

        if not abstract_tree:
            ax.text(
                0.5,
                0.5,
                "Building hierarchy...",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            return

        G = nx.DiGraph()
        G.add_node("root", label="Root", node_type="root")

        pos = {"root": (0.5, 0.9)}
        num_communities = len(community_order)

        # Type prefix: R=ring, F=functional, S=singleton
        type_prefix = {"ring": "R", "functional": "F", "singleton": "S"}

        if num_communities > 0:
            comm_spacing = 0.8 / max(num_communities, 1)
            for idx, comm_id in enumerate(community_order):
                comm_type = community_types.get(comm_id, "singleton")
                prefix = type_prefix.get(comm_type, "?")
                comm_node = f"C{comm_id}"
                comm_label = f"{prefix}{comm_id}"
                G.add_node(
                    comm_node,
                    label=comm_label,
                    node_type="community",
                    comm_type=comm_type,
                )
                G.add_edge("root", comm_node)
                comm_x = 0.1 + idx * comm_spacing + comm_spacing / 2
                pos[comm_node] = (comm_x, 0.6)

                leaf_nodes = abstract_tree.get(comm_id, [])
                if leaf_nodes:
                    leaf_spacing = comm_spacing / max(len(leaf_nodes) + 1, 1)
                    for leaf_idx, leaf_id in enumerate(leaf_nodes):
                        leaf_node = f"L{leaf_id}"
                        G.add_node(leaf_node, label=str(leaf_id), node_type="leaf")
                        G.add_edge(comm_node, leaf_node)
                        leaf_x = (
                            comm_x - comm_spacing / 2 + (leaf_idx + 1) * leaf_spacing
                        )
                        pos[leaf_node] = (leaf_x, 0.2)

        # Draw tree edges (root -> community -> leaf)
        for u, v in G.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color="gray", linewidth=1.5, zorder=1)

        # Draw super-graph connections between communities (purple dotted arcs)
        super_pairs = token_state.get("super_edge_community_pairs", set())
        for src_comm, dst_comm in super_pairs:
            src_node = f"C{src_comm}"
            dst_node = f"C{dst_comm}"
            if src_node in pos and dst_node in pos:
                ax.annotate(
                    "",
                    xy=pos[dst_node],
                    xytext=pos[src_node],
                    arrowprops=dict(
                        arrowstyle="-",
                        color="#9932CC",
                        lw=1.5,
                        linestyle="--",
                        connectionstyle="arc3,rad=0.3",
                    ),
                    zorder=1,
                )

        # Draw nodes
        for node in G.nodes():
            node_type = G.nodes[node].get("node_type", "")
            label = G.nodes[node].get("label", "")
            x, y = pos[node]

            if node_type == "root":
                color = "#808080"
                size = 400
            elif node_type == "community":
                comm_type = G.nodes[node].get("comm_type", "singleton")
                color = HDTC_TYPE_COLORS.get(comm_type, "#808080")
                size = 500
                # Extract comm_id for highlight check
                comm_id_str = node[1:]  # Strip "C" prefix
                try:
                    comm_id = int(comm_id_str)
                except ValueError:
                    comm_id = -1
                if comm_id == current_community:
                    ax.scatter(x, y, s=800, c="gold", alpha=0.5, zorder=0)
            else:
                color = "#404040"
                size = 300

            ax.scatter(
                x, y, s=size, c=color, edgecolors="black", linewidths=1.5, zorder=2
            )
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        # Type legend
        for i, (tname, tcolor) in enumerate(HDTC_TYPE_COLORS.items()):
            ax.scatter(
                [], [], c=tcolor, s=60, label=tname.capitalize(), edgecolors="black"
            )
        ax.legend(loc="lower center", fontsize=7, ncol=3, framealpha=0.8)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.0)


# =============================================================================
# HSENT Visualizer
# =============================================================================


class HSENTVisualizer(BaseGenerationVisualizer):
    """Visualizer for HSENT with block diagram side panel."""

    def _has_side_panel(self) -> bool:
        return True

    def _draw_side_panel(self, ax: plt.Axes, token_state: dict) -> None:
        self._draw_block_diagram(ax, token_state)

    def _parse_token_state(self, tokens: list[int]) -> dict:
        """Hybrid parser: parse_tokens() for graph data, manual scan for progress."""
        state = {
            "phase": "Initializing",
            "communities": {},
            "visible_nodes": set(),
            "bipartite_edges": set(),
            "current_community": -1,
            "communities_info": {},
            "bipartite_connections": [],
        }

        # --- Ground-truth graph data via tokenizer.parse_tokens() ---
        eos = self.tokenizer.EOS
        tokens_with_eos = list(tokens)
        if tokens_with_eos[-1] != eos:
            tokens_with_eos.append(eos)

        tokens_tensor = torch.tensor(tokens_with_eos, dtype=torch.long)

        try:
            hg = self.tokenizer.parse_tokens(tokens_tensor)
        except Exception:
            state["phase"] = "Parsing..."
            return state

        # Extract visible_nodes and communities from parsed hierarchy
        for part in hg.partitions:
            part_id = part.part_id
            nodes = part.global_node_indices
            for node_idx in nodes:
                state["visible_nodes"].add(node_idx)
                state["communities"][node_idx] = part_id

        # Extract bipartite edges (cross-community) from parsed hierarchy
        for bipart in hg.bipartites:
            left_part = hg.get_partition(bipart.left_part_id)
            right_part = hg.get_partition(bipart.right_part_id)
            if bipart.edge_index.numel() > 0:
                ei = bipart.edge_index.numpy()
                for e in range(ei.shape[1]):
                    left_local = int(ei[0, e])
                    right_local = int(ei[1, e])
                    if (
                        left_local < left_part.num_nodes
                        and right_local < right_part.num_nodes
                    ):
                        lg = left_part.local_to_global(left_local)
                        rg = right_part.local_to_global(right_local)
                        state["bipartite_edges"].add((lg, rg))
                        state["bipartite_edges"].add((rg, lg))
                        state["bipartite_connections"].append(
                            (bipart.left_part_id, bipart.right_part_id, lg, rg)
                        )

        # --- Manual scan for phase/progress tracking (for block diagram) ---
        LCOM = self.tokenizer.LCOM
        RCOM = self.tokenizer.RCOM
        LBIP = self.tokenizer.LBIP
        SEP = self.tokenizer.SEP
        idx_offset = self.tokenizer.IDX_OFFSET

        current_part_id = -1
        in_community_header = False
        in_community_sent = False
        header_position = 0
        header_global_indices: list[int] = []
        sent_node_count = 0

        for tok in tokens:
            if tok == LCOM:
                in_community_header = True
                in_community_sent = False
                header_position = 0
                header_global_indices = []
                sent_node_count = 0
                current_part_id = -1
            elif tok == SEP:
                in_community_header = False
                in_community_sent = True
                sent_node_count = 0
                if current_part_id >= 0:
                    state["communities_info"][current_part_id] = {
                        "total_nodes": len(header_global_indices),
                        "nodes_visited": 0,
                        "global_indices": list(header_global_indices),
                    }
            elif tok == RCOM:
                if (
                    current_part_id >= 0
                    and current_part_id in state["communities_info"]
                ):
                    state["communities_info"][current_part_id]["nodes_visited"] = (
                        sent_node_count
                    )
                in_community_header = False
                in_community_sent = False
            elif tok == LBIP:
                in_community_header = False
                in_community_sent = False
            elif tok >= idx_offset:
                val = tok - idx_offset
                if in_community_header:
                    if header_position == 0:
                        current_part_id = val
                    elif header_position >= 2:
                        header_global_indices.append(val)
                    header_position += 1
                elif in_community_sent:
                    sent_node_count += 1
                    if (
                        current_part_id >= 0
                        and current_part_id in state["communities_info"]
                    ):
                        state["communities_info"][current_part_id]["nodes_visited"] = (
                            sent_node_count
                        )

        # Determine current phase by scanning backward
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == LBIP:
                # Find the pair IDs from subsequent tokens
                bip_left = -1
                bip_right = -1
                bip_pos = 0
                for j in range(i + 1, min(i + 4, len(tokens))):
                    if tokens[j] >= idx_offset:
                        if bip_pos == 0:
                            bip_left = tokens[j] - idx_offset
                        elif bip_pos == 1:
                            bip_right = tokens[j] - idx_offset
                            break
                        bip_pos += 1
                state["phase"] = f"Bipartite: P{bip_left} <-> P{bip_right}"
                state["current_community"] = -1
                break
            if tokens[i] == LCOM:
                for j in range(i + 1, min(i + 3, len(tokens))):
                    if tokens[j] >= idx_offset:
                        cid = tokens[j] - idx_offset
                        state["current_community"] = cid
                        state["phase"] = f"Community {cid}: Local SENT"
                        break
                break

        if not state["visible_nodes"]:
            state["phase"] = "Building Structure"

        return state

    def _draw_block_diagram(self, ax: plt.Axes, token_state: dict) -> None:
        """Draw block diagram with fill bars showing generation progress."""
        ax.set_aspect("auto")
        ax.axis("off")
        ax.set_title("Partition Blocks", fontsize=12, fontweight="bold")

        communities_info = token_state.get("communities_info", {})
        current_community = token_state.get("current_community", -1)
        bipartite_connections = token_state.get("bipartite_connections", [])

        if not communities_info:
            ax.text(
                0.5,
                0.5,
                "Defining partitions...",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            return

        comm_ids = sorted(communities_info.keys())
        n_comms = len(comm_ids)

        # Layout: vertical stack of blocks
        block_height = min(0.12, 0.8 / max(n_comms, 1))
        block_width = 0.7
        start_y = 0.9

        block_positions = {}  # comm_id -> (center_x, center_y)

        for idx, cid in enumerate(comm_ids):
            info = communities_info[cid]
            total = max(info["total_nodes"], 1)
            visited = info["nodes_visited"]
            progress = min(visited / total, 1.0)

            y = start_y - idx * (block_height + 0.04)
            x = 0.15

            color = COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)]
            block_positions[cid] = (x + block_width / 2, y - block_height / 2)

            # Background block
            bg = mpatches.FancyBboxPatch(
                (x, y - block_height),
                block_width,
                block_height,
                boxstyle="round,pad=0.02",
                facecolor="white",
                edgecolor=color,
                linewidth=3 if cid == current_community else 1.5,
            )
            ax.add_patch(bg)

            # Gold glow for current community
            if cid == current_community:
                glow = mpatches.FancyBboxPatch(
                    (x - 0.02, y - block_height - 0.02),
                    block_width + 0.04,
                    block_height + 0.04,
                    boxstyle="round,pad=0.02",
                    facecolor="gold",
                    edgecolor="gold",
                    alpha=0.3,
                    linewidth=0,
                    zorder=0,
                )
                ax.add_patch(glow)

            # Fill bar
            fill_width = block_width * progress
            fill = mpatches.Rectangle(
                (x, y - block_height),
                fill_width,
                block_height,
                facecolor=color,
                alpha=0.4,
                zorder=1,
            )
            ax.add_patch(fill)

            # Label
            ax.text(
                x + block_width / 2,
                y - block_height / 2,
                f"P{cid} ({visited}/{total})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=2,
            )

        # Draw bipartite connection arrows between blocks
        seen_pairs = set()
        for left_part, right_part, _lg, _rg in bipartite_connections:
            pair = (min(left_part, right_part), max(left_part, right_part))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if left_part in block_positions and right_part in block_positions:
                lpos = block_positions[left_part]
                rpos = block_positions[right_part]
                ax.annotate(
                    "",
                    xy=rpos,
                    xytext=lpos,
                    arrowprops=dict(
                        arrowstyle="<->",
                        color="#9932CC",
                        lw=1.5,
                        connectionstyle="arc3,rad=0.4",
                    ),
                    zorder=3,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.0)


# =============================================================================
# Factory
# =============================================================================


def create_visualizer(
    tokenizer: object, tokenizer_type: str
) -> BaseGenerationVisualizer:
    """Create the appropriate visualizer for the given tokenizer type."""
    if tokenizer_type == "hdt":
        return HDTVisualizer(tokenizer, tokenizer_type)
    elif tokenizer_type == "hdtc":
        return HDTCVisualizer(tokenizer, tokenizer_type)
    elif tokenizer_type == "hsent":
        return HSENTVisualizer(tokenizer, tokenizer_type)
    else:
        return SENTVisualizer(tokenizer, tokenizer_type)


# =============================================================================
# Model Loading (with all bug fixes from test.py)
# =============================================================================


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_type: str,
    labeled_graph: bool = True,
    coarsening_strategy: str = "spectral",
) -> tuple[GraphGeneratorModule, object]:
    """Load model and create appropriate tokenizer.

    Args:
        checkpoint_path: Path to the model checkpoint.
        tokenizer_type: One of "hdt", "hsent", "sent", "hdtc".
        labeled_graph: Whether the model uses labeled graphs.
        coarsening_strategy: Coarsening strategy for hierarchical tokenizers.

    Returns:
        Tuple of (model, tokenizer).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract vocab size from checkpoint
    wte_key = "model.model.transformer.wte.weight"
    if "state_dict" in checkpoint and wte_key in checkpoint["state_dict"]:
        checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
    else:
        raise ValueError(
            f"Cannot determine vocab size from checkpoint: {checkpoint_path}"
        )

    # Create tokenizer with correct params
    if tokenizer_type == "hdt":
        tokenizer = HDTTokenizer(
            max_length=2048,
            labeled_graph=labeled_graph,
            coarsening_strategy=coarsening_strategy,
        )
        idx_offset = tokenizer.IDX_OFFSET
    elif tokenizer_type == "hdtc":
        tokenizer = HDTCTokenizer(
            max_length=2048,
            labeled_graph=labeled_graph,
        )
        idx_offset = tokenizer.IDX_OFFSET
    elif tokenizer_type == "hsent":
        tokenizer = HSENTTokenizer(
            max_length=2048,
            labeled_graph=labeled_graph,
            coarsening_strategy=coarsening_strategy,
        )
        idx_offset = tokenizer.IDX_OFFSET
    else:
        tokenizer = SENTTokenizer(
            max_length=2048,
            labeled_graph=labeled_graph,
        )
        idx_offset = tokenizer.idx_offset

    # Calculate max_num_nodes from checkpoint vocab (matches test.py pattern)
    if labeled_graph:
        checkpoint_max_num_nodes = (
            checkpoint_vocab_size - idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES
        )
        if checkpoint_max_num_nodes <= 0:
            print(
                f"  Warning: labeled formula gives non-positive max_num_nodes "
                f"({checkpoint_max_num_nodes}), falling back to unlabeled"
            )
            labeled_graph = False
            tokenizer.labeled_graph = False
            checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset
    else:
        checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset

    # Force-set max_num_nodes directly (set_num_nodes() only increases)
    tokenizer.max_num_nodes = checkpoint_max_num_nodes

    if labeled_graph:
        tokenizer.set_num_node_and_edge_types(
            num_node_types=NUM_ATOM_TYPES,
            num_edge_types=NUM_BOND_TYPES,
        )

    # Validate vocab size matches checkpoint
    assert tokenizer.vocab_size == checkpoint_vocab_size, (
        f"Vocab mismatch: tokenizer={tokenizer.vocab_size}, "
        f"checkpoint={checkpoint_vocab_size} "
        f"(type={tokenizer_type}, max_num_nodes={checkpoint_max_num_nodes}, "
        f"labeled={labeled_graph})"
    )

    # Extract max position embeddings from checkpoint (GPT-2 wpe)
    wpe_key = "model.model.transformer.wpe.weight"
    load_kwargs: dict = {"tokenizer": tokenizer, "weights_only": False}
    if "state_dict" in checkpoint and wpe_key in checkpoint["state_dict"]:
        checkpoint_max_length = checkpoint["state_dict"][wpe_key].shape[0]
        load_kwargs["sampling_max_length"] = checkpoint_max_length

    model = GraphGeneratorModule.load_from_checkpoint(checkpoint_path, **load_kwargs)
    model.eval()

    return model, tokenizer


# =============================================================================
# Side-by-side comparison GIF
# =============================================================================


def create_side_by_side_gif(
    gif_paths: dict[str, Path],
    output_path: Path,
    fps: int = 2,
    max_width_per_gif: int = 600,
) -> None:
    """Create a side-by-side comparison GIF with memory optimization."""
    from PIL import Image

    valid_paths = {name: path for name, path in gif_paths.items() if path.exists()}

    if not valid_paths:
        return

    if len(valid_paths) == 1:
        import shutil

        single_path = list(valid_paths.values())[0]
        shutil.copy(single_path, output_path)
        return

    gif_info = {}
    for name, path in valid_paths.items():
        img = Image.open(path)
        width, height = img.size
        n_frames = 0
        try:
            while True:
                n_frames += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        img.close()

        scale = min(1.0, max_width_per_gif / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gif_info[name] = {
            "path": path,
            "n_frames": n_frames,
            "width": new_width,
            "height": new_height,
            "scale": scale,
        }

    max_frames = max(info["n_frames"] for info in gif_info.values())
    names = list(gif_info.keys())
    combined_width = sum(info["width"] for info in gif_info.values())
    combined_height = max(info["height"] for info in gif_info.values())

    batch_size = 20
    all_combined_frames = []

    for batch_start in range(0, max_frames, batch_size):
        batch_end = min(batch_start + batch_size, max_frames)

        batch_frames = {name: [] for name in names}
        for name in names:
            info = gif_info[name]
            img = Image.open(info["path"])
            for frame_idx in range(batch_start, batch_end):
                actual_idx = min(frame_idx, info["n_frames"] - 1)
                try:
                    img.seek(actual_idx)
                    frame = img.copy()
                    if info["scale"] < 1.0:
                        frame = frame.resize(
                            (info["width"], info["height"]), Image.Resampling.LANCZOS
                        )
                    if frame.mode != "RGB":
                        frame = frame.convert("RGB")
                    batch_frames[name].append(frame)
                except EOFError:
                    if batch_frames[name]:
                        batch_frames[name].append(batch_frames[name][-1].copy())
            img.close()

        for local_idx in range(batch_end - batch_start):
            combined = Image.new("RGB", (combined_width, combined_height), "white")
            x_offset = 0
            for name in names:
                info = gif_info[name]
                if local_idx < len(batch_frames[name]):
                    frame = batch_frames[name][local_idx]
                    y_offset = (combined_height - info["height"]) // 2
                    combined.paste(frame, (x_offset, y_offset))
                x_offset += info["width"]
            all_combined_frames.append(combined)

        del batch_frames

    if all_combined_frames:
        all_combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=all_combined_frames[1:],
            duration=1000 // fps,
            loop=0,
            optimize=True,
        )


# =============================================================================
# Main (Hydra)
# =============================================================================


def generate_demo(
    cfg: DictConfig,
    output_dir: Path,
) -> None:
    """Generate demo GIFs for each model in the config."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = cfg.generation
    anim_cfg = cfg.animation
    motif_cfg = cfg.get("motif", None)
    num_samples = gen_cfg.num_samples

    print(f"Output directory: {output_dir}")
    print(f"Generating {num_samples} samples per model\n")

    models_list = OmegaConf.to_container(cfg.models, resolve=True)
    if not models_list:
        print("Error: No models configured")
        return

    for sample_idx in range(num_samples):
        print(f"\n{'=' * 60}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print("=" * 60)

        sample_gif_paths = {}

        for model_cfg in models_list:
            name = model_cfg["name"]
            ckpt_path = model_cfg["checkpoint_path"]
            tokenizer_type = model_cfg["tokenizer_type"]
            labeled = model_cfg.get("labeled_graph", True)
            coarsening = model_cfg.get("coarsening_strategy", "spectral")

            if not Path(ckpt_path).exists():
                print(f"\n{name}: Checkpoint not found: {ckpt_path}")
                continue

            print(f"\n{name} ({tokenizer_type.upper()}): Loading from {ckpt_path}")

            try:
                model, tokenizer = load_model_and_tokenizer(
                    ckpt_path, tokenizer_type, labeled, coarsening
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                visualizer = create_visualizer(tokenizer, tokenizer_type)

                print("  Generating tokens...")
                tokens, graphs_history = visualizer.generate_with_history(
                    model,
                    max_length=gen_cfg.get("max_length", 512),
                    top_k=gen_cfg.get("top_k", 10),
                    temperature=gen_cfg.get("temperature", 1.0),
                )

                print(f"  Generated {len(tokens)} tokens, {len(graphs_history)} frames")

                if graphs_history:
                    final_graph = graphs_history[-1]
                    smiles = graph_to_smiles(final_graph)
                    print(f"  SMILES: {smiles or '(invalid)'}")

                gif_path = output_dir / f"{name}_sample_{sample_idx + 1}.gif"
                print(f"  Creating animation: {gif_path}")
                visualizer.create_animation(
                    tokens,
                    graphs_history,
                    gif_path,
                    fps=anim_cfg.get("fps", 2),
                    max_frames=anim_cfg.get("max_frames", 150),
                    figsize=tuple(anim_cfg.get("figsize", [10, 12])),
                    show_tokens=anim_cfg.get("show_tokens", True),
                    side_panel_width=anim_cfg.get("side_panel_width", 4),
                    motif_cfg=motif_cfg,
                )
                sample_gif_paths[name] = gif_path

            except Exception as e:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()

        out_cfg = cfg.get("output", {})
        if out_cfg.get("create_comparison", True) and len(sample_gif_paths) > 1:
            # Filter to configured comparison models (if specified)
            comparison_models = out_cfg.get("comparison_models", None)
            if comparison_models:
                comparison_models = list(comparison_models)
                comparison_paths = {
                    name: path
                    for name, path in sample_gif_paths.items()
                    if name in comparison_models
                }
                # Preserve config order
                comparison_paths = {
                    name: comparison_paths[name]
                    for name in comparison_models
                    if name in comparison_paths
                }
            else:
                comparison_paths = sample_gif_paths

            if len(comparison_paths) > 1:
                combined_path = output_dir / f"comparison_sample_{sample_idx + 1}.gif"
                print(f"\nCreating comparison: {combined_path}")
                create_side_by_side_gif(
                    comparison_paths,
                    combined_path,
                    fps=anim_cfg.get("fps", 2),
                    max_width_per_gif=out_cfg.get("comparison_max_width", 600),
                )

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"GIFs saved to: {output_dir}")


@hydra.main(
    config_path="../../configs",
    config_name="generation_demo",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra config."""
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(cfg.output.get("dir", "outputs/generation_demo"))

    generate_demo(cfg, output_dir)


if __name__ == "__main__":
    main()
