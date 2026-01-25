#!/usr/bin/env python
"""Generation demo script for visualizing autoregressive molecule generation.

This script generates molecules using trained models with different tokenization
schemes (HDT, HSENT, SENT) and creates animated GIFs showing the step-by-step
generation process.

Usage:
    python scripts/generation_demo.py
    python scripts/generation_demo.py --num-samples 5 --fps 3
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

# Default checkpoint paths
DEFAULT_HDT_CKPT = "outputs/train/moses_hdt_n50000_20260122-185129/best.ckpt"
DEFAULT_HSENT_CKPT = "outputs/train/moses_hsent_n50000_20260122-093526/best.ckpt"
DEFAULT_SENT_CKPT = "outputs/train/moses_sent_n50000_20260123-140906/best.ckpt"

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
from src.tokenizers import HDTTokenizer, HSENTTokenizer, SENTTokenizer  # noqa: E402
from src.tokenizers.motif.detection import detect_motifs_from_smiles  # noqa: E402

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


class GenerationVisualizer:
    """Visualizer for step-by-step molecule generation."""

    def __init__(self, tokenizer, tokenizer_type: str) -> None:
        """Initialize the visualizer."""
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type

    def _parse_token_state(self, tokens: list[int]) -> dict:
        """Parse tokens to get current phase, communities, and visible nodes.

        Returns dict with:
            - phase: str describing current generation phase
            - communities: dict mapping node_id -> community_id
            - visible_nodes: set of node ids that should be visible
            - bipartite_edges: set of (src, dst) tuples for cross-community edges
        """
        state = {
            "phase": "Initializing",
            "communities": {},
            "visible_nodes": set(),
            "bipartite_edges": set(),
            "current_community": -1,
        }

        if self.tokenizer_type == "hdt":
            return self._parse_hdt_state(tokens, state)
        elif self.tokenizer_type == "hsent":
            return self._parse_hsent_state(tokens, state)
        else:  # sent
            return self._parse_sent_state(tokens, state)

    def _parse_hdt_state(self, tokens: list[int], state: dict) -> dict:
        """Parse HDT tokens for visualization state using the actual tokenizer.

        Uses tokenizer.parse_tokens() to get accurate community assignments
        and cross-community edges from the HierarchicalGraph structure.
        """
        # Use the tokenizer's actual parsing to get HierarchicalGraph
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

        # Build community info from HierarchicalGraph
        state["abstract_tree"] = {}
        state["community_order"] = []
        state["cross_community_edges"] = set()

        # Get nodes and their communities from partitions
        for part in hg.partitions:
            part_id = part.part_id
            nodes = part.global_node_indices
            if nodes:
                state["abstract_tree"][part_id] = list(nodes)
                state["community_order"].append(part_id)
                for node_idx in nodes:
                    state["visible_nodes"].add(node_idx)
                    state["communities"][node_idx] = part_id

        # Get cross-community edges from bipartites
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

        # Determine current phase from token parsing
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

    def _parse_hsent_state(self, tokens: list[int], state: dict) -> dict:
        """Parse HSENT tokens for visualization state."""
        idx_offset = self.tokenizer.IDX_OFFSET
        LCOM = self.tokenizer.LCOM
        RCOM = self.tokenizer.RCOM
        LBIP = self.tokenizer.LBIP
        RBIP = self.tokenizer.RBIP
        SEP = self.tokenizer.SEP
        LADJ = self.tokenizer.LADJ
        RADJ = self.tokenizer.RADJ

        # For labeled graphs, distinguish node indices from atom/bond types
        is_labeled = getattr(self.tokenizer, "labeled_graph", False)
        node_idx_offset = getattr(self.tokenizer, "node_idx_offset", None)
        max_num_nodes = getattr(self.tokenizer, "max_num_nodes", 100)

        in_community_header = False
        in_community_sent = False
        in_bipartite = False
        in_bracket = False  # Track back-edge brackets
        current_part_id = -1
        header_position = 0
        current_global_indices = []
        communities_map = {}  # part_id -> list of global indices
        sent_node_count = 0  # Track nodes in current community's SENT

        bip_left_part = -1
        bip_right_part = -1
        bip_position = 0

        def is_node_index_token(tok: int) -> bool:
            """Check if token is a node index (not atom/bond type)."""
            if is_labeled and node_idx_offset is not None:
                return idx_offset <= tok < node_idx_offset
            return idx_offset <= tok < idx_offset + max_num_nodes

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            if tok == LCOM:
                in_community_header = True
                in_community_sent = False
                in_bipartite = False
                in_bracket = False
                header_position = 0
                current_part_id = -1
                current_global_indices = []
                sent_node_count = 0
                state["phase"] = "Local Community SENT"
            elif tok == SEP:
                in_community_header = False
                in_community_sent = True
                in_bracket = False
                sent_node_count = 0
                state["current_community"] = current_part_id
                state["phase"] = f"Community {current_part_id}: Local SENT"
            elif tok == RCOM:
                if current_part_id >= 0 and current_global_indices:
                    communities_map[current_part_id] = list(current_global_indices)
                in_community_header = False
                in_community_sent = False
                in_bracket = False
            elif tok == LBIP:
                in_bipartite = True
                in_community_header = False
                in_community_sent = False
                in_bracket = False
                bip_position = 0
                state["phase"] = "Bipartite Connection"
            elif tok == RBIP:
                in_bipartite = False
            elif tok == LADJ:
                in_bracket = True
            elif tok == RADJ:
                in_bracket = False
            elif is_node_index_token(tok):
                val = tok - idx_offset

                if in_community_header:
                    if header_position == 0:
                        current_part_id = val
                    elif header_position >= 2:
                        current_global_indices.append(val)
                    header_position += 1
                elif in_community_sent and not in_bracket:
                    # This is a node in the SENT sequence (not a back-edge target)
                    # val is the local index within the community
                    if val < len(current_global_indices):
                        global_idx = current_global_indices[val]
                        state["visible_nodes"].add(global_idx)
                        state["communities"][global_idx] = current_part_id
                    sent_node_count += 1
                elif in_bipartite and not in_bracket:
                    if bip_position == 0:
                        bip_left_part = val
                    elif bip_position == 1:
                        bip_right_part = val
                        state["phase"] = (
                            f"Bipartite: Comm {bip_left_part} ↔ Comm {bip_right_part}"
                        )
                    elif bip_position >= 3:
                        pair_pos = bip_position - 3
                        if pair_pos % 2 == 0:
                            state["_bip_left_local"] = val
                        else:
                            left_local = state.get("_bip_left_local", 0)
                            right_local = val
                            if (
                                bip_left_part in communities_map
                                and bip_right_part in communities_map
                            ):
                                left_nodes = communities_map[bip_left_part]
                                right_nodes = communities_map[bip_right_part]
                                if left_local < len(left_nodes) and right_local < len(
                                    right_nodes
                                ):
                                    lg = left_nodes[left_local]
                                    rg = right_nodes[right_local]
                                    state["bipartite_edges"].add((lg, rg))
                                    state["bipartite_edges"].add((rg, lg))
                    bip_position += 1

            i += 1

        return state

    def _parse_sent_state(self, tokens: list[int], state: dict) -> dict:
        """Parse SENT tokens for visualization state."""
        idx_offset = self.tokenizer.idx_offset
        RESET = self.tokenizer.reset
        LADJ = self.tokenizer.ladj
        RADJ = self.tokenizer.radj

        # For labeled graphs, we need to distinguish node indices from atom/bond types
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
                # For labeled graphs, only count tokens in the node index range
                # Node indices: [idx_offset, node_idx_offset)
                # Atom types: [node_idx_offset, edge_idx_offset)
                # Bond types: [edge_idx_offset, vocab_size)
                if is_labeled and node_idx_offset is not None:
                    if tok < node_idx_offset:
                        # This is a node index token
                        state["visible_nodes"].add(node_count)
                        node_count += 1
                    # Skip atom/bond type tokens
                else:
                    # Unlabeled graph: simple check
                    node_id = tok - idx_offset
                    if node_id < max_num_nodes:
                        state["visible_nodes"].add(node_count)
                        node_count += 1

        state["phase"] = f"Random Walk (Trail {trail_count})"
        return state

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

                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                top_k_probs = torch.softmax(top_k_logits / temperature, dim=-1)
                next_token_idx = torch.multinomial(top_k_probs, 1).item()
                next_token = top_k_indices[next_token_idx].item()

                tokens.append(next_token)

                # Try to decode partial sequence
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

    def _detect_motifs_with_edges(self, smiles: str) -> list[dict]:
        """Detect motifs from SMILES and get their required edges.

        Returns list of dicts with:
            - name: motif name
            - atoms: frozenset of atom indices
            - edges: set of (i, j) tuples (sorted, i < j) for required bonds
        """
        try:
            from rdkit import Chem
        except ImportError:
            return []

        motifs = detect_motifs_from_smiles(smiles)
        if not motifs:
            return []

        # Get the molecule to find bonds within each motif
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        result = []
        for motif in motifs:
            atoms = motif.atom_indices
            # Find all bonds where BOTH atoms are in the motif
            edges = set()
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in atoms and a2 in atoms:
                    edges.add((min(a1, a2), max(a1, a2)))

            result.append(
                {
                    "name": motif.name,
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

        # Pre-compute layout from final graph
        final_graph = graphs_sampled[-1]
        fixed_positions = self._compute_layout(final_graph)

        # Detect motifs from final SMILES (for shading when complete)
        final_smiles = graph_to_smiles(final_graph)
        detected_motifs = []
        if final_smiles:
            detected_motifs = self._detect_motifs_with_edges(final_smiles)

        # Pre-compute token states for each frame (for gradual reveal and phase info)
        token_states = []
        for token_idx in token_indices:
            tokens_so_far = tokens[: token_idx + 2]  # +2 to include current token
            state = self._parse_token_state(tokens_so_far)
            token_states.append(state)

        # For HDT, add an abstract tree panel on the left
        show_abstract_tree = self.tokenizer_type == "hdt"

        # Use consistent graph panel size across all tokenizers
        graph_width = figsize[0]  # Same graph width for all
        tree_width = 4  # Additional width for tree panel

        if show_abstract_tree:
            if show_tokens:
                # Total width = tree_width + graph_width
                # Graph panel should be exactly graph_width
                fig = plt.figure(figsize=(tree_width + graph_width, figsize[1]))
                gs = fig.add_gridspec(
                    2,
                    2,
                    width_ratios=[tree_width, graph_width],
                    height_ratios=[4, 1],
                )
                ax_tree = fig.add_subplot(gs[0, 0])
                ax_graph = fig.add_subplot(gs[0, 1])
                ax_tokens = fig.add_subplot(gs[1, :])
            else:
                fig = plt.figure(figsize=(tree_width + graph_width, figsize[1] - 2))
                gs = fig.add_gridspec(1, 2, width_ratios=[tree_width, graph_width])
                ax_tree = fig.add_subplot(gs[0, 0])
                ax_graph = fig.add_subplot(gs[0, 1])
                ax_tokens = None
        else:
            ax_tree = None
            if show_tokens:
                fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[4, 1])
                ax_graph, ax_tokens = axes
            else:
                fig, ax_graph = plt.subplots(figsize=figsize)
                ax_tokens = None

        def update(frame_idx: int) -> list:
            ax_graph.clear()
            graph = graphs_sampled[frame_idx]
            token_up_to = token_indices[frame_idx] + 1
            state = token_states[frame_idx]

            # Draw abstract tree for HDT
            if ax_tree is not None:
                ax_tree.clear()
                self._draw_abstract_tree(ax_tree, state)

            # Draw graph with community colors, edge styles, and gradual reveal
            self._draw_graph(
                ax_graph,
                graph,
                frame_idx,
                len(graphs_sampled),
                fixed_positions,
                token_state=state,
                motifs=detected_motifs,
            )

            # Title with phase annotation
            phase = state.get("phase", "Generating")
            title = f"{self.tokenizer_type.upper()}: {phase}\nStep {frame_idx + 1}/{len(graphs_sampled)}"
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
        """Compute node positions for a graph using RDKit 2D coordinates.

        Uses RDKit's molecular coordinate generation for chemical-looking layouts.
        Falls back to NetworkX spring layout if RDKit fails.
        """
        if graph.num_nodes == 0:
            return {}

        # Try RDKit-based layout first for molecular structures
        pos = self._compute_rdkit_layout(graph)
        if pos is not None:
            return pos

        # Fallback to NetworkX layout
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))

        if graph.edge_index.numel() > 0:
            edge_index = graph.edge_index.numpy()
            for k in range(edge_index.shape[1]):
                i, j = int(edge_index[0, k]), int(edge_index[1, k])
                if i < j and i < graph.num_nodes and j < graph.num_nodes:
                    G.add_edge(i, j)

        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100, scale=1.5)
        else:
            pos = nx.circular_layout(G, scale=1.5)

        return pos

    def _compute_rdkit_layout(
        self, graph: Data
    ) -> Optional[dict[int, tuple[float, float]]]:
        """Compute 2D molecular layout using RDKit.

        Args:
            graph: PyG Data object representing a molecule.

        Returns:
            Dictionary mapping node indices to (x, y) positions, or None if failed.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            return None

        # Try to convert graph to RDKit molecule
        try:
            num_nodes = graph.num_nodes
            edge_index = graph.edge_index

            mol = Chem.RWMol()

            # Detect if using integer labels or one-hot features
            labeled = graph.x.dtype == torch.long or graph.x.dtype == torch.int64

            # Add atoms
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

            # Add bonds
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

            mol = mol.GetMol()

            # Compute 2D coordinates
            AllChem.Compute2DCoords(mol)
            conformer = mol.GetConformer()

            # Extract positions
            pos = {}
            for i in range(num_nodes):
                atom_pos = conformer.GetAtomPosition(i)
                pos[i] = (atom_pos.x, atom_pos.y)

            # Normalize positions to similar scale as spring_layout
            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
                y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1
                scale = max(x_range, y_range) if max(x_range, y_range) > 0 else 1
                x_center = (max(x_coords) + min(x_coords)) / 2
                y_center = (max(y_coords) + min(y_coords)) / 2

                # Scale to [-1.5, 1.5] range (similar to spring_layout scale=1.5)
                for node_id in pos:
                    x, y = pos[node_id]
                    pos[node_id] = (
                        (x - x_center) / scale * 3.0,
                        (y - y_center) / scale * 3.0,
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
    ) -> None:
        """Draw a molecular graph with community colors, edge styles, and motif shading.

        Args:
            ax: Matplotlib axis.
            graph: PyG Data object.
            step: Current step.
            total_steps: Total steps.
            fixed_positions: Pre-computed node positions.
            token_state: Token parsing state with communities, visible_nodes, bipartite_edges.
            motifs: List of detected motifs with atoms and required edges.
        """
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

        # Get visible nodes from token state (for gradual reveal)
        visible_nodes = None
        communities = {}
        cross_community_edges = set()
        if token_state:
            visible_nodes = token_state.get("visible_nodes")
            communities = token_state.get("communities", {})
            # For HSENT: bipartite_edges, for HDT: cross_community_edges
            cross_community_edges = token_state.get("bipartite_edges", set())
            cross_community_edges = cross_community_edges.union(
                token_state.get("cross_community_edges", set())
            )

        # Determine which nodes to show
        if visible_nodes is not None and len(visible_nodes) > 0:
            # Show only nodes that have actually been visited/revealed
            # Intersect with valid node range for safety
            nodes_to_show = visible_nodes & set(range(graph.num_nodes))
        else:
            # Fallback: show all nodes in the decoded graph
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

                    # Check if this is a cross-community edge (purple dotted)
                    is_cross_community = (i, j) in cross_community_edges or (
                        j,
                        i,
                    ) in cross_community_edges
                    G.add_edge(
                        i, j, bond_type=bond_type, is_cross_community=is_cross_community
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

        # Draw motif shading (blue convex hull) for complete motifs
        # A motif is complete when ALL its atoms AND ALL its edges are present
        if motifs and pos:
            current_edges = set()
            for u, v in G.edges():
                current_edges.add((min(u, v), max(u, v)))

            for motif in motifs:
                motif_atoms = motif["atoms"]
                motif_edges = motif["edges"]

                # Check if all atoms are present
                all_atoms_present = all(atom in G.nodes() for atom in motif_atoms)
                if not all_atoms_present:
                    continue

                # Check if all required edges are present
                all_edges_present = all(edge in current_edges for edge in motif_edges)
                if not all_edges_present:
                    continue

                # Draw blue convex hull around the motif atoms
                motif_positions = [pos[atom] for atom in motif_atoms if atom in pos]
                if len(motif_positions) >= 3:
                    try:
                        from scipy.spatial import ConvexHull

                        points = np.array(motif_positions)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        # Close the polygon
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.fill(
                            hull_points[:, 0],
                            hull_points[:, 1],
                            color="#4169E1",
                            alpha=0.25,
                            zorder=0,
                        )
                    except Exception:
                        pass
                elif len(motif_positions) == 2:
                    # For 2 atoms, draw an ellipse between them
                    p1, p2 = motif_positions
                    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                    width = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) + 0.3
                    from matplotlib.patches import Ellipse

                    ellipse = Ellipse(
                        center,
                        width=width,
                        height=0.4,
                        angle=np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])),
                        color="#4169E1",
                        alpha=0.25,
                        zorder=0,
                    )
                    ax.add_patch(ellipse)

        # Draw edges (solid black for intra-community, curved purple for cross-community)
        from matplotlib.patches import FancyArrowPatch

        for u, v, data in G.edges(data=True):
            bond_type = data.get("bond_type", 0)
            is_cross_community = data.get("is_cross_community", False)
            style = BOND_STYLES.get(bond_type, BOND_STYLES[0])

            if is_cross_community:
                # Draw curved arc for cross-community edges
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
                # Draw straight line for intra-community edges
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

        # Draw nodes with community colors
        node_colors = []
        node_edge_colors = []
        node_labels = {}
        for node in G.nodes():
            atom = G.nodes[node].get("atom", "C")
            community = G.nodes[node].get("community", -1)

            # Use community color as edge color, atom color as fill
            node_colors.append(ATOM_COLORS.get(atom, DEFAULT_ATOM_COLOR))

            if community >= 0:
                edge_color = COMMUNITY_COLORS[community % len(COMMUNITY_COLORS)]
            else:
                edge_color = "black"
            node_edge_colors.append(edge_color)

            # Label with atom symbol and community number
            if community >= 0:
                node_labels[node] = f"{atom}\n(C{community})"
            else:
                node_labels[node] = atom

        node_list = list(G.nodes())
        node_positions = np.array([pos[n] for n in node_list])

        # Draw nodes with thick community-colored borders
        for idx, node in enumerate(node_list):
            ax.scatter(
                node_positions[idx, 0],
                node_positions[idx, 1],
                s=1200,
                c=[node_colors[idx]],
                edgecolors=[node_edge_colors[idx]],
                linewidths=4,
                zorder=2,
            )

        # Draw node labels
        for node, (x, y) in pos.items():
            label = node_labels.get(node, "")
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        # Add padding
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            margin = 0.5
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

        # Info box with visible/total nodes
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

        # Add community legend entries
        unique_communities = set(
            G.nodes[n].get("community", -1)
            for n in G.nodes()
            if G.nodes[n].get("community", -1) >= 0
        )
        for comm_id in sorted(unique_communities):
            color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
            legend_elements.append(
                mpatches.Patch(
                    facecolor=color, edgecolor=color, label=f"Community {comm_id}"
                )
            )

        # Add edge type legend if we have cross-community edges
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

    def _draw_abstract_tree(
        self,
        ax: plt.Axes,
        token_state: dict,
    ) -> None:
        """Draw the abstract tree for HDT showing communities and leaf nodes."""
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

        # Build tree layout
        # Root at top, communities below, leaf nodes at bottom
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

                # Add leaf nodes under this community
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

        # Draw edges
        for u, v in G.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color="gray", linewidth=1.5, zorder=1)

        # Draw nodes
        for node in G.nodes():
            node_type = G.nodes[node].get("node_type", "")
            label = G.nodes[node].get("label", "")
            x, y = pos[node]

            if node_type == "root":
                color = "#808080"
                size = 400
            elif node_type == "community":
                comm_id = int(label[1:])  # Extract community ID
                color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
                size = 500
                # Highlight current community
                if comm_id == current_community:
                    ax.scatter(x, y, s=800, c="yellow", alpha=0.5, zorder=0)
            else:  # leaf
                color = "#404040"  # Default carbon color
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


def create_side_by_side_gif(
    gif_paths: dict[str, Path],
    output_path: Path,
    fps: int = 2,
    max_width_per_gif: int = 600,
) -> None:
    """Create a side-by-side comparison GIF with memory optimization.

    Args:
        gif_paths: Dictionary mapping names to GIF file paths.
        output_path: Output path for the combined GIF.
        fps: Frames per second.
        max_width_per_gif: Maximum width for each individual GIF (for memory savings).
    """
    from PIL import Image

    valid_paths = {name: path for name, path in gif_paths.items() if path.exists()}

    if not valid_paths:
        return

    if len(valid_paths) == 1:
        import shutil

        single_path = list(valid_paths.values())[0]
        shutil.copy(single_path, output_path)
        return

    # Get frame counts and sizes without loading all frames
    gif_info = {}
    for name, path in valid_paths.items():
        img = Image.open(path)
        width, height = img.size
        # Count frames
        n_frames = 0
        try:
            while True:
                n_frames += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        img.close()

        # Calculate scale factor if needed
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

    # Process frames in batches to save memory
    batch_size = 20
    all_combined_frames = []

    for batch_start in range(0, max_frames, batch_size):
        batch_end = min(batch_start + batch_size, max_frames)

        # Load only the frames we need for this batch
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
                    # Repeat last frame
                    if batch_frames[name]:
                        batch_frames[name].append(batch_frames[name][-1].copy())
            img.close()

        # Combine frames for this batch
        for local_idx in range(batch_end - batch_start):
            combined = Image.new("RGB", (combined_width, combined_height), "white")
            x_offset = 0
            for name in names:
                info = gif_info[name]
                if local_idx < len(batch_frames[name]):
                    frame = batch_frames[name][local_idx]
                    # Center vertically if needed
                    y_offset = (combined_height - info["height"]) // 2
                    combined.paste(frame, (x_offset, y_offset))
                x_offset += info["width"]
            all_combined_frames.append(combined)

        # Clear batch frames to free memory
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


def load_model_and_tokenizer(
    checkpoint_path: str, tokenizer_type: str
) -> tuple[GraphGeneratorModule, object]:
    """Load model and create appropriate tokenizer."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    wte_key = "model.model.transformer.wte.weight"
    if "state_dict" in checkpoint and wte_key in checkpoint["state_dict"]:
        vocab_size = checkpoint["state_dict"][wte_key].shape[0]
    else:
        vocab_size = 100

    if tokenizer_type == "hdt":
        tokenizer = HDTTokenizer(max_length=2048, labeled_graph=True)
        idx_offset = tokenizer.IDX_OFFSET
    elif tokenizer_type == "hsent":
        tokenizer = HSENTTokenizer(max_length=2048, labeled_graph=True)
        idx_offset = tokenizer.IDX_OFFSET
    else:
        tokenizer = SENTTokenizer(max_length=2048, labeled_graph=True)
        idx_offset = tokenizer.idx_offset

    max_num_nodes_labeled = vocab_size - idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES
    if max_num_nodes_labeled > 0 and max_num_nodes_labeled <= 200:
        tokenizer.set_num_nodes(max_num_nodes_labeled)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)
    else:
        max_num_nodes = vocab_size - idx_offset
        if max_num_nodes > 0:
            tokenizer.labeled_graph = False
            tokenizer.set_num_nodes(max_num_nodes)

    model = GraphGeneratorModule.load_from_checkpoint(
        checkpoint_path, tokenizer=tokenizer
    )
    model.eval()

    return model, tokenizer


def generate_demo(
    checkpoint_paths: dict[str, Optional[str]],
    output_dir: Path,
    num_samples: int = 3,
    fps: int = 2,
    max_frames: int = 150,
    top_k: int = 10,
    temperature: float = 1.0,
    max_length: int = 512,
    show_tokens: bool = True,
) -> None:
    """Generate demo GIFs for each tokenizer type."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Generating {num_samples} samples per tokenizer\n")

    active_checkpoints = {k: v for k, v in checkpoint_paths.items() if v is not None}

    if not active_checkpoints:
        print("Error: No checkpoint paths provided")
        return

    for sample_idx in range(num_samples):
        print(f"\n{'=' * 60}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print("=" * 60)

        sample_gif_paths = {}

        for tokenizer_type, ckpt_path in active_checkpoints.items():
            print(f"\n{tokenizer_type.upper()}: Loading model from {ckpt_path}")

            try:
                model, tokenizer = load_model_and_tokenizer(ckpt_path, tokenizer_type)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                visualizer = GenerationVisualizer(tokenizer, tokenizer_type)

                print("  Generating tokens...")
                tokens, graphs_history = visualizer.generate_with_history(
                    model, max_length=max_length, top_k=top_k, temperature=temperature
                )

                print(f"  Generated {len(tokens)} tokens, {len(graphs_history)} frames")

                if graphs_history:
                    final_graph = graphs_history[-1]
                    smiles = graph_to_smiles(final_graph)
                    print(f"  SMILES: {smiles or '(invalid)'}")

                gif_path = output_dir / f"{tokenizer_type}_sample_{sample_idx + 1}.gif"
                print(f"  Creating animation: {gif_path}")
                visualizer.create_animation(
                    tokens,
                    graphs_history,
                    gif_path,
                    fps=fps,
                    max_frames=max_frames,
                    show_tokens=show_tokens,
                )
                sample_gif_paths[tokenizer_type] = gif_path

            except Exception as e:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()

        if len(sample_gif_paths) > 1:
            combined_path = output_dir / f"comparison_sample_{sample_idx + 1}.gif"
            print(f"\nCreating comparison: {combined_path}")
            create_side_by_side_gif(sample_gif_paths, combined_path, fps=fps)

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"GIFs saved to: {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate molecule generation demo GIFs"
    )

    parser.add_argument("--hdt-ckpt", type=str, default=DEFAULT_HDT_CKPT)
    parser.add_argument("--hsent-ckpt", type=str, default=DEFAULT_HSENT_CKPT)
    parser.add_argument("--sent-ckpt", type=str, default=DEFAULT_SENT_CKPT)
    parser.add_argument("--output-dir", type=str, default="outputs/generation_demo")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tokens", action="store_true")

    args = parser.parse_args()

    hdt_ckpt = args.hdt_ckpt if Path(args.hdt_ckpt).exists() else None
    hsent_ckpt = args.hsent_ckpt if Path(args.hsent_ckpt).exists() else None
    sent_ckpt = args.sent_ckpt if Path(args.sent_ckpt).exists() else None

    if not any([hdt_ckpt, hsent_ckpt, sent_ckpt]):
        parser.error("No valid checkpoint files found.")

    print("Checkpoint paths:")
    print(f"  HDT:   {hdt_ckpt or '(not found)'}")
    print(f"  HSENT: {hsent_ckpt or '(not found)'}")
    print(f"  SENT:  {sent_ckpt or '(not found)'}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    generate_demo(
        checkpoint_paths={"hdt": hdt_ckpt, "hsent": hsent_ckpt, "sent": sent_ckpt},
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        fps=args.fps,
        max_frames=args.max_frames,
        top_k=args.top_k,
        temperature=args.temperature,
        max_length=args.max_length,
        show_tokens=not args.no_tokens,
    )


if __name__ == "__main__":
    main()
