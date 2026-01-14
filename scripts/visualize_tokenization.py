#!/usr/bin/env python
"""Visualization script comparing SENT and H-SENT tokenization.

This script provides side-by-side comparison of flat SENT tokenization
and hierarchical H-SENT tokenization for molecular graphs.

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
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.tokenizers.sent import SENTTokenizer
from src.tokenizers.hierarchical import HSENTTokenizer


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
    return Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())


def plot_graph(ax: plt.Axes, data: Data, title: str = "Graph") -> None:
    """Plot a graph using networkx."""
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color="lightblue",
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_graph_with_walk(ax: plt.Axes, data: Data, tokens: list[int],
                         tokenizer: SENTTokenizer) -> None:
    """Plot graph with random walk path highlighted."""
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    # Extract walk order from tokens
    walk_order = []
    for tok in tokens:
        if tok >= tokenizer.idx_offset:
            walk_order.append(tok - tokenizer.idx_offset)

    # Draw base graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color="lightgray",
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=1)

    # Highlight walk path with gradient colors
    if len(walk_order) > 1:
        cmap = plt.cm.viridis
        colors = [cmap(i / len(walk_order)) for i in range(len(walk_order))]

        # Draw walk edges
        walk_edges = []
        for i in range(len(walk_order) - 1):
            if walk_order[i] < data.num_nodes and walk_order[i+1] < data.num_nodes:
                walk_edges.append((walk_order[i], walk_order[i+1]))

        for idx, (u, v) in enumerate(walk_edges):
            if G.has_edge(u, v):
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                                       edge_color=[colors[idx]], width=3, alpha=0.8)

        # Draw nodes with walk order colors
        for idx, node in enumerate(walk_order):
            if node < data.num_nodes:
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node],
                                       node_size=400, node_color=[colors[idx]],
                                       edgecolors="black", linewidths=2)

    # Add labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    ax.set_title("SENT: Random Walk Traversal", fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_graph_with_communities(ax: plt.Axes, data: Data, hg, title: str = "") -> None:
    """Plot graph with community coloring."""
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    # Color by community
    cmap = plt.cm.Set3
    colors = []
    for node in range(data.num_nodes):
        comm = hg.community_assignment[node] if node < len(hg.community_assignment) else 0
        colors.append(cmap(comm % 12))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color=colors,
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Add legend
    unique_comms = sorted(set(hg.community_assignment))
    patches = [mpatches.Patch(color=cmap(c % 12), label=f"C{c}")
               for c in unique_comms[:6]]
    if patches:
        ax.legend(handles=patches, loc="upper left", fontsize=8)

    ax.set_title(title or "H-SENT: Community Structure", fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_tokens(ax: plt.Axes, tokens: list[int], tokenizer, title: str,
                max_tokens: int = 60) -> None:
    """Plot token sequence as colored blocks."""
    # Token type colors
    if hasattr(tokenizer, 'SOS'):
        # H-SENT tokenizer
        token_colors = {
            tokenizer.SOS: "#2ecc71",      # Green - SOS
            tokenizer.EOS: "#e74c3c",      # Red - EOS
            tokenizer.PAD: "#bdc3c7",      # Gray - PAD
            tokenizer.RESET: "#9b59b6",    # Purple - RESET
            tokenizer.LADJ: "#3498db",     # Blue - LADJ
            tokenizer.RADJ: "#3498db",     # Blue - RADJ
            tokenizer.LCOM: "#f39c12",     # Orange - LCOM
            tokenizer.RCOM: "#f39c12",     # Orange - RCOM
            tokenizer.LBIP: "#1abc9c",     # Teal - LBIP
            tokenizer.RBIP: "#1abc9c",     # Teal - RBIP
            tokenizer.SEP: "#95a5a6",      # Gray - SEP
        }
        idx_offset = tokenizer.IDX_OFFSET
    else:
        # SENT tokenizer
        token_colors = {
            tokenizer.sos: "#2ecc71",      # Green - SOS
            tokenizer.eos: "#e74c3c",      # Red - EOS
            tokenizer.pad: "#bdc3c7",      # Gray - PAD
            tokenizer.reset: "#9b59b6",    # Purple - RESET
            tokenizer.ladj: "#3498db",     # Blue - LADJ
            tokenizer.radj: "#3498db",     # Blue - RADJ
        }
        idx_offset = tokenizer.idx_offset

    # Truncate if needed
    display_tokens = tokens[:max_tokens]
    truncated = len(tokens) > max_tokens

    # Calculate grid dimensions
    cols = min(20, len(display_tokens))
    rows = (len(display_tokens) + cols - 1) // cols

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)

    for idx, tok in enumerate(display_tokens):
        row = idx // cols
        col = idx % cols
        y = rows - 1 - row  # Flip y-axis

        # Get color
        if tok in token_colors:
            color = token_colors[tok]
        elif tok >= idx_offset:
            # Node index - use gradient
            node_idx = tok - idx_offset
            color = plt.cm.Pastel1(node_idx % 9)
        else:
            color = "#ecf0f1"

        # Draw rectangle
        rect = plt.Rectangle((col - 0.45, y - 0.45), 0.9, 0.9,
                             facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)

        # Add token text
        if tok >= idx_offset:
            label = str(tok - idx_offset)
        else:
            # Get token name
            if hasattr(tokenizer, 'SOS'):
                names = {tokenizer.SOS: "S", tokenizer.EOS: "E", tokenizer.PAD: "P",
                        tokenizer.RESET: "R", tokenizer.LADJ: "[", tokenizer.RADJ: "]",
                        tokenizer.LCOM: "{", tokenizer.RCOM: "}",
                        tokenizer.LBIP: "<", tokenizer.RBIP: ">", tokenizer.SEP: "|"}
            else:
                names = {tokenizer.sos: "S", tokenizer.eos: "E", tokenizer.pad: "P",
                        tokenizer.reset: "R", tokenizer.ladj: "[", tokenizer.radj: "]"}
            label = names.get(tok, "?")

        ax.text(col, y, label, ha="center", va="center", fontsize=7, fontweight="bold")

    title_text = title
    if truncated:
        title_text += f" (showing {max_tokens}/{len(tokens)})"
    ax.set_title(title_text, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_block_matrix(ax: plt.Axes, hg) -> None:
    """Plot block adjacency matrix."""
    n = sum(len(p.global_node_indices) for p in hg.partitions)
    if n == 0:
        ax.text(0.5, 0.5, "Empty", ha="center", va="center")
        ax.axis("off")
        return

    # Create reordered adjacency matrix
    node_order = []
    for part in hg.partitions:
        node_order.extend(part.global_node_indices)

    adj = np.zeros((n, n))

    # Fill diagonal blocks (partition edges)
    offset = 0
    for part in hg.partitions:
        part_size = len(part.global_node_indices)
        if part.edge_index.numel() > 0:
            for i in range(part.edge_index.shape[1]):
                src, dst = part.edge_index[0, i].item(), part.edge_index[1, i].item()
                adj[offset + src, offset + dst] = 1
        offset += part_size

    # Fill off-diagonal blocks (bipartite edges)
    for bipart in hg.bipartites:
        left_offset = sum(len(hg.partitions[i].global_node_indices)
                         for i in range(bipart.left_part_id))
        right_offset = sum(len(hg.partitions[i].global_node_indices)
                          for i in range(bipart.right_part_id))

        if bipart.edge_index.numel() > 0:
            for i in range(bipart.edge_index.shape[1]):
                src = bipart.edge_index[0, i].item()
                dst = bipart.edge_index[1, i].item()
                adj[left_offset + src, right_offset + dst] = 1
                adj[right_offset + dst, left_offset + src] = 1

    # Plot
    ax.imshow(adj, cmap="Blues", aspect="equal")

    # Draw partition boundaries
    offset = 0
    for part in hg.partitions[:-1]:
        offset += len(part.global_node_indices)
        ax.axhline(y=offset - 0.5, color="red", linewidth=2)
        ax.axvline(x=offset - 0.5, color="red", linewidth=2)

    ax.set_title("Block Adjacency Matrix", fontsize=11, fontweight="bold")
    ax.set_xlabel("Nodes (reordered)")
    ax.set_ylabel("Nodes (reordered)")


def compare_tokenization(
    smiles: str,
    name: str | None = None,
    output: str | None = None,
    show: bool = True,
    seed: int = 42,
    motif_aware: bool = False,
    motif_alpha: float = 10.0,
) -> plt.Figure | None:
    """Create side-by-side comparison of SENT and H-SENT tokenization.

    Args:
        smiles: SMILES string.
        name: Optional molecule name for title.
        output: Output path to save figure.
        show: Whether to display the plot.
        seed: Random seed.
        motif_aware: Enable motif-aware coarsening for H-SENT.
        motif_alpha: Motif affinity weight (alpha parameter).

    Returns:
        Matplotlib figure or None if invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    data = smiles_to_graph(smiles)
    if data is None:
        return None

    # Store SMILES on data for motif detection
    data.smiles = smiles

    # Create tokenizers
    sent_tokenizer = SENTTokenizer(seed=seed)
    sent_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    hsent_tokenizer = HSENTTokenizer(
        seed=seed,
        motif_aware=motif_aware,
        motif_alpha=motif_alpha,
    )
    hsent_tokenizer.set_num_nodes(max(100, data.num_nodes + 20))

    # Tokenize
    sent_tokens = sent_tokenizer.tokenize(data)
    hsent_tokens = hsent_tokenizer.tokenize(data)
    hg = hsent_tokenizer.coarsener.build_hierarchy(data)

    # Create figure: 2 columns (SENT | H-SENT)
    fig = plt.figure(figsize=(16, 10))

    # Title
    title = name or smiles[:40]
    if len(smiles) > 40:
        title += "..."
    fig.suptitle(
        f"{title}\n({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)",
        fontsize=14, fontweight="bold"
    )

    # Left column: SENT
    ax1 = fig.add_subplot(2, 2, 1)
    plot_graph_with_walk(ax1, data, sent_tokens.tolist(), sent_tokenizer)

    ax3 = fig.add_subplot(2, 2, 3)
    plot_tokens(ax3, sent_tokens.tolist(), sent_tokenizer,
                f"SENT Tokens ({len(sent_tokens)} tokens)")

    # Right column: H-SENT
    ax2 = fig.add_subplot(2, 2, 2)
    hsent_label = "H-SENT"
    if motif_aware:
        hsent_label += f" (motif-aware, α={motif_alpha})"
    plot_graph_with_communities(ax2, data, hg,
                                f"{hsent_label}: {hg.num_communities} Communities")

    ax4 = fig.add_subplot(2, 2, 4)
    plot_tokens(ax4, hsent_tokens.tolist(), hsent_tokenizer,
                f"H-SENT Tokens ({len(hsent_tokens)} tokens)")

    plt.tight_layout()

    # Add comparison stats as text
    stats_text = (
        f"SENT: {len(sent_tokens)} tokens  |  "
        f"H-SENT: {len(hsent_tokens)} tokens ({hg.num_communities} communities)"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10,
             style="italic", color="gray")

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")

    if show:
        plt.show()

    return fig


def run_demo(
    output_dir: str | None = None,
    show: bool = True,
    motif_aware: bool = False,
    motif_alpha: float = 10.0,
):
    """Run demo with several molecules."""
    demo_molecules = [
        ("benzene", MOLECULES["benzene"]),
        ("caffeine", MOLECULES["caffeine"]),
        ("aspirin", MOLECULES["aspirin"]),
        ("dopamine", MOLECULES["dopamine"]),
    ]

    for name, smiles in demo_molecules:
        print(f"\n{'='*60}")
        print(f"Comparing tokenization: {name}")
        print(f"SMILES: {smiles}")
        if motif_aware:
            print(f"Motif-aware coarsening: α={motif_alpha}")
        print("=" * 60)

        output = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            suffix = "_motif" if motif_aware else ""
            output = f"{output_dir}/{name}_comparison{suffix}.png"

        compare_tokenization(
            smiles,
            name=name.replace("_", " ").title(),
            output=output,
            show=show,
            motif_aware=motif_aware,
            motif_alpha=motif_alpha,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare SENT and H-SENT tokenization of molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    %(prog)s --smiles "c1ccccc1"
    %(prog)s --name caffeine --output caffeine_compare.png
    %(prog)s --demo --output-dir ./figures
    %(prog)s --name cholesterol --motif-aware --alpha 10.0
    %(prog)s --demo --motif-aware --output-dir ./figures

Available molecules: {', '.join(sorted(MOLECULES.keys()))}
""",
    )
    parser.add_argument("--smiles", type=str, help="SMILES string to visualize")
    parser.add_argument("--name", type=str, help="Molecule name from predefined list")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--output-dir", type=str, help="Output directory for demo")
    parser.add_argument("--demo", action="store_true", help="Run demo with multiple molecules")
    parser.add_argument("--list", action="store_true", help="List available molecules")
    parser.add_argument("--no-show", action="store_true", help="Don't display (only save)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--motif-aware", action="store_true",
        help="Enable motif-aware coarsening for H-SENT"
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0,
        help="Motif affinity weight for motif-aware coarsening (default: 10.0)"
    )

    args = parser.parse_args()

    if args.list:
        print("Available molecules:")
        for name, smiles in sorted(MOLECULES.items()):
            mol = Chem.MolFromSmiles(smiles)
            atoms = mol.GetNumAtoms() if mol else "?"
            print(f"  {name:15} ({atoms:2} atoms): {smiles}")
        return

    if args.demo:
        run_demo(
            output_dir=args.output_dir,
            show=not args.no_show,
            motif_aware=args.motif_aware,
            motif_alpha=args.alpha,
        )
        return

    # Get SMILES
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

    compare_tokenization(
        smiles,
        name=name,
        output=args.output,
        show=not args.no_show,
        seed=args.seed,
        motif_aware=args.motif_aware,
        motif_alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
