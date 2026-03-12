#!/usr/bin/env python
"""Generation gallery: paper-quality figure comparing molecules across MOSAIC tokenizers.

Produces a grid figure with reference molecules (rows) vs. model variants (columns),
enabling visual comparison of generation quality across SENT, HSENT (SC/HAC/MC),
HDT (SC/HAC/MC), and HDTC.

Usage:
    python scripts/visualization/generation_gallery.py --dataset moses --num-samples 5
    python scripts/visualization/generation_gallery.py --dataset coconut --num-samples 3 --no-show
    python scripts/visualization/generation_gallery.py --dataset moses --num-samples 2 --num-generate 16 --no-show
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from rdkit import Chem, RDLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.coconut_loader import CoconutLoader  # noqa: E402
from src.data.molecular import (  # noqa: E402
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    graph_to_smiles,
    load_moses_dataset,
    smiles_to_graph,
)
from src.models.transformer import GraphGeneratorModule  # noqa: E402
from src.tokenizers import (  # noqa: E402
    HDTCTokenizer,
    HDTTokenizer,
    HSENTTokenizer,
    SENTTokenizer,
)

# Import drawing utilities from pipeline_overview (safe, no Hydra)
from pipeline_overview import (  # noqa: E402
    compute_rdkit_2d_layout,
    draw_molecule,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Reference molecule loading from real datasets
# ============================================================================


def _count_heavy_atoms(smiles: str) -> int | None:
    """Count heavy atoms in a SMILES string. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def load_reference_molecules(
    dataset: str,
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n reference molecules spanning the atom-count range of the dataset.

    Loads a pool of SMILES from the corresponding dataset (MOSES test set or
    COCONUT), bins them by heavy-atom count, and picks n evenly-spaced
    representatives so the gallery rows cover small-to-large molecules.

    Args:
        dataset: "moses" or "coconut".
        n: Number of reference molecules to select.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys "smiles" and "atoms", sorted by atom count.
    """
    rng = np.random.RandomState(seed)

    if dataset == "moses":
        # Load from MOSES test split (smaller, ~26K molecules)
        print("Loading MOSES test set for references...")
        all_smiles = load_moses_dataset(split="test", max_molecules=10000, seed=seed)
    else:
        # Load from COCONUT
        print("Loading COCONUT dataset for references...")
        loader = CoconutLoader(
            min_atoms=20, max_atoms=100, min_rings=3,
            data_file="data/coconut_complex.smi",
        )
        all_smiles = loader.load_smiles(n_samples=10000, seed=seed)

    # Count atoms and filter invalid
    pool: list[dict] = []
    for smi in all_smiles:
        natoms = _count_heavy_atoms(smi)
        if natoms is not None and natoms >= 5:
            pool.append({"smiles": smi, "atoms": natoms})

    if not pool:
        raise RuntimeError(f"No valid molecules found in {dataset} dataset")

    pool.sort(key=lambda m: m["atoms"])
    print(f"  Pool: {len(pool)} valid molecules, "
          f"atom range [{pool[0]['atoms']}, {pool[-1]['atoms']}]")

    # Pick n evenly-spaced by atom count
    atom_counts = np.array([m["atoms"] for m in pool])
    target_atoms = np.linspace(atom_counts.min(), atom_counts.max(), n)

    selected: list[dict] = []
    used_indices: set[int] = set()

    for target in target_atoms:
        # Find closest unused molecule
        diffs = np.abs(atom_counts - target)
        order = np.argsort(diffs)
        for idx in order:
            if idx not in used_indices:
                used_indices.add(idx)
                selected.append(pool[idx])
                break

    selected.sort(key=lambda m: m["atoms"])
    return selected


# ============================================================================
# Checkpoint configs
# ============================================================================

MOSES_CHECKPOINTS = {
    "SENT": {
        "path": "outputs/benchmark/moses_sent_20260221-021200/last.ckpt",
        "tokenizer_type": "sent",
        "coarsening_strategy": "spectral",
    },
    "HSENT_SC": {
        "path": "outputs/benchmark/moses_500k_hsent_sc_20260227-013849/last.ckpt",
        "tokenizer_type": "hsent",
        "coarsening_strategy": "spectral",
    },
    "HSENT_HAC": {
        "path": "outputs/benchmark/moses_500k_hsent_hac_20260227-014850/last.ckpt",
        "tokenizer_type": "hsent",
        "coarsening_strategy": "hac",
    },
    "HSENT_MC": {
        "path": "outputs/benchmark/moses_hsent_mc_20260221-060545/last.ckpt",
        "tokenizer_type": "hsent",
        "coarsening_strategy": "motif_community",
    },
    "HDT_SC": {
        "path": "outputs/benchmark/moses_500k_hdt_sc_20260227-015353/last.ckpt",
        "tokenizer_type": "hdt",
        "coarsening_strategy": "spectral",
    },
    "HDT_HAC": {
        "path": "outputs/benchmark/moses_500k_hdt_hac_20260227-015913/last.ckpt",
        "tokenizer_type": "hdt",
        "coarsening_strategy": "hac",
    },
    "HDT_MC": {
        "path": "outputs/benchmark/moses_hdt_mc_20260221-124557/last.ckpt",
        "tokenizer_type": "hdt",
        "coarsening_strategy": "motif_community",
    },
    "HDTC": {
        "path": "outputs/benchmark/moses_hdtc_20260221-224537/last.ckpt",
        "tokenizer_type": "hdtc",
        "coarsening_strategy": "spectral",
    },
}

COCONUT_CHECKPOINTS = {
    "SENT": {
        "path": "outputs/benchmark_coconut/coconut_sent_20260305-015418/last.ckpt",
        "tokenizer_type": "sent",
        "coarsening_strategy": "spectral",
    },
    "HSENT_SC": {
        "path": "outputs/benchmark_coconut/coconut_hsent_sc_20260306-033600/last.ckpt",
        "tokenizer_type": "hsent",
        "coarsening_strategy": "spectral",
    },
    "HSENT_HAC": {
        "path": "outputs/benchmark_coconut/coconut_hsent_hac_20260306-033435/last.ckpt",
        "tokenizer_type": "hsent",
        "coarsening_strategy": "hac",
    },
    "HSENT_MC": {
        "path": "outputs/benchmark_coconut/coconut_hsent_mc_20260305-015514/last.ckpt",
        "tokenizer_type": "hsent",
        "coarsening_strategy": "motif_community",
    },
    "HDT_SC": {
        "path": "outputs/benchmark_coconut/coconut_hdt_sc_20260306-033705/last.ckpt",
        "tokenizer_type": "hdt",
        "coarsening_strategy": "spectral",
    },
    "HDT_HAC": {
        "path": "outputs/benchmark_coconut/coconut_hdt_hac_20260306-033803/last.ckpt",
        "tokenizer_type": "hdt",
        "coarsening_strategy": "hac",
    },
    "HDT_MC": {
        "path": "outputs/benchmark_coconut/coconut_hdt_mc_20260305-015609/last.ckpt",
        "tokenizer_type": "hdt",
        "coarsening_strategy": "motif_community",
    },
    "HDTC": {
        "path": "outputs/benchmark_coconut/coconut_hdtc_20260305-015709/last.ckpt",
        "tokenizer_type": "hdtc",
        "coarsening_strategy": "spectral",
    },
}


# ============================================================================
# Model loading (copied from generation_demo.py to avoid Hydra side-effects)
# ============================================================================


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_type: str,
    labeled_graph: bool = True,
    coarsening_strategy: str = "spectral",
) -> tuple:
    """Load model and create appropriate tokenizer.

    Args:
        checkpoint_path: Path to the model checkpoint.
        tokenizer_type: One of "hdt", "hsent", "sent", "hdtc".
        labeled_graph: Whether the model uses labeled graphs.
        coarsening_strategy: Coarsening strategy for hierarchical tokenizers.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    wte_key = "model.model.transformer.wte.weight"
    if "state_dict" in checkpoint and wte_key in checkpoint["state_dict"]:
        checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
    else:
        raise ValueError(
            f"Cannot determine vocab size from checkpoint: {checkpoint_path}"
        )

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

    # Force-set (set_num_nodes() only increases)
    tokenizer.max_num_nodes = checkpoint_max_num_nodes

    if labeled_graph:
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

    wpe_key = "model.model.transformer.wpe.weight"
    load_kwargs: dict = {"tokenizer": tokenizer, "weights_only": False}
    if "state_dict" in checkpoint and wpe_key in checkpoint["state_dict"]:
        checkpoint_max_length = checkpoint["state_dict"][wpe_key].shape[0]
        load_kwargs["sampling_max_length"] = checkpoint_max_length

    model = GraphGeneratorModule.load_from_checkpoint(checkpoint_path, **load_kwargs)
    model.eval()

    return model, tokenizer


# ============================================================================
# Generation + selection
# ============================================================================


def generate_molecules(model, num_generate: int) -> list[tuple]:
    """Generate molecules and return list of (graph, smiles) for valid ones."""
    print(f"  Generating {num_generate} molecules...")
    graphs, avg_time, token_lengths = model.generate(
        num_samples=num_generate, show_progress=True
    )

    valid = []
    for g in graphs:
        smi = graph_to_smiles(g)
        if smi is not None:
            valid.append((g, smi))

    print(
        f"  Valid: {len(valid)}/{len(graphs)} "
        f"({100 * len(valid) / max(len(graphs), 1):.0f}%)"
    )
    return valid


def select_closest_molecules(
    valid_pool: list[tuple],
    target_atom_counts: list[int],
) -> list[tuple | None]:
    """Greedily pick one molecule per target atom count (no duplicates).

    Returns a list aligned with target_atom_counts; entries may be None.
    """
    remaining = list(valid_pool)
    results: list[tuple | None] = []

    for target in target_atom_counts:
        if not remaining:
            results.append(None)
            continue
        best_idx = min(
            range(len(remaining)),
            key=lambda i: abs(remaining[i][0].num_nodes - target),
        )
        results.append(remaining.pop(best_idx))

    return results


# ============================================================================
# Figure assembly
# ============================================================================


def draw_placeholder(ax: plt.Axes, message: str = "No valid\nmolecule") -> None:
    """Draw a red-tinted placeholder for failed generation."""
    ax.set_facecolor("#fff0f0")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=8,
        color="#cc0000",
        fontstyle="italic",
        transform=ax.transAxes,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def draw_mol_from_smiles(ax: plt.Axes, smiles: str, subtitle: str = "") -> None:
    """Draw a molecule from SMILES onto the given axes."""
    data = smiles_to_graph(smiles, labeled=True)
    if data is None:
        draw_placeholder(ax, "Parse error")
        return
    positions = compute_rdkit_2d_layout(smiles)
    if positions is None:
        draw_placeholder(ax, "Layout error")
        return
    draw_molecule(ax, smiles, data, positions)
    if subtitle:
        ax.set_title(subtitle, fontsize=7, pad=2, color="#555555")


def create_gallery_figure(
    references: list[dict],
    model_names: list[str],
    model_results: dict[str, list[tuple | None]],
    dpi: int = 200,
) -> plt.Figure:
    """Assemble the gallery grid figure.

    Args:
        references: List of reference molecule dicts.
        model_names: Ordered list of model display names.
        model_results: model_name -> list of (graph, smiles) or None per row.
        dpi: Output DPI.

    Returns:
        Matplotlib Figure.
    """
    n_rows = len(references)
    n_model_cols = len(model_names)
    # Columns: reference | divider | model_1 | model_2 | ... | model_N
    n_cols = 1 + 1 + n_model_cols

    width_ratios = [1.0] + [0.08] + [1.0] * n_model_cols
    fig_width = 2.4 * (1 + n_model_cols) + 0.3
    fig_height = 2.4 * n_rows + 1.2

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(
        n_rows + 1,  # +1 for header row
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.3,
        wspace=0.05,
        height_ratios=[0.15] + [1.0] * n_rows,
    )

    # --- Header row ---
    ax_header_ref = fig.add_subplot(gs[0, 0])
    ax_header_ref.text(
        0.5, 0.3, "Reference", ha="center", va="center",
        fontsize=10, fontweight="bold", transform=ax_header_ref.transAxes,
    )
    ax_header_ref.axis("off")

    # Divider column header
    ax_div_header = fig.add_subplot(gs[0, 1])
    ax_div_header.axis("off")

    for col_idx, mname in enumerate(model_names):
        ax_h = fig.add_subplot(gs[0, 2 + col_idx])
        ax_h.text(
            0.5, 0.3, mname, ha="center", va="center",
            fontsize=9, fontweight="bold", transform=ax_h.transAxes,
        )
        ax_h.axis("off")

    # --- Data rows ---
    for row_idx, ref in enumerate(references):
        grid_row = row_idx + 1  # offset by header

        # Reference column (light gray background)
        ax_ref = fig.add_subplot(gs[grid_row, 0])
        ax_ref.set_facecolor("#f5f5f5")
        subtitle_ref = f"{ref['atoms']} atoms"
        draw_mol_from_smiles(ax_ref, ref["smiles"], subtitle=subtitle_ref)

        # Divider column
        ax_div = fig.add_subplot(gs[grid_row, 1])
        ax_div.axvline(0.5, color="#cccccc", linewidth=1.5)
        ax_div.axis("off")

        # Model columns
        for col_idx, mname in enumerate(model_names):
            ax_gen = fig.add_subplot(gs[grid_row, 2 + col_idx])
            result = model_results[mname][row_idx]

            if result is None:
                draw_placeholder(ax_gen)
            else:
                g, smi = result
                positions = compute_rdkit_2d_layout(smi)
                if positions is None:
                    draw_placeholder(ax_gen, "Layout error")
                else:
                    draw_molecule(ax_gen, smi, g, positions)
                    ax_gen.set_title(
                        f"{g.num_nodes} atoms",
                        fontsize=7, pad=2, color="#555555",
                    )

    fig.suptitle(
        "MOSAIC Generation Gallery",
        fontsize=14, fontweight="bold", y=0.99,
    )

    return fig


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a gallery figure comparing MOSAIC tokenizer variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="moses", choices=["moses", "coconut"],
        help="Dataset / checkpoint set (default: moses)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of rows / reference molecules (default: 5)",
    )
    parser.add_argument(
        "--num-generate", type=int, default=64,
        help="Molecules generated per model for selection (default: 64)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./figures",
        help="Output directory (default: ./figures)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI (default: 600)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--no-show", action="store_true", help="Suppress plt.show()",
    )
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Select references from the real dataset
    references = load_reference_molecules(args.dataset, args.num_samples, seed=args.seed)
    target_atom_counts = [ref["atoms"] for ref in references]
    print(f"\nReference molecules ({len(references)}):")
    for ref in references:
        print(f"  {ref['smiles'][:50]}... ({ref['atoms']} atoms)")

    # Select checkpoint set
    checkpoints = MOSES_CHECKPOINTS if args.dataset == "moses" else COCONUT_CHECKPOINTS
    model_names = list(checkpoints.keys())
    model_results: dict[str, list[tuple | None]] = {}

    for mname, ckpt_cfg in checkpoints.items():
        ckpt_path = ckpt_cfg["path"]
        print(f"\n{'=' * 60}")
        print(f"Model: {mname}")
        print(f"  Checkpoint: {ckpt_path}")

        if not Path(ckpt_path).exists():
            print(f"  SKIPPED (checkpoint not found)")
            model_results[mname] = [None] * len(references)
            continue

        try:
            model, tokenizer = load_model_and_tokenizer(
                ckpt_path,
                ckpt_cfg["tokenizer_type"],
                labeled_graph=True,
                coarsening_strategy=ckpt_cfg["coarsening_strategy"],
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            valid_pool = generate_molecules(model, args.num_generate)
            selected = select_closest_molecules(valid_pool, target_atom_counts)
            model_results[mname] = selected

            for i, (ref, sel) in enumerate(zip(references, selected)):
                if sel is not None:
                    g, smi = sel
                    print(f"  Row {i} (target={ref['atoms']}): "
                          f"{smi} ({g.num_nodes} atoms)")
                else:
                    print(f"  Row {i} (target={ref['atoms']}): No valid molecule")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            model_results[mname] = [None] * len(references)

    # Assemble figure
    print(f"\n{'=' * 60}")
    print("Assembling gallery figure...")
    fig = create_gallery_figure(references, model_names, model_results, dpi=args.dpi)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"generation_gallery_{args.dataset}.png"
    fig.savefig(str(output_path), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    pdf_path = output_dir / f"generation_gallery_{args.dataset}.pdf"
    fig.savefig(str(pdf_path), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
