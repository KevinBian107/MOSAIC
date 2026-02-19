#!/usr/bin/env python
"""Run valence violation error analysis across multiple COCONUT models.

Generates molecules from each model, analyzes where valence violations
occur, and produces a 3-panel figure comparing error locations.

Usage:
    python property_experiment/error_analysis/run_experiment.py
    python property_experiment/error_analysis/run_experiment.py generation.num_samples=100
"""

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import (
    HDTCTokenizer,
    HDTTokenizer,
    HSENTTokenizer,
    SENTTokenizer,
)

from property_experiment.error_analysis.analysis import analyze_batch
from property_experiment.error_analysis.visualize import create_figure


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_type: str,
    labeled_graph: bool = True,
    coarsening_strategy: str = "spectral",
) -> tuple:
    """Load model and create appropriate tokenizer.

    Battle-tested checkpoint loading logic from generation_demo.py.

    Args:
        checkpoint_path: Path to the model checkpoint.
        tokenizer_type: One of "hdt", "hsent", "sent", "hdtc".
        labeled_graph: Whether the model uses labeled graphs.
        coarsening_strategy: Coarsening strategy for hierarchical tokenizers.

    Returns:
        Tuple of (model, tokenizer).
    """
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
    load_kwargs = {"tokenizer": tokenizer, "weights_only": False}
    if "state_dict" in checkpoint and wpe_key in checkpoint["state_dict"]:
        checkpoint_max_length = checkpoint["state_dict"][wpe_key].shape[0]
        load_kwargs["sampling_max_length"] = checkpoint_max_length

    model = GraphGeneratorModule.load_from_checkpoint(checkpoint_path, **load_kwargs)
    model.eval()

    return model, tokenizer


@hydra.main(
    config_path="../../configs/property_experiment",
    config_name="error_analysis",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run the valence violation error analysis experiment."""
    # Resolve output dir (Hydra changes cwd, so use absolute path)
    output_dir = Path(cfg.output.dir)
    if not output_dir.is_absolute():
        output_dir = Path(hydra.utils.get_original_cwd()) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)

    all_results = {}

    for model_cfg in cfg.models:
        name = model_cfg.name
        checkpoint_path = model_cfg.checkpoint_path
        if not Path(checkpoint_path).is_absolute():
            checkpoint_path = str(
                Path(hydra.utils.get_original_cwd()) / checkpoint_path
            )

        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        if not Path(checkpoint_path).exists():
            print(f"  SKIP: checkpoint not found at {checkpoint_path}")
            continue

        # Load model
        print("  Loading model...")
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=checkpoint_path,
            tokenizer_type=model_cfg.tokenizer_type,
            labeled_graph=True,
            coarsening_strategy=model_cfg.get("coarsening_strategy", "spectral"),
        )

        # Override generation settings from config
        model.sampling_batch_size = cfg.generation.batch_size
        model.sampling_top_k = cfg.generation.top_k
        model.sampling_temperature = cfg.generation.temperature

        # Generate molecules
        print(f"  Generating {cfg.generation.num_samples} molecules...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        graphs, avg_time, token_lengths = model.generate(
            num_samples=cfg.generation.num_samples,
            show_progress=True,
        )
        print(f"  Generated {len(graphs)} molecules (avg {avg_time:.3f}s/sample)")

        # Analyze for valence violations
        print("  Analyzing valence violations...")
        batch_results = analyze_batch(graphs)

        print(f"  Valid: {batch_results['num_valid']}/{batch_results['total']} "
              f"({batch_results['validity_rate']:.1%})")
        print(f"  Total violations: {batch_results['total_violations']}")
        print(f"  Role counts: {batch_results['role_counts']}")
        print(f"  Boundary ratio: {batch_results['boundary_ratio']:.3f}")

        all_results[name] = batch_results

    if not all_results:
        print("\nNo models were successfully loaded. Exiting.")
        return

    # Create figure
    print(f"\n{'='*60}")
    print("Creating visualization...")
    create_figure(all_results, str(output_dir))

    # Save raw results as JSON (excluding per-molecule details for size)
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            k: v for k, v in res.items() if k != "per_molecule"
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")

    # Also save full per-molecule data
    full_path = output_dir / "results_full.json"
    with open(full_path, "w") as f:
        json.dump(
            {name: res for name, res in all_results.items()},
            f,
            indent=2,
        )
    print(f"Full results saved to {full_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
