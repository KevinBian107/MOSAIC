#!/usr/bin/env python
"""Scaffold-primed generation evaluation script.

This script evaluates scaffold priming using complex natural products from COCONUT.
It extracts Murcko scaffolds from complex molecules, primes the model with scaffold
tokens, generates completions, and evaluates how well they match the original.

Usage:
    # First, prepare the data (one-time)
    python scripts/prepare_coconut_data.py

    # Run evaluation
    python scripts/primed_gen.py

    # Customize evaluation
    python scripts/primed_gen.py \
        data_source.n_molecules=50 \
        evaluation.samples_per_molecule=5

    # Use different tokenizer
    python scripts/primed_gen.py tokenizer=hsent
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress RDKit error messages
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.coconut_loader import CoconutLoader  # noqa: E402
from src.data.molecular import (  # noqa: E402
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)
from src.models.transformer import GraphGeneratorModule  # noqa: E402
from src.tokenizers import (  # noqa: E402
    HDTCTokenizer,
    HDTTokenizer,
    HSENTTokenizer,
    SENTTokenizer,
)
from src.transfer_learning import PrimedGenerator  # noqa: E402
from src.transfer_learning.datasets.complex_molecule_dataset import (  # noqa: E402
    ComplexMoleculeDataset,
)
from src.transfer_learning.evaluation.priming_evaluator import (  # noqa: E402
    PrimingEvaluator,
)
from src.transfer_learning.evaluation.visualization import (  # noqa: E402
    create_summary_grid,
    visualize_evaluation_results,
)

log = logging.getLogger(__name__)


def get_tokenizer(cfg: DictConfig):
    """Create tokenizer based on configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        Configured tokenizer instance.
    """
    tokenizer_type = cfg.tokenizer.get("type", "hdt").lower()

    if tokenizer_type == "hdt":
        log.info("Using HDT tokenizer")
        tokenizer = HDTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", "spectral"),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hsent":
        log.info("Using H-SENT tokenizer")
        tokenizer = HSENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", "spectral"),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hdtc":
        log.info("Using HDTC tokenizer")
        tokenizer = HDTCTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            include_rings=cfg.tokenizer.get("include_rings", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "sent":
        log.info("Using SENT tokenizer")
        tokenizer = SENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            undirected=cfg.tokenizer.get("undirected", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer


def configure_tokenizer_from_checkpoint(
    tokenizer,
    checkpoint_path: str,
) -> None:
    """Configure tokenizer vocab size from checkpoint.

    Args:
        tokenizer: Tokenizer instance to configure.
        checkpoint_path: Path to model checkpoint.
    """
    log.info(f"Extracting vocab size from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        wte_key = "model.model.transformer.wte.weight"
        if wte_key in checkpoint["state_dict"]:
            checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
            log.info(f"Checkpoint vocab size: {checkpoint_vocab_size}")

            idx_offset = getattr(tokenizer, "idx_offset", None) or getattr(
                tokenizer, "IDX_OFFSET", 7
            )

            # Try labeled first
            checkpoint_max_num_nodes_labeled = (
                checkpoint_vocab_size - idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES
            )

            if (
                checkpoint_max_num_nodes_labeled > 0
                and checkpoint_max_num_nodes_labeled <= 100
            ):
                log.info("Detected labeled checkpoint")
                tokenizer.labeled_graph = True
                tokenizer.set_num_nodes(checkpoint_max_num_nodes_labeled)
                tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)
                log.info(
                    f"Set tokenizer: max_num_nodes={checkpoint_max_num_nodes_labeled}, "
                    f"labeled_graph=True"
                )
            else:
                checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset
                log.info(
                    f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}"
                )
                tokenizer.set_num_nodes(checkpoint_max_num_nodes)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="primed_gen",
)
def main(cfg: DictConfig) -> None:
    """Main scaffold-primed generation evaluation function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.checkpoint_path is None:
        raise ValueError(
            "model.checkpoint_path must be specified. "
            "Example: model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt"
        )

    # Create output directory
    output_dir = Path(cfg.logs.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")

    pl.seed_everything(cfg.seed, workers=True)

    # Create tokenizer
    tokenizer = get_tokenizer(cfg)

    # Configure tokenizer from checkpoint
    configure_tokenizer_from_checkpoint(tokenizer, cfg.model.checkpoint_path)

    # Load model
    log.info(f"Loading model from {cfg.model.checkpoint_path}...")
    model = GraphGeneratorModule.load_from_checkpoint(
        cfg.model.checkpoint_path,
        tokenizer=tokenizer,
        sampling_batch_size=cfg.generation.batch_size,
        sampling_top_k=cfg.sampling.top_k,
        sampling_temperature=cfg.sampling.temperature,
        sampling_max_length=cfg.sampling.max_length,
    )
    model.eval()

    # Create primed generator
    log.info("Creating primed generator...")
    generator = PrimedGenerator(model)

    # Run evaluation
    log.info("\n" + "=" * 60)
    log.info("SCAFFOLD PRIMING EVALUATION")
    log.info("=" * 60)

    # Create data loader
    data_file = cfg.data_source.data_file
    if not Path(data_file).exists():
        log.error(f"Data file not found: {data_file}")
        log.error("Please run: python scripts/prepare_coconut_data.py")
        return

    loader = CoconutLoader(
        min_atoms=cfg.data_source.min_atoms,
        max_atoms=cfg.data_source.max_atoms,
        min_rings=cfg.data_source.min_rings,
        min_scaffold_atoms=cfg.data_source.min_scaffold_atoms,
        data_file=data_file,
    )

    # Create dataset
    log.info(f"Loading {cfg.data_source.n_molecules} complex molecules...")
    dataset = ComplexMoleculeDataset(
        coconut_loader=loader,
        n_samples=cfg.data_source.n_molecules,
        seed=cfg.seed,
    )

    log.info(f"Loaded {len(dataset)} molecules with valid scaffolds")

    if len(dataset) == 0:
        log.error("No valid molecules found! Check data file and filtering criteria.")
        return

    # Print dataset summary
    summary = dataset.summary()
    log.info("Dataset summary:")
    log.info(f"  Unique scaffolds: {summary.get('n_unique_scaffolds', 0)}")
    log.info(f"  Scaffold size: {summary.get('scaffold_size_mean', 0):.1f} atoms (mean)")
    log.info(f"  Molecule size: {summary.get('mol_size_mean', 0):.1f} atoms (mean)")

    # Create evaluator
    evaluator = PrimingEvaluator()

    # Run evaluation (with per-sample results for visualization)
    samples_per_scaffold = cfg.evaluation.samples_per_molecule
    min_new_tokens = cfg.sampling.get("min_new_tokens", None)
    primer_fraction = cfg.sampling.get("primer_fraction", None)

    log.info(f"\nGenerating {samples_per_scaffold} samples per scaffold...")
    if primer_fraction is not None and primer_fraction < 1.0:
        log.info(f"  primer_fraction={primer_fraction} (using partial scaffold as primer)")
    if min_new_tokens:
        log.info(f"  min_new_tokens={min_new_tokens} (encouraging longer completions)")

    results = evaluator.evaluate_dataset(
        dataset,
        generator,
        samples_per_scaffold=samples_per_scaffold,
        verbose=True,
        return_per_sample=True,
        min_new_tokens=min_new_tokens,
        primer_fraction=primer_fraction,
    )

    # Log results
    log.info("\n" + "=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"Molecules evaluated: {results.get('n_molecules', 0)}")
    log.info(f"Samples per scaffold: {results.get('samples_per_scaffold', 0)}")
    log.info("")
    log.info(
        f"Scaffold preservation: {results.get('scaffold_preservation_mean', 0):.1%} "
        f"(+/-{results.get('scaffold_preservation_std', 0):.1%})"
    )
    log.info(
        f"Tanimoto similarity: {results.get('tanimoto_mean', 0):.3f} "
        f"(+/-{results.get('tanimoto_std', 0):.3f})"
    )
    log.info(f"Best Tanimoto (max of maxes): {results.get('tanimoto_max_max', 0):.3f}")
    log.info(
        f"Validity rate: {results.get('valid_rate_mean', 0):.1%} "
        f"(+/-{results.get('valid_rate_std', 0):.1%})"
    )
    log.info(
        f"Atom ratio: {results.get('atom_ratio_mean', 0):.2f} "
        f"(+/-{results.get('atom_ratio_std', 0):.2f})"
    )

    # Generate visualizations
    if cfg.output.get("visualize", True):
        per_sample_results = results.get("per_sample_results", [])
        if per_sample_results:
            log.info("\nGenerating visualizations...")

            # Create individual sample visualizations
            vis_dir = output_dir / "visualizations"
            saved_paths = visualize_evaluation_results(
                per_sample_results,
                vis_dir,
                max_samples=min(10, len(per_sample_results)),
                n_generated_per_sample=3,
            )
            log.info(f"  Saved {len(saved_paths)} sample visualizations to {vis_dir}")

            # Create summary grid
            summary_path = output_dir / "summary_grid.png"
            try:
                create_summary_grid(
                    per_sample_results,
                    summary_path,
                    n_samples=min(6, len(per_sample_results)),
                )
                log.info(f"  Saved summary grid to {summary_path}")
            except Exception as e:
                log.warning(f"  Could not create summary grid: {e}")

    # Save evaluation results
    if cfg.output.get("save_evaluation", True):
        eval_file = output_dir / "evaluation_results.json"
        with open(eval_file, "w") as f:
            # Convert numpy values to Python types for JSON serialization
            # Exclude per_sample_results from JSON (too large)
            serializable_results = {
                k: float(v) if hasattr(v, "item") else v
                for k, v in results.items()
                if k != "per_sample_results"
            }
            json.dump(serializable_results, f, indent=2)
        log.info(f"\nEvaluation results saved to {eval_file}")

    log.info("\nScaffold priming evaluation complete!")


if __name__ == "__main__":
    main()
