#!/usr/bin/env python
"""Testing script for evaluating trained molecular graph generation models.

This script loads a trained model and evaluates its generation quality
using molecular metrics (validity, uniqueness, novelty, FCD, etc.)
and motif distribution metrics.

Supports both MOSAIC and AutoGraph pretrained checkpoints.

Usage:
    python scripts/test.py model.checkpoint_path=/path/to/model.ckpt
    python scripts/test.py data.dataset_name=qm9 model.checkpoint_path=outputs/model.ckpt
    python scripts/test.py model.checkpoint_path=/path/to/autograph.ckpt model.is_autograph=true
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import MolecularDataModule
from src.data.molecular import graph_to_smiles
from src.evaluation.molecular_metrics import MolecularMetrics, compute_fcd
from src.evaluation.motif_distribution import MotifDistributionMetric
from src.models.transformer import GraphGeneratorModule
from src.tokenizers.sent import SENTTokenizer

# Import AutoGraph conversion functions for handling AutoGraph checkpoints
try:
    autograph_path = Path(__file__).parent.parent / "tmp" / "AutoGraph"
    if str(autograph_path) not in sys.path:
        sys.path.insert(0, str(autograph_path))
    from autograph.evaluation.molsets import build_molecule, mol2smiles
    AUTOGRAPH_AVAILABLE = True
except ImportError:
    AUTOGRAPH_AVAILABLE = False

log = logging.getLogger(__name__)


def autograph_graph_to_smiles(graph, atom_decoder: list[str]) -> Optional[str]:
    """Convert AutoGraph PyG Data object to SMILES using AutoGraph's functions.

    Args:
        graph: PyG Data object with integer-encoded atom/bond types (AutoGraph format).
        atom_decoder: List mapping atom type indices to atom symbols.

    Returns:
        SMILES string or None if conversion fails.
    """
    if not AUTOGRAPH_AVAILABLE:
        return None
    try:
        mol = build_molecule(graph, atom_decoder)
        return mol2smiles(mol)
    except Exception:
        return None


def is_autograph_checkpoint(checkpoint_path: str) -> bool:
    """Detect if a checkpoint is from AutoGraph or MOSAIC.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        True if AutoGraph checkpoint, False if MOSAIC checkpoint.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # AutoGraph checkpoints have a 'cfg' in hyper_parameters
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            # AutoGraph has 'cfg' key, MOSAIC doesn't
            return "cfg" in hparams
        return False
    except Exception as e:
        log.warning(f"Could not detect checkpoint type: {e}")
        return False


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
    """Main testing function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.checkpoint_path is None:
        raise ValueError("model.checkpoint_path must be specified")

    pl.seed_everything(cfg.seed, workers=True)

    tokenizer = SENTTokenizer(
        max_length=cfg.tokenizer.max_length,
        truncation_length=cfg.tokenizer.truncation_length,
        undirected=cfg.tokenizer.undirected,
        seed=cfg.seed,
    )

    datamodule = MolecularDataModule(
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_train=cfg.data.num_train,
        num_val=cfg.data.num_val,
        num_test=cfg.data.num_test,
        include_hydrogens=cfg.data.get("include_hydrogens", False),
        seed=cfg.seed,
        data_root=cfg.data.get("data_root", "data"),
    )

    datamodule.setup(stage="test")

    # Detect checkpoint type
    use_autograph = cfg.model.get("is_autograph", False)
    if not use_autograph:
        use_autograph = is_autograph_checkpoint(cfg.model.checkpoint_path)

    # For MOSAIC checkpoints, extract vocab size and update tokenizer
    if not use_autograph:
        log.info(f"Extracting vocab size from checkpoint: {cfg.model.checkpoint_path}")
        checkpoint = torch.load(cfg.model.checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            # Extract vocab size from embedding weight shape
            wte_key = "model.model.transformer.wte.weight"
            if wte_key in checkpoint["state_dict"]:
                checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
                log.info(f"Checkpoint vocab size: {checkpoint_vocab_size}")

                # Detect if this is a labeled graph model
                # Unlabeled: vocab_size = idx_offset (6) + max_num_nodes
                # Labeled: vocab_size = idx_offset (6) + max_num_nodes + num_node_types + num_edge_types
                from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES

                # Try labeled first
                checkpoint_max_num_nodes_labeled = checkpoint_vocab_size - tokenizer.idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES

                if checkpoint_max_num_nodes_labeled > 0 and checkpoint_max_num_nodes_labeled <= 100:
                    # This is likely a labeled graph model
                    log.info("Detected labeled SENT checkpoint")
                    tokenizer.labeled_graph = True
                    tokenizer.set_num_nodes(checkpoint_max_num_nodes_labeled)
                    tokenizer.set_num_node_and_edge_types(
                        num_node_types=NUM_ATOM_TYPES,
                        num_edge_types=NUM_BOND_TYPES,
                    )
                    log.info(f"Set tokenizer: max_num_nodes={checkpoint_max_num_nodes_labeled}, labeled_graph=True")
                else:
                    # Unlabeled model
                    checkpoint_max_num_nodes = checkpoint_vocab_size - tokenizer.idx_offset
                    log.info(f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}")
                    tokenizer.set_num_nodes(checkpoint_max_num_nodes)

    if use_autograph:
        log.info(f"Detected AutoGraph checkpoint at {cfg.model.checkpoint_path}")
        log.info("Loading model using AutoGraph adapter...")
        from src.models.autograph_adapter import AutoGraphAdapter

        model = AutoGraphAdapter.load_from_checkpoint(
            cfg.model.checkpoint_path,
            tokenizer=tokenizer,
            sampling_batch_size=cfg.sampling.get("batch_size", 32),
            sampling_top_k=cfg.sampling.get("top_k", 10),
            sampling_temperature=cfg.sampling.get("temperature", 1.0),
            sampling_max_length=cfg.sampling.get("max_length", 2048),
        )
    else:
        log.info(f"Loading MOSAIC model from {cfg.model.checkpoint_path}...")
        model = GraphGeneratorModule.load_from_checkpoint(
            cfg.model.checkpoint_path,
            tokenizer=tokenizer,
        )
    model.eval()

    num_test = len(datamodule.test_smiles)

    num_samples = cfg.sampling.num_samples
    if num_samples < 0:
        num_samples = num_test

    log.info(f"Generating {num_samples} molecules...")
    generated_graphs, gen_time = model.generate(num_samples=num_samples)
    log.info(f"Generated {len(generated_graphs)} graphs")
    log.info(f"Average generation time: {gen_time:.4f}s per sample")

    # Convert to SMILES using appropriate converter
    # IMPORTANT: Include all attempts (even failures) for accurate validity metric
    # Use a sentinel value for failed conversions that RDKit will reject
    INVALID_SMILES_SENTINEL = "INVALID"
    generated_smiles = []
    if use_autograph:
        # AutoGraph models - use AutoGraph's conversion functions
        # MOSES atom decoder (from AutoGraph's MOSESDataset): ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
        atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
        log.info("Converting AutoGraph graphs to SMILES...")
        for g in generated_graphs:
            smiles = autograph_graph_to_smiles(g, atom_decoder)
            generated_smiles.append(smiles if smiles else INVALID_SMILES_SENTINEL)
    else:
        # MOSAIC models - use MOSAIC's conversion function
        log.info("Converting MOSAIC graphs to SMILES...")
        for g in generated_graphs:
            smiles = graph_to_smiles(g)
            generated_smiles.append(smiles if smiles else INVALID_SMILES_SENTINEL)

    valid_count = sum(1 for s in generated_smiles if s != INVALID_SMILES_SENTINEL)
    log.info(f"Successfully converted {valid_count}/{len(generated_smiles)} graphs to SMILES")

    log.info("\n" + "=" * 50)
    log.info("MOLECULAR METRICS")
    log.info("=" * 50)

    mol_metrics = MolecularMetrics(
        reference_smiles=datamodule.train_smiles,
    )
    mol_results = mol_metrics(generated_smiles)

    for name, value in mol_results.items():
        log.info(f"  {name:20s}: {value:.6f}")

    log.info("\n" + "=" * 50)
    log.info("MOTIF DISTRIBUTION METRICS")
    log.info("=" * 50)

    motif_metrics = MotifDistributionMetric(
        reference_smiles=datamodule.train_smiles,
    )
    motif_results = motif_metrics(generated_smiles)

    for name, value in motif_results.items():
        log.info(f"  {name:20s}: {value:.6f}")

    # Try to compute FCD if available
    log.info("\n" + "=" * 50)
    log.info("FCD METRIC")
    log.info("=" * 50)

    fcd_score = None
    try:
        fcd_score = compute_fcd(generated_smiles, datamodule.test_smiles)
        if not (fcd_score != fcd_score):  # Check for NaN
            log.info(f"  FCD: {fcd_score:.6f}")
        else:
            log.info("  FCD: Not available (install moses or fcd package)")
            fcd_score = None
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log.error(f"  FCD: Failed with error: {e}")
        log.info("  FCD: Skipping due to error")
        fcd_score = None

    # Get motif summary for reference
    log.info("\n" + "=" * 50)
    log.info("MOTIF SUMMARY (Top 10)")
    log.info("=" * 50)

    summary = motif_metrics.get_motif_summary(generated_smiles[:100])
    log.info("\nSMARTS Motifs found:")
    for name, count in list(summary["smarts_motifs"].items())[:10]:
        log.info(f"  {name}: {count}")

    log.info("\nFunctional Groups found:")
    for name, count in list(summary["functional_groups"].items())[:10]:
        log.info(f"  {name}: {count}")

    # Compile all results including motif summary
    all_results = {
        **mol_results,
        **motif_results,
        "fcd": fcd_score,
        "generation_time": gen_time,
        "num_samples": num_samples,
        "num_valid_smiles": valid_count,
        "motif_summary": summary,  # Add detailed motif counts
    }

    # Save results
    output_path = Path(cfg.logs.path)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {results_file}")

    # Save generated SMILES (only valid ones, excluding sentinel values)
    smiles_file = output_path / "generated_smiles.txt"
    with open(smiles_file, "w") as f:
        for smi in generated_smiles:
            if smi != INVALID_SMILES_SENTINEL:
                f.write(smi + "\n")
    log.info(f"Generated SMILES saved to {smiles_file}")

    log.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
