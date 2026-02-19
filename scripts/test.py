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
import random
import statistics
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress RDKit error messages for invalid SMILES parsing
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.datamodule import MolecularDataModule
from src.data.molecular import graph_to_smiles, smiles_to_graph
from src.evaluation.molecular_metrics import MolecularMetrics, compute_fcd
from src.evaluation.motif_distribution import MotifDistributionMetric
from src.evaluation.polygraph_metric import PolygraphMetric
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer
from src.visualization import visualize_generated_molecules

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

    # Create output directory and save configuration
    output_dir = Path(cfg.logs.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")

    pl.seed_everything(cfg.seed, workers=True)

    # Select tokenizer based on config
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()
    if tokenizer_type == "hdt":
        motif_aware = cfg.tokenizer.get("motif_aware", False)
        if motif_aware:
            log.info("Using hierarchical HDT tokenizer with motif-aware coarsening")
            log.info(f"  motif_alpha: {cfg.tokenizer.get('motif_alpha', 1.0)}")
        else:
            log.info("Using hierarchical HDT tokenizer with spectral coarsening")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  min_community_size: {cfg.tokenizer.get('min_community_size', 4)}")

        tokenizer = HDTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", None),
            motif_aware=motif_aware,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hsent":
        motif_aware = cfg.tokenizer.get("motif_aware", False)
        if motif_aware:
            log.info("Using hierarchical H-SENT tokenizer with motif-aware coarsening")
            log.info(f"  motif_alpha: {cfg.tokenizer.get('motif_alpha', 1.0)}")
        else:
            log.info("Using hierarchical H-SENT tokenizer with spectral coarsening")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  min_community_size: {cfg.tokenizer.get('min_community_size', 4)}")

        tokenizer = HSENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", None),
            motif_aware=motif_aware,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hdtc":
        log.info("Using HDTC (compositional) tokenizer with functional hierarchy")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  include_rings: {cfg.tokenizer.get('include_rings', True)}")

        tokenizer = HDTCTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            include_rings=cfg.tokenizer.get("include_rings", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    else:
        log.info("Using flat SENT tokenizer")
        tokenizer = SENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            undirected=cfg.tokenizer.get("undirected", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", False),
            seed=cfg.seed,
        )

    datamodule = MolecularDataModule(
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        test_num_workers=cfg.data.get("test_num_workers", 0),
        num_train=cfg.data.num_train,
        num_val=cfg.data.num_val,
        num_test=cfg.data.num_test,
        include_hydrogens=cfg.data.get("include_hydrogens", False),
        seed=cfg.seed,
        data_root=cfg.data.get("data_root", "data"),
        use_cache=cfg.data.get("use_cache", False),
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
        data_file=cfg.data.get("data_file", None),
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
    )

    datamodule.setup(stage="test")

    # Detect checkpoint type
    use_autograph = cfg.model.get("is_autograph", False)
    if not use_autograph:
        use_autograph = is_autograph_checkpoint(cfg.model.checkpoint_path)

    # For MOSAIC checkpoints, extract vocab size and update tokenizer
    checkpoint_max_length = None
    if not use_autograph:
        log.info(f"Extracting vocab size from checkpoint: {cfg.model.checkpoint_path}")
        checkpoint = torch.load(
            cfg.model.checkpoint_path, map_location="cpu", weights_only=False
        )
        if "state_dict" in checkpoint:
            # Extract vocab size from embedding weight shape
            wte_key = "model.model.transformer.wte.weight"
            if wte_key in checkpoint["state_dict"]:
                checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
                log.info(f"Checkpoint vocab size: {checkpoint_vocab_size}")

                # Determine vocab layout from tokenizer's labeled_graph setting
                # Unlabeled: vocab_size = idx_offset + max_num_nodes
                # Labeled: vocab_size = idx_offset + max_num_nodes + num_node_types + num_edge_types
                from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES

                # Get idx_offset (handle both lowercase and uppercase)
                idx_offset = getattr(tokenizer, "idx_offset", None) or getattr(
                    tokenizer, "IDX_OFFSET", 6
                )

                is_labeled = getattr(tokenizer, "labeled_graph", False)

                if is_labeled:
                    checkpoint_max_num_nodes = (
                        checkpoint_vocab_size
                        - idx_offset
                        - NUM_ATOM_TYPES
                        - NUM_BOND_TYPES
                    )
                    if checkpoint_max_num_nodes <= 0:
                        log.warning(
                            f"Labeled formula gives non-positive max_num_nodes "
                            f"({checkpoint_max_num_nodes}), falling back to unlabeled"
                        )
                        is_labeled = False
                        tokenizer.labeled_graph = False
                        checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset

                    if is_labeled:
                        # Force-set max_num_nodes to match checkpoint exactly;
                        # set_num_nodes() only increases and won't shrink a value
                        # inflated by datamodule.setup()
                        tokenizer.max_num_nodes = checkpoint_max_num_nodes
                        tokenizer.set_num_node_and_edge_types(
                            num_node_types=NUM_ATOM_TYPES,
                            num_edge_types=NUM_BOND_TYPES,
                        )
                        log.info(
                            f"Set tokenizer: max_num_nodes={checkpoint_max_num_nodes}, labeled_graph=True"
                        )
                    else:
                        tokenizer.max_num_nodes = checkpoint_max_num_nodes
                        log.info(
                            f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}"
                        )
                else:
                    # Unlabeled model
                    checkpoint_max_num_nodes = checkpoint_vocab_size - idx_offset
                    log.info(
                        f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}"
                    )
                    # Force-set to match checkpoint exactly
                    tokenizer.max_num_nodes = checkpoint_max_num_nodes

            # Extract max position embeddings from checkpoint (GPT-2 wpe)
            wpe_key = "model.model.transformer.wpe.weight"
            if wpe_key in checkpoint["state_dict"]:
                checkpoint_max_length = checkpoint["state_dict"][wpe_key].shape[0]
                log.info(f"Checkpoint max position embeddings: {checkpoint_max_length}")

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
        load_kwargs: dict = {"tokenizer": tokenizer, "weights_only": False}
        if checkpoint_max_length is not None:
            load_kwargs["sampling_max_length"] = checkpoint_max_length
        model = GraphGeneratorModule.load_from_checkpoint(
            cfg.model.checkpoint_path,
            **load_kwargs,
        )
    model.eval()

    num_test = len(datamodule.test_smiles)

    num_samples = cfg.sampling.num_samples
    if num_samples < 0:
        num_samples = num_test

    log.info(f"Generating {num_samples} molecules...")
    log.info(
        "(Progress bar shows batches; slow because generation is autoregressive, one token per step.)"
    )
    gen_result = model.generate(num_samples=num_samples, show_progress=True)
    generated_graphs = gen_result[0]
    gen_time = gen_result[1]
    token_lengths = gen_result[2] if len(gen_result) > 2 else None
    log.info(f"Generated {len(generated_graphs)} graphs")
    log.info(f"Average generation time: {gen_time:.4f}s per sample")
    if token_lengths:
        log.info(
            f"Token lengths per generation: min={min(token_lengths)}, max={max(token_lengths)}, "
            f"mean={statistics.mean(token_lengths):.1f}, median={statistics.median(token_lengths):.0f}"
        )

    # Convert to SMILES using appropriate converter
    # IMPORTANT: Include all attempts (even failures) for accurate validity metric
    # Use a sentinel value for failed conversions that RDKit will reject
    INVALID_SMILES_SENTINEL = "INVALID"
    generated_smiles = []
    if use_autograph:
        # AutoGraph models - use AutoGraph's conversion functions
        # MOSES atom decoder (from AutoGraph's MOSESDataset): ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
        atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]
        log.info("Converting AutoGraph graphs to SMILES...")
        for g in tqdm(generated_graphs, desc="Converting to SMILES"):
            smiles = autograph_graph_to_smiles(g, atom_decoder)
            generated_smiles.append(smiles if smiles else INVALID_SMILES_SENTINEL)
    else:
        # MOSAIC models - use MOSAIC's conversion function
        log.info("Converting MOSAIC graphs to SMILES...")
        for g in tqdm(generated_graphs, desc="Converting to SMILES"):
            smiles = graph_to_smiles(g)
            generated_smiles.append(smiles if smiles else INVALID_SMILES_SENTINEL)

    valid_count = sum(1 for s in generated_smiles if s != INVALID_SMILES_SENTINEL)
    log.info(
        f"Successfully converted {valid_count}/{len(generated_smiles)} graphs to SMILES"
    )

    # Core-only mode: compute only validity, uniqueness, novelty (no FCD, PGD, motif, etc.)
    if cfg.metrics.get("core_only", False):
        reference_split = cfg.metrics.get("reference_split", "test")
        train_smiles = list(datamodule.train_smiles) if hasattr(datamodule, "train_smiles") else []
        mol_metrics = MolecularMetrics(
            reference_smiles=[],  # not used for validity/uniqueness/novelty
            train_smiles=train_smiles,
        )
        mol_results = mol_metrics(generated_smiles)
        log.info("Core metrics (core_only mode):")
        for name in ("validity", "uniqueness", "novelty"):
            log.info(f"  {name:20s}: {mol_results.get(name, 0):.6f}")
        all_results = {
            "validity": mol_results.get("validity"),
            "uniqueness": mol_results.get("uniqueness"),
            "novelty": mol_results.get("novelty"),
            "generation_time": gen_time,
            "num_samples": num_samples,
            "num_valid_smiles": valid_count,
            "reference_split": reference_split,
        }
        output_path = Path(cfg.logs.path)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\nResults saved to {results_file}")
        smiles_file = output_path / "generated_smiles.txt"
        with open(smiles_file, "w") as f:
            for smi in generated_smiles:
                if smi != INVALID_SMILES_SENTINEL:
                    f.write(smi + "\n")
        log.info(f"Generated SMILES saved to {smiles_file}")
        log.info("\nEvaluation complete (core only: validity, uniqueness, novelty).")
        return

    # Visualization (if enabled)
    if cfg.get("visualization", {}).get("enabled", False):
        log.info("Generating molecule visualizations...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        visualize_generated_molecules(
            generated_graphs=generated_graphs,
            generated_smiles=generated_smiles,
            tokenizer=tokenizer,
            output_dir=viz_dir,
            max_molecules=cfg.visualization.get("max_molecules", 12),
            dpi=cfg.visualization.get("dpi", 150),
        )
        log.info(f"Visualizations saved to {viz_dir}")

    # Reference sets for metrics
    # Distributional metrics (FCD, SNN, Frag, Scaf) use test or train+test as reference
    # Novelty always uses training set (measures memorization)
    ref_size = cfg.metrics.get("reference_size", 100000)
    reference_split = cfg.metrics.get("reference_split", "test")
    train_smiles = datamodule.train_smiles

    if reference_split == "full":
        if len(train_smiles) == 0:
            log.warning(
                "reference_split='full' but train_smiles is empty; "
                "falling back to test-only reference"
            )
            reference_smiles = datamodule.test_smiles[:ref_size]
        else:
            combined = list(train_smiles) + list(datamodule.test_smiles)
            random.Random(cfg.seed).shuffle(combined)
            reference_smiles = combined[:ref_size]
        ref_label = "train+test"
    else:
        reference_smiles = datamodule.test_smiles[:ref_size]
        ref_label = "test"

    log.info(
        f"Using {len(reference_smiles)} {ref_label} SMILES for distributional metrics"
    )
    log.info(f"Using {len(train_smiles)} train SMILES for novelty")

    log.info("\n" + "=" * 50)
    log.info("MOLECULAR METRICS")
    log.info("=" * 50)

    mol_metrics = MolecularMetrics(
        reference_smiles=reference_smiles,
        train_smiles=train_smiles,
    )
    mol_results = mol_metrics(generated_smiles)

    for name, value in mol_results.items():
        log.info(f"  {name:20s}: {value:.6f}")

    # Motif Distribution Metrics
    log.info("\n" + "=" * 50)
    log.info("MOTIF DISTRIBUTION METRICS")
    log.info("=" * 50)

    motif_results = {}
    motif_summary = {}
    if cfg.metrics.get("compute_motif", True):  # Default enabled
        try:
            motif_metrics = MotifDistributionMetric(
                reference_smiles=reference_smiles,
            )
            motif_results = motif_metrics(generated_smiles)

            for name, value in motif_results.items():
                log.info(f"  {name:20s}: {value:.6f}")

            # Get motif summary for top 100 molecules
            motif_summary = motif_metrics.get_motif_summary(generated_smiles[:100])

        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"  Motif metrics computation failed: {e}")
            motif_results = {}
            motif_summary = {}
    else:
        log.info("  [Motif metrics computation disabled in config]")

    # PolyGraph Discrepancy Metric
    log.info("\n" + "=" * 50)
    log.info("POLYGRAPH DISCREPANCY METRIC")
    log.info("=" * 50)

    pgd_score = None
    if cfg.metrics.get("compute_pgd", True):  # Default enabled
        try:
            max_ref_size = cfg.metrics.get("pgd_reference_size", 100)
            ref_graphs_path = cfg.metrics.get("reference_graphs_path")

            if ref_graphs_path and Path(ref_graphs_path).exists():
                log.info(f"Loading precomputed reference graphs from {ref_graphs_path}")
                reference_graphs = torch.load(
                    ref_graphs_path, map_location="cpu", weights_only=False
                )
                log.info(f"Loaded {len(reference_graphs)} reference graphs")
            else:
                # Convert reference SMILES to graphs for PGD
                if reference_split == "full":
                    pgd_reference_smiles = reference_smiles[:max_ref_size]
                else:
                    pgd_reference_smiles = datamodule.test_smiles[:max_ref_size]

                log.info(
                    f"Converting {len(pgd_reference_smiles)} reference SMILES to graphs..."
                )
                reference_graphs = []
                for smi in tqdm(
                    pgd_reference_smiles, desc="Converting reference to graphs"
                ):
                    try:
                        g = smiles_to_graph(smi)
                        if g is not None and g.num_nodes > 0:
                            reference_graphs.append(g)
                    except Exception:
                        continue  # Skip invalid SMILES

                log.info(f"Successfully converted {len(reference_graphs)} reference graphs")

            if len(reference_graphs) > 0:
                polygraph_metric = PolygraphMetric(
                    reference_graphs=reference_graphs,
                    max_reference_size=max_ref_size,
                )
                polygraph_results = polygraph_metric(generated_graphs)
                pgd_score = polygraph_results.get("pgd")

                if pgd_score is not None:
                    log.info(f"  pgd                 : {pgd_score:.6f}")
                    log.info(
                        "  (Lower is better: <0.1 excellent, <0.3 good, <0.5 moderate)"
                    )
                else:
                    log.info("  PGD computation returned None")
            else:
                log.info("  No valid reference graphs - skipping PGD")
        except ImportError:
            log.info("  PolyGraph not installed - skipping PGD metric")
            log.info("  Install with: pip install polygraph-benchmark")
            pgd_score = None
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"  PGD computation failed: {e}")
            pgd_score = None
    else:
        log.info("  [PGD computation disabled in config]")

    # Try to compute FCD if available
    log.info("\n" + "=" * 50)
    log.info("FCD METRIC")
    log.info("=" * 50)

    fcd_score = None
    if cfg.metrics.get("compute_fcd", True):  # Default enabled
        try:
            fcd_score = compute_fcd(generated_smiles, reference_smiles)
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
    else:
        log.info("  [FCD computation disabled in config]")

    # Get motif summary for reference (if motif metrics were computed)
    if cfg.metrics.get("compute_motif", True) and len(motif_summary) > 0:
        log.info("\n" + "=" * 50)
        log.info("MOTIF SUMMARY (Top 10)")
        log.info("=" * 50)

        log.info("\nSMARTS Motifs found:")
        for name, count in list(motif_summary["smarts_motifs"].items())[:10]:
            log.info(f"  {name}: {count}")

        log.info("\nFunctional Groups found:")
        for name, count in list(motif_summary["functional_groups"].items())[:10]:
            log.info(f"  {name}: {count}")

    # Compile all results
    all_results = {
        **mol_results,
        **motif_results,
        "pgd": pgd_score,
        "fcd": fcd_score,
        "generation_time": gen_time,
        "num_samples": num_samples,
        "num_valid_smiles": valid_count,
        "reference_split": reference_split,
    }
    if token_lengths:
        all_results["token_lengths"] = token_lengths
        all_results["token_length_mean"] = float(statistics.mean(token_lengths))
        all_results["token_length_min"] = min(token_lengths)
        all_results["token_length_max"] = max(token_lengths)

    # Add motif summary only if computed
    if len(motif_summary) > 0:
        all_results["motif_summary"] = motif_summary

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
