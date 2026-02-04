#!/usr/bin/env python
"""Evaluation script for fine-tuned models on COCONUT.

This script evaluates fine-tuned models by comparing the distribution of
generated molecules to the reference COCONUT distribution.

Usage:
    python scripts/eval_finetune.py \
        model.checkpoint_path=outputs/coconut_finetune/best.ckpt

    python scripts/eval_finetune.py \
        model.checkpoint_path=outputs/coconut_finetune/best.ckpt \
        generation.num_samples=1000 \
        data.n_reference=1000
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.coconut_loader import CoconutLoader
from src.data.molecular import graph_to_smiles, load_moses_dataset
from src.evaluation.molecular_metrics import MolecularMetrics
from src.evaluation.motif_distribution import (
    MotifCooccurrenceMetric,
    MotifDistributionMetric,
    MotifHistogramMetric,
)
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_finetune")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Evaluation configuration:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.seed, workers=True)

    # Create output directory
    output_dir = Path(cfg.logs.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference COCONUT molecules
    log.info("Loading reference COCONUT molecules...")
    loader = CoconutLoader(
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
        data_file=cfg.data.reference_file,
    )
    reference_smiles = loader.load_smiles(n_samples=cfg.data.n_reference, seed=cfg.seed)
    log.info(f"Loaded {len(reference_smiles)} reference molecules")

    # Select tokenizer
    tokenizer_type = cfg.tokenizer.get("type", "hdtc").lower()
    log.info(f"Using {tokenizer_type.upper()} tokenizer")

    if tokenizer_type == "hdtc":
        tokenizer = HDTCTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            include_rings=cfg.tokenizer.get("include_rings", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hdt":
        tokenizer = HDTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", "spectral"),
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hsent":
        tokenizer = HSENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", "spectral"),
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    else:
        tokenizer = SENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            undirected=cfg.tokenizer.get("undirected", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", False),
            seed=cfg.seed,
        )

    # Set max num nodes and labeled graph types to match training setup
    tokenizer.set_num_nodes(cfg.data.get("max_atoms", 100))
    if hasattr(tokenizer, "labeled_graph") and tokenizer.labeled_graph:
        from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES

        tokenizer.set_num_node_and_edge_types(
            num_node_types=NUM_ATOM_TYPES,
            num_edge_types=NUM_BOND_TYPES,
        )

    # Load model
    log.info(f"Loading model from {cfg.model.checkpoint_path}")
    checkpoint = torch.load(cfg.model.checkpoint_path, map_location="cpu")

    model = GraphGeneratorModule(
        tokenizer=tokenizer,
        model_name=cfg.model.get("model_name", "gpt2-xs"),
        sampling_top_k=cfg.sampling.top_k,
        sampling_temperature=cfg.sampling.temperature,
        sampling_max_length=cfg.sampling.max_length,
        sampling_batch_size=cfg.generation.batch_size,
    )

    # Load state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle vocabulary size mismatch by resizing embeddings
    new_vocab_size = len(tokenizer)
    wte_key = "model.model.transformer.wte.weight"
    lm_head_key = "model.model.lm_head.weight"

    if wte_key in state_dict:
        old_vocab_size = state_dict[wte_key].shape[0]
        if old_vocab_size != new_vocab_size:
            log.info(
                f"Resizing embeddings: {old_vocab_size} -> {new_vocab_size} tokens"
            )
            hidden_size = state_dict[wte_key].shape[1]

            # Create new embedding weights with random init for new tokens
            new_wte = torch.randn(new_vocab_size, hidden_size) * 0.02
            new_lm_head = torch.randn(new_vocab_size, hidden_size) * 0.02

            # Copy pretrained weights for existing tokens
            copy_size = min(old_vocab_size, new_vocab_size)
            new_wte[:copy_size] = state_dict[wte_key][:copy_size]
            if lm_head_key in state_dict:
                new_lm_head[:copy_size] = state_dict[lm_head_key][:copy_size]

            state_dict[wte_key] = new_wte
            state_dict[lm_head_key] = new_lm_head
            log.info(f"  Copied {copy_size} pretrained token embeddings")

    # Handle position embedding size mismatch
    wpe_key = "model.model.transformer.wpe.weight"
    new_max_positions = cfg.sampling.max_length
    if wpe_key in state_dict:
        old_max_positions = state_dict[wpe_key].shape[0]
        if old_max_positions != new_max_positions:
            log.info(
                f"Resizing position embeddings: {old_max_positions} -> {new_max_positions}"
            )
            hidden_size = state_dict[wpe_key].shape[1]

            # Create new position embeddings
            new_wpe = torch.randn(new_max_positions, hidden_size) * 0.02

            # Copy pretrained position embeddings
            copy_size = min(old_max_positions, new_max_positions)
            new_wpe[:copy_size] = state_dict[wpe_key][:copy_size]

            state_dict[wpe_key] = new_wpe
            log.info(f"  Copied {copy_size} pretrained position embeddings")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        log.warning(f"Unexpected keys: {len(unexpected)}")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    log.info(f"Model loaded on {device}")

    # Generate molecules
    log.info(f"Generating {cfg.generation.num_samples} molecules...")
    generated_graphs, gen_time = model.generate(
        num_samples=cfg.generation.num_samples, show_progress=True
    )
    log.info(f"Generated {len(generated_graphs)} graphs in {gen_time:.4f}s per sample")

    # Convert to SMILES
    INVALID = "INVALID"
    generated_smiles = []
    for g in tqdm(generated_graphs, desc="Converting to SMILES"):
        smiles = graph_to_smiles(g)
        generated_smiles.append(smiles if smiles else INVALID)

    valid_smiles = [s for s in generated_smiles if s != INVALID]
    log.info(f"Valid molecules: {len(valid_smiles)}/{len(generated_smiles)}")

    # Initialize results
    results = {
        "num_generated": len(generated_smiles),
        "num_valid": len(valid_smiles),
        "validity": len(valid_smiles) / max(len(generated_smiles), 1),
        "generation_time_per_sample": gen_time,
    }

    # Compute molecular metrics
    if cfg.metrics.get("validity", True):
        log.info("Computing molecular metrics...")
        mol_metrics = MolecularMetrics(reference_smiles=reference_smiles)
        mol_results = mol_metrics(generated_smiles)
        results.update(mol_results)
        for name, value in mol_results.items():
            log.info(f"  {name}: {value:.6f}")

    # Compute motif distribution metrics
    if cfg.metrics.get("motif_mmd", True):
        log.info("Computing motif distribution metrics...")
        motif_metrics = MotifDistributionMetric(reference_smiles=reference_smiles)
        motif_results = motif_metrics(generated_smiles)
        results.update(motif_results)
        for name, value in motif_results.items():
            log.info(f"  {name}: {value:.6f}")

    # Compute motif histogram metrics (vs COCONUT reference)
    if cfg.metrics.get("motif_histogram", True):
        log.info("Computing motif histogram metrics (vs COCONUT)...")
        hist_metrics = MotifHistogramMetric(
            reference_smiles=reference_smiles, distance_fn="kl"
        )
        hist_results = hist_metrics(generated_smiles)
        results["motif_hist_mean"] = hist_results["motif_hist_mean"]
        results["motif_hist_max"] = hist_results["motif_hist_max"]
        log.info(f"  motif_hist_mean (COCONUT): {hist_results['motif_hist_mean']:.6f}")
        log.info(f"  motif_hist_max (COCONUT): {hist_results['motif_hist_max']:.6f}")

        # Also compute vs MOSES reference to measure domain shift
        log.info("Computing motif histogram metrics (vs MOSES)...")
        moses_smiles = load_moses_dataset(
            split="train", max_molecules=cfg.data.n_reference, seed=cfg.seed
        )
        hist_metrics_moses = MotifHistogramMetric(
            reference_smiles=moses_smiles, distance_fn="kl"
        )
        hist_results_moses = hist_metrics_moses(generated_smiles)
        results["motif_hist_mean_moses"] = hist_results_moses["motif_hist_mean"]
        results["motif_hist_max_moses"] = hist_results_moses["motif_hist_max"]
        log.info(f"  motif_hist_mean (MOSES): {hist_results_moses['motif_hist_mean']:.6f}")
        log.info(f"  motif_hist_max (MOSES): {hist_results_moses['motif_hist_max']:.6f}")

    # Compute motif co-occurrence metrics
    if cfg.metrics.get("motif_cooccurrence", True):
        log.info("Computing motif co-occurrence metrics...")
        cooccur_metrics = MotifCooccurrenceMetric(reference_smiles=reference_smiles)
        cooccur_results = cooccur_metrics(generated_smiles)
        results.update(cooccur_results)
        for name, value in cooccur_results.items():
            log.info(f"  {name}: {value:.6f}")

    # Save results
    if cfg.output.save_results:
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {results_path}")

    # Save generated SMILES
    if cfg.output.save_smiles:
        smiles_path = output_dir / "generated_smiles.txt"
        with open(smiles_path, "w") as f:
            for smiles in generated_smiles:
                f.write(f"{smiles}\n")
        log.info(f"Generated SMILES saved to {smiles_path}")

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"Checkpoint: {cfg.model.checkpoint_path}")
    log.info(f"Generated: {results['num_generated']} molecules")
    log.info(f"Valid: {results['num_valid']} ({results['validity'] * 100:.1f}%)")
    if "uniqueness" in results:
        log.info(f"Uniqueness: {results['uniqueness'] * 100:.1f}%")
    if "novelty" in results:
        log.info(f"Novelty: {results['novelty'] * 100:.1f}%")
    if "motif_mmd" in results:
        log.info(f"Motif MMD: {results['motif_mmd']:.6f}")
    log.info("=" * 60)

    log.info("Evaluation complete!")


if __name__ == "__main__":
    main()
