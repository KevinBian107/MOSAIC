#!/usr/bin/env python
"""Realistic generation script.

This script generates molecules unconditionally and analyzes how well
they match the structural patterns of training data.

Usage:
    # Generate and analyze with HDT
    python scripts/realistic_gen.py \
        model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt

    # Generate and analyze with SENT
    python scripts/realistic_gen.py \
        model.checkpoint_path=outputs/train/moses_sent_*/best.ckpt \
        tokenizer=sent

    # Custom number of samples
    python scripts/realistic_gen.py \
        model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
        generation.num_samples=500
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress RDKit error messages
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.datamodule import MolecularDataModule  # noqa: E402
from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES, graph_to_smiles  # noqa: E402
from src.models.transformer import GraphGeneratorModule  # noqa: E402
from src.realistic_gen import (  # noqa: E402
    analyze_benzene_substitution,
    analyze_functional_groups,
    compare_distributions,
    draw_molecule_comparison,
    generate_molecules,
    plot_combined_analysis,
)
from src.tokenizers import (  # noqa: E402
    HDTCTokenizer,
    HDTTokenizer,
    HSENTTokenizer,
    SENTTokenizer,
)

log = logging.getLogger(__name__)


def extract_training_info(checkpoint_path: str) -> dict:
    """Extract training metadata from a checkpoint for results reporting.

    Reads global_step from the checkpoint and attempts to recover
    effective_batch_size from the co-located training config.yaml.
    Returns a dict with global_step, effective_batch_size, and samples_seen
    (or None for fields that can't be recovered).
    """
    info: dict = {
        "global_step": None,
        "effective_batch_size": None,
        "samples_seen": None,
    }
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        info["global_step"] = ckpt.get("global_step")
    except Exception as e:
        log.warning(f"Could not read global_step from checkpoint: {e}")
        return info

    # Try to recover effective_batch_size from co-located config.yaml
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists():
        try:
            train_cfg = OmegaConf.load(config_path)
            batch_size = OmegaConf.select(train_cfg, "data.batch_size", default=None)
            devices = OmegaConf.select(train_cfg, "trainer.devices", default=1)
            num_nodes = OmegaConf.select(train_cfg, "trainer.num_nodes", default=1)
            accum = OmegaConf.select(
                train_cfg, "trainer.accumulate_grad_batches", default=1
            )
            if batch_size is not None:
                eff = int(batch_size) * int(devices) * int(num_nodes) * int(accum)
                info["effective_batch_size"] = eff
        except Exception as e:
            log.warning(f"Could not parse training config for effective_batch_size: {e}")

    # Compute samples_seen if both pieces are available
    if info["global_step"] is not None and info["effective_batch_size"] is not None:
        info["samples_seen"] = info["global_step"] * info["effective_batch_size"]

    return info


def _resolve_coarsening_strategy(cfg: DictConfig, tokenizer_name: str) -> tuple[str, bool]:
    """Resolve coarsening strategy with backward-compatible motif_aware fallback."""
    strategy = cfg.tokenizer.get("coarsening_strategy")
    motif_aware = bool(cfg.tokenizer.get("motif_aware", False))

    if strategy is None:
        if motif_aware:
            strategy = "motif_aware_spectral"
            log.warning(
                "tokenizer.motif_aware is deprecated for %s. "
                "Using tokenizer.coarsening_strategy=motif_aware_spectral.",
                tokenizer_name,
            )
        else:
            strategy = "spectral"
    elif motif_aware and strategy != "motif_aware_spectral":
        log.warning(
            "Both tokenizer.motif_aware=true and tokenizer.coarsening_strategy=%s "
            "are set for %s. Using coarsening_strategy and ignoring motif_aware.",
            strategy,
            tokenizer_name,
        )

    return strategy, motif_aware


def get_tokenizer(cfg: DictConfig):
    """Create tokenizer based on configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        Configured tokenizer instance.
    """
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()

    if tokenizer_type == "hdt":
        coarsening_strategy, motif_aware = _resolve_coarsening_strategy(cfg, "hdt")
        log.info(f"Using HDT tokenizer with {coarsening_strategy} coarsening")
        tokenizer = HDTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=coarsening_strategy,
            motif_aware=motif_aware,
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hsent":
        coarsening_strategy, motif_aware = _resolve_coarsening_strategy(cfg, "hsent")
        log.info(
            f"Using hierarchical H-SENT tokenizer with {coarsening_strategy} coarsening"
        )
        if coarsening_strategy in ("motif_aware_spectral", "motif_community"):
            log.info(f"  motif_alpha: {cfg.tokenizer.get('motif_alpha', 1.0)}")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  min_community_size: {cfg.tokenizer.get('min_community_size', 4)}")

        tokenizer = HSENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=coarsening_strategy,
            motif_aware=motif_aware,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
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
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer


def configure_tokenizer_from_checkpoint(
    tokenizer,
    checkpoint_path: str,
) -> Optional[int]:
    """Configure tokenizer vocab size from checkpoint.

    Args:
        tokenizer: Tokenizer instance to configure.
        checkpoint_path: Path to model checkpoint.

    Returns:
        Max position embedding length from checkpoint, or None if not found.
    """
    log.info(f"Extracting vocab size from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_max_length = None

    if "state_dict" in checkpoint:
        wte_key = "model.model.transformer.wte.weight"
        if wte_key in checkpoint["state_dict"]:
            checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
            log.info(f"Checkpoint vocab size: {checkpoint_vocab_size}")

            idx_offset = getattr(tokenizer, "idx_offset", None) or getattr(
                tokenizer, "IDX_OFFSET", 6
            )

            is_labeled = getattr(tokenizer, "labeled_graph", False)

            if is_labeled:
                checkpoint_max_num_nodes = (
                    checkpoint_vocab_size - idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES
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
                        NUM_ATOM_TYPES, NUM_BOND_TYPES
                    )
                    log.info(
                        f"Set tokenizer: max_num_nodes={checkpoint_max_num_nodes}, "
                        f"labeled_graph=True"
                    )
                else:
                    tokenizer.max_num_nodes = checkpoint_max_num_nodes
                    log.info(
                        f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}"
                    )
            else:
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

    return checkpoint_max_length


def filter_by_motif(smiles_list: list[str], motif_smiles: str) -> list[str]:
    """Filter SMILES to only those containing the motif.

    Args:
        smiles_list: List of SMILES strings.
        motif_smiles: SMILES of the motif to filter by.

    Returns:
        Filtered list of SMILES.
    """
    motif = Chem.MolFromSmiles(motif_smiles)
    if motif is None:
        return smiles_list

    filtered = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.HasSubstructMatch(motif):
            filtered.append(smi)

    return filtered


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="realistic_gen",
)
def main(cfg: DictConfig) -> None:
    """Main realistic generation function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.checkpoint_path is None:
        raise ValueError("model.checkpoint_path must be specified")

    # Extract training metadata (global_step, effective_batch_size, samples_seen)
    training_info = extract_training_info(cfg.model.checkpoint_path)
    if training_info["global_step"] is not None:
        log.info(f"Checkpoint global_step: {training_info['global_step']:,}")
    if training_info["effective_batch_size"] is not None:
        log.info(f"Training effective_batch_size: {training_info['effective_batch_size']}")
    if training_info["samples_seen"] is not None:
        log.info(f"Training samples_seen: {training_info['samples_seen']:,}")

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
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()

    # Load test data for reference comparison (before model loading so
    # datamodule.setup() doesn't mutate the tokenizer after the model is built)
    log.info("Loading test data for reference comparison...")
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
        data_file=cfg.data.get("data_file", None),
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
        use_precomputed_smiles=cfg.data.get("use_precomputed_smiles", False),
        precomputed_smiles_dir=cfg.data.get("precomputed_smiles_dir", None),
    )
    datamodule.setup(stage="test")

    # Build reference set based on reference_split config.
    # Cap the reference pool to avoid spending hours on RDKit analysis
    # of 500K molecules.  Default cap = 5000 (aligned with COCONUT scale).
    reference_split = cfg.get("metrics", {}).get("reference_split", "test")
    reference_cap = int(cfg.get("metrics", {}).get("reference_size", 5000))
    train_smiles = datamodule.train_smiles

    if reference_split == "full":
        if len(train_smiles) == 0:
            log.warning(
                "reference_split='full' but train_smiles is empty; "
                "falling back to test-only reference"
            )
            ref_smiles = list(datamodule.test_smiles)
        else:
            combined = list(train_smiles) + list(datamodule.test_smiles)
            random.Random(cfg.seed).shuffle(combined)
            ref_smiles = combined[:reference_cap]
        ref_label = "train+test"
    else:
        ref_smiles = list(datamodule.test_smiles)[:reference_cap]
        ref_label = "test"

    log.info(f"Loaded {len(ref_smiles)} {ref_label} SMILES for reference (cap={reference_cap})")
    log.info("=" * 70)
    log.info("RESOLVED REALISTIC_GEN CONFIG SUMMARY")
    log.info("=" * 70)
    log.info(f"Dataset: {cfg.data.dataset_name}")
    log.info(f"Tokenizer: {tokenizer_type}")
    if tokenizer_type in ("hdt", "hsent"):
        log.info(
            f"Coarsening strategy: {cfg.tokenizer.get('coarsening_strategy', 'spectral')}"
        )
    log.info(
        "Generation: "
        f"num_samples={cfg.generation.num_samples}, batch_size={cfg.generation.batch_size}"
    )
    log.info(
        "Sampling: "
        f"top_k={cfg.sampling.top_k}, temperature={cfg.sampling.temperature}, "
        f"max_length={cfg.sampling.max_length}"
    )
    log.info(
        "Reference: "
        f"split={reference_split}, reference_count={len(ref_smiles)}"
    )
    log.info("=" * 70)

    reuse_generated = bool(cfg.generation.get("reuse_generated_smiles", False))
    generated_smiles_path_cfg = cfg.generation.get("generated_smiles_path")

    generated_graphs = []
    generated_smiles = []
    gen_time = 0.0
    num_attempted = 0

    if reuse_generated:
        if generated_smiles_path_cfg:
            smiles_file = Path(generated_smiles_path_cfg)
        else:
            smiles_file = output_dir / "generated_smiles.txt"
        if not smiles_file.exists():
            raise FileNotFoundError(
                f"reuse_generated_smiles=true but file not found: {smiles_file}"
            )
        log.info("\n" + "=" * 60)
        log.info("REUSING GENERATED SMILES")
        log.info("=" * 60)
        with open(smiles_file, "r") as f:
            generated_smiles = [line.strip() for line in f if line.strip()]
        log.info(f"Loaded {len(generated_smiles)} generated SMILES from {smiles_file}")

        # Require explicit attempted-count metadata from the source directory
        # so denominator semantics are preserved exactly in reuse mode.
        source_metadata_file = smiles_file.parent / "generated_metadata.json"
        if not source_metadata_file.exists():
            raise FileNotFoundError(
                "reuse_generated_smiles=true requires generated_metadata.json next to the source SMILES file "
                f"(missing: {source_metadata_file})."
            )
        with open(source_metadata_file, "r") as f:
            metadata = json.load(f)
        num_attempted = int(metadata.get("num_attempted"))
        metadata_num_valid = int(metadata.get("num_valid", len(generated_smiles)))
        if metadata_num_valid != len(generated_smiles):
            raise ValueError(
                "reuse artifact mismatch: generated_metadata.json num_valid "
                f"({metadata_num_valid}) != loaded valid SMILES count ({len(generated_smiles)}) from {smiles_file}."
            )
        log.info(
            f"Loaded attempted-count denominator from {source_metadata_file} "
            f"(num_attempted={num_attempted})"
        )

        # Keep a local copy in output_dir for consistency.
        out_smiles_file = output_dir / "generated_smiles.txt"
        if smiles_file.resolve() != out_smiles_file.resolve():
            with open(out_smiles_file, "w") as f:
                for smi in generated_smiles:
                    f.write(smi + "\n")
            log.info(f"Copied generated SMILES to {out_smiles_file}")
        metadata_file = output_dir / "generated_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "num_attempted": int(num_attempted),
                    "num_valid": int(len(generated_smiles)),
                },
                f,
                indent=2,
            )
        log.info(f"Generated metadata saved to {metadata_file}")
    else:
        # Configure tokenizer from checkpoint (force-corrects max_num_nodes
        # after any inflation by datamodule.setup())
        checkpoint_max_length = configure_tokenizer_from_checkpoint(
            tokenizer, cfg.model.checkpoint_path
        )

        # Load model
        log.info(f"Loading model from {cfg.model.checkpoint_path}...")
        model = GraphGeneratorModule.load_from_checkpoint(
            cfg.model.checkpoint_path,
            tokenizer=tokenizer,
            sampling_batch_size=cfg.generation.batch_size,
            sampling_top_k=cfg.sampling.top_k,
            sampling_temperature=cfg.sampling.temperature,
            sampling_max_length=checkpoint_max_length or cfg.sampling.max_length,
        )
        model.eval()

        # Generate molecules
        num_samples = cfg.generation.num_samples
        num_attempted = int(num_samples)
        log.info("\n" + "=" * 60)
        log.info("GENERATION")
        log.info("=" * 60)
        log.info(f"Generating {num_samples} molecules...")

        generated_graphs, gen_time = generate_molecules(
            model=model,
            num_samples=num_samples,
            show_progress=True,
        )
        log.info(f"Generated {len(generated_graphs)} graphs")
        log.info(f"Average time: {gen_time:.4f}s per sample")

        # Convert to SMILES
        generated_smiles = []
        for g in generated_graphs:
            smiles = graph_to_smiles(g)
            if smiles:
                generated_smiles.append(smiles)

        log.info(f"Valid SMILES: {len(generated_smiles)}/{len(generated_graphs)}")

        # Save generated SMILES
        smiles_file = output_dir / "generated_smiles.txt"
        with open(smiles_file, "w") as f:
            for smi in generated_smiles:
                f.write(smi + "\n")
        log.info(f"Generated SMILES saved to {smiles_file}")
        metadata_file = output_dir / "generated_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "num_attempted": int(num_attempted),
                    "num_valid": int(len(generated_smiles)),
                },
                f,
                indent=2,
            )
        log.info(f"Generated metadata saved to {metadata_file}")

    # Filter by motif for analysis
    motif_smiles = cfg.analysis.motif_smiles
    log.info(f"\nFiltering for motif: {motif_smiles}")

    ref_filtered = filter_by_motif(ref_smiles, motif_smiles)
    gen_filtered = filter_by_motif(generated_smiles, motif_smiles)

    log.info(
        f"  Reference ({ref_label}): {len(ref_filtered)}/{len(ref_smiles)} contain motif"
    )
    log.info(f"  Generated: {len(gen_filtered)}/{len(generated_smiles)} contain motif")

    if len(ref_filtered) == 0 or len(gen_filtered) == 0:
        log.warning("Not enough molecules with motif for analysis")
        return

    # Analyze substitution patterns
    log.info("\n" + "=" * 60)
    log.info("SUBSTITUTION PATTERN ANALYSIS")
    log.info("=" * 60)

    ref_sub = analyze_benzene_substitution(ref_filtered)
    gen_sub = analyze_benzene_substitution(gen_filtered)

    log.info("\nSubstitution Count Distribution:")
    log.info(f"  {'Pattern':<15} {'Reference':>12} {'Generated':>12}")
    log.info(f"  {'-' * 15} {'-' * 12} {'-' * 12}")

    ref_total = sum(ref_sub["substitution_count"].values()) or 1
    gen_total = sum(gen_sub["substitution_count"].values()) or 1

    for pattern in ["unsubstituted", "mono", "di", "tri", "poly"]:
        ref_pct = 100 * ref_sub["substitution_count"].get(pattern, 0) / ref_total
        gen_pct = 100 * gen_sub["substitution_count"].get(pattern, 0) / gen_total
        log.info(f"  {pattern:<15} {ref_pct:>11.1f}% {gen_pct:>11.1f}%")

    log.info("\nDi-substitution Patterns (ortho/meta/para):")
    ref_di_total = sum(ref_sub["disubstitution_pattern"].values()) or 1
    gen_di_total = sum(gen_sub["disubstitution_pattern"].values()) or 1

    for pattern in ["ortho", "meta", "para"]:
        ref_pct = 100 * ref_sub["disubstitution_pattern"].get(pattern, 0) / ref_di_total
        gen_pct = 100 * gen_sub["disubstitution_pattern"].get(pattern, 0) / gen_di_total
        log.info(f"  {pattern:<15} {ref_pct:>11.1f}% {gen_pct:>11.1f}%")

    # Analyze functional groups
    log.info("\n" + "=" * 60)
    log.info("FUNCTIONAL GROUP ANALYSIS")
    log.info("=" * 60)

    ref_fg = analyze_functional_groups(ref_filtered)
    gen_fg = analyze_functional_groups(gen_filtered)

    log.info("\nFunctional Groups Attached to Benzene:")
    log.info(f"  {'Group':<20} {'Reference':>12} {'Generated':>12}")
    log.info(f"  {'-' * 20} {'-' * 12} {'-' * 12}")

    ref_fg_total = sum(ref_fg.values()) or 1
    gen_fg_total = sum(gen_fg.values()) or 1

    combined = ref_fg + gen_fg
    for group, _ in combined.most_common(12):
        ref_pct = 100 * ref_fg.get(group, 0) / ref_fg_total
        gen_pct = 100 * gen_fg.get(group, 0) / gen_fg_total
        log.info(f"  {group:<20} {ref_pct:>11.1f}% {gen_pct:>11.1f}%")

    # Compute similarity metrics
    log.info("\n" + "=" * 60)
    log.info("DISTRIBUTION SIMILARITY METRICS")
    log.info("=" * 60)

    sub_metrics = compare_distributions(
        ref_sub["substitution_count"],
        gen_sub["substitution_count"],
    )
    fg_metrics = compare_distributions(ref_fg, gen_fg)

    log.info("\nSubstitution Pattern Similarity:")
    log.info(f"  Total Variation Distance: {sub_metrics['total_variation']:.4f}")
    log.info(f"  KL Divergence: {sub_metrics['kl_divergence']:.4f}")

    log.info("\nFunctional Group Similarity:")
    log.info(f"  Total Variation Distance: {fg_metrics['total_variation']:.4f}")
    log.info(f"  KL Divergence: {fg_metrics['kl_divergence']:.4f}")

    log.info("\n(Lower values = more similar to reference distribution)")

    # Generate visualizations
    log.info("\n" + "=" * 60)
    log.info("GENERATING VISUALIZATIONS")
    log.info("=" * 60)

    # Bar chart analysis
    chart_path = output_dir / f"analysis_{tokenizer_type}.png"
    fig = plot_combined_analysis(
        ref_filtered,
        gen_filtered,
        output_path=str(chart_path),
        title_prefix=f"{tokenizer_type.upper()} Generation",
    )
    log.info(f"Bar chart saved to: {chart_path}")
    plt.close(fig)

    # Molecule structure visualizations
    log.info("Generating molecule structure visualizations...")
    mol_files = draw_molecule_comparison(
        ref_filtered,
        gen_filtered,
        output_dir=str(output_dir),
        tokenizer_name=tokenizer_type,
        motif_smiles=motif_smiles,
        seed=cfg.seed,
    )

    for viz_type, filepath in mol_files.items():
        log.info(f"  {viz_type}: {filepath}")

    # Save results
    results = {
        "tokenizer_type": tokenizer_type,
        "num_generated": int(num_attempted),
        "num_valid": len(generated_smiles),
        "num_with_motif": len(gen_filtered),
        "motif_rate": len(gen_filtered) / len(generated_smiles)
        if generated_smiles
        else 0,
        "generation_time": gen_time,
        "substitution_tv": sub_metrics["total_variation"],
        "substitution_kl": sub_metrics["kl_divergence"],
        "functional_group_tv": fg_metrics["total_variation"],
        "functional_group_kl": fg_metrics["kl_divergence"],
        "reference_split": reference_split,
        # Training metadata (from checkpoint + co-located config.yaml)
        "global_step": training_info["global_step"],
        "effective_batch_size": training_info["effective_batch_size"],
        "samples_seen": training_info["samples_seen"],
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {results_file}")

    log.info("\nRealistic generation complete!")


if __name__ == "__main__":
    main()
