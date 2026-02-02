#!/usr/bin/env python
"""Scaffold-primed generation script.

This script generates molecules starting from scaffold structures using trained
models. It enables zero-shot generation of complex molecules by priming with
known structural motifs.

Usage:
    # Generate from a named scaffold
    python scripts/primed_gen.py \
        model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
        scaffold.name=naphthalene

    # Generate from custom SMILES
    python scripts/primed_gen.py \
        model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
        scaffold.smiles="c1ccc2ccccc2c1"

    # Generate from all Tier 2 scaffolds (fused bicyclic)
    python scripts/primed_gen.py \
        model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
        scaffold.tier=2

    # Use HSENT tokenizer
    python scripts/primed_gen.py \
        model.checkpoint_path=outputs/train/moses_hsent_*/best.ckpt \
        tokenizer=hsent \
        scaffold.name=carbazole
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

from src.data.molecular import (  # noqa: E402
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
from src.transfer_learning import PrimedGenerator  # noqa: E402

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
    """Main scaffold-primed generation function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.checkpoint_path is None:
        raise ValueError(
            "model.checkpoint_path must be specified. "
            "Example: model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt"
        )

    # Validate scaffold configuration
    scaffold_name = cfg.scaffold.get("name")
    scaffold_smiles = cfg.scaffold.get("smiles")
    scaffold_tier = cfg.scaffold.get("tier")

    if not any([scaffold_name, scaffold_smiles, scaffold_tier]):
        raise ValueError(
            "Must specify one of: scaffold.name, scaffold.smiles, or scaffold.tier. "
            "Example: scaffold.name=naphthalene"
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
    tokenizer_type = cfg.tokenizer.get("type", "hdt").lower()

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

    # List available scaffolds
    log.info(f"Available scaffolds: {len(generator.list_available_scaffolds())}")

    # Generate based on scaffold configuration
    num_samples = cfg.scaffold.num_samples
    priming_level = cfg.scaffold.get("priming_level", "scaffold_only")

    results = {}

    log.info("\n" + "=" * 60)
    log.info("SCAFFOLD-PRIMED GENERATION")
    log.info("=" * 60)

    if scaffold_name:
        # Generate from named scaffold
        log.info(f"\nGenerating {num_samples} samples from scaffold: {scaffold_name}")
        scaffold_info = generator.get_scaffold_info(scaffold_name)
        log.info(f"  SMILES: {scaffold_info['smiles']}")
        log.info(f"  Tier: {scaffold_info['tier']}")
        log.info(f"  Atoms: {scaffold_info['num_atoms']}")

        graphs, gen_time = generator.generate_from_scaffold(
            scaffold_name,
            num_samples=num_samples,
            priming_level=priming_level,
        )
        results[scaffold_name] = {
            "graphs": graphs,
            "time": gen_time,
            "info": scaffold_info,
        }

    elif scaffold_smiles:
        # Generate from custom SMILES
        log.info(f"\nGenerating {num_samples} samples from SMILES: {scaffold_smiles}")

        graphs, gen_time = generator.generate_from_smiles(
            scaffold_smiles,
            num_samples=num_samples,
            priming_level=priming_level,
        )
        results["custom"] = {
            "graphs": graphs,
            "time": gen_time,
            "info": {"smiles": scaffold_smiles, "tier": 0, "num_atoms": 0},
        }

    elif scaffold_tier:
        # Generate from all scaffolds in tier
        log.info(f"\nGenerating from all Tier {scaffold_tier} scaffolds")
        tier_results, total_time = generator.generate_by_tier(
            tier=scaffold_tier,
            samples_per_scaffold=num_samples,
        )

        for name, graphs in tier_results.items():
            scaffold_info = generator.get_scaffold_info(name)
            results[name] = {
                "graphs": graphs,
                "time": total_time / len(tier_results),
                "info": scaffold_info,
            }

    # Process and save results
    log.info("\n" + "=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    all_smiles = []
    summary = []

    for scaffold_name, data in results.items():
        graphs = data["graphs"]
        info = data["info"]

        # Convert to SMILES
        smiles_list = []
        for g in graphs:
            smiles = graph_to_smiles(g)
            if smiles:
                smiles_list.append(smiles)
                all_smiles.append(smiles)

        valid_rate = len(smiles_list) / len(graphs) if graphs else 0

        log.info(f"\n{scaffold_name}:")
        log.info(f"  Scaffold SMILES: {info.get('smiles', 'N/A')}")
        log.info(f"  Generated: {len(graphs)} graphs")
        log.info(f"  Valid SMILES: {len(smiles_list)} ({valid_rate:.1%})")
        log.info(f"  Time: {data['time']:.4f}s per sample")

        if smiles_list:
            log.info("  Examples:")
            for smi in smiles_list[:3]:
                log.info(f"    - {smi}")

        summary.append(
            {
                "scaffold": scaffold_name,
                "scaffold_smiles": info.get("smiles", ""),
                "tier": info.get("tier", 0),
                "num_generated": len(graphs),
                "num_valid": len(smiles_list),
                "valid_rate": valid_rate,
                "time_per_sample": data["time"],
            }
        )

    # Save generated SMILES
    if cfg.output.save_smiles and all_smiles:
        smiles_file = output_dir / "generated_smiles.txt"
        with open(smiles_file, "w") as f:
            for smi in all_smiles:
                f.write(smi + "\n")
        log.info(f"\nGenerated SMILES saved to {smiles_file}")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "tokenizer_type": tokenizer_type,
                "total_generated": sum(s["num_generated"] for s in summary),
                "total_valid": sum(s["num_valid"] for s in summary),
                "scaffolds": summary,
            },
            f,
            indent=2,
        )
    log.info(f"Summary saved to {summary_file}")

    log.info("\nScaffold-primed generation complete!")


if __name__ == "__main__":
    main()
