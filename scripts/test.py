#!/usr/bin/env python
"""Testing script for evaluating trained molecular graph generation models.

This script loads a trained model and evaluates its generation quality
using molecular metrics (validity, uniqueness, novelty, FCD, etc.)
and motif distribution metrics.

Usage:
    python scripts/test.py model.checkpoint_path=/path/to/model.ckpt
    python scripts/test.py data.dataset_name=qm9 model.checkpoint_path=outputs/model.ckpt
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import MolecularDataModule
from src.data.molecular import graph_to_smiles
from src.evaluation.molecular_metrics import MolecularMetrics, compute_fcd
from src.evaluation.motif_distribution import MotifDistributionMetric
from src.models.transformer import GraphGeneratorModule
from src.tokenizers.sent import SENTTokenizer

log = logging.getLogger(__name__)


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

    log.info(f"Loading model from {cfg.model.checkpoint_path}...")
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

    # Convert to SMILES
    generated_smiles = []
    for g in generated_graphs:
        smiles = graph_to_smiles(g)
        if smiles:
            generated_smiles.append(smiles)
    log.info(f"Successfully converted {len(generated_smiles)} graphs to SMILES")

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

    fcd_score = compute_fcd(generated_smiles, datamodule.test_smiles)
    if not (fcd_score != fcd_score):  # Check for NaN
        log.info(f"  FCD: {fcd_score:.6f}")
    else:
        log.info("  FCD: Not available (install moses or fcd package)")

    # Compile all results
    all_results = {
        **mol_results,
        **motif_results,
        "fcd": fcd_score if not (fcd_score != fcd_score) else None,
        "generation_time": gen_time,
        "num_samples": num_samples,
        "num_valid_smiles": len(generated_smiles),
    }

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

    # Save results
    output_path = Path(cfg.logs.path)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {results_file}")

    # Save generated SMILES
    smiles_file = output_path / "generated_smiles.txt"
    with open(smiles_file, "w") as f:
        for smi in generated_smiles:
            f.write(smi + "\n")
    log.info(f"Generated SMILES saved to {smiles_file}")

    log.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
