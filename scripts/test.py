#!/usr/bin/env python
"""Testing script for evaluating trained graph generation models.

This script loads a trained model and evaluates its generation quality
using both standard graph metrics and motif-specific metrics.

Usage:
    python scripts/test.py model.checkpoint_path=/path/to/model.ckpt
    python scripts/test.py experiment=synthetic model.checkpoint_path=outputs/model.ckpt
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

from src.data.datamodule import GraphDataModule
from src.evaluation.metrics import GraphMetrics, compute_validity_metrics
from src.evaluation.motif_metrics import MotifMetrics
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

    datamodule = GraphDataModule(
        generator_configs=OmegaConf.to_container(cfg.data.generators),
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_train=cfg.data.num_train,
        num_val=cfg.data.num_val,
        num_test=cfg.data.num_test,
        seed=cfg.seed,
    )

    datamodule.setup(stage="test")

    log.info(f"Loading model from {cfg.model.checkpoint_path}...")
    model = GraphGeneratorModule.load_from_checkpoint(
        cfg.model.checkpoint_path,
        tokenizer=tokenizer,
    )
    model.eval()

    test_graphs = [datamodule.test_dataset[i] for i in range(len(datamodule.test_dataset))]
    num_test = len(test_graphs)

    num_samples = cfg.sampling.num_samples
    if num_samples < 0:
        num_samples = num_test

    log.info(f"Generating {num_samples} graphs...")
    generated_graphs, gen_time = model.generate(num_samples=num_samples)
    log.info(f"Generated {len(generated_graphs)} graphs")
    log.info(f"Average generation time: {gen_time:.4f}s per sample")

    log.info("\n" + "=" * 50)
    log.info("STANDARD GRAPH METRICS")
    log.info("=" * 50)

    graph_metrics = GraphMetrics(
        reference_graphs=test_graphs,
        compute_emd=cfg.metrics.compute_emd,
        metrics_list=cfg.metrics.metrics_list,
    )
    standard_results = graph_metrics(generated_graphs)

    for name, value in standard_results.items():
        log.info(f"  {name:15s}: {value:.6f}")

    log.info("\n" + "=" * 50)
    log.info("VALIDITY METRICS")
    log.info("=" * 50)

    validity_results = compute_validity_metrics(generated_graphs, test_graphs)
    for name, value in validity_results.items():
        log.info(f"  {name:15s}: {value:.4f}")

    log.info("\n" + "=" * 50)
    log.info("MOTIF METRICS")
    log.info("=" * 50)

    motif_metrics = MotifMetrics(reference_graphs=test_graphs)
    motif_results = motif_metrics(generated_graphs)

    for name, value in motif_results.items():
        log.info(f"  {name:20s}: {value:.6f}")

    all_results = {
        **standard_results,
        **validity_results,
        **motif_results,
        "generation_time": gen_time,
        "num_samples": num_samples,
    }

    output_path = Path(cfg.logs.path)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {results_file}")

    log.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
