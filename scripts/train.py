#!/usr/bin/env python
"""Training script for molecular graph generation models.

This script trains a transformer model on molecular graph data using
either SENT (flat) or H-SENT (hierarchical) tokenization.

Usage:
    python scripts/train.py
    python scripts/train.py data.dataset_name=qm9
    python scripts/train.py tokenizer.type=hsent  # Use hierarchical tokenization
    python scripts/train.py model.model_name=llama-s trainer.max_steps=200000
    python scripts/train.py wandb.enabled=true wandb.project=my-project
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import MolecularDataModule
from src.evaluation.molecular_metrics import MolecularMetrics
from src.evaluation.motif_distribution import MotifDistributionMetric
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import SENTTokenizer, HSENTTokenizer

log = logging.getLogger(__name__)


def setup_wandb_logger(cfg: DictConfig) -> Optional[pl.loggers.WandbLogger]:
    """Set up Weights & Biases logger with full configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        WandbLogger instance or None if disabled.
    """
    if not cfg.wandb.enabled:
        return None

    run_name = cfg.wandb.name
    if run_name is None:
        run_name = f"{cfg.model.model_name}_{cfg.data.dataset_name}_s{cfg.seed}"

    tags = list(cfg.wandb.tags) if cfg.wandb.tags else []
    tags.append(cfg.model.model_name.split("-")[0])
    tags.append(cfg.data.dataset_name)

    wandb_logger = pl.loggers.WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=tags,
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.logs.path,
        log_model=cfg.wandb.log_model,
    )

    return wandb_logger


def log_generated_molecules_to_wandb(
    wandb_logger: pl.loggers.WandbLogger,
    smiles_list: list[str],
    prefix: str = "generated",
    max_molecules: int = 9,
) -> None:
    """Log generated molecule visualizations to WandB.

    Args:
        wandb_logger: WandB logger instance.
        smiles_list: List of generated SMILES strings.
        prefix: Prefix for the logged image name.
        max_molecules: Maximum number of molecules to visualize.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb
        from rdkit import Chem
        from rdkit.Chem import Draw

        valid_mols = []
        valid_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and len(valid_mols) < max_molecules:
                valid_mols.append(mol)
                valid_smiles.append(smiles)

        if not valid_mols:
            return

        # Draw molecules in a grid
        n_mols = len(valid_mols)
        n_cols = 3
        n_rows = (n_mols + n_cols - 1) // n_cols

        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=n_cols,
            subImgSize=(300, 300),
            legends=valid_smiles[:n_mols],
        )

        wandb_logger.experiment.log({f"{prefix}/molecules": wandb.Image(img)})

    except ImportError as e:
        log.warning(f"Could not log molecules to WandB: {e}")
    except Exception as e:
        log.warning(f"Error logging molecules to WandB: {e}")


def log_final_metrics_to_wandb(
    wandb_logger: pl.loggers.WandbLogger,
    metrics: dict,
    prefix: str = "final",
) -> None:
    """Log final evaluation metrics to WandB.

    Args:
        wandb_logger: WandB logger instance.
        metrics: Dictionary of metric names to values.
        prefix: Prefix for metric names.
    """
    if wandb_logger is None:
        return

    log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb_logger.experiment.log(log_dict)


def save_model_artifact(
    wandb_logger: pl.loggers.WandbLogger,
    checkpoint_path: str,
    artifact_name: str = "model",
    artifact_type: str = "model",
) -> None:
    """Save model checkpoint as WandB artifact.

    Args:
        wandb_logger: WandB logger instance.
        checkpoint_path: Path to the checkpoint file.
        artifact_name: Name for the artifact.
        artifact_type: Type of artifact.
    """
    if wandb_logger is None:
        return

    try:
        import wandb

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description="Trained molecular graph generation model",
        )
        artifact.add_file(checkpoint_path)
        wandb_logger.experiment.log_artifact(artifact)
        log.info(f"Saved model artifact: {artifact_name}")
    except Exception as e:
        log.warning(f"Could not save model artifact: {e}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.seed, workers=True)

    # Select tokenizer based on config
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()
    if tokenizer_type == "hsent":
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
            motif_aware=motif_aware,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
            seed=cfg.seed,
        )
    else:
        log.info("Using flat SENT tokenizer")
        tokenizer = SENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            undirected=cfg.tokenizer.get("undirected", True),
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

    datamodule.setup()

    model = GraphGeneratorModule(
        tokenizer=tokenizer,
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.model.max_steps,
        sampling_top_k=cfg.sampling.top_k,
        sampling_temperature=cfg.sampling.temperature,
        sampling_max_length=cfg.sampling.max_length,
        sampling_num_samples=cfg.sampling.num_samples,
        sampling_batch_size=cfg.sampling.batch_size,
    )

    loggers = []
    loggers.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    wandb_logger = setup_wandb_logger(cfg)
    if wandb_logger is not None:
        loggers.append(wandb_logger)
        log.info(f"WandB logging enabled: {cfg.wandb.project}")

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=cfg.logs.path,
            filename=cfg.model.model_name,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,
        callbacks=callbacks,
    )

    log.info("Starting training...")
    trainer.fit(model, datamodule)

    log.info("Saving final checkpoint...")
    final_checkpoint_path = f"{cfg.logs.path}/{cfg.model.model_name}-last.ckpt"
    trainer.save_checkpoint(final_checkpoint_path)

    if wandb_logger is not None and cfg.wandb.log_model:
        save_model_artifact(
            wandb_logger,
            final_checkpoint_path,
            artifact_name=f"{cfg.model.model_name}-final",
        )

    log.info("Running evaluation on test set...")
    trainer.test(model, datamodule)

    log.info("Generating samples for evaluation...")
    model.eval()
    generated_graphs, gen_time = model.generate(num_samples=cfg.sampling.num_samples)
    log.info(f"Generated {len(generated_graphs)} graphs in {gen_time:.4f}s per sample")

    # Convert generated graphs to SMILES for molecular metrics
    from src.data.molecular import graph_to_smiles

    generated_smiles = []
    for g in generated_graphs:
        smiles = graph_to_smiles(g)
        if smiles:
            generated_smiles.append(smiles)
    log.info(f"Successfully converted {len(generated_smiles)} graphs to SMILES")

    if wandb_logger is not None and cfg.wandb.log_graphs:
        log.info("Logging generated molecules to WandB...")
        log_generated_molecules_to_wandb(
            wandb_logger, generated_smiles, prefix="final"
        )

    log.info("Computing molecular metrics...")
    mol_metrics = MolecularMetrics(reference_smiles=datamodule.train_smiles)
    metrics = mol_metrics(generated_smiles)
    for name, value in metrics.items():
        log.info(f"  {name}: {value:.6f}")

    log.info("Computing motif distribution metrics...")
    motif_metrics = MotifDistributionMetric(reference_smiles=datamodule.train_smiles)
    motif_results = motif_metrics(generated_smiles)
    for name, value in motif_results.items():
        log.info(f"  {name}: {value:.6f}")

    all_metrics = {
        **metrics,
        **motif_results,
        "generation_time": gen_time,
    }
    log_final_metrics_to_wandb(wandb_logger, all_metrics)

    if wandb_logger is not None:
        import wandb
        wandb.finish()
        log.info("WandB run finished")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
