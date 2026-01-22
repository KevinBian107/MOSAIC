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
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_lightning.callbacks import Callback

from src.data.datamodule import MolecularDataModule
from src.data.molecular import graph_to_smiles
from src.evaluation.molecular_metrics import MolecularMetrics
from src.evaluation.motif_distribution import (
    MOLECULAR_MOTIFS,
    MotifCooccurrenceMetric,
    MotifDistributionMetric,
    MotifHistogramMetric,
    get_motif_counts,
)
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import HDTTokenizer, HSENTTokenizer, SENTTokenizer

log = logging.getLogger(__name__)

# Categorical colormap for motif types (organized by category)
MOTIF_COLORS = {
    # Aromatic rings - blues/purples
    "benzene": (0.2, 0.4, 0.8),
    "pyridine": (0.3, 0.3, 0.9),
    "pyrrole": (0.4, 0.2, 0.8),
    "furan": (0.5, 0.3, 0.7),
    "thiophene": (0.3, 0.5, 0.8),
    "imidazole": (0.4, 0.4, 0.9),
    "pyrimidine": (0.2, 0.5, 0.7),
    "naphthalene": (0.1, 0.3, 0.9),
    # Functional groups - greens/yellows
    "hydroxyl": (0.2, 0.8, 0.3),
    "carboxyl": (0.8, 0.6, 0.2),
    "carbonyl": (0.9, 0.7, 0.1),
    "aldehyde": (0.7, 0.8, 0.2),
    "ester": (0.6, 0.7, 0.3),
    "amide": (0.5, 0.8, 0.4),
    "amine_primary": (0.3, 0.9, 0.5),
    "amine_secondary": (0.4, 0.85, 0.5),
    "amine_tertiary": (0.5, 0.8, 0.5),
    "nitro": (0.9, 0.2, 0.2),
    "nitrile": (0.7, 0.3, 0.5),
    # Halogens - oranges/reds
    "halogen": (1.0, 0.5, 0.0),
    "fluorine": (0.9, 0.6, 0.1),
    "chlorine": (0.8, 0.5, 0.2),
    "bromine": (0.7, 0.4, 0.1),
    "iodine": (0.6, 0.3, 0.2),
    # Others - teals/cyans
    "ether": (0.2, 0.7, 0.7),
    "thioether": (0.3, 0.6, 0.6),
    "sulfone": (0.4, 0.5, 0.7),
    "sulfonamide": (0.5, 0.6, 0.8),
    "phosphate": (0.6, 0.4, 0.6),
}


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


def log_molecules_with_motifs_to_wandb(
    wandb_logger: pl.loggers.WandbLogger,
    smiles_list: list[str],
    prefix: str = "generated",
    max_molecules: int = 12,
) -> None:
    """Log generated molecules with color-coded motif highlighting to WandB.

    Creates:
    1. Grid image with per-motif-type colored highlights
    2. WandB Table with per-molecule metadata
    3. Color legend for motif types

    Args:
        wandb_logger: WandB logger instance.
        smiles_list: List of generated SMILES strings.
        prefix: Prefix for logged items.
        max_molecules: Maximum molecules to visualize.
    """
    if wandb_logger is None:
        return

    try:
        import wandb
        from rdkit import Chem
        from rdkit.Chem import Draw

        valid_data = []
        for smiles in smiles_list:
            if not smiles or smiles in ["INVALID", ""]:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and len(valid_data) < max_molecules:
                motifs = get_motif_counts(smiles)
                valid_data.append({
                    "smiles": smiles,
                    "mol": mol,
                    "motifs": motifs,
                })

        if not valid_data:
            return

        # --- 1. Create Grid with Color-Coded Motif Highlights ---
        mols = [d["mol"] for d in valid_data]
        legends = []
        highlight_atoms_list = []
        highlight_atom_colors_list = []

        for d in valid_data:
            mol = d["mol"]
            motifs = d["motifs"]

            # Collect atoms with per-motif colors
            highlight_atoms = []
            atom_colors = {}

            for motif_name in motifs:
                if motif_name in MOLECULAR_MOTIFS:
                    pattern = Chem.MolFromSmarts(MOLECULAR_MOTIFS[motif_name])
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        color = MOTIF_COLORS.get(motif_name, (0.5, 0.5, 0.5))
                        for match in matches:
                            for atom_idx in match:
                                if atom_idx not in atom_colors:
                                    highlight_atoms.append(atom_idx)
                                    atom_colors[atom_idx] = color

            highlight_atoms_list.append(highlight_atoms)
            highlight_atom_colors_list.append(atom_colors)

            # Create legend with top motifs
            top_motifs = sorted(motifs.items(), key=lambda x: -x[1])[:3]
            motif_str = ", ".join(f"{k}:{v}" for k, v in top_motifs)
            legends.append(f"{d['smiles'][:30]}\n{motif_str}")

        # Draw grid with color-coded highlights
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(350, 350),
            legends=legends,
            highlightAtomLists=highlight_atoms_list,
            highlightAtomColors=highlight_atom_colors_list,
        )

        wandb_logger.experiment.log({f"{prefix}/molecules_with_motifs": wandb.Image(img)})

        # --- 2. Create WandB Table with Metadata ---
        columns = ["smiles", "num_atoms", "num_motifs", "motif_list"]
        table_data = []

        for d in valid_data:
            mol = d["mol"]
            motifs = d["motifs"]

            table_data.append([
                d["smiles"],
                mol.GetNumAtoms(),
                sum(motifs.values()),
                ", ".join(f"{k}({v})" for k, v in motifs.items()),
            ])

        table = wandb.Table(columns=columns, data=table_data)
        wandb_logger.experiment.log({f"{prefix}/molecule_details": table})

        # --- 3. Log Color Legend ---
        motifs_found = set()
        for d in valid_data:
            motifs_found.update(d["motifs"].keys())
        _log_motif_color_legend(wandb_logger, motifs_found, prefix)

    except ImportError as e:
        log.warning(f"Could not log molecules with motifs to WandB: {e}")
    except Exception as e:
        log.warning(f"Error logging molecules with motifs to WandB: {e}")


def _log_motif_color_legend(
    wandb_logger: pl.loggers.WandbLogger,
    motifs_found: set[str],
    prefix: str,
) -> None:
    """Log a color legend for the motif types found.

    Args:
        wandb_logger: WandB logger instance.
        motifs_found: Set of motif names that were found.
        prefix: Prefix for logged items.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import wandb

        # Only show motifs that were actually found
        legend_items = []
        for motif_name in sorted(motifs_found):
            if motif_name in MOTIF_COLORS:
                color = MOTIF_COLORS[motif_name]
                patch = mpatches.Patch(color=color, label=motif_name)
                legend_items.append(patch)

        if not legend_items:
            return

        fig, ax = plt.subplots(figsize=(4, max(2, len(legend_items) * 0.3)))
        ax.axis("off")
        ax.legend(handles=legend_items, loc="center", frameon=False, fontsize=8)
        plt.tight_layout()

        wandb_logger.experiment.log({f"{prefix}/motif_color_legend": wandb.Image(fig)})
        plt.close(fig)

    except Exception as e:
        log.warning(f"Could not log motif color legend: {e}")


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


class IntermediateEvaluationCallback(Callback):
    """Callback to run generation and log metrics during training.

    Triggers every `eval_every_n_val` validation epochs to:
    1. Generate sample molecules
    2. Compute molecular metrics
    3. Compute motif metrics (including histogram and co-occurrence)
    4. Log molecules with color-coded motif highlights to WandB

    Attributes:
        tokenizer: Tokenizer for decoding generated tokens.
        reference_smiles: Training set SMILES for metric computation.
        eval_every_n_val: Run evaluation every N validation epochs.
        num_samples: Number of molecules to generate.
        max_logged_molecules: Maximum molecules to visualize.
        wandb_logger: WandB logger instance.
    """

    def __init__(
        self,
        tokenizer,
        reference_smiles: list[str],
        eval_every_n_val: int = 5,
        num_samples: int = 50,
        max_logged_molecules: int = 12,
        wandb_logger: Optional[pl.loggers.WandbLogger] = None,
    ) -> None:
        """Initialize the intermediate evaluation callback.

        Args:
            tokenizer: Tokenizer for decoding generated tokens.
            reference_smiles: Reference SMILES (typically training set).
            eval_every_n_val: Run evaluation every N validation epochs.
            num_samples: Number of molecules to generate per evaluation.
            max_logged_molecules: Maximum molecules to visualize in WandB.
            wandb_logger: WandB logger instance (optional).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.reference_smiles = reference_smiles
        self.eval_every_n_val = eval_every_n_val
        self.num_samples = num_samples
        self.max_logged_molecules = max_logged_molecules
        self.wandb_logger = wandb_logger

        self._val_epoch_count = 0

        # Metric evaluators (lazily initialized)
        self._mol_metrics: Optional[MolecularMetrics] = None
        self._motif_metrics: Optional[MotifDistributionMetric] = None
        self._hist_metrics: Optional[MotifHistogramMetric] = None
        self._cooccur_metrics: Optional[MotifCooccurrenceMetric] = None

    def _lazy_init_metrics(self) -> None:
        """Lazily initialize metric evaluators on first use."""
        if self._mol_metrics is not None:
            return

        self._mol_metrics = MolecularMetrics(self.reference_smiles)
        self._motif_metrics = MotifDistributionMetric(self.reference_smiles)
        self._hist_metrics = MotifHistogramMetric(
            self.reference_smiles, distance_fn="kl"
        )
        self._cooccur_metrics = MotifCooccurrenceMetric(self.reference_smiles)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Run intermediate evaluation after validation epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: The Lightning module being trained.
        """
        self._val_epoch_count += 1

        # Skip if not at evaluation interval
        if self._val_epoch_count % self.eval_every_n_val != 0:
            return

        # Skip if no WandB logger
        if self.wandb_logger is None:
            return

        self._lazy_init_metrics()

        log.info(
            f"Running intermediate evaluation (val epoch {self._val_epoch_count})..."
        )

        # Generate samples
        pl_module.eval()
        generated_graphs, gen_time = pl_module.generate(
            num_samples=self.num_samples, show_progress=True
        )

        # Convert to SMILES
        INVALID = "INVALID"
        generated_smiles = []
        for g in tqdm(generated_graphs, desc="Converting to SMILES"):
            smiles = graph_to_smiles(g)
            generated_smiles.append(smiles if smiles else INVALID)

        valid_count = sum(1 for s in generated_smiles if s != INVALID)
        log.info(f"  Generated {valid_count}/{len(generated_smiles)} valid molecules")

        # Compute metrics
        mol_results = self._mol_metrics(generated_smiles)
        motif_results = self._motif_metrics(generated_smiles)
        hist_results = self._hist_metrics(generated_smiles)
        cooccur_results = self._cooccur_metrics(generated_smiles)

        # Combine all metrics with intermediate prefix
        step = trainer.global_step
        all_metrics = {
            **{f"intermediate/{k}": v for k, v in mol_results.items()},
            **{f"intermediate/{k}": v for k, v in motif_results.items()},
            "intermediate/motif_hist_mean": hist_results["motif_hist_mean"],
            "intermediate/motif_hist_max": hist_results["motif_hist_max"],
            **{f"intermediate/{k}": v for k, v in cooccur_results.items()},
            "intermediate/generation_time": gen_time,
            "intermediate/valid_fraction": valid_count / max(len(generated_smiles), 1),
        }

        # Log metrics
        self.wandb_logger.experiment.log(all_metrics, step=step)

        # Log molecule visualizations with motif highlighting
        log_molecules_with_motifs_to_wandb(
            self.wandb_logger,
            generated_smiles,
            prefix="intermediate",
            max_molecules=self.max_logged_molecules,
        )

        log.info(f"  Logged intermediate metrics at step {step}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.seed, workers=True)

    # Create output directory and save configuration
    output_dir = Path(cfg.logs.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")

    # Check for checkpoint BEFORE loading data (saves time if resuming)
    ckpt_path = None
    if cfg.get("resume", True):
        output_dir = cfg.logs.path
        last_ckpt = os.path.join(output_dir, "last.ckpt")
        best_ckpt = os.path.join(output_dir, "best.ckpt")

        if os.path.exists(last_ckpt):
            log.info(f"✓ Found checkpoint: {last_ckpt}")
            log.info("Will resume training after setup...")
            ckpt_path = last_ckpt
        elif os.path.exists(best_ckpt):
            log.info(f"✓ Found checkpoint: {best_ckpt}")
            log.info("Will resume training after setup...")
            ckpt_path = best_ckpt
        else:
            log.info("No checkpoint found. Starting fresh training.")
    else:
        log.info("Resume disabled (resume=false). Starting fresh training.")

    # Now load data (expensive operation)
    log.info("Setting up dataset and tokenizer...")

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
            motif_aware=motif_aware,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
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
        num_train=cfg.data.num_train,
        num_val=cfg.data.num_val,
        num_test=cfg.data.num_test,
        include_hydrogens=cfg.data.get("include_hydrogens", False),
        seed=cfg.seed,
        data_root=cfg.data.get("data_root", "data"),
        use_cache=cfg.data.get("use_cache", False),
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
    )

    datamodule.setup()

    # Calculate steps per epoch and adjust val_check_interval if needed
    train_dataset_size = len(datamodule.train_dataset)
    steps_per_epoch = train_dataset_size // cfg.data.batch_size
    val_check_interval = cfg.trainer.val_check_interval

    # If val_check_interval exceeds steps per epoch, use steps per epoch (1 eval per epoch)
    if val_check_interval > steps_per_epoch:
        log.warning(
            f"val_check_interval ({val_check_interval}) exceeds steps per epoch ({steps_per_epoch}). "
            f"Setting val_check_interval={steps_per_epoch} (1 validation per epoch)"
        )
        val_check_interval = steps_per_epoch

    log.info(f"Training dataset size: {train_dataset_size:,} samples")
    log.info(f"Steps per epoch: {steps_per_epoch:,}")
    log.info(f"Validation check interval: {val_check_interval:,} steps")

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
            filename="best",  # Simple name for best checkpoint
            save_last=True,  # Saves last.ckpt automatically
            mode="min",
        ),
    ]

    # Add intermediate evaluation callback if WandB enabled and configured
    eval_every_n_val = cfg.wandb.get("eval_every_n_val", 0)
    if wandb_logger is not None and eval_every_n_val > 0:
        eval_callback = IntermediateEvaluationCallback(
            tokenizer=tokenizer,
            reference_smiles=datamodule.train_smiles,
            eval_every_n_val=eval_every_n_val,
            num_samples=cfg.wandb.get("eval_num_samples", 50),
            max_logged_molecules=cfg.wandb.get("max_logged_molecules", 12),
            wandb_logger=wandb_logger,
        )
        callbacks.append(eval_callback)
        log.info(
            f"Intermediate evaluation enabled every {eval_every_n_val} validation epochs"
        )

    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        val_check_interval=val_check_interval,  # Use calculated value
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,
        callbacks=callbacks,
    )

    # Checkpoint detection was already done earlier to save time
    if ckpt_path:
        log.info(f"Resuming training from: {ckpt_path}")
    else:
        log.info("Starting fresh training...")

    log.info("Starting training...")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    log.info("Running evaluation on test set...")
    trainer.test(model, datamodule)

    log.info("Generating samples for evaluation...")
    model.eval()
    generated_graphs, gen_time = model.generate(
        num_samples=cfg.sampling.num_samples, show_progress=True
    )
    log.info(f"Generated {len(generated_graphs)} graphs in {gen_time:.4f}s per sample")

    # Convert generated graphs to SMILES for molecular metrics
    # Use sentinel value for failed conversions to compute accurate metrics
    INVALID_SMILES_SENTINEL = "INVALID"
    generated_smiles = []
    for g in tqdm(generated_graphs, desc="Converting to SMILES"):
        smiles = graph_to_smiles(g)
        generated_smiles.append(smiles if smiles else INVALID_SMILES_SENTINEL)

    valid_count = sum(1 for s in generated_smiles if s != INVALID_SMILES_SENTINEL)
    log.info(f"Successfully converted {valid_count}/{len(generated_smiles)} graphs to SMILES")

    if wandb_logger is not None and cfg.wandb.log_graphs:
        log.info("Logging generated molecules to WandB...")
        # Simple molecule grid
        log_generated_molecules_to_wandb(
            wandb_logger, generated_smiles, prefix="final"
        )
        # Enhanced visualization with color-coded motif highlighting
        log_molecules_with_motifs_to_wandb(
            wandb_logger,
            generated_smiles,
            prefix="final",
            max_molecules=cfg.wandb.get("max_logged_molecules", 12),
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

    log.info("Computing motif histogram distribution metrics...")
    hist_metrics = MotifHistogramMetric(
        reference_smiles=datamodule.train_smiles,
        distance_fn="kl",
    )
    hist_results = hist_metrics(generated_smiles)
    log.info(f"  motif_hist_mean: {hist_results['motif_hist_mean']:.6f}")
    log.info(f"  motif_hist_max: {hist_results['motif_hist_max']:.6f}")

    log.info("Computing motif co-occurrence metrics...")
    cooccur_metrics = MotifCooccurrenceMetric(
        reference_smiles=datamodule.train_smiles,
    )
    cooccur_results = cooccur_metrics(generated_smiles)
    for name, value in cooccur_results.items():
        log.info(f"  {name}: {value:.6f}")

    all_metrics = {
        **metrics,
        **motif_results,
        "motif_hist_mean": hist_results["motif_hist_mean"],
        "motif_hist_max": hist_results["motif_hist_max"],
        **cooccur_results,
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
