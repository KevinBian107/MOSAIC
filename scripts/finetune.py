#!/usr/bin/env python
"""Fine-tuning script for transfer learning on COCONUT.

This script fine-tunes a pretrained model on COCONUT complex natural products
and evaluates the fine-tuned model with full metrics.

Usage:
    # Fine-tune from MOSES checkpoint on COCONUT
    python scripts/finetune.py \
        model.pretrained_path=outputs/moses_hdtc/best.ckpt \
        data.num_train=5000

    # Fine-tune with custom settings
    python scripts/finetune.py \
        model.pretrained_path=outputs/moses_hdtc/best.ckpt \
        experiment=coconut \
        trainer.max_steps=50000 \
        model.learning_rate=1e-5
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main fine-tuning function.

    This is a wrapper around train.py that sets sensible defaults for fine-tuning:
    - Uses COCONUT dataset
    - Lower learning rate
    - Requires pretrained_path to be set

    Args:
        cfg: Hydra configuration.
    """
    # Set COCONUT defaults if not already set
    if cfg.data.dataset_name != "coconut":
        log.info("Setting dataset to COCONUT for fine-tuning")
        cfg.data.dataset_name = "coconut"

    # Warn if no pretrained path
    if not cfg.model.get("pretrained_path"):
        log.warning(
            "No pretrained_path specified. For transfer learning, provide a pretrained "
            "checkpoint: model.pretrained_path=path/to/best.ckpt"
        )

    # Use lower learning rate for fine-tuning if not explicitly set
    if cfg.model.learning_rate >= 1e-4:
        log.info(
            f"Using fine-tuning learning rate: 1e-5 (original: {cfg.model.learning_rate})"
        )
        cfg.model.learning_rate = 1e-5

    # CRITICAL FIX: Ensure tokenizer truncation_length <= model's max_length
    # This prevents CUDA index out of bounds errors when sequences exceed
    # the model's position embedding size
    max_seq_length = cfg.sampling.max_length
    tokenizer_trunc = cfg.tokenizer.get("truncation_length", 2048)
    if tokenizer_trunc > max_seq_length:
        log.warning(
            f"Tokenizer truncation_length ({tokenizer_trunc}) > sampling.max_length ({max_seq_length}). "
            f"Overriding to {max_seq_length} to avoid position embedding overflow."
        )
        cfg.tokenizer.truncation_length = max_seq_length

    log.info(f"Fine-tuning configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Import and run the main training function

    # Call train_main with the modified config
    # Note: We can't call it directly since it's decorated with @hydra.main
    # Instead, we'll replicate the essential training logic here

    import pytorch_lightning as pl
    import torch
    from tqdm import tqdm

    from src.data.datamodule import MolecularDataModule
    from src.data.molecular import graph_to_smiles
    from src.models.transformer import GraphGeneratorModule
    from src.tokenizers import (
        HDTCTokenizer,
        HDTTokenizer,
        HSENTTokenizer,
        SENTTokenizer,
    )

    # Lightweight eval callback for logging during training
    class FinetuneEvalCallback(pl.Callback):
        """Callback to log generated molecules during finetuning.

        Logs a small number of generated molecules and reference molecules
        every N steps to monitor training progress.
        """

        def __init__(
            self,
            reference_smiles: list[str],
            eval_every_n_steps: int = 1000,
            num_samples: int = 5,
            output_dir: str = "outputs",
        ):
            """Initialize callback.

            Args:
                reference_smiles: List of reference SMILES from training data.
                eval_every_n_steps: How often to run evaluation.
                num_samples: Number of molecules to generate per eval.
                output_dir: Directory to save generated SMILES.
            """
            super().__init__()
            self.reference_smiles = reference_smiles
            self.eval_every_n_steps = eval_every_n_steps
            self.num_samples = num_samples
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            """Called after each training batch."""
            global_step = trainer.global_step

            # Only evaluate at specified intervals
            if global_step == 0 or global_step % self.eval_every_n_steps != 0:
                return

            log.info(f"[Step {global_step}] Running intermediate evaluation...")

            # Generate a few molecules
            pl_module.eval()
            try:
                generated_graphs, gen_time, _ = pl_module.generate(
                    num_samples=self.num_samples, show_progress=False
                )

                # Suppress RDKit stderr noise from invalid generated molecules
                from rdkit import RDLogger

                RDLogger.DisableLog("rdApp.*")

                # Convert to SMILES
                generated_smiles = []
                valid_count = 0
                for g in generated_graphs:
                    smiles = graph_to_smiles(g)
                    if smiles:
                        generated_smiles.append(smiles)
                        valid_count += 1
                    else:
                        generated_smiles.append("INVALID")

                # Log validity
                validity = (
                    valid_count / len(generated_graphs) if generated_graphs else 0.0
                )
                log.info(
                    f"  [Eval] Validity: {valid_count}/{len(generated_graphs)} ({validity:.1%})"
                )

                # Log generated SMILES samples
                log.info(f"  [Eval] Generated SMILES (sample):")
                for i, smi in enumerate(generated_smiles[:3]):
                    log.info(f"    {i + 1}. {smi[:80]}...")

                # Log reference SMILES samples for comparison
                log.info(f"  [Eval] Reference SMILES (sample):")
                import random

                ref_sample = random.sample(
                    self.reference_smiles, min(3, len(self.reference_smiles))
                )
                for i, smi in enumerate(ref_sample):
                    log.info(f"    {i + 1}. {smi[:80]}...")

                # Save to file
                eval_file = self.output_dir / f"eval_step_{global_step}.txt"
                with open(eval_file, "w") as f:
                    f.write(f"# Step {global_step} - Validity: {validity:.1%}\n")
                    f.write("# Generated:\n")
                    for smi in generated_smiles:
                        f.write(f"{smi}\n")
                    f.write("\n# Reference samples:\n")
                    for smi in ref_sample:
                        f.write(f"{smi}\n")

                # Re-enable RDKit logging
                RDLogger.EnableLog("rdApp.*")

                # Log to trainer loggers if available
                if trainer.logger:
                    trainer.logger.log_metrics(
                        {
                            "eval/validity": validity,
                            "eval/valid_count": valid_count,
                            "eval/gen_time_per_sample": gen_time,
                        },
                        step=global_step,
                    )

                # Log molecule images to WandB
                self._log_molecule_images(
                    trainer, generated_smiles, ref_sample, global_step
                )

            except Exception as e:
                log.warning(f"  [Eval] Error during intermediate evaluation: {e}")
            finally:
                pl_module.train()

        def _log_molecule_images(
            self,
            trainer: pl.Trainer,
            generated_smiles: list[str],
            reference_smiles: list[str],
            global_step: int,
        ) -> None:
            """Log generated and reference molecule images to WandB.

            Args:
                trainer: PyTorch Lightning trainer.
                generated_smiles: List of generated SMILES (may include "INVALID").
                reference_smiles: List of reference SMILES for comparison.
                global_step: Current training step.
            """
            try:
                import wandb
                from rdkit import Chem
                from rdkit.Chem import Draw

                # Find WandB logger
                wandb_logger = None
                for logger in trainer.loggers:
                    if isinstance(logger, pl.loggers.WandbLogger):
                        wandb_logger = logger
                        break
                if wandb_logger is None:
                    return

                def _render_grid(smiles_list: list[str]) -> "wandb.Image | None":
                    mols = []
                    legends = []
                    for smi in smiles_list:
                        if smi == "INVALID":
                            continue
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            mols.append(mol)
                            # Truncate long SMILES for legend
                            legends.append(smi[:50] + "..." if len(smi) > 50 else smi)
                    if not mols:
                        return None
                    img = Draw.MolsToGridImage(
                        mols,
                        molsPerRow=min(3, len(mols)),
                        subImgSize=(300, 300),
                        legends=legends,
                    )
                    return wandb.Image(img)

                gen_img = _render_grid(generated_smiles)
                ref_img = _render_grid(reference_smiles)

                log_dict = {}
                if gen_img is not None:
                    log_dict["eval/generated_molecules"] = gen_img
                if ref_img is not None:
                    log_dict["eval/reference_molecules"] = ref_img

                if log_dict:
                    wandb_logger.experiment.log(log_dict, step=global_step)

            except Exception as e:
                log.debug(f"Could not log molecule images to WandB: {e}")

    pl.seed_everything(cfg.seed, workers=True)

    # Create output directory
    output_dir = Path(cfg.logs.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")

    # Select tokenizer
    tokenizer_type = cfg.tokenizer.get("type", "hdtc").lower()
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
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
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
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
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

    log.info(f"Using {tokenizer_type.upper()} tokenizer")

    # Create data module
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
        data_file=cfg.data.get("data_file"),
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
    )
    datamodule.setup()

    log.info(f"Training dataset size: {len(datamodule.train_dataset):,}")
    log.info(f"Validation dataset size: {len(datamodule.val_dataset):,}")
    log.info(f"Max num nodes: {datamodule.max_num_nodes}")
    log.info(f"Tokenizer vocab_size: {len(tokenizer)}")
    if hasattr(tokenizer, "labeled_graph") and tokenizer.labeled_graph:
        log.info(f"  node_idx_offset: {tokenizer.node_idx_offset}")
        log.info(f"  edge_idx_offset: {tokenizer.edge_idx_offset}")
        log.info(f"  num_node_types: {tokenizer.num_node_types}")
        log.info(f"  num_edge_types: {tokenizer.num_edge_types}")

    # Auto-adjust val_check_interval if dataset is too small
    num_train_batches = len(datamodule.train_dataset) // cfg.data.batch_size
    if num_train_batches < cfg.trainer.val_check_interval:
        new_interval = max(1, num_train_batches // 2)
        log.warning(
            f"Dataset too small for val_check_interval={cfg.trainer.val_check_interval}. "
            f"Adjusting to {new_interval} (have {num_train_batches} batches)"
        )
        cfg.trainer.val_check_interval = new_interval

    # Create model
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

    # Load pretrained weights
    pretrained_path = cfg.model.get("pretrained_path")
    if pretrained_path:
        log.info(f"Loading pretrained weights from {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in pretrained:
            state_dict = pretrained["state_dict"]
        else:
            state_dict = pretrained

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

                old_wte = state_dict[wte_key]
                old_lm_head = state_dict.get(lm_head_key)

                # Semantic embedding mapping for labeled graph tokenizers
                # Vocab layout: [special (IDX_OFFSET)] [node IDs (max_num_nodes)] [atom types] [bond types]
                # When max_num_nodes changes, atom/bond type tokens shift positions.
                # We must copy embeddings by semantic role, not raw index.
                if (
                    hasattr(tokenizer, "labeled_graph")
                    and tokenizer.labeled_graph
                    and hasattr(tokenizer, "node_idx_offset")
                    and tokenizer.node_idx_offset > 0
                ):
                    from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES

                    # HDT/HDTC/HSENT use IDX_OFFSET class constant, SENT uses idx_offset
                    if hasattr(tokenizer, "IDX_OFFSET"):
                        IDX_OFFSET = tokenizer.IDX_OFFSET
                    else:
                        IDX_OFFSET = tokenizer.idx_offset
                    new_node_off = tokenizer.node_idx_offset
                    new_edge_off = tokenizer.edge_idx_offset

                    # Infer old layout from old vocab size
                    old_max_nodes = (
                        old_vocab_size - IDX_OFFSET - NUM_ATOM_TYPES - NUM_BOND_TYPES
                    )
                    old_node_off = IDX_OFFSET + old_max_nodes
                    old_edge_off = old_node_off + NUM_ATOM_TYPES

                    log.info(f"  Semantic embedding mapping (labeled graph):")
                    log.info(
                        f"    Old layout: nodes@[{IDX_OFFSET},{old_node_off}) "
                        f"atoms@[{old_node_off},{old_edge_off}) "
                        f"bonds@[{old_edge_off},{old_vocab_size})"
                    )
                    log.info(
                        f"    New layout: nodes@[{IDX_OFFSET},{new_node_off}) "
                        f"atoms@[{new_node_off},{new_edge_off}) "
                        f"bonds@[{new_edge_off},{new_vocab_size})"
                    )

                    def _copy_range(src, dst, old_start, old_end, new_start, new_end):
                        n = min(old_end - old_start, new_end - new_start)
                        dst[new_start : new_start + n] = src[old_start : old_start + n]
                        return n

                    # 1. Special tokens (same positions)
                    n = _copy_range(old_wte, new_wte, 0, IDX_OFFSET, 0, IDX_OFFSET)
                    if old_lm_head is not None:
                        _copy_range(
                            old_lm_head, new_lm_head, 0, IDX_OFFSET, 0, IDX_OFFSET
                        )
                    log.info(f"    Copied {n} special token embeddings")

                    # 2. Node IDs (may have different range sizes)
                    n = _copy_range(
                        old_wte,
                        new_wte,
                        IDX_OFFSET,
                        old_node_off,
                        IDX_OFFSET,
                        new_node_off,
                    )
                    if old_lm_head is not None:
                        _copy_range(
                            old_lm_head,
                            new_lm_head,
                            IDX_OFFSET,
                            old_node_off,
                            IDX_OFFSET,
                            new_node_off,
                        )
                    log.info(f"    Copied {n} node ID embeddings")

                    # 3. Atom types (same count, different positions)
                    n = _copy_range(
                        old_wte,
                        new_wte,
                        old_node_off,
                        old_edge_off,
                        new_node_off,
                        new_edge_off,
                    )
                    if old_lm_head is not None:
                        _copy_range(
                            old_lm_head,
                            new_lm_head,
                            old_node_off,
                            old_edge_off,
                            new_node_off,
                            new_edge_off,
                        )
                    log.info(f"    Copied {n} atom type embeddings")

                    # 4. Bond types (same count, different positions)
                    n = _copy_range(
                        old_wte,
                        new_wte,
                        old_edge_off,
                        old_vocab_size,
                        new_edge_off,
                        new_vocab_size,
                    )
                    if old_lm_head is not None:
                        _copy_range(
                            old_lm_head,
                            new_lm_head,
                            old_edge_off,
                            old_vocab_size,
                            new_edge_off,
                            new_vocab_size,
                        )
                    log.info(f"    Copied {n} bond type embeddings")

                else:
                    # Unlabeled tokenizer or no layout shift: simple positional copy
                    copy_size = min(old_vocab_size, new_vocab_size)
                    new_wte[:copy_size] = old_wte[:copy_size]
                    if old_lm_head is not None:
                        new_lm_head[:copy_size] = old_lm_head[:copy_size]
                    log.info(f"  Copied {copy_size} pretrained token embeddings")

                state_dict[wte_key] = new_wte
                state_dict[lm_head_key] = new_lm_head

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
            log.warning(f"Missing keys: {missing}")
        if unexpected:
            log.warning(f"Unexpected keys: {unexpected}")
        log.info("Pretrained weights loaded successfully")

    # Setup trainer
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=cfg.logs.path,
            filename="best",
            save_last=True,
            mode="min",
        ),
    ]

    # Add finetuning eval callback if enabled
    eval_every_n_steps = cfg.get("eval_every_n_steps", 1000)
    eval_num_samples = cfg.get("eval_num_samples", 5)
    if eval_every_n_steps > 0:
        callbacks.append(
            FinetuneEvalCallback(
                reference_smiles=datamodule.train_smiles,
                eval_every_n_steps=eval_every_n_steps,
                num_samples=eval_num_samples,
                output_dir=cfg.logs.path,
            )
        )
        log.info(
            f"Eval callback enabled: every {eval_every_n_steps} steps, "
            f"{eval_num_samples} samples"
        )

    # Setup loggers
    loggers = [pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")]

    # Add WandB logger if enabled
    wandb_logger = None
    if cfg.wandb.get("enabled", False):
        run_name = cfg.wandb.get("name")
        if run_name is None:
            run_name = f"finetune_{cfg.data.dataset_name}_{tokenizer_type}"

        wandb_logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            tags=list(cfg.wandb.tags) if cfg.wandb.get("tags") else ["finetune"],
            notes=cfg.wandb.get("notes"),
            save_dir=cfg.logs.path,
            log_model=cfg.wandb.get("log_model", True),
        )
        loggers.append(wandb_logger)
        log.info(f"WandB logging enabled: {cfg.wandb.project}/{run_name}")

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

    # Train
    log.info("Starting fine-tuning...")
    trainer.fit(model, datamodule)

    # Evaluate
    log.info("Evaluating fine-tuned model...")
    model.eval()
    generated_graphs, gen_time, _ = model.generate(
        num_samples=cfg.sampling.num_samples, show_progress=True
    )
    log.info(f"Generated {len(generated_graphs)} graphs in {gen_time:.4f}s per sample")

    # Convert to SMILES
    INVALID = "INVALID"
    generated_smiles = []
    for g in tqdm(generated_graphs, desc="Converting to SMILES"):
        smiles = graph_to_smiles(g)
        generated_smiles.append(smiles if smiles else INVALID)

    valid_count = sum(1 for s in generated_smiles if s != INVALID)
    log.info(f"Valid molecules: {valid_count}/{len(generated_smiles)}")

    # Log final validity and generation time
    # Full metric computation is handled by the test script (scripts/test.py)
    if wandb_logger is not None:
        wandb_logger.experiment.log(
            {
                "final/valid_count": valid_count,
                "final/validity": valid_count / max(len(generated_smiles), 1),
                "final/generation_time_per_sample": gen_time,
            }
        )

        import wandb

        wandb.finish()
        log.info("WandB run finished")

    log.info("Fine-tuning complete!")
    log.info(f"Best checkpoint saved to: {cfg.logs.path}/best.ckpt")


if __name__ == "__main__":
    main()
