#!/usr/bin/env python
"""Compute cross-entropy loss on the held-out test split.

This script loads a trained model checkpoint and evaluates it on the test
split of the dataset, computing the same autoregressive cross-entropy loss
used during training. This provides a direct measure of model quality
independent of downstream generation metrics.

Usage:
    python scripts/test_loss.py model.checkpoint_path=/path/to/model.ckpt
    python scripts/test_loss.py experiment=coconut model.checkpoint_path=/path/to/model.ckpt
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import MolecularDataModule
from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_loss")
def main(cfg: DictConfig) -> None:
    """Compute test loss for a trained model.

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

    # Select tokenizer based on config (mirrors test.py logic)
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()
    if tokenizer_type == "hdt":
        tokenizer = HDTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", None),
            motif_aware=cfg.tokenizer.get("motif_aware", False),
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
            coarsening_strategy=cfg.tokenizer.get("coarsening_strategy", None),
            motif_aware=cfg.tokenizer.get("motif_aware", False),
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hdtc":
        tokenizer = HDTCTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            include_rings=cfg.tokenizer.get("include_rings", True),
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
        data_file=cfg.data.get("data_file", None),
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
    )

    datamodule.setup(stage="test")

    # Extract vocab size from checkpoint and configure tokenizer
    log.info(f"Extracting vocab size from checkpoint: {cfg.model.checkpoint_path}")
    checkpoint = torch.load(
        cfg.model.checkpoint_path, map_location="cpu", weights_only=False
    )
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
                    tokenizer.max_num_nodes = checkpoint_max_num_nodes
                    tokenizer.set_num_node_and_edge_types(
                        num_node_types=NUM_ATOM_TYPES,
                        num_edge_types=NUM_BOND_TYPES,
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
                tokenizer.max_num_nodes = checkpoint_max_num_nodes
                log.info(
                    f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}"
                )

        wpe_key = "model.model.transformer.wpe.weight"
        if wpe_key in checkpoint["state_dict"]:
            checkpoint_max_length = checkpoint["state_dict"][wpe_key].shape[0]
            log.info(f"Checkpoint max position embeddings: {checkpoint_max_length}")

    # Load model
    log.info(f"Loading model from {cfg.model.checkpoint_path}...")
    load_kwargs: dict = {"tokenizer": tokenizer, "weights_only": False}
    if checkpoint_max_length is not None:
        load_kwargs["sampling_max_length"] = checkpoint_max_length
    model = GraphGeneratorModule.load_from_checkpoint(
        cfg.model.checkpoint_path,
        **load_kwargs,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Compute test loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad)
    test_loader = datamodule.test_dataloader()

    log.info(f"Computing test loss on {len(datamodule.test_smiles)} molecules...")
    log.info(f"Device: {device}")

    total_loss = 0.0
    total_tokens = 0
    batch_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing test loss"):
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            logits = model(x)
            logits = logits.view(-1, logits.shape[-1])
            y_flat = y.reshape(-1)

            loss = loss_fn(logits, y_flat)

            # Count non-pad tokens for weighted average
            non_pad_mask = y_flat != tokenizer.pad
            num_tokens = non_pad_mask.sum().item()

            batch_losses.append(
                {
                    "loss": loss.item(),
                    "num_tokens": num_tokens,
                }
            )

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    log.info("\n" + "=" * 50)
    log.info("TEST LOSS RESULTS")
    log.info("=" * 50)
    log.info(f"  Average loss:  {avg_loss:.6f}")
    log.info(f"  Perplexity:    {perplexity:.4f}")
    log.info(f"  Total tokens:  {total_tokens}")
    log.info(f"  Num batches:   {len(batch_losses)}")
    log.info(f"  Num molecules: {len(datamodule.test_smiles)}")

    # Save results
    results = {
        "test_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_batches": len(batch_losses),
        "num_molecules": len(datamodule.test_smiles),
        "batch_losses": batch_losses,
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {results_file}")

    log.info("\nTest loss computation complete!")


if __name__ == "__main__":
    main()
