#!/usr/bin/env python
"""Preprocess and cache tokenized molecular datasets.

This script tokenizes molecular datasets and saves the token sequences to disk
for faster training. This is especially useful for H-SENT and HDT tokenizers
which have spectral clustering operations.

With optimized spectral coarsening (n_init=10, vectorized modularity, discretize):
- 25x speedup over original implementation
- Equivalent quality (statistically no difference in modularity scores)
- ~7.3 hours to preprocess 500k molecules (serial)

Usage:
    python scripts/preprocess_dataset.py tokenizer=hsent
    python scripts/preprocess_dataset.py tokenizer=hdt
    python scripts/preprocess_dataset.py tokenizer=sent
"""

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.molecular import MolecularDataset
from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer

log = logging.getLogger(__name__)


def get_cache_filename(
    dataset_name: str,
    split: str,
    tokenizer_type: str,
    num_samples: int,
    tokenizer_config: dict[str, Any],
) -> str:
    """Generate cache filename based on dataset and tokenizer config.

    Args:
        dataset_name: Name of dataset (moses, qm9).
        split: Dataset split (train, val, test).
        tokenizer_type: Type of tokenizer (sent, hsent, hdt).
        num_samples: Number of samples in this split.
        tokenizer_config: Tokenizer configuration dict.

    Returns:
        Cache filename.
    """
    # Create deterministic hash from config
    import hashlib
    import json

    config_str = json.dumps(tokenizer_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    return f"{dataset_name}_{split}_{tokenizer_type}_{num_samples}_{config_hash}.pt"


def _load_coconut_split(
    split: str,
    num_samples: int | None,
    include_hydrogens: bool,
    labeled: bool,
    data_file: str,
    min_atoms: int,
    max_atoms: int,
    min_rings: int,
    num_train: int,
    num_val: int,
) -> MolecularDataset:
    """Load a COCONUT split with sequential ordering matching datamodule cache mode.

    Args:
        split: Dataset split (train, val, test).
        num_samples: Expected number of samples for this split.
        include_hydrogens: Whether to include explicit hydrogens.
        labeled: Whether to include atom/bond labels.
        data_file: Path to COCONUT SMILES file.
        min_atoms: Minimum atom count filter.
        max_atoms: Maximum atom count filter.
        min_rings: Minimum ring count filter.
        num_train: Total training samples (for split offset calculation).
        num_val: Total validation samples (for split offset calculation).

    Returns:
        MolecularDataset for the requested split.
    """
    from src.data.coconut_loader import CoconutLoader

    loader = CoconutLoader(
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_rings=min_rings,
        data_file=data_file,
    )

    # Load ALL molecules in file order (no shuffle) for sequential splitting
    all_smiles = loader.load_smiles()
    log.info(f"Loaded {len(all_smiles)} COCONUT molecules passing filters")

    # Sequential split matching datamodule cache-mode ordering
    if split == "train":
        smiles = all_smiles[:num_train]
    elif split == "val":
        smiles = all_smiles[num_train : num_train + num_val]
    else:  # test
        smiles = all_smiles[num_train + num_val : num_train + num_val + (num_samples or 500)]

    return MolecularDataset(
        smiles,
        dataset_name=f"coconut_{split}",
        include_hydrogens=include_hydrogens,
        labeled=labeled,
    )


def preprocess_split(
    dataset_name: str,
    split: str,
    tokenizer: Any,
    num_samples: int | None,
    cache_dir: Path,
    tokenizer_type: str,
    tokenizer_config: dict[str, Any],
    include_hydrogens: bool = False,
    data_file: str = "",
    min_atoms: int = 0,
    max_atoms: int = 100,
    min_rings: int = 0,
    num_train: int = 0,
    num_val: int = 0,
    use_precomputed_smiles: bool = False,
    precomputed_smiles_dir: str | None = None,
) -> Path:
    """Preprocess and cache a single dataset split.

    Args:
        dataset_name: Name of dataset.
        split: Dataset split name.
        tokenizer: Tokenizer instance.
        num_samples: Number of samples to process.
        cache_dir: Directory to save cache file.
        tokenizer_type: Type of tokenizer.
        tokenizer_config: Tokenizer configuration.
        include_hydrogens: Whether to include explicit hydrogens.
        data_file: Path to data file (for COCONUT).
        min_atoms: Minimum atom filter (for COCONUT).
        max_atoms: Maximum atom filter (for COCONUT).
        min_rings: Minimum ring filter (for COCONUT).
        num_train: Training split size (for COCONUT split offsets).
        num_val: Validation split size (for COCONUT split offsets).

    Returns:
        Path to saved cache file.
    """
    # Check if cache already exists (skip reprocessing)
    if num_samples is not None:
        cache_filename = get_cache_filename(
            dataset_name, split, tokenizer_type, num_samples, tokenizer_config
        )
        cache_path = cache_dir / cache_filename
        if cache_path.exists():
            log.info(f"Cache already exists for {split}: {cache_path} (skipping)")
            return cache_path

    log.info(f"Preprocessing {split} split ({num_samples or 'all'} samples)...")

    # Load molecular dataset
    labeled = hasattr(tokenizer, "labeled_graph") and tokenizer.labeled_graph

    log.info(f"Loading {dataset_name} dataset...")
    if dataset_name == "moses":
        mol_dataset = MolecularDataset.from_moses(
            split="train" if split == "train" else "test",
            max_molecules=num_samples,
            include_hydrogens=include_hydrogens,
            labeled=labeled,
            use_precomputed_smiles=use_precomputed_smiles,
            precomputed_smiles_dir=precomputed_smiles_dir,
        )
    elif dataset_name == "qm9":
        mol_dataset = MolecularDataset.from_qm9(
            split=split,
            max_molecules=num_samples,
            include_hydrogens=include_hydrogens,
            labeled=labeled,
        )
    elif dataset_name == "coconut":
        mol_dataset = _load_coconut_split(
            split=split,
            num_samples=num_samples,
            include_hydrogens=include_hydrogens,
            labeled=labeled,
            data_file=data_file,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_rings=min_rings,
            num_train=num_train,
            num_val=num_val,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Configure tokenizer with dataset info
    max_num_nodes = mol_dataset.max_num_nodes
    tokenizer.set_num_nodes(max_num_nodes)

    if labeled:
        from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES

        tokenizer.set_num_node_and_edge_types(
            num_node_types=NUM_ATOM_TYPES,
            num_edge_types=NUM_BOND_TYPES,
        )

    # Tokenize all samples with progress bar
    tokenized_data = []
    smiles_list = []

    log.info(f"Tokenizing {len(mol_dataset)} molecules...")
    for i in tqdm(range(len(mol_dataset)), desc=f"Tokenizing {split}"):
        graph = mol_dataset[i]
        tokens = tokenizer(graph)  # Returns token tensor

        tokenized_data.append(tokens)
        smiles_list.append(mol_dataset.smiles_list[i])

    # Save to cache
    cache_filename = get_cache_filename(
        dataset_name, split, tokenizer_type, len(mol_dataset), tokenizer_config
    )
    cache_path = cache_dir / cache_filename

    cache_data = {
        "tokens": tokenized_data,
        "smiles": smiles_list,
        "vocab_size": tokenizer.vocab_size,
        "max_num_nodes": max_num_nodes,
        "tokenizer_type": tokenizer_type,
        "tokenizer_config": tokenizer_config,
        "dataset_name": dataset_name,
        "split": split,
        "num_samples": len(mol_dataset),
        "labeled": labeled,
    }

    log.info(f"Saving cache to {cache_path}...")
    torch.save(cache_data, cache_path)

    log.info(f"✓ Cached {len(tokenized_data)} tokenized sequences")
    log.info(f"  Vocab size: {tokenizer.vocab_size}")
    log.info(f"  Max nodes: {max_num_nodes}")
    log.info(f"  Cache file: {cache_path}")

    return cache_path


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main preprocessing function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create cache directory
    cache_dir = Path(cfg.data.get("cache_dir", "data/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Cache directory: {cache_dir}")

    # Select tokenizer based on config
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()

    # Store tokenizer config for cache key
    tokenizer_config = {
        "type": tokenizer_type,
        "max_length": cfg.tokenizer.max_length,
        "truncation_length": cfg.tokenizer.truncation_length,
        "labeled_graph": cfg.tokenizer.get("labeled_graph", False),
    }

    if tokenizer_type == "hdtc":
        tokenizer_config.update(
            {
                "node_order": cfg.tokenizer.get("node_order", "BFS"),
                "include_rings": cfg.tokenizer.get("include_rings", True),
            }
        )

        log.info("Using HDTC tokenizer")

        tokenizer = HDTCTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            include_rings=cfg.tokenizer.get("include_rings", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )

    elif tokenizer_type == "hdt":
        coarsening_strategy = cfg.tokenizer.get("coarsening_strategy", "spectral")
        motif_aware = cfg.tokenizer.get("motif_aware", False)
        tokenizer_config.update(
            {
                "node_order": cfg.tokenizer.get("node_order", "BFS"),
                "min_community_size": cfg.tokenizer.get("min_community_size", 4),
                "coarsening_strategy": coarsening_strategy,
                "motif_aware": motif_aware,
                "motif_alpha": cfg.tokenizer.get("motif_alpha", 1.0),
                "normalize_by_motif_size": cfg.tokenizer.get(
                    "normalize_by_motif_size", False
                ),
                "undirected": cfg.tokenizer.get("undirected", True),
            }
        )
        # Add spectral coarsening parameters for cache hash consistency
        if coarsening_strategy == "spectral":
            tokenizer_config["n_init"] = cfg.tokenizer.get("n_init", 10)
            tokenizer_config["k_min_factor"] = cfg.tokenizer.get("k_min_factor", 0.7)
            tokenizer_config["k_max_factor"] = cfg.tokenizer.get("k_max_factor", 1.3)

        log.info(
            f"Using HDT tokenizer ({coarsening_strategy}{' + motif-aware' if motif_aware else ''})"
        )

        tokenizer_kwargs = {
            "max_length": cfg.tokenizer.max_length,
            "truncation_length": cfg.tokenizer.truncation_length,
            "node_order": cfg.tokenizer.get("node_order", "BFS"),
            "min_community_size": cfg.tokenizer.get("min_community_size", 4),
            "coarsening_strategy": coarsening_strategy,
            "motif_aware": motif_aware,
            "motif_alpha": cfg.tokenizer.get("motif_alpha", 1.0),
            "normalize_by_motif_size": cfg.tokenizer.get("normalize_by_motif_size", False),
            "labeled_graph": cfg.tokenizer.get("labeled_graph", True),
            "seed": cfg.seed,
        }
        # Add spectral parameters if using spectral coarsening
        if coarsening_strategy == "spectral":
            tokenizer_kwargs["n_init"] = cfg.tokenizer.get("n_init", 10)
            tokenizer_kwargs["k_min_factor"] = cfg.tokenizer.get("k_min_factor", 0.7)
            tokenizer_kwargs["k_max_factor"] = cfg.tokenizer.get("k_max_factor", 1.3)
        tokenizer = HDTTokenizer(**tokenizer_kwargs)

    elif tokenizer_type == "hsent":
        coarsening_strategy = cfg.tokenizer.get("coarsening_strategy", "spectral")
        motif_aware = cfg.tokenizer.get("motif_aware", False)
        tokenizer_config.update(
            {
                "node_order": cfg.tokenizer.get("node_order", "BFS"),
                "min_community_size": cfg.tokenizer.get("min_community_size", 4),
                "coarsening_strategy": coarsening_strategy,
                "motif_aware": motif_aware,
                "motif_alpha": cfg.tokenizer.get("motif_alpha", 1.0),
                "normalize_by_motif_size": cfg.tokenizer.get(
                    "normalize_by_motif_size", False
                ),
                "undirected": cfg.tokenizer.get("undirected", True),
            }
        )
        # Add spectral coarsening parameters for cache hash consistency
        if coarsening_strategy == "spectral":
            tokenizer_config["n_init"] = cfg.tokenizer.get("n_init", 10)
            tokenizer_config["k_min_factor"] = cfg.tokenizer.get("k_min_factor", 0.7)
            tokenizer_config["k_max_factor"] = cfg.tokenizer.get("k_max_factor", 1.3)

        log.info(
            f"Using H-SENT tokenizer ({coarsening_strategy}{' + motif-aware' if motif_aware else ''})"
        )

        tokenizer_kwargs = {
            "max_length": cfg.tokenizer.max_length,
            "truncation_length": cfg.tokenizer.truncation_length,
            "node_order": cfg.tokenizer.get("node_order", "BFS"),
            "min_community_size": cfg.tokenizer.get("min_community_size", 4),
            "coarsening_strategy": coarsening_strategy,
            "motif_aware": motif_aware,
            "motif_alpha": cfg.tokenizer.get("motif_alpha", 1.0),
            "normalize_by_motif_size": cfg.tokenizer.get("normalize_by_motif_size", False),
            "labeled_graph": cfg.tokenizer.get("labeled_graph", True),
            "seed": cfg.seed,
        }
        # Add spectral parameters if using spectral coarsening
        if coarsening_strategy == "spectral":
            tokenizer_kwargs["n_init"] = cfg.tokenizer.get("n_init", 10)
            tokenizer_kwargs["k_min_factor"] = cfg.tokenizer.get("k_min_factor", 0.7)
            tokenizer_kwargs["k_max_factor"] = cfg.tokenizer.get("k_max_factor", 1.3)
        tokenizer = HSENTTokenizer(**tokenizer_kwargs)

    else:
        tokenizer_config.update(
            {
                "undirected": cfg.tokenizer.get("undirected", True),
            }
        )

        log.info("Using SENT tokenizer")

        tokenizer = SENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            undirected=cfg.tokenizer.get("undirected", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", False),
            seed=cfg.seed,
        )

    # Preprocess each split
    splits = []

    if cfg.data.num_train is not None and cfg.data.num_train > 0:
        splits.append(("train", cfg.data.num_train))

    if cfg.data.num_val is not None and cfg.data.num_val > 0:
        splits.append(("val", cfg.data.num_val))

    if cfg.data.num_test is not None and cfg.data.num_test > 0:
        splits.append(("test", cfg.data.num_test))

    if not splits:
        log.error("No splits specified! Set data.num_train, data.num_val, or data.num_test")
        return

    # COCONUT-specific parameters for split offset calculation
    coconut_kwargs = {}
    if cfg.data.dataset_name == "coconut":
        coconut_kwargs = {
            "data_file": cfg.data.get("data_file", "data/coconut_complex.smi"),
            "min_atoms": cfg.data.get("min_atoms", 20),
            "max_atoms": cfg.data.get("max_atoms", 100),
            "min_rings": cfg.data.get("min_rings", 3),
            "num_train": cfg.data.num_train or 5000,
            "num_val": cfg.data.num_val or 500,
        }

    for split_name, num_samples in splits:
        preprocess_split(
            dataset_name=cfg.data.dataset_name,
            split=split_name,
            tokenizer=tokenizer,
            num_samples=num_samples,
            cache_dir=cache_dir,
            tokenizer_type=tokenizer_type,
            tokenizer_config=tokenizer_config,
            include_hydrogens=cfg.data.get("include_hydrogens", False),
            use_precomputed_smiles=cfg.data.get("use_precomputed_smiles", False),
            precomputed_smiles_dir=cfg.data.get("precomputed_smiles_dir", None),
            **coconut_kwargs,
        )

    log.info("\n" + "=" * 50)
    log.info("PREPROCESSING COMPLETE")
    log.info("=" * 50)
    log.info(f"Cache directory: {cache_dir}")
    log.info(
        f"Preprocessed {len(splits)} splits: {', '.join(s[0] for s in splits)}"
    )
    log.info("\nTo use cached data during training:")
    log.info(
        f"  python scripts/train.py tokenizer={tokenizer_type} data.use_cache=true"
    )


if __name__ == "__main__":
    main()
