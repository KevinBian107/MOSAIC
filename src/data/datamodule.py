"""PyTorch Lightning data module for molecular graph datasets.

This module provides a unified interface for loading molecular graph datasets
with tokenization support for training graph generation models.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.molecular import MolecularDataset

log = logging.getLogger(__name__)


class MolecularGraphDataset(Dataset):
    """PyTorch Dataset wrapper for molecular graphs.

    Attributes:
        molecular_dataset: Underlying MolecularDataset.
        transform: Optional transform to apply.
    """

    def __init__(
        self,
        molecular_dataset: MolecularDataset,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            molecular_dataset: MolecularDataset instance.
            transform: Optional transform to apply to each sample.
        """
        self.molecular_dataset = molecular_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.molecular_dataset)

    def __getitem__(self, idx: int):
        """Get a single sample."""
        data = self.molecular_dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @property
    def smiles_list(self) -> list[str]:
        """Return list of SMILES strings."""
        return self.molecular_dataset.smiles_list


class CachedTokenDataset(Dataset):
    """PyTorch Dataset for pre-tokenized molecular graphs.

    This dataset loads pre-tokenized sequences from disk, avoiding
    expensive on-the-fly tokenization during training.

    Attributes:
        tokens: List of token tensors.
        smiles: List of SMILES strings.
        vocab_size: Vocabulary size.
        max_num_nodes: Maximum number of nodes.
    """

    def __init__(self, cache_path: str) -> None:
        """Initialize from cached file.

        Args:
            cache_path: Path to cached .pt file.
        """
        log.info(f"Loading cached dataset from {cache_path}")
        cache_data = torch.load(cache_path)

        self.tokens = cache_data["tokens"]
        self.smiles_list = cache_data["smiles"]
        self.vocab_size = cache_data["vocab_size"]
        self.max_num_nodes = cache_data["max_num_nodes"]
        self.tokenizer_type = cache_data.get("tokenizer_type", "unknown")
        self.labeled = cache_data.get("labeled", False)

        log.info(f"  Loaded {len(self.tokens)} pre-tokenized samples")
        log.info(f"  Tokenizer: {self.tokenizer_type}")
        log.info(f"  Vocab size: {self.vocab_size}")
        log.info(f"  Max nodes: {self.max_num_nodes}")
        log.info(f"  Labeled: {self.labeled}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a pre-tokenized sample.

        Args:
            idx: Sample index.

        Returns:
            Token tensor.
        """
        return self.tokens[idx]


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
    config_str = json.dumps(tokenizer_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    return f"{dataset_name}_{split}_{tokenizer_type}_{num_samples}_{config_hash}.pt"


class MolecularDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for molecular datasets.

    This module handles data loading, tokenization, and batching for
    training molecular graph generation models.

    Attributes:
        dataset_name: Name of the dataset ('moses' or 'qm9').
        tokenizer: Graph tokenizer for converting graphs to sequences.
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.
    """

    def __init__(
        self,
        dataset_name: str = "moses",
        tokenizer: Optional[Any] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        include_hydrogens: bool = False,
        seed: int = 42,
        data_root: str = "data",
        use_cache: bool = False,
        cache_dir: str = "data/cache",
        data_file: Optional[str] = None,
        min_atoms: int = 20,
        max_atoms: int = 100,
        min_rings: int = 3,
    ) -> None:
        """Initialize the data module.

        Args:
            dataset_name: Name of dataset to use ('moses', 'qm9', or 'coconut').
            tokenizer: Graph tokenizer instance.
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            num_train: Number of training molecules (None for all).
            num_val: Number of validation molecules (None for all).
            num_test: Number of test molecules (None for all).
            include_hydrogens: Whether to include explicit hydrogens.
            seed: Random seed for reproducibility.
            data_root: Root directory for data storage.
            use_cache: Whether to use cached pre-tokenized data.
            cache_dir: Directory containing cached data files.
            data_file: Path to data file (for coconut dataset).
            min_atoms: Minimum atoms filter (for coconut dataset).
            max_atoms: Maximum atoms filter (for coconut dataset).
            min_rings: Minimum rings filter (for coconut dataset).
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.include_hydrogens = include_hydrogens
        self.seed = seed
        self.data_root = data_root
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.data_file = data_file
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_rings = min_rings

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.train_smiles: list[str] = []
        self.val_smiles: list[str] = []
        self.test_smiles: list[str] = []

        self.max_num_nodes: int = 0

    def prepare_data(self) -> None:
        """Download data if needed (called once on main process)."""
        if self.dataset_name == "qm9":
            # Trigger download of QM9
            from torch_geometric.datasets import QM9

            QM9(root=f"{self.data_root}/qm9")

    def _get_tokenizer_config(self) -> dict[str, Any]:
        """Extract tokenizer configuration for cache lookup.

        Returns:
            Dictionary of tokenizer configuration parameters.
        """
        if self.tokenizer is None:
            return {}

        config = {
            "type": getattr(self.tokenizer, "tokenizer_type", "unknown"),
            "max_length": getattr(self.tokenizer, "max_length", -1),
            "truncation_length": getattr(self.tokenizer, "truncation_length", 2048),
            "labeled_graph": getattr(self.tokenizer, "labeled_graph", False),
        }

        # Add tokenizer-specific config
        if hasattr(self.tokenizer, "node_order"):
            config["node_order"] = self.tokenizer.node_order

        if hasattr(self.tokenizer, "min_community_size"):
            config["min_community_size"] = self.tokenizer.min_community_size

        if hasattr(self.tokenizer, "include_rings"):
            config["include_rings"] = self.tokenizer.include_rings

        if hasattr(self.tokenizer, "motif_aware"):
            config.update(
                {
                    "motif_aware": self.tokenizer.motif_aware,
                    "motif_alpha": getattr(self.tokenizer, "motif_alpha", 1.0),
                    "normalize_by_motif_size": getattr(
                        self.tokenizer, "normalize_by_motif_size", False
                    ),
                }
            )

        if hasattr(self.tokenizer, "undirected"):
            config["undirected"] = self.tokenizer.undirected

        return config

    def _try_load_cache(
        self, split: str, num_samples: Optional[int]
    ) -> Optional[CachedTokenDataset]:
        """Try to load cached dataset for a split.

        Args:
            split: Dataset split name.
            num_samples: Number of samples expected.

        Returns:
            CachedTokenDataset if cache exists and matches, None otherwise.
        """
        if not self.use_cache:
            return None

        if num_samples is None:
            log.info(f"Cannot use cache for {split}: num_samples not specified")
            return None

        tokenizer_type = getattr(self.tokenizer, "tokenizer_type", "unknown")
        tokenizer_config = self._get_tokenizer_config()

        cache_filename = get_cache_filename(
            self.dataset_name, split, tokenizer_type, num_samples, tokenizer_config
        )
        cache_path = self.cache_dir / cache_filename

        if not cache_path.exists():
            log.info(f"Cache not found for {split}: {cache_path}")
            log.info(
                f"  Run: python scripts/preprocess_dataset.py tokenizer={tokenizer_type} "
                f"data.num_{split}={num_samples}"
            )
            return None

        try:
            cached_dataset = CachedTokenDataset(str(cache_path))

            # Update max_num_nodes from cache
            self.max_num_nodes = max(self.max_num_nodes, cached_dataset.max_num_nodes)

            return cached_dataset
        except Exception as e:
            log.warning(f"Failed to load cache for {split}: {e}")
            return None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self.dataset_name == "moses":
            self._setup_moses(stage)
        elif self.dataset_name == "qm9":
            self._setup_qm9(stage)
        elif self.dataset_name == "coconut":
            self._setup_coconut(stage)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Update tokenizer with max nodes
        # Add safety margin for hierarchical tokenizers that encode partition metadata
        # (num_communities, part_ids, num_edges, etc.) using the same index space
        if self.tokenizer is not None:
            effective_max_nodes = self.max_num_nodes
            tokenizer_name = type(self.tokenizer).__name__
            if tokenizer_name in ("HSENTTokenizer", "HDTTokenizer", "HDTCTokenizer"):
                # Hierarchical tokenizers may need extra headroom for:
                # - partition IDs, community counts, node counts per partition
                # - bipartite edge counts (can be large for complex molecules)
                # Use 3x margin to be safe for complex COCONUT molecules
                effective_max_nodes = int(self.max_num_nodes * 3)
                log.info(
                    f"Hierarchical tokenizer: max_num_nodes {self.max_num_nodes} -> {effective_max_nodes}"
                )
            self.tokenizer.set_num_nodes(effective_max_nodes)

            # Configure labeled graph support (AutoGraph format)
            if (
                hasattr(self.tokenizer, "labeled_graph")
                and self.tokenizer.labeled_graph
            ):
                from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES

                self.tokenizer.set_num_node_and_edge_types(
                    num_node_types=NUM_ATOM_TYPES,
                    num_edge_types=NUM_BOND_TYPES,
                )

    def _setup_moses(self, stage: Optional[str] = None) -> None:
        """Set up MOSES dataset."""
        labeled = (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "labeled_graph")
            and self.tokenizer.labeled_graph
        )

        if stage == "fit" or stage is None:
            # Try to load cached training data
            cached_train = self._try_load_cache("train", self.num_train)

            if cached_train is not None:
                log.info("✓ Using cached training data")
                self.train_dataset = cached_train
                self.train_smiles = cached_train.smiles_list
            else:
                # Fall back to on-the-fly tokenization
                log.info("Loading training data (on-the-fly tokenization)")
                train_mol = MolecularDataset.from_moses(
                    split="train",
                    max_molecules=self.num_train,
                    include_hydrogens=self.include_hydrogens,
                    labeled=labeled,
                )
                self.train_smiles = train_mol.smiles_list
                self.max_num_nodes = max(self.max_num_nodes, train_mol.max_num_nodes)
                self.train_dataset = MolecularGraphDataset(
                    train_mol, transform=self.tokenizer
                )

            # Try to load cached validation data
            val_size = (
                self.num_val
                if self.num_val
                else min(10000, len(self.train_smiles) // 10)
            )
            cached_val = self._try_load_cache("val", val_size)

            if cached_val is not None:
                log.info("✓ Using cached validation data")
                self.val_dataset = cached_val
                self.val_smiles = cached_val.smiles_list
            else:
                # Fall back to on-the-fly tokenization
                log.info("Loading validation data (on-the-fly tokenization)")
                val_mol = MolecularDataset.from_moses(
                    split="test",
                    max_molecules=val_size,
                    include_hydrogens=self.include_hydrogens,
                    labeled=labeled,
                )
                self.val_smiles = val_mol.smiles_list
                self.max_num_nodes = max(self.max_num_nodes, val_mol.max_num_nodes)
                self.val_dataset = MolecularGraphDataset(
                    val_mol, transform=self.tokenizer
                )

        if stage == "test" or stage is None:
            # Load train SMILES for metrics even in test mode
            if stage == "test" and len(self.train_smiles) == 0:
                # Try cache first
                cached_train = self._try_load_cache("train", self.num_train)
                if cached_train is not None:
                    self.train_smiles = cached_train.smiles_list
                    self.max_num_nodes = max(
                        self.max_num_nodes, cached_train.max_num_nodes
                    )
                else:
                    train_mol = MolecularDataset.from_moses(
                        split="train",
                        max_molecules=self.num_train,
                        include_hydrogens=self.include_hydrogens,
                        labeled=labeled,
                    )
                    self.train_smiles = train_mol.smiles_list
                    self.max_num_nodes = max(
                        self.max_num_nodes, train_mol.max_num_nodes
                    )

            # Try to load cached test data
            cached_test = self._try_load_cache("test", self.num_test)

            if cached_test is not None:
                log.info("✓ Using cached test data")
                self.test_dataset = cached_test
                self.test_smiles = cached_test.smiles_list
            else:
                # Fall back to on-the-fly tokenization
                log.info("Loading test data (on-the-fly tokenization)")
                test_mol = MolecularDataset.from_moses(
                    split="test",
                    max_molecules=self.num_test,
                    include_hydrogens=self.include_hydrogens,
                    labeled=labeled,
                )
                self.test_smiles = test_mol.smiles_list
                self.max_num_nodes = max(self.max_num_nodes, test_mol.max_num_nodes)
                self.test_dataset = MolecularGraphDataset(
                    test_mol, transform=self.tokenizer
                )

    def _setup_qm9(self, stage: Optional[str] = None) -> None:
        """Set up QM9 dataset with train/val/test splits."""
        from torch_geometric.datasets import QM9
        import numpy as np

        labeled = (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "labeled_graph")
            and self.tokenizer.labeled_graph
        )

        # Load full QM9 dataset
        full_dataset = QM9(root=f"{self.data_root}/qm9")
        num_molecules = len(full_dataset)

        # Create reproducible split (80/10/10)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(num_molecules)

        train_size = int(0.8 * num_molecules)
        val_size = int(0.1 * num_molecules)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Apply size limits
        if self.num_train:
            train_indices = train_indices[: self.num_train]
        if self.num_val:
            val_indices = val_indices[: self.num_val]
        if self.num_test:
            test_indices = test_indices[: self.num_test]

        # Extract SMILES and create datasets
        def get_smiles_subset(indices):
            smiles_list = []
            for idx in indices:
                data = full_dataset[int(idx)]
                if hasattr(data, "smiles"):
                    smiles_list.append(data.smiles)
            return smiles_list

        if stage == "fit" or stage is None:
            train_smiles = get_smiles_subset(train_indices)
            train_mol = MolecularDataset(
                train_smiles,
                dataset_name="qm9_train",
                include_hydrogens=self.include_hydrogens,
                labeled=labeled,
            )
            self.train_smiles = train_mol.smiles_list
            self.max_num_nodes = max(self.max_num_nodes, train_mol.max_num_nodes)
            self.train_dataset = MolecularGraphDataset(
                train_mol, transform=self.tokenizer
            )

            val_smiles = get_smiles_subset(val_indices)
            val_mol = MolecularDataset(
                val_smiles,
                dataset_name="qm9_val",
                include_hydrogens=self.include_hydrogens,
                labeled=labeled,
            )
            self.val_smiles = val_mol.smiles_list
            self.max_num_nodes = max(self.max_num_nodes, val_mol.max_num_nodes)
            self.val_dataset = MolecularGraphDataset(val_mol, transform=self.tokenizer)

        if stage == "test" or stage is None:
            test_smiles = get_smiles_subset(test_indices)
            test_mol = MolecularDataset(
                test_smiles,
                dataset_name="qm9_test",
                include_hydrogens=self.include_hydrogens,
                labeled=labeled,
            )
            self.test_smiles = test_mol.smiles_list
            self.max_num_nodes = max(self.max_num_nodes, test_mol.max_num_nodes)
            self.test_dataset = MolecularGraphDataset(
                test_mol, transform=self.tokenizer
            )

    def _setup_coconut(self, stage: Optional[str] = None) -> None:
        """Set up COCONUT dataset for fine-tuning.

        Loads complex natural products from COCONUT and creates train/val/test
        splits for transfer learning experiments.
        """
        import numpy as np

        from src.data.coconut_loader import CoconutLoader

        labeled = (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "labeled_graph")
            and self.tokenizer.labeled_graph
        )

        # Determine data file path
        data_file = self.data_file or "data/coconut_complex.smi"

        # Load all molecules that pass filtering
        loader = CoconutLoader(
            min_atoms=self.min_atoms,
            max_atoms=self.max_atoms,
            min_rings=self.min_rings,
            data_file=data_file,
        )

        # Calculate total samples needed
        total_train = self.num_train or 5000
        total_val = self.num_val or 500
        total_test = self.num_test or 500
        total_needed = total_train + total_val + total_test

        log.info(f"Loading COCONUT molecules from {data_file}")
        all_smiles = loader.load_smiles(n_samples=total_needed, seed=self.seed)
        log.info(f"Loaded {len(all_smiles)} molecules passing complexity filters")

        if len(all_smiles) < total_needed:
            log.warning(
                f"Only {len(all_smiles)} molecules available, adjusting split sizes"
            )
            # Adjust proportionally
            ratio = len(all_smiles) / total_needed
            total_train = int(total_train * ratio)
            total_val = int(total_val * ratio)
            total_test = len(all_smiles) - total_train - total_val

        # Split into train/val/test
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(all_smiles))

        train_smiles = [all_smiles[i] for i in indices[:total_train]]
        val_smiles = [
            all_smiles[i] for i in indices[total_train : total_train + total_val]
        ]
        test_smiles = [
            all_smiles[i]
            for i in indices[
                total_train + total_val : total_train + total_val + total_test
            ]
        ]

        log.info(
            f"Split sizes: train={len(train_smiles)}, val={len(val_smiles)}, test={len(test_smiles)}"
        )

        if stage == "fit" or stage is None:
            train_mol = MolecularDataset(
                train_smiles,
                dataset_name="coconut_train",
                include_hydrogens=self.include_hydrogens,
                labeled=labeled,
            )
            self.train_smiles = train_mol.smiles_list
            self.max_num_nodes = max(self.max_num_nodes, train_mol.max_num_nodes)
            self.train_dataset = MolecularGraphDataset(
                train_mol, transform=self.tokenizer
            )

            val_mol = MolecularDataset(
                val_smiles,
                dataset_name="coconut_val",
                include_hydrogens=self.include_hydrogens,
                labeled=labeled,
            )
            self.val_smiles = val_mol.smiles_list
            self.max_num_nodes = max(self.max_num_nodes, val_mol.max_num_nodes)
            self.val_dataset = MolecularGraphDataset(val_mol, transform=self.tokenizer)

        if stage == "test" or stage is None:
            # Load train SMILES for metrics even in test mode
            if stage == "test" and len(self.train_smiles) == 0:
                train_mol = MolecularDataset(
                    train_smiles,
                    dataset_name="coconut_train",
                    include_hydrogens=self.include_hydrogens,
                    labeled=labeled,
                )
                self.train_smiles = train_mol.smiles_list
                self.max_num_nodes = max(self.max_num_nodes, train_mol.max_num_nodes)

            test_mol = MolecularDataset(
                test_smiles,
                dataset_name="coconut_test",
                include_hydrogens=self.include_hydrogens,
                labeled=labeled,
            )
            self.test_smiles = test_mol.smiles_list
            self.max_num_nodes = max(self.max_num_nodes, test_mol.max_num_nodes)
            self.test_dataset = MolecularGraphDataset(
                test_mol, transform=self.tokenizer
            )

    def _collate_fn(self, batch: list) -> torch.Tensor:
        """Collate function for batching tokenized graphs.

        Args:
            batch: List of tokenized sequences.

        Returns:
            Padded tensor batch.
        """
        if self.tokenizer is None:
            return batch

        return self.tokenizer.batch_converter()(batch)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        assert self.train_dataset is not None, "Call setup() first"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn if self.tokenizer else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        assert self.val_dataset is not None, "Call setup() first"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn if self.tokenizer else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        assert self.test_dataset is not None, "Call setup() first"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn if self.tokenizer else None,
        )
