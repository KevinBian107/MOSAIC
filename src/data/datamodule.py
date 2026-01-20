"""PyTorch Lightning data module for molecular graph datasets.

This module provides a unified interface for loading molecular graph datasets
with tokenization support for training graph generation models.
"""

from functools import partial
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.molecular import MolecularDataset


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
    ) -> None:
        """Initialize the data module.

        Args:
            dataset_name: Name of dataset to use ('moses' or 'qm9').
            tokenizer: Graph tokenizer instance.
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            num_train: Number of training molecules (None for all).
            num_val: Number of validation molecules (None for all).
            num_test: Number of test molecules (None for all).
            include_hydrogens: Whether to include explicit hydrogens.
            seed: Random seed for reproducibility.
            data_root: Root directory for data storage.
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

        self.train_dataset: Optional[MolecularGraphDataset] = None
        self.val_dataset: Optional[MolecularGraphDataset] = None
        self.test_dataset: Optional[MolecularGraphDataset] = None

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self.dataset_name == "moses":
            self._setup_moses(stage)
        elif self.dataset_name == "qm9":
            self._setup_qm9(stage)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Update tokenizer with max nodes
        if self.tokenizer is not None:
            self.tokenizer.set_num_nodes(self.max_num_nodes)

            # Configure labeled graph support (AutoGraph format)
            if hasattr(self.tokenizer, 'labeled_graph') and self.tokenizer.labeled_graph:
                from src.data.molecular import NUM_ATOM_TYPES, NUM_BOND_TYPES
                self.tokenizer.set_num_node_and_edge_types(
                    num_node_types=NUM_ATOM_TYPES,
                    num_edge_types=NUM_BOND_TYPES,
                )

    def _setup_moses(self, stage: Optional[str] = None) -> None:
        """Set up MOSES dataset."""
        labeled = (self.tokenizer is not None and
                  hasattr(self.tokenizer, 'labeled_graph') and
                  self.tokenizer.labeled_graph)

        if stage == "fit" or stage is None:
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

            # Use subset of training data for validation if not specified
            val_size = self.num_val if self.num_val else min(10000, len(train_mol) // 10)
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
            if stage == "test" and not hasattr(self, 'train_smiles'):
                train_mol = MolecularDataset.from_moses(
                    split="train",
                    max_molecules=self.num_train,
                    include_hydrogens=self.include_hydrogens,
                    labeled=labeled,
                )
                self.train_smiles = train_mol.smiles_list
                self.max_num_nodes = max(self.max_num_nodes, train_mol.max_num_nodes)

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

        labeled = (self.tokenizer is not None and
                  hasattr(self.tokenizer, 'labeled_graph') and
                  self.tokenizer.labeled_graph)

        # Load full QM9 dataset
        full_dataset = QM9(root=f"{self.data_root}/qm9")
        num_molecules = len(full_dataset)

        # Create reproducible split (80/10/10)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(num_molecules)

        train_size = int(0.8 * num_molecules)
        val_size = int(0.1 * num_molecules)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Apply size limits
        if self.num_train:
            train_indices = train_indices[:self.num_train]
        if self.num_val:
            val_indices = val_indices[:self.num_val]
        if self.num_test:
            test_indices = test_indices[:self.num_test]

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
            self.val_dataset = MolecularGraphDataset(
                val_mol, transform=self.tokenizer
            )

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
