"""PyTorch Lightning data module for graph datasets.

This module provides a unified interface for loading graph datasets with
tokenization support for training graph generation models.
"""

from functools import partial
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch_geometric.data import Data, InMemoryDataset

from src.data.synthetic import SyntheticGraphGenerator


class SyntheticInMemoryDataset(InMemoryDataset):
    """In-memory dataset for synthetic graphs.

    Attributes:
        graphs: List of pre-generated graphs.
        dataset_name: Name identifier for the dataset.
    """

    def __init__(
        self,
        graphs: list[Data],
        dataset_name: str = "synthetic",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            graphs: List of PyG Data objects.
            dataset_name: Name identifier.
            transform: Transform to apply on each access.
            pre_transform: Transform to apply once during processing.
        """
        self.graphs = graphs
        self.dataset_name = dataset_name
        self._transform = transform
        self._pre_transform = pre_transform

        self._data_list = []
        for g in graphs:
            if pre_transform is not None:
                g = pre_transform(g)
            g.dataset_name = dataset_name
            self._data_list.append(g)

        super().__init__(root=None, transform=transform)

    @property
    def raw_file_names(self) -> list[str]:
        """Return empty list - no raw files needed."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Return empty list - no processed files needed."""
        return []

    def download(self) -> None:
        """No download needed for synthetic data."""
        pass

    def process(self) -> None:
        """No processing needed - data is in memory."""
        pass

    def len(self) -> int:
        """Return the number of graphs."""
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        """Get a single graph by index."""
        data = self._data_list[idx]
        if self._transform is not None:
            data = self._transform(data)
        return data


def add_dataset_name(data: Data, dataset_name: str) -> Data:
    """Add dataset name attribute to a Data object.

    Args:
        data: PyG Data object.
        dataset_name: Name to add.

    Returns:
        Modified Data object.
    """
    data.dataset_name = dataset_name
    return data


class GraphDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for graph datasets.

    This module handles data loading, tokenization, and batching for
    training graph generation models.

    Attributes:
        tokenizer: Graph tokenizer for converting graphs to sequences.
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.
    """

    def __init__(
        self,
        generator_configs: Optional[list[dict[str, Any]]] = None,
        tokenizer: Optional[Any] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        num_train: int = 1000,
        num_val: int = 100,
        num_test: int = 100,
        seed: int = 42,
    ) -> None:
        """Initialize the data module.

        Args:
            generator_configs: List of dicts with 'name' and optional params.
            tokenizer: Graph tokenizer instance.
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            num_train: Number of training graphs per generator.
            num_val: Number of validation graphs per generator.
            num_test: Number of test graphs per generator.
            seed: Random seed.
        """
        super().__init__()

        if generator_configs is None:
            generator_configs = [{"name": "erdos_renyi"}]

        self.generator_configs = generator_configs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed

        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None
        self.test_dataset: Optional[ConcatDataset] = None

        self.max_num_nodes: int = 0

    def prepare_data(self) -> None:
        """Generate all synthetic data (called once)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for i, config in enumerate(self.generator_configs):
            gen_name = config["name"]
            gen_params = {k: v for k, v in config.items() if k != "name"}

            generator = SyntheticGraphGenerator(
                gen_name,
                seed=self.seed + i,
            )

            splits = generator.generate_dataset(
                num_train=self.num_train,
                num_val=self.num_val,
                num_test=self.num_test,
                **gen_params,
            )

            max_nodes = max(g.num_nodes for g in splits["train"])
            self.max_num_nodes = max(self.max_num_nodes, max_nodes)

            transform = self.tokenizer if self.tokenizer is not None else None
            pre_transform = partial(add_dataset_name, dataset_name=gen_name)

            train_datasets.append(
                SyntheticInMemoryDataset(
                    splits["train"], gen_name, transform, pre_transform
                )
            )
            val_datasets.append(
                SyntheticInMemoryDataset(
                    splits["val"], gen_name, transform, pre_transform
                )
            )
            test_datasets.append(
                SyntheticInMemoryDataset(
                    splits["test"], gen_name, transform, pre_transform
                )
            )

        if self.tokenizer is not None:
            self.tokenizer.set_num_nodes(self.max_num_nodes)

        if stage == "fit" or stage is None:
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)

        if stage == "test" or stage is None:
            self.test_dataset = ConcatDataset(test_datasets)

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
