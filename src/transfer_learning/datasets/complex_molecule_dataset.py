"""Complex molecule dataset for scaffold priming evaluation.

This module provides the ComplexMoleculeDataset class for managing complex
molecules and their extracted scaffolds for priming evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from torch_geometric.data import Data

from src.data.coconut_loader import CoconutLoader
from src.data.molecular import smiles_to_graph
from src.transfer_learning.scaffolds.murcko_extractor import MurckoExtractor


@dataclass
class ComplexMoleculeSample:
    """A complex molecule with its extracted scaffold.

    Attributes:
        molecule: RDKit Mol object of the complete molecule.
        molecule_smiles: Canonical SMILES of the molecule.
        scaffold: RDKit Mol object of the Murcko scaffold.
        scaffold_smiles: Canonical SMILES of the scaffold.
        scaffold_graph: PyG Data object of the scaffold for tokenization.
        complexity_metrics: Dictionary of complexity metrics.
    """

    molecule: Chem.Mol
    molecule_smiles: str
    scaffold: Chem.Mol
    scaffold_smiles: str
    scaffold_graph: Optional[Data] = field(default=None, repr=False)
    complexity_metrics: dict = field(default_factory=dict)

    def get_scaffold_graph(self, labeled: bool = True) -> Optional[Data]:
        """Get or create the PyG graph for the scaffold.

        Args:
            labeled: If True, use integer labels (AutoGraph format).

        Returns:
            PyG Data object or None if conversion fails.
        """
        if self.scaffold_graph is None:
            self.scaffold_graph = smiles_to_graph(self.scaffold_smiles, labeled=labeled)
        return self.scaffold_graph


class ComplexMoleculeDataset:
    """Dataset of complex molecules with scaffolds for priming evaluation.

    This dataset loads complex molecules from COCONUT, extracts their Murcko
    scaffolds, and provides samples for scaffold priming evaluation.

    Attributes:
        samples: List of ComplexMoleculeSample objects.
        coconut_loader: COCONUT loader instance.
        murcko_extractor: Murcko extractor instance.
    """

    def __init__(
        self,
        coconut_loader: Optional[CoconutLoader] = None,
        murcko_extractor: Optional[MurckoExtractor] = None,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        """Initialize the complex molecule dataset.

        Args:
            coconut_loader: COCONUT loader instance. If None, creates default.
            murcko_extractor: Murcko extractor instance. If None, creates default.
            n_samples: Number of samples to load.
            seed: Random seed for sampling.
        """
        self.coconut_loader = coconut_loader or CoconutLoader()
        self.murcko_extractor = murcko_extractor or MurckoExtractor()
        self.samples: list[ComplexMoleculeSample] = []

        self._prepare_samples(n_samples, seed)

    def _prepare_samples(self, n_samples: int, seed: int) -> None:
        """Prepare samples from COCONUT molecules.

        Args:
            n_samples: Number of samples to prepare.
            seed: Random seed for sampling.
        """
        molecules = self.coconut_loader.load_molecules(n_samples=n_samples, seed=seed)

        for mol in molecules:
            scaffold = self.murcko_extractor.extract_scaffold(mol)
            if scaffold is None:
                continue

            mol_smiles = Chem.MolToSmiles(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)

            complexity_metrics = self.coconut_loader.get_complexity_metrics(mol)

            sample = ComplexMoleculeSample(
                molecule=mol,
                molecule_smiles=mol_smiles,
                scaffold=scaffold,
                scaffold_smiles=scaffold_smiles,
                complexity_metrics=complexity_metrics,
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> ComplexMoleculeSample:
        """Get a sample by index."""
        return self.samples[idx]

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)

    def get_scaffolds(self, labeled: bool = True) -> list[Data]:
        """Get all scaffold graphs.

        Args:
            labeled: If True, use integer labels.

        Returns:
            List of PyG Data objects.
        """
        graphs = []
        for sample in self.samples:
            graph = sample.get_scaffold_graph(labeled=labeled)
            if graph is not None:
                graphs.append(graph)
        return graphs

    def get_unique_scaffolds(self) -> dict[str, list[ComplexMoleculeSample]]:
        """Group samples by unique scaffold SMILES.

        Returns:
            Dictionary mapping scaffold SMILES to list of samples
            with that scaffold.
        """
        scaffold_groups: dict[str, list[ComplexMoleculeSample]] = {}
        for sample in self.samples:
            if sample.scaffold_smiles not in scaffold_groups:
                scaffold_groups[sample.scaffold_smiles] = []
            scaffold_groups[sample.scaffold_smiles].append(sample)
        return scaffold_groups

    def filter_by_scaffold_size(
        self,
        min_atoms: int = 0,
        max_atoms: int = 1000,
    ) -> "ComplexMoleculeDataset":
        """Create a filtered dataset by scaffold size.

        Args:
            min_atoms: Minimum scaffold atoms.
            max_atoms: Maximum scaffold atoms.

        Returns:
            New ComplexMoleculeDataset with filtered samples.
        """
        new_dataset = ComplexMoleculeDataset.__new__(ComplexMoleculeDataset)
        new_dataset.coconut_loader = self.coconut_loader
        new_dataset.murcko_extractor = self.murcko_extractor
        new_dataset.samples = [
            s
            for s in self.samples
            if min_atoms <= s.scaffold.GetNumAtoms() <= max_atoms
        ]
        return new_dataset

    def summary(self) -> dict:
        """Get summary statistics of the dataset.

        Returns:
            Dictionary of summary statistics.
        """
        if not self.samples:
            return {"n_samples": 0}

        scaffold_sizes = [s.scaffold.GetNumAtoms() for s in self.samples]
        mol_sizes = [s.molecule.GetNumAtoms() for s in self.samples]
        unique_scaffolds = len(self.get_unique_scaffolds())

        return {
            "n_samples": len(self.samples),
            "n_unique_scaffolds": unique_scaffolds,
            "scaffold_size_mean": sum(scaffold_sizes) / len(scaffold_sizes),
            "scaffold_size_min": min(scaffold_sizes),
            "scaffold_size_max": max(scaffold_sizes),
            "mol_size_mean": sum(mol_sizes) / len(mol_sizes),
            "mol_size_min": min(mol_sizes),
            "mol_size_max": max(mol_sizes),
        }

    @classmethod
    def from_smiles_list(
        cls,
        smiles_list: list[str],
        murcko_extractor: Optional[MurckoExtractor] = None,
    ) -> "ComplexMoleculeDataset":
        """Create dataset from a list of SMILES.

        Args:
            smiles_list: List of SMILES strings.
            murcko_extractor: Murcko extractor instance.

        Returns:
            New ComplexMoleculeDataset.
        """
        dataset = cls.__new__(cls)
        dataset.coconut_loader = None
        dataset.murcko_extractor = murcko_extractor or MurckoExtractor()
        dataset.samples = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            scaffold = dataset.murcko_extractor.extract_scaffold(mol)
            if scaffold is None:
                continue

            mol_smiles = Chem.MolToSmiles(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)

            sample = ComplexMoleculeSample(
                molecule=mol,
                molecule_smiles=mol_smiles,
                scaffold=scaffold,
                scaffold_smiles=scaffold_smiles,
                complexity_metrics={},
            )
            dataset.samples.append(sample)

        return dataset
