"""COCONUT data loader for complex molecules.

This module provides the CoconutLoader class for loading and filtering
complex natural products from COCONUT for scaffold priming evaluation.
"""

from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class CoconutLoader:
    """Load and filter complex natural products from COCONUT.

    Loads molecules from a pre-filtered SMILES file and applies additional
    complexity filtering to ensure molecules are suitable for scaffold
    priming evaluation.

    Attributes:
        min_atoms: Minimum number of atoms required.
        max_atoms: Maximum number of atoms allowed.
        min_rings: Minimum number of rings required.
        min_scaffold_atoms: Minimum atoms in the Murcko scaffold.
        data_file: Path to the SMILES file.
    """

    def __init__(
        self,
        min_atoms: int = 30,
        max_atoms: int = 100,
        min_rings: int = 4,
        min_scaffold_atoms: int = 15,
        data_file: str = "data/coconut_complex.smi",
    ) -> None:
        """Initialize the COCONUT loader.

        Args:
            min_atoms: Minimum number of atoms required.
            max_atoms: Maximum number of atoms allowed.
            min_rings: Minimum number of rings required.
            min_scaffold_atoms: Minimum atoms in the Murcko scaffold.
            data_file: Path to the SMILES file containing pre-filtered molecules.
        """
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_rings = min_rings
        self.min_scaffold_atoms = min_scaffold_atoms
        self.data_file = Path(data_file)

    def load_molecules(
        self,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> list[Chem.Mol]:
        """Load filtered complex molecules from pre-filtered SMILES file.

        Args:
            n_samples: Maximum number of molecules to return. If None, returns
                all molecules that pass filtering.
            seed: Random seed for shuffling when n_samples is specified.

        Returns:
            List of RDKit Mol objects that pass complexity filtering.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"COCONUT data file not found: {self.data_file}. "
                "Generate it with: python scripts/prepare_coconut_data.py"
            )

        molecules: list[Chem.Mol] = []

        with open(self.data_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Handle SMILES with optional name/ID after whitespace
                parts = line.split()
                smiles = parts[0]

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                if self.filter_by_complexity(mol):
                    molecules.append(mol)

        # Shuffle and limit if n_samples is specified
        if n_samples is not None and n_samples < len(molecules):
            import random

            random.seed(seed)
            random.shuffle(molecules)
            molecules = molecules[:n_samples]

        return molecules

    def filter_by_complexity(self, mol: Chem.Mol) -> bool:
        """Check if molecule meets complexity requirements.

        Args:
            mol: RDKit Mol object to check.

        Returns:
            True if molecule meets all complexity requirements.
        """
        # Check atom count
        num_atoms = mol.GetNumAtoms()
        if num_atoms < self.min_atoms or num_atoms > self.max_atoms:
            return False

        # Check ring count
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        if num_rings < self.min_rings:
            return False

        # Check scaffold atoms if min_scaffold_atoms is set
        if self.min_scaffold_atoms > 0:
            try:
                from rdkit.Chem.Scaffolds import MurckoScaffold

                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold.GetNumAtoms() < self.min_scaffold_atoms:
                    return False
            except Exception:
                return False

        return True

    def get_complexity_metrics(self, mol: Chem.Mol) -> dict:
        """Calculate complexity metrics for a molecule.

        Args:
            mol: RDKit Mol object.

        Returns:
            Dictionary of complexity metrics.
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold

        metrics = {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "molecular_weight": Descriptors.MolWt(mol),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        }

        # Add scaffold metrics
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            metrics["scaffold_num_atoms"] = scaffold.GetNumAtoms()
            metrics["scaffold_num_rings"] = rdMolDescriptors.CalcNumRings(scaffold)
        except Exception:
            metrics["scaffold_num_atoms"] = 0
            metrics["scaffold_num_rings"] = 0

        return metrics

    def load_smiles(
        self,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> list[str]:
        """Load SMILES strings instead of Mol objects.

        Convenience method that returns SMILES strings for molecules
        that pass complexity filtering.

        Args:
            n_samples: Maximum number of SMILES to return.
            seed: Random seed for shuffling.

        Returns:
            List of SMILES strings.
        """
        molecules = self.load_molecules(n_samples=n_samples, seed=seed)
        return [Chem.MolToSmiles(mol) for mol in molecules]
