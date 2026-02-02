"""Scaffold library for managing scaffold structures.

This module provides the Scaffold dataclass and ScaffoldLibrary class for
managing scaffold structures used in priming.
"""

from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from torch_geometric.data import Data

from src.data.molecular import smiles_to_graph
from src.transfer_learning.scaffolds.tier_patterns import (
    ALL_SCAFFOLDS,
    SCAFFOLD_TIERS,
)


@dataclass
class Scaffold:
    """A molecular scaffold structure.

    Attributes:
        name: Scaffold identifier name.
        smiles: Canonical SMILES representation.
        tier: Complexity tier (1=simple, 2=fused, 3=complex).
        category: Structural classification.
        num_atoms: Number of atoms in the scaffold.
        graph: Optional PyG Data object (lazy-loaded).
    """

    name: str
    smiles: str
    tier: int
    category: str
    num_atoms: int = field(default=0)
    graph: Optional[Data] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize num_atoms from SMILES if not provided."""
        if self.num_atoms == 0:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is not None:
                self.num_atoms = mol.GetNumAtoms()

    def get_graph(self, labeled: bool = True) -> Optional[Data]:
        """Get or create the PyG graph representation.

        Args:
            labeled: If True, use integer labels (AutoGraph format).

        Returns:
            PyG Data object or None if conversion fails.
        """
        if self.graph is None:
            self.graph = smiles_to_graph(self.smiles, labeled=labeled)
        return self.graph

    def to_mol(self) -> Optional[Chem.Mol]:
        """Convert scaffold to RDKit Mol object.

        Returns:
            RDKit Mol object or None if conversion fails.
        """
        return Chem.MolFromSmiles(self.smiles)


class ScaffoldLibrary:
    """Library for managing and accessing scaffold structures.

    This class provides methods for retrieving scaffolds by name, tier,
    or category, as well as creating scaffolds from custom SMILES.

    Attributes:
        scaffolds: Dictionary mapping scaffold names to Scaffold objects.
    """

    def __init__(self) -> None:
        """Initialize the scaffold library with default patterns."""
        self.scaffolds: dict[str, Scaffold] = {}
        self._load_default_scaffolds()

    def _load_default_scaffolds(self) -> None:
        """Load all default scaffold patterns."""
        for name, info in ALL_SCAFFOLDS.items():
            tier = SCAFFOLD_TIERS.get(name, 1)
            self.scaffolds[name] = Scaffold(
                name=name,
                smiles=info["smiles"],
                tier=tier,
                category=info["category"],
            )

    def get_scaffold(self, name: str) -> Scaffold:
        """Get a scaffold by name.

        Args:
            name: Scaffold name (e.g., "benzene", "naphthalene").

        Returns:
            Scaffold object.

        Raises:
            KeyError: If scaffold name not found.
        """
        if name not in self.scaffolds:
            raise KeyError(
                f"Scaffold '{name}' not found. Available: {list(self.scaffolds.keys())}"
            )
        return self.scaffolds[name]

    def get_scaffolds_by_tier(self, tier: int) -> list[Scaffold]:
        """Get all scaffolds for a given complexity tier.

        Args:
            tier: Complexity tier (1, 2, or 3).

        Returns:
            List of Scaffold objects for that tier.

        Raises:
            ValueError: If tier is not 1, 2, or 3.
        """
        if tier not in (1, 2, 3):
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        return [s for s in self.scaffolds.values() if s.tier == tier]

    def get_scaffolds_by_category(self, category: str) -> list[Scaffold]:
        """Get all scaffolds for a given category.

        Args:
            category: Structural category (e.g., "aromatic_6", "fused_hetero_56").

        Returns:
            List of Scaffold objects for that category.
        """
        return [s for s in self.scaffolds.values() if s.category == category]

    def from_smiles(
        self,
        smiles: str,
        name: str = "custom",
        tier: int = 0,
        category: str = "custom",
    ) -> Scaffold:
        """Create a scaffold from a SMILES string.

        Args:
            smiles: SMILES string representation.
            name: Optional name for the scaffold.
            tier: Complexity tier (0 = custom/unknown).
            category: Structural category.

        Returns:
            New Scaffold object.

        Raises:
            ValueError: If SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Canonicalize SMILES
        canonical_smiles = Chem.MolToSmiles(mol)

        return Scaffold(
            name=name,
            smiles=canonical_smiles,
            tier=tier,
            category=category,
            num_atoms=mol.GetNumAtoms(),
        )

    def add_scaffold(self, scaffold: Scaffold) -> None:
        """Add a scaffold to the library.

        Args:
            scaffold: Scaffold object to add.
        """
        self.scaffolds[scaffold.name] = scaffold

    def list_scaffolds(self) -> list[str]:
        """List all available scaffold names.

        Returns:
            List of scaffold names.
        """
        return list(self.scaffolds.keys())

    def list_categories(self) -> list[str]:
        """List all unique scaffold categories.

        Returns:
            List of unique category names.
        """
        return list(set(s.category for s in self.scaffolds.values()))

    def __len__(self) -> int:
        """Return number of scaffolds in library."""
        return len(self.scaffolds)

    def __contains__(self, name: str) -> bool:
        """Check if scaffold name exists in library."""
        return name in self.scaffolds

    def __getitem__(self, name: str) -> Scaffold:
        """Get scaffold by name using indexing syntax."""
        return self.get_scaffold(name)
