"""Murcko scaffold extraction utilities.

This module provides the MurckoExtractor class for extracting Murcko scaffolds
from molecules for scaffold priming evaluation.
"""

from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class MurckoExtractor:
    """Extract Murcko scaffolds from molecules.

    The Murcko scaffold is the core ring structure of a molecule, consisting
    of ring systems and the linkers between them, with all side chains removed.

    Attributes:
        include_sidechains: Whether to include sidechains in scaffold.
    """

    def __init__(self, include_sidechains: bool = False) -> None:
        """Initialize the Murcko extractor.

        Args:
            include_sidechains: If True, includes sidechains attached to rings.
                Default is False (pure Murcko scaffold).
        """
        self.include_sidechains = include_sidechains

    def extract_scaffold(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Extract Murcko scaffold (ring systems + linkers).

        The Murcko scaffold consists of all ring systems and the atoms
        that connect them (linker atoms), but not the side chains.

        Args:
            mol: RDKit Mol object.

        Returns:
            RDKit Mol object of the scaffold, or None if extraction fails.
        """
        try:
            if self.include_sidechains:
                return MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)
                )
            return MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            return None

    def extract_core(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Extract core scaffold (rings only, no linkers).

        The core scaffold consists only of ring atoms, without the
        linker atoms between ring systems.

        Args:
            mol: RDKit Mol object.

        Returns:
            RDKit Mol object of the core scaffold, or None if extraction fails.
        """
        try:
            return MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
        except Exception:
            return None

    def get_scaffold_smiles(self, mol: Chem.Mol) -> Optional[str]:
        """Get canonical SMILES of the Murcko scaffold.

        Args:
            mol: RDKit Mol object.

        Returns:
            Canonical SMILES string of scaffold, or None if extraction fails.
        """
        scaffold = self.extract_scaffold(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)

    def get_core_smiles(self, mol: Chem.Mol) -> Optional[str]:
        """Get canonical SMILES of the core scaffold.

        Args:
            mol: RDKit Mol object.

        Returns:
            Canonical SMILES string of core scaffold, or None if extraction fails.
        """
        core = self.extract_core(mol)
        if core is None:
            return None
        return Chem.MolToSmiles(core)

    def scaffold_from_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Extract scaffold from a SMILES string.

        Convenience method that parses SMILES and extracts scaffold.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            RDKit Mol object of the scaffold, or None if extraction fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.extract_scaffold(mol)

    def scaffold_smiles_from_smiles(self, smiles: str) -> Optional[str]:
        """Get scaffold SMILES from a molecule SMILES.

        Convenience method that parses SMILES and returns scaffold SMILES.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Canonical SMILES string of scaffold, or None if extraction fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.get_scaffold_smiles(mol)

    def get_scaffold_ratio(self, mol: Chem.Mol) -> float:
        """Calculate the ratio of scaffold atoms to total atoms.

        A higher ratio indicates more of the molecule is in the core
        ring structure rather than side chains.

        Args:
            mol: RDKit Mol object.

        Returns:
            Float between 0 and 1 representing scaffold/total atoms.
        """
        scaffold = self.extract_scaffold(mol)
        if scaffold is None:
            return 0.0
        return scaffold.GetNumAtoms() / mol.GetNumAtoms()
