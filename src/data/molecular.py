"""Molecular dataset loading and processing.

This module provides utilities for loading molecular datasets (MOSES, QM9)
and converting them to PyTorch Geometric graph format.
"""

from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


# Atom types supported (common in drug-like molecules)
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES) + 1  # +1 for unknown

# Bond types
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPE_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}
NUM_BOND_TYPES = len(BOND_TYPES) + 1  # +1 for unknown


def smiles_to_graph(
    smiles: str,
    include_hydrogens: bool = False,
    compute_2d_coords: bool = False,
) -> Optional[Data]:
    """Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string representation of the molecule.
        include_hydrogens: Whether to include explicit hydrogens.
        compute_2d_coords: Whether to compute 2D coordinates.

    Returns:
        PyG Data object with node and edge features, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if include_hydrogens:
        mol = Chem.AddHs(mol)

    if compute_2d_coords:
        AllChem.Compute2DCoords(mol)

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        # Atom type (one-hot)
        atom_type_idx = ATOM_TYPE_TO_IDX.get(atom.GetSymbol(), NUM_ATOM_TYPES - 1)
        atom_type_onehot = [0] * NUM_ATOM_TYPES
        atom_type_onehot[atom_type_idx] = 1

        # Additional features
        features = atom_type_onehot + [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetDegree(),
        ]
        node_features.append(features)

    if len(node_features) == 0:
        return None

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index and features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Bond type (one-hot)
        bond_type = bond.GetBondType()
        bond_type_idx = BOND_TYPE_TO_IDX.get(bond_type, NUM_BOND_TYPES - 1)
        bond_type_onehot = [0] * NUM_BOND_TYPES
        bond_type_onehot[bond_type_idx] = 1

        # Additional bond features
        bond_feat = bond_type_onehot + [
            int(bond.GetIsAromatic()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
        ]

        # Add both directions (undirected graph)
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bond_feat, bond_feat])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, NUM_BOND_TYPES + 3), dtype=torch.float)

    # Store 2D coordinates if computed
    pos = None
    if compute_2d_coords:
        conformer = mol.GetConformer()
        pos = torch.tensor(
            [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y]
             for i in range(mol.GetNumAtoms())],
            dtype=torch.float,
        )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        num_nodes=mol.GetNumAtoms(),
    )
    data.smiles = smiles

    return data


def graph_to_smiles(data: Data) -> Optional[str]:
    """Convert a PyTorch Geometric Data object back to SMILES.

    Args:
        data: PyG Data object with node and edge features.

    Returns:
        SMILES string or None if conversion fails.
    """
    try:
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Create empty editable molecule
        mol = Chem.RWMol()

        # Add atoms
        for i in range(num_nodes):
            node_feat = data.x[i].numpy()
            # Extract atom type from one-hot encoding
            atom_type_idx = int(np.argmax(node_feat[:NUM_ATOM_TYPES]))
            if atom_type_idx < len(ATOM_TYPES):
                atom_symbol = ATOM_TYPES[atom_type_idx]
            else:
                atom_symbol = "C"  # Default to carbon for unknown

            atom = Chem.Atom(atom_symbol)

            # Set formal charge if available
            if len(node_feat) > NUM_ATOM_TYPES + 1:
                formal_charge = int(node_feat[NUM_ATOM_TYPES + 1])
                atom.SetFormalCharge(formal_charge)

            mol.AddAtom(atom)

        # Add bonds (only process each edge once)
        added_bonds = set()
        for k in range(edge_index.size(1)):
            i = int(edge_index[0, k])
            j = int(edge_index[1, k])
            if i < j and (i, j) not in added_bonds:
                added_bonds.add((i, j))

                # Extract bond type from edge features
                if data.edge_attr is not None and data.edge_attr.size(0) > k:
                    edge_feat = data.edge_attr[k].numpy()
                    bond_type_idx = int(np.argmax(edge_feat[:NUM_BOND_TYPES]))
                    if bond_type_idx < len(BOND_TYPES):
                        bond_type = BOND_TYPES[bond_type_idx]
                    else:
                        bond_type = Chem.rdchem.BondType.SINGLE
                else:
                    bond_type = Chem.rdchem.BondType.SINGLE

                mol.AddBond(i, j, bond_type)

        # Sanitize and convert to SMILES
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    except Exception:
        return None


def load_moses_dataset(split: str = "train") -> list[str]:
    """Load MOSES dataset.

    Args:
        split: Dataset split ('train', 'test', 'test_scaffolds').

    Returns:
        List of SMILES strings.
    """
    try:
        import moses
        return moses.get_dataset(split)
    except ImportError:
        raise ImportError(
            "MOSES package not installed. Install with: pip install molsets"
        )


def load_qm9_smiles(
    root: str = "data/qm9",
    max_molecules: Optional[int] = None,
) -> list[str]:
    """Load QM9 dataset SMILES.

    Args:
        root: Root directory for data storage.
        max_molecules: Maximum number of molecules to load.

    Returns:
        List of SMILES strings.
    """
    try:
        from torch_geometric.datasets import QM9

        dataset = QM9(root=root)
        smiles_list = []

        for i, data in enumerate(dataset):
            if max_molecules is not None and i >= max_molecules:
                break
            if hasattr(data, "smiles"):
                smiles_list.append(data.smiles)

        return smiles_list

    except Exception as e:
        raise RuntimeError(f"Failed to load QM9 dataset: {e}")


class MolecularDataset:
    """Dataset class for molecular graphs.

    Attributes:
        smiles_list: List of SMILES strings.
        graphs: List of PyG Data objects.
        dataset_name: Name of the dataset.
    """

    def __init__(
        self,
        smiles_list: list[str],
        dataset_name: str = "molecular",
        include_hydrogens: bool = False,
        max_molecules: Optional[int] = None,
    ) -> None:
        """Initialize molecular dataset.

        Args:
            smiles_list: List of SMILES strings.
            dataset_name: Name identifier for the dataset.
            include_hydrogens: Whether to include explicit hydrogens.
            max_molecules: Maximum number of molecules to include.
        """
        self.dataset_name = dataset_name
        self.include_hydrogens = include_hydrogens

        if max_molecules is not None:
            smiles_list = smiles_list[:max_molecules]

        self.smiles_list = []
        self.graphs = []

        for smiles in smiles_list:
            graph = smiles_to_graph(smiles, include_hydrogens=include_hydrogens)
            if graph is not None:
                graph.dataset_name = dataset_name
                self.smiles_list.append(smiles)
                self.graphs.append(graph)

    def __len__(self) -> int:
        """Return number of molecules."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Get a single molecular graph."""
        return self.graphs[idx]

    @property
    def max_num_nodes(self) -> int:
        """Return maximum number of nodes across all graphs."""
        return max(g.num_nodes for g in self.graphs) if self.graphs else 0

    @classmethod
    def from_moses(
        cls,
        split: str = "train",
        max_molecules: Optional[int] = None,
        include_hydrogens: bool = False,
    ) -> "MolecularDataset":
        """Create dataset from MOSES.

        Args:
            split: Dataset split ('train', 'test', 'test_scaffolds').
            max_molecules: Maximum number of molecules to load.
            include_hydrogens: Whether to include explicit hydrogens.

        Returns:
            MolecularDataset instance.
        """
        smiles_list = load_moses_dataset(split)
        return cls(
            smiles_list,
            dataset_name=f"moses_{split}",
            include_hydrogens=include_hydrogens,
            max_molecules=max_molecules,
        )

    @classmethod
    def from_qm9(
        cls,
        root: str = "data/qm9",
        max_molecules: Optional[int] = None,
        include_hydrogens: bool = False,
    ) -> "MolecularDataset":
        """Create dataset from QM9.

        Args:
            root: Root directory for data storage.
            max_molecules: Maximum number of molecules to load.
            include_hydrogens: Whether to include explicit hydrogens.

        Returns:
            MolecularDataset instance.
        """
        smiles_list = load_qm9_smiles(root, max_molecules)
        return cls(
            smiles_list,
            dataset_name="qm9",
            include_hydrogens=include_hydrogens,
            max_molecules=max_molecules,
        )
