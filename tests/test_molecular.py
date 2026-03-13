"""Tests for molecular data module."""

import pytest

from src.data.molecular import (
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    MolecularDataset,
    graph_to_smiles,
    smiles_to_graph,
)

# Test SMILES strings
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
PHENOL = "c1ccccc1O"
ACETIC_ACID = "CC(=O)O"
INVALID_SMILES = "not_a_molecule"


class TestSmilesToGraph:
    """Tests for SMILES to graph conversion."""

    def test_ethanol_num_nodes(self) -> None:
        """Test ethanol has 3 heavy atoms."""
        data = smiles_to_graph(ETHANOL)
        assert data is not None
        assert data.num_nodes == 3

    def test_benzene_num_nodes(self) -> None:
        """Test benzene has 6 atoms."""
        data = smiles_to_graph(BENZENE)
        assert data is not None
        assert data.num_nodes == 6

    def test_benzene_edges(self) -> None:
        """Test benzene has 12 edges (6 bonds * 2 for undirected)."""
        data = smiles_to_graph(BENZENE)
        assert data is not None
        assert data.edge_index.size(1) == 12

    def test_node_features_shape(self) -> None:
        """Test node features have correct shape."""
        data = smiles_to_graph(ETHANOL)
        assert data is not None
        # NUM_ATOM_TYPES + atomic_num + formal_charge + total_Hs + is_aromatic + is_in_ring + degree
        expected_dim = NUM_ATOM_TYPES + 6
        assert data.x.shape == (3, expected_dim)

    def test_edge_features_shape(self) -> None:
        """Test edge features have correct shape."""
        data = smiles_to_graph(ETHANOL)
        assert data is not None
        # NUM_BOND_TYPES + is_aromatic + is_in_ring + is_conjugated
        expected_dim = NUM_BOND_TYPES + 3
        assert data.edge_attr.shape[1] == expected_dim

    def test_invalid_smiles_returns_none(self) -> None:
        """Test invalid SMILES returns None."""
        data = smiles_to_graph(INVALID_SMILES)
        assert data is None

    def test_stores_smiles(self) -> None:
        """Test SMILES is stored in data object."""
        data = smiles_to_graph(ETHANOL)
        assert data is not None
        assert data.smiles == ETHANOL

    def test_aromatic_atom_features(self) -> None:
        """Test aromatic atoms have correct features."""
        data = smiles_to_graph(BENZENE)
        assert data is not None
        # Check that aromatic flag is set for all atoms
        aromatic_idx = NUM_ATOM_TYPES + 3  # Position of is_aromatic
        assert all(data.x[:, aromatic_idx] == 1)


class TestGraphToSmiles:
    """Tests for graph to SMILES conversion."""

    def test_roundtrip_ethanol(self) -> None:
        """Test roundtrip conversion for ethanol."""
        data = smiles_to_graph(ETHANOL)
        assert data is not None
        smiles = graph_to_smiles(data)
        # May not be identical due to canonicalization
        assert smiles is not None

    def test_roundtrip_benzene(self) -> None:
        """Test roundtrip conversion for benzene."""
        data = smiles_to_graph(BENZENE)
        assert data is not None
        smiles = graph_to_smiles(data)
        assert smiles is not None


class TestMolecularDataset:
    """Tests for MolecularDataset class."""

    @pytest.fixture
    def sample_smiles(self) -> list[str]:
        """Sample SMILES for testing."""
        return [ETHANOL, BENZENE, PHENOL, ACETIC_ACID]

    def test_init(self, sample_smiles: list[str]) -> None:
        """Test dataset initialization."""
        dataset = MolecularDataset(sample_smiles)
        assert len(dataset) == 4

    def test_getitem(self, sample_smiles: list[str]) -> None:
        """Test dataset indexing."""
        dataset = MolecularDataset(sample_smiles)
        data = dataset[0]
        assert data.num_nodes == 3  # Ethanol

    def test_max_num_nodes(self, sample_smiles: list[str]) -> None:
        """Test max_num_nodes property."""
        dataset = MolecularDataset(sample_smiles)
        # Phenol has the most atoms (7)
        assert dataset.max_num_nodes == 7

    def test_smiles_list(self, sample_smiles: list[str]) -> None:
        """Test smiles_list property."""
        dataset = MolecularDataset(sample_smiles)
        assert dataset.smiles_list == sample_smiles

    def test_invalid_smiles_filtered(self) -> None:
        """Test invalid SMILES are filtered out."""
        smiles = [ETHANOL, INVALID_SMILES, BENZENE]
        dataset = MolecularDataset(smiles)
        assert len(dataset) == 2

    def test_max_molecules_limit(self, sample_smiles: list[str]) -> None:
        """Test max_molecules parameter."""
        dataset = MolecularDataset(sample_smiles, max_molecules=2)
        assert len(dataset) == 2
