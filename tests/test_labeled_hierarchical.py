"""Tests for labeled graph support in hierarchical tokenizers.

To run these tests:
    conda activate mosaic
    pytest tests/test_labeled_hierarchical.py -v

To run with verbose output:
    conda activate mosaic
    pytest tests/test_labeled_hierarchical.py -vv

To run a specific test:
    conda activate mosaic
    pytest tests/test_labeled_hierarchical.py::test_hdt_vocab_size_unlabeled -v
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers.hierarchical.hdt import HDTTokenizer
from src.tokenizers.hierarchical.hsent import HSENTTokenizer

# Test constants
NUM_ATOM_TYPES = 10  # Example: C, N, O, F, etc.
NUM_BOND_TYPES = 4  # Example: single, double, triple, aromatic


class TestHDTVocabularySize:
    """Test vocabulary size calculations for HDT tokenizer."""

    def test_unlabeled_vocab_size(self):
        """Test HDT vocab size for unlabeled graphs."""
        tokenizer = HDTTokenizer(labeled_graph=False)
        tokenizer.set_num_nodes(50)
        expected = 7 + 50  # IDX_OFFSET (7) + max_num_nodes
        assert tokenizer.vocab_size == expected

    def test_labeled_vocab_size(self):
        """Test HDT vocab size for labeled graphs."""
        tokenizer = HDTTokenizer(labeled_graph=True)
        tokenizer.set_num_nodes(50)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)
        expected = 7 + 50 + NUM_ATOM_TYPES + NUM_BOND_TYPES  # 71
        assert tokenizer.vocab_size == expected

    def test_labeled_offsets(self):
        """Test that node and edge offsets are calculated correctly."""
        tokenizer = HDTTokenizer(labeled_graph=True)
        tokenizer.set_num_nodes(50)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        # Node offset should be IDX_OFFSET + max_num_nodes
        assert tokenizer.node_idx_offset == 7 + 50

        # Edge offset should be node_offset + num_node_types
        assert tokenizer.edge_idx_offset == 7 + 50 + NUM_ATOM_TYPES

    def test_cannot_set_types_when_unlabeled(self):
        """Test that setting node/edge types fails when labeled_graph=False."""
        tokenizer = HDTTokenizer(labeled_graph=False)
        tokenizer.set_num_nodes(50)

        with pytest.raises(ValueError, match="Cannot set node/edge types"):
            tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)


class TestHSENTVocabularySize:
    """Test vocabulary size calculations for H-SENT tokenizer."""

    def test_unlabeled_vocab_size(self):
        """Test H-SENT vocab size for unlabeled graphs."""
        tokenizer = HSENTTokenizer(labeled_graph=False)
        tokenizer.set_num_nodes(50)
        expected = 11 + 50  # IDX_OFFSET (11) + max_num_nodes
        assert tokenizer.vocab_size == expected

    def test_labeled_vocab_size(self):
        """Test H-SENT vocab size for labeled graphs."""
        tokenizer = HSENTTokenizer(labeled_graph=True)
        tokenizer.set_num_nodes(50)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)
        expected = 11 + 50 + NUM_ATOM_TYPES + NUM_BOND_TYPES  # 75
        assert tokenizer.vocab_size == expected

    def test_labeled_offsets(self):
        """Test that node and edge offsets are calculated correctly."""
        tokenizer = HSENTTokenizer(labeled_graph=True)
        tokenizer.set_num_nodes(50)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        # Node offset should be IDX_OFFSET + max_num_nodes
        assert tokenizer.node_idx_offset == 11 + 50

        # Edge offset should be node_offset + num_node_types
        assert tokenizer.edge_idx_offset == 11 + 50 + NUM_ATOM_TYPES


class TestHDTSimpleGraphRoundtrip:
    """Test HDT roundtrip with simple labeled graphs."""

    def test_triangle_labeled_roundtrip(self):
        """Test HDT roundtrip with a triangle graph with atom/bond types."""
        # Create triangle: 0-1-2-0
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        x = torch.tensor([1, 2, 3], dtype=torch.long)  # Atom types
        edge_attr = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)  # Bond types

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=3)

        # Tokenize
        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=2)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)

        # Decode back to Data object
        reconstructed = tokenizer.decode(tokens)

        # Verify structure
        assert reconstructed.num_nodes == 3
        assert reconstructed.edge_index.shape[1] == 6

        # Verify node features
        assert reconstructed.x is not None
        assert torch.allclose(reconstructed.x, x)

        # Verify edge features (order may differ)
        assert reconstructed.edge_attr is not None
        assert set(reconstructed.edge_attr.tolist()) == set(edge_attr.tolist())

    def test_path_labeled_roundtrip(self):
        """Test HDT roundtrip with a path graph."""
        # Create path: 0-1-2-3
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        x = torch.tensor([5, 6, 7, 8], dtype=torch.long)  # Atom types
        edge_attr = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.long)  # Bond types

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=4)

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=2)
        tokenizer.set_num_nodes(20)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 4
        assert reconstructed.edge_index.shape[1] == 6
        assert torch.allclose(reconstructed.x, x)
        assert set(reconstructed.edge_attr.tolist()) == set(edge_attr.tolist())

    def test_single_node_with_features(self):
        """Test HDT with single node (edge case)."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        x = torch.tensor([3], dtype=torch.long)

        data = Data(edge_index=edge_index, x=x, num_nodes=1)

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 1
        assert reconstructed.x is not None
        assert reconstructed.x[0] == 3


class TestHSENTSimpleGraphRoundtrip:
    """Test H-SENT roundtrip with simple labeled graphs."""

    def test_triangle_labeled_roundtrip(self):
        """Test H-SENT roundtrip with a triangle graph."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        x = torch.tensor([1, 2, 3], dtype=torch.long)
        edge_attr = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=3)

        tokenizer = HSENTTokenizer(labeled_graph=True, min_community_size=2, seed=42)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 3
        assert reconstructed.edge_index.shape[1] == 6
        assert torch.allclose(reconstructed.x, x)
        assert set(reconstructed.edge_attr.tolist()) == set(edge_attr.tolist())


class TestTokenSequenceFormat:
    """Test that token sequences have correct structure."""

    def test_hdt_has_special_tokens(self):
        """Verify HDT tokens include expected special tokens."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([5, 7], dtype=torch.long)
        edge_attr = torch.tensor([2, 2], dtype=torch.long)

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=2)

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)

        # Should contain SOS, EOS, ENTER, EXIT, LEDGE, REDGE
        assert tokenizer.SOS in tokens
        assert tokenizer.EOS in tokens
        assert tokenizer.ENTER in tokens
        assert tokenizer.EXIT in tokens
        assert tokenizer.LEDGE in tokens
        assert tokenizer.REDGE in tokens

    def test_hsent_has_special_tokens(self):
        """Verify H-SENT tokens include expected special tokens."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([5, 7], dtype=torch.long)
        edge_attr = torch.tensor([2, 2], dtype=torch.long)

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=2)

        tokenizer = HSENTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)

        # Should contain SOS, EOS, LCOM, RCOM
        assert tokenizer.SOS in tokens
        assert tokenizer.EOS in tokens
        assert tokenizer.LCOM in tokens
        assert tokenizer.RCOM in tokens

    def test_hdt_atom_types_in_vocab_range(self):
        """Verify atom type tokens are in correct vocabulary range."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([5, 7], dtype=torch.long)
        edge_attr = torch.tensor([2, 2], dtype=torch.long)

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=2)

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)

        # Find atom type tokens (should be in range [node_idx_offset, edge_idx_offset))
        atom_tokens = [
            t
            for t in tokens
            if tokenizer.node_idx_offset <= t < tokenizer.edge_idx_offset
        ]

        # Should have at least 2 atom type tokens (one for each node)
        assert len(atom_tokens) >= 2

    def test_hdt_bond_types_in_vocab_range(self):
        """Verify bond type tokens are in correct vocabulary range."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([5, 7], dtype=torch.long)
        edge_attr = torch.tensor([2, 2], dtype=torch.long)

        data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=2)

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)

        # Find bond type tokens (should be >= edge_idx_offset)
        bond_tokens = [t for t in tokens if t >= tokenizer.edge_idx_offset]

        # Should have at least some bond type tokens
        # (exact count depends on how edges are encoded)
        assert len(bond_tokens) > 0


class TestUnlabeledBackwardCompatibility:
    """Test that unlabeled mode still works (backward compatibility)."""

    def test_hdt_unlabeled_roundtrip(self):
        """Test HDT roundtrip without labels."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=3)

        tokenizer = HDTTokenizer(labeled_graph=False, min_community_size=2)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 3
        assert reconstructed.edge_index.shape[1] == 6
        # No features expected
        assert reconstructed.x is None
        assert reconstructed.edge_attr is None

    def test_hsent_unlabeled_roundtrip(self):
        """Test H-SENT roundtrip without labels."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=3)

        tokenizer = HSENTTokenizer(labeled_graph=False, min_community_size=2)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 3
        assert reconstructed.edge_index.shape[1] == 6
        assert reconstructed.x is None
        assert reconstructed.edge_attr is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_hdt_empty_graph(self):
        """Test HDT with empty graph (no nodes, no edges)."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=0)

        tokenizer = HDTTokenizer(labeled_graph=False)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 0
        assert reconstructed.edge_index.shape[1] == 0

    def test_hdt_missing_node_features(self):
        """Test HDT labeled mode when input data lacks node features."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=2)  # No x attribute

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        # Should handle gracefully (use default atom type 0)
        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        # Should create node features with default values
        assert reconstructed.num_nodes == 2
        if reconstructed.x is not None:
            # All atom types should be 0 (default)
            assert torch.all(reconstructed.x == 0)

    def test_hdt_missing_edge_features(self):
        """Test HDT labeled mode when input data lacks edge features."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([1, 2], dtype=torch.long)
        data = Data(edge_index=edge_index, x=x, num_nodes=2)  # No edge_attr

        tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
        tokenizer.set_num_nodes(10)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        # Should handle gracefully (use default bond type 0)
        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == 2
        if reconstructed.edge_attr is not None:
            # All bond types should be 0 (default)
            assert torch.all(reconstructed.edge_attr == 0)


# Optional: Molecular graph tests (requires RDKit)
try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMolecularGraphs:
    """Test with real molecular graphs (requires RDKit)."""

    def smiles_to_graph(self, smiles: str) -> Data:
        """Convert SMILES to PyG Data with atom/bond types.

        This is a simplified conversion for testing purposes.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get atom types (simplified: atomic number % 10)
        atom_types = [atom.GetAtomicNum() % 10 for atom in mol.GetAtoms()]
        x = torch.tensor(atom_types, dtype=torch.long)

        # Get edges and bond types
        edges = []
        bond_types = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = int(bond.GetBondTypeAsDouble())  # 1, 2, 3, 1.5 (aromatic)
            edges.append((i, j))
            edges.append((j, i))
            bond_types.append(bond_type)
            bond_types.append(bond_type)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(bond_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)

        return Data(
            edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=len(atom_types)
        )

    def test_hdt_ethanol_roundtrip(self):
        """Test HDT with ethanol molecule."""
        data = self.smiles_to_graph("CCO")

        tokenizer = HDTTokenizer(labeled_graph=True)
        tokenizer.set_num_nodes(50)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == data.num_nodes
        assert torch.allclose(reconstructed.x, data.x)
        # Edge attributes should match (possibly reordered)
        assert sorted(reconstructed.edge_attr.tolist()) == sorted(
            data.edge_attr.tolist()
        )

    def test_hsent_ethanol_roundtrip(self):
        """Test H-SENT with ethanol molecule."""
        data = self.smiles_to_graph("CCO")

        tokenizer = HSENTTokenizer(labeled_graph=True, seed=42)
        tokenizer.set_num_nodes(50)
        tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        assert reconstructed.num_nodes == data.num_nodes
        assert torch.allclose(reconstructed.x, data.x)
        assert sorted(reconstructed.edge_attr.tolist()) == sorted(
            data.edge_attr.tolist()
        )
