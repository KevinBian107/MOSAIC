"""Tests for tokenizer module."""

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers.base import BatchConverter, Tokenizer
from src.tokenizers.sent import SENTTokenizer
from tests.fixtures.graphs import (
    triangle_graph,
    square_graph,
    erdos_renyi_graph,
    disconnected_graph,
)


class TestSENTTokenizer:
    """Tests for SENTTokenizer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        tokenizer = SENTTokenizer()
        assert tokenizer.sos == 0
        assert tokenizer.eos == 4
        assert tokenizer.pad == 5

    def test_set_num_nodes(self) -> None:
        """Test setting maximum number of nodes."""
        tokenizer = SENTTokenizer()
        tokenizer.set_num_nodes(100)
        assert tokenizer.max_num_nodes == 100
        assert tokenizer.vocab_size == 100 + 6

    def test_tokenize_triangle(self, triangle_graph: Data) -> None:
        """Test tokenization of a triangle graph."""
        tokenizer = SENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(triangle_graph)

        assert tokens[0] == tokenizer.sos
        assert tokens[-1] == tokenizer.eos
        assert len(tokens) > 2

    def test_tokenize_deterministic(self, triangle_graph: Data) -> None:
        """Test that tokenization is deterministic with same seed."""
        tokenizer1 = SENTTokenizer(seed=42)
        tokenizer1.set_num_nodes(10)

        tokenizer2 = SENTTokenizer(seed=42)
        tokenizer2.set_num_nodes(10)

        tokens1 = tokenizer1.tokenize(triangle_graph)
        tokens2 = tokenizer2.tokenize(triangle_graph)

        assert torch.equal(tokens1, tokens2)

    def test_decode_triangle(self, triangle_graph: Data) -> None:
        """Test decode recovers graph structure."""
        tokenizer = SENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(triangle_graph)
        decoded = tokenizer.decode(tokens)

        assert decoded.num_nodes == triangle_graph.num_nodes
        assert decoded.edge_index.shape[1] == triangle_graph.edge_index.shape[1]

    def test_tokenize_decode_roundtrip(self, erdos_renyi_graph: Data) -> None:
        """Test encode-decode roundtrip preserves node count."""
        tokenizer = SENTTokenizer(seed=42)
        tokenizer.set_num_nodes(30)

        tokens = tokenizer.tokenize(erdos_renyi_graph)
        decoded = tokenizer.decode(tokens)

        assert decoded.num_nodes == erdos_renyi_graph.num_nodes

    def test_tokenize_disconnected(self, disconnected_graph: Data) -> None:
        """Test tokenization handles disconnected graphs."""
        tokenizer = SENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens = tokenizer.tokenize(disconnected_graph)

        assert tokens[0] == tokenizer.sos
        assert tokens[-1] == tokenizer.eos
        assert tokenizer.reset in tokens.tolist()

    def test_callable_interface(self, triangle_graph: Data) -> None:
        """Test __call__ works same as tokenize."""
        tokenizer = SENTTokenizer(seed=42)
        tokenizer.set_num_nodes(10)

        tokens1 = tokenizer.tokenize(triangle_graph)
        tokens2 = tokenizer(triangle_graph)

        assert torch.equal(tokens1, tokens2)

    def test_vocab_size_error_without_num_nodes(self) -> None:
        """Test vocab_size raises error if num_nodes not set."""
        tokenizer = SENTTokenizer()
        with pytest.raises(ValueError):
            _ = tokenizer.vocab_size


class TestBatchConverter:
    """Tests for BatchConverter class."""

    def test_pad_sequences(self) -> None:
        """Test padding sequences to equal length."""
        tokenizer = SENTTokenizer()
        tokenizer.set_num_nodes(10)
        converter = BatchConverter(tokenizer)

        seq1 = torch.tensor([0, 6, 7, 4])
        seq2 = torch.tensor([0, 6, 7, 8, 9, 4])

        batch = converter([seq1, seq2])

        assert batch.shape == (2, 6)
        assert batch[0, -1] == tokenizer.pad
        assert batch[0, -2] == tokenizer.pad
        assert batch[1, -1] == 4

    def test_truncation(self) -> None:
        """Test sequence truncation."""
        tokenizer = SENTTokenizer()
        tokenizer.set_num_nodes(10)
        converter = BatchConverter(tokenizer, truncation_length=4)

        seq = torch.tensor([0, 6, 7, 8, 9, 10, 4])
        batch = converter([seq])

        assert batch.shape == (1, 4)
