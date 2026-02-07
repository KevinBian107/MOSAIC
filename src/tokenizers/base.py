"""Base tokenizer interface for graph tokenization.

This module defines the abstract interface that all graph tokenizers must
implement, enabling interchangeable tokenization schemes.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import torch
from torch_geometric.data import Data


class Tokenizer(ABC):
    """Abstract base class for graph tokenizers.

    All tokenizers must implement methods for converting graphs to token
    sequences and back, as well as providing a batch collation function.

    Attributes:
        sos: Start-of-sequence token index.
        eos: End-of-sequence token index.
        pad: Padding token index.
    """

    sos: int = 0
    eos: int = 4
    pad: int = 5

    @abstractmethod
    def tokenize(self, data: Data) -> torch.Tensor:
        """Convert a graph to a token sequence.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            1D tensor of token indices.
        """
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> Data:
        """Convert a token sequence back to a graph.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            PyTorch Geometric Data object.
        """
        pass

    @abstractmethod
    def batch_converter(self) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        """Return a callable for batching tokenized sequences.

        Returns:
            Callable that takes a list of token sequences and returns a
            padded batch tensor.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size.

        Returns:
            Total number of unique tokens.
        """
        pass

    def __call__(self, data: Data) -> torch.Tensor:
        """Tokenize a graph (callable interface).

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            1D tensor of token indices.
        """
        tokens = self.tokenize(data)

        # Validate tokens are within vocab bounds
        vocab_size = len(self)
        max_token = int(tokens.max().item())
        if max_token >= vocab_size:
            invalid_mask = tokens >= vocab_size
            invalid_idx = int(invalid_mask.nonzero()[0].item())
            invalid_token = int(tokens[invalid_idx].item())
            print(f"\n{'='*60}")
            print(f"TOKEN VALIDATION ERROR")
            print(f"{'='*60}")
            print(f"Invalid token: {invalid_token} at position {invalid_idx}")
            print(f"Vocab size: {vocab_size}")
            print(f"Graph nodes: {data.num_nodes}")
            print(f"Tokenizer: {type(self).__name__}")
            print(f"max_num_nodes: {getattr(self, 'max_num_nodes', 'N/A')}")
            print(f"IDX_OFFSET: {getattr(self, 'IDX_OFFSET', 'N/A')}")
            print(f"node_idx_offset: {getattr(self, 'node_idx_offset', 'N/A')}")
            print(f"edge_idx_offset: {getattr(self, 'edge_idx_offset', 'N/A')}")
            print(f"Token sequence length: {len(tokens)}")
            print(f"First 20 tokens: {tokens[:20].tolist()}")
            print(f"Tokens around error: {tokens[max(0,invalid_idx-5):invalid_idx+5].tolist()}")
            print(f"{'='*60}\n")
            raise ValueError(
                f"Token {invalid_token} at position {invalid_idx} exceeds vocab_size {vocab_size}. "
                f"Graph has {data.num_nodes} nodes. "
                f"Tokenizer: {type(self).__name__}, max_num_nodes={getattr(self, 'max_num_nodes', 'N/A')}"
            )

        return tokens

    def __len__(self) -> int:
        """Return vocabulary size.

        Returns:
            Total number of unique tokens.
        """
        return self.vocab_size

    def set_num_nodes(self, max_num_nodes: int) -> None:
        """Set the maximum number of nodes (for vocabulary sizing).

        Args:
            max_num_nodes: Maximum number of nodes in any graph.
        """
        pass


class BatchConverter:
    """Collation function for batching tokenized sequences.

    This class pads sequences to equal length and stacks them into a batch.

    Attributes:
        tokenizer: The tokenizer instance (for padding token).
        truncation_length: Optional maximum sequence length.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        truncation_length: int | None = None,
    ) -> None:
        """Initialize the batch converter.

        Args:
            tokenizer: Tokenizer instance for padding token.
            truncation_length: Maximum sequence length (truncates if exceeded).
        """
        self.tokenizer = tokenizer
        self.truncation_length = truncation_length

    def __call__(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        """Convert a list of sequences to a padded batch.

        Args:
            batch: List of 1D token tensors.

        Returns:
            2D tensor of shape [batch_size, max_length].
        """
        batch_size = len(batch)
        max_len = max(len(b) for b in batch)

        if self.truncation_length is not None:
            max_len = min(max_len, self.truncation_length)

        batched = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad,
            dtype=batch[0].dtype,
        )

        for i, seq in enumerate(batch):
            if self.truncation_length is not None and len(seq) > self.truncation_length:
                start_idx = torch.randint(
                    0, len(seq) - self.truncation_length, (1,)
                ).item()
                seq = seq[start_idx : start_idx + self.truncation_length]
            batched[i, : len(seq)] = seq

        return batched
