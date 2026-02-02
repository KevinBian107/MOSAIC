"""Base class for tokenizer primers.

This module defines the abstract TokenizerPrimer interface that all
tokenizer-specific primers must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.tokenizers.base import Tokenizer
    from src.transfer_learning.scaffolds.library import Scaffold


class TokenizerPrimer(ABC):
    """Abstract base class for tokenizer-specific primers.

    A primer converts scaffold structures into token sequences that can be
    used to prime the generation process. The resulting token sequence starts
    with SOS but does NOT end with EOS, allowing the model to continue
    generating from the scaffold.

    Attributes:
        tokenizer: The tokenizer instance to use for priming.
    """

    def __init__(self, tokenizer: "Tokenizer") -> None:
        """Initialize the primer with a tokenizer.

        Args:
            tokenizer: Tokenizer instance (HSENT, HDT, or HDTC).
        """
        self.tokenizer = tokenizer

    @abstractmethod
    def create_primer(
        self,
        scaffold: "Scaffold",
        priming_level: str = "scaffold_only",
    ) -> Tensor:
        """Create primer tokens from a scaffold.

        The primer tokens should start with SOS and NOT end with EOS,
        representing a partial sequence that the model can complete.

        Args:
            scaffold: Scaffold object to create primer from.
            priming_level: Level of priming detail:
                - "scaffold_only": Just the scaffold structure
                - "with_attachment": Include attachment point hints

        Returns:
            1D tensor of token indices (SOS + scaffold tokens, without EOS).
        """
        pass

    @abstractmethod
    def validate_primer(self, primer_tokens: Tensor) -> bool:
        """Validate the structure of primer tokens.

        Checks that:
        - Primer starts with SOS
        - Primer does NOT end with EOS
        - Primer contains valid token indices

        Args:
            primer_tokens: 1D tensor of token indices.

        Returns:
            True if valid, False otherwise.
        """
        pass

    @abstractmethod
    def find_valid_cut_points(self, tokens: Tensor) -> list[int]:
        """Find indices where tokens can be safely cut for priming.

        Each tokenizer has specific structural boundaries where it's safe
        to cut the token sequence. For example:
        - HDTC: After COMM_END tokens (complete communities)
        - HDT: After EXIT tokens when back at root level
        - HSENT: After RCOM tokens (complete communities)

        Cutting at invalid points could result in malformed partial sequences
        that the model cannot properly complete.

        Args:
            tokens: 1D tensor of token indices (full tokenization with EOS).

        Returns:
            List of valid cut point indices (sorted ascending).
        """
        pass

    def create_primer_at_level(
        self,
        scaffold: "Scaffold",
        cut_level: int = -1,
    ) -> Tensor:
        """Create primer tokens cut at a specific structural level.

        Instead of simply stripping EOS, this method cuts the token sequence
        at a structurally valid boundary (e.g., after a complete community).

        Args:
            scaffold: Scaffold object to create primer from.
            cut_level: Index into the list of valid cut points.
                -1 means the last valid cut point (most complete).
                0 means the first valid cut point (least complete).

        Returns:
            1D tensor of token indices cut at the specified level.

        Raises:
            ValueError: If no valid cut points exist.
        """
        # Get full tokenization (includes SOS and EOS)
        graph = scaffold.get_graph(labeled=True)
        if graph is None:
            raise ValueError(f"Failed to create graph from scaffold: {scaffold.smiles}")

        # Ensure max_num_nodes is set
        if self.tokenizer.max_num_nodes is None:
            self.tokenizer.set_num_nodes(graph.num_nodes)
        elif graph.num_nodes > self.tokenizer.max_num_nodes:
            self.tokenizer.set_num_nodes(graph.num_nodes)

        tokens = self.tokenizer.tokenize(graph)

        # Find valid cut points
        cut_points = self.find_valid_cut_points(tokens)

        if not cut_points:
            # No valid cut points, fall back to just SOS
            return torch.tensor([self.tokenizer.sos], dtype=torch.long)

        # Get the requested cut point
        cut_idx = cut_points[cut_level]

        # Return tokens up to and including the cut point
        return tokens[: cut_idx + 1]

    def batch_primers(
        self,
        scaffolds: list["Scaffold"],
        priming_level: str = "scaffold_only",
    ) -> Tensor:
        """Create a batch of primers with padding.

        Args:
            scaffolds: List of scaffolds to create primers for.
            priming_level: Level of priming detail.

        Returns:
            2D tensor of shape [batch_size, max_primer_length] with padding.
        """
        primers = [self.create_primer(s, priming_level) for s in scaffolds]
        max_len = max(len(p) for p in primers)

        # Pad primers to max length
        batch = torch.full(
            (len(primers), max_len),
            self.tokenizer.pad,
            dtype=torch.long,
        )

        for i, primer in enumerate(primers):
            batch[i, : len(primer)] = primer

        return batch

    def get_special_tokens(self) -> dict[str, int]:
        """Get the special token mappings from the tokenizer.

        Returns:
            Dictionary mapping token names to indices.
        """
        return {
            "sos": self.tokenizer.sos,
            "eos": self.tokenizer.eos,
            "pad": self.tokenizer.pad,
        }

    def primer_length(self, scaffold: "Scaffold") -> int:
        """Calculate the primer length for a scaffold without creating it.

        Args:
            scaffold: Scaffold to calculate length for.

        Returns:
            Expected number of tokens in the primer.
        """
        # Default implementation: just create and measure
        primer = self.create_primer(scaffold)
        return len(primer)
