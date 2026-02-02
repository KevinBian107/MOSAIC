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
