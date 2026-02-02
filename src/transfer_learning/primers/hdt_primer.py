"""HDT-specific primer for scaffold priming.

This module implements the HDTPrimer for creating primers compatible
with the HDTTokenizer.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.transfer_learning.primers.base import TokenizerPrimer

if TYPE_CHECKING:
    from src.tokenizers.hdt import HDTTokenizer
    from src.transfer_learning.scaffolds.library import Scaffold


class HDTPrimer(TokenizerPrimer):
    """Primer for HDT tokenization.

    Creates primers by tokenizing scaffolds with HDTTokenizer and stripping
    the EOS token to allow model completion.

    Token vocabulary (from HDTTokenizer):
        0: SOS (start of sequence)
        1: EOS (end of sequence)
        2: PAD (padding)
        3: ENTER (enter super node)
        4: EXIT (exit current super node)
        5: LEDGE (left edge bracket)
        6: REDGE (right edge bracket)
        7+: Node indices (offset by IDX_OFFSET)

    Attributes:
        tokenizer: HDTTokenizer instance.
    """

    def __init__(self, tokenizer: "HDTTokenizer") -> None:
        """Initialize with an HDTTokenizer.

        Args:
            tokenizer: HDTTokenizer instance.
        """
        super().__init__(tokenizer)
        self._tokenizer: "HDTTokenizer" = tokenizer

    def create_primer(
        self,
        scaffold: "Scaffold",
        priming_level: str = "scaffold_only",
    ) -> Tensor:
        """Create HDT primer tokens from scaffold.

        Tokenizes the scaffold using HDT tokenization and removes the EOS
        token at the end to create a primer.

        Args:
            scaffold: Scaffold object to create primer from.
            priming_level: Level of priming detail (currently only
                "scaffold_only" is supported).

        Returns:
            1D tensor of token indices (SOS + scaffold tokens, without EOS).

        Raises:
            ValueError: If scaffold graph cannot be created.
        """
        # Get graph representation (labeled for tokenization)
        graph = scaffold.get_graph(labeled=True)
        if graph is None:
            raise ValueError(f"Failed to create graph from scaffold: {scaffold.smiles}")

        # Ensure max_num_nodes is set
        if self._tokenizer.max_num_nodes is None:
            self._tokenizer.set_num_nodes(graph.num_nodes)
        elif graph.num_nodes > self._tokenizer.max_num_nodes:
            self._tokenizer.set_num_nodes(graph.num_nodes)

        # Tokenize the scaffold
        tokens = self._tokenizer.tokenize(graph)

        # Strip EOS token (last token)
        if len(tokens) > 0 and tokens[-1].item() == self._tokenizer.EOS:
            tokens = tokens[:-1]

        return tokens

    def validate_primer(self, primer_tokens: Tensor) -> bool:
        """Validate HDT primer structure.

        Checks that:
        - Primer starts with SOS
        - Primer does NOT end with EOS
        - All tokens are valid indices

        Args:
            primer_tokens: 1D tensor of token indices.

        Returns:
            True if valid, False otherwise.
        """
        if len(primer_tokens) == 0:
            return False

        # Check starts with SOS
        if primer_tokens[0].item() != self._tokenizer.SOS:
            return False

        # Check does NOT end with EOS
        if primer_tokens[-1].item() == self._tokenizer.EOS:
            return False

        # Check all tokens are valid
        max_token = self._tokenizer.vocab_size
        for tok in primer_tokens:
            tok_val = tok.item()
            if tok_val < 0 or tok_val >= max_token:
                return False

        return True

    def find_valid_cut_points(self, tokens: Tensor) -> list[int]:
        """Find indices after EXIT tokens when back at root level.

        HDT tokenization structure:
            SOS [ENTER nodes EXIT back_edges]* EOS

        Valid cut points are after EXIT tokens when the depth returns
        to 0 (root level), marking complete subtrees.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            List of valid cut point indices (after EXIT at root level).
        """
        cut_points: list[int] = []
        depth = 0

        for i, tok in enumerate(tokens):
            tok_val = tok.item()
            if tok_val == self._tokenizer.ENTER:
                depth += 1
            elif tok_val == self._tokenizer.EXIT:
                depth -= 1
                if depth == 0:
                    cut_points.append(i)

        return cut_points

    def get_special_tokens(self) -> dict[str, int]:
        """Get HDT special token mappings.

        Returns:
            Dictionary mapping token names to indices.
        """
        return {
            "sos": self._tokenizer.SOS,
            "eos": self._tokenizer.EOS,
            "pad": self._tokenizer.PAD,
            "enter": self._tokenizer.ENTER,
            "exit": self._tokenizer.EXIT,
            "ledge": self._tokenizer.LEDGE,
            "redge": self._tokenizer.REDGE,
            "idx_offset": self._tokenizer.IDX_OFFSET,
        }

    def decode_primer_to_graph(self, primer_tokens: Tensor) -> "Scaffold":
        """Decode primer tokens back to scaffold for verification.

        Adds EOS token back and decodes through tokenizer to verify
        the primer represents the expected structure.

        Args:
            primer_tokens: 1D tensor of primer tokens.

        Returns:
            Scaffold object reconstructed from tokens.
        """
        from torch_geometric.data import Data

        from src.transfer_learning.scaffolds.library import Scaffold

        # Add EOS token for decoding
        tokens_with_eos = torch.cat(
            [
                primer_tokens,
                torch.tensor([self._tokenizer.EOS], dtype=torch.long),
            ]
        )

        # Decode through tokenizer
        graph: Data = self._tokenizer.decode(tokens_with_eos)

        return Scaffold(
            name="decoded",
            smiles="",
            tier=0,
            category="decoded",
            num_atoms=graph.num_nodes,
            graph=graph,
        )
