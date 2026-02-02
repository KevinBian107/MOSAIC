"""HDTC-specific primer for scaffold priming.

This module implements the HDTCPrimer for creating primers compatible
with the HDTCTokenizer.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.transfer_learning.primers.base import TokenizerPrimer

if TYPE_CHECKING:
    from src.tokenizers.hdtc import HDTCTokenizer
    from src.transfer_learning.scaffolds.library import Scaffold


class HDTCPrimer(TokenizerPrimer):
    """Primer for HDTC tokenization.

    Creates primers by tokenizing scaffolds with HDTCTokenizer and stripping
    the EOS token to allow model completion.

    Token vocabulary (from HDTCTokenizer):
        0: SOS (start of sequence)
        1: EOS (end of sequence)
        2: PAD (padding)
        3: COMM_START (start community block)
        4: COMM_END (end community block)
        5: LEDGE (left edge bracket)
        6: REDGE (right edge bracket)
        7: SUPER_START (start super-graph block)
        8: SUPER_END (end super-graph block)
        9: TYPE_RING (community type: ring)
        10: TYPE_FUNC (community type: functional group)
        11: TYPE_SINGLETON (community type: singleton)
        12+: Node indices (offset by IDX_OFFSET)

    Attributes:
        tokenizer: HDTCTokenizer instance.
    """

    def __init__(self, tokenizer: "HDTCTokenizer") -> None:
        """Initialize with an HDTCTokenizer.

        Args:
            tokenizer: HDTCTokenizer instance.
        """
        super().__init__(tokenizer)
        self._tokenizer: "HDTCTokenizer" = tokenizer

    def create_primer(
        self,
        scaffold: "Scaffold",
        priming_level: str = "scaffold_only",
    ) -> Tensor:
        """Create HDTC primer tokens from scaffold.

        Tokenizes the scaffold using HDTC tokenization and removes the EOS
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
        """Validate HDTC primer structure.

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
        """Find indices after COMM_END tokens.

        HDTC tokenization structure:
            SOS [COMM_START nodes COMM_END edges]* EOS

        Valid cut points are after COMM_END tokens, which mark the end
        of a complete community block including its internal edges.

        Args:
            tokens: 1D tensor of token indices.

        Returns:
            List of valid cut point indices (after each COMM_END).
        """
        cut_points: list[int] = []
        for i, tok in enumerate(tokens):
            if tok.item() == self._tokenizer.COMM_END:
                cut_points.append(i)
        return cut_points

    def get_special_tokens(self) -> dict[str, int]:
        """Get HDTC special token mappings.

        Returns:
            Dictionary mapping token names to indices.
        """
        return {
            "sos": self._tokenizer.SOS,
            "eos": self._tokenizer.EOS,
            "pad": self._tokenizer.PAD,
            "comm_start": self._tokenizer.COMM_START,
            "comm_end": self._tokenizer.COMM_END,
            "ledge": self._tokenizer.LEDGE,
            "redge": self._tokenizer.REDGE,
            "super_start": self._tokenizer.SUPER_START,
            "super_end": self._tokenizer.SUPER_END,
            "type_ring": self._tokenizer.TYPE_RING,
            "type_func": self._tokenizer.TYPE_FUNC,
            "type_singleton": self._tokenizer.TYPE_SINGLETON,
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
            smiles="",  # Would need graph_to_smiles
            tier=0,
            category="decoded",
            num_atoms=graph.num_nodes,
            graph=graph,
        )
