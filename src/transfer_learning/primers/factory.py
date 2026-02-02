"""Factory for creating tokenizer-specific primers.

This module provides the PrimerFactory class that automatically detects
the tokenizer type and returns the appropriate primer.
"""

from typing import TYPE_CHECKING, Union

from src.transfer_learning.primers.base import TokenizerPrimer
from src.transfer_learning.primers.hdt_primer import HDTPrimer
from src.transfer_learning.primers.hdtc_primer import HDTCPrimer
from src.transfer_learning.primers.hsent_primer import HSENTPrimer

if TYPE_CHECKING:
    from src.tokenizers.base import Tokenizer
    from src.tokenizers.hdt import HDTTokenizer
    from src.tokenizers.hdtc import HDTCTokenizer
    from src.tokenizers.hsent import HSENTTokenizer


class PrimerFactory:
    """Factory for creating tokenizer-specific primers.

    This factory detects the tokenizer type and returns the appropriate
    primer implementation.

    Example:
        >>> from src.tokenizers import HDTCTokenizer
        >>> tokenizer = HDTCTokenizer()
        >>> primer = PrimerFactory.create(tokenizer)
        >>> isinstance(primer, HDTCPrimer)
        True
    """

    # Mapping of tokenizer types to primer classes
    PRIMER_CLASSES: dict[str, type[TokenizerPrimer]] = {
        "hsent": HSENTPrimer,
        "hdt": HDTPrimer,
        "hdtc": HDTCPrimer,
    }

    @classmethod
    def create(
        cls,
        tokenizer: Union[
            "Tokenizer", "HSENTTokenizer", "HDTTokenizer", "HDTCTokenizer"
        ],
    ) -> TokenizerPrimer:
        """Create a primer for the given tokenizer.

        Automatically detects the tokenizer type and returns the appropriate
        primer implementation.

        Args:
            tokenizer: Tokenizer instance (HSENT, HDT, or HDTC).

        Returns:
            TokenizerPrimer instance appropriate for the tokenizer.

        Raises:
            ValueError: If tokenizer type is not supported.
        """
        # Detect tokenizer type
        tokenizer_type = cls._detect_tokenizer_type(tokenizer)

        if tokenizer_type not in cls.PRIMER_CLASSES:
            raise ValueError(
                f"Unsupported tokenizer type: {tokenizer_type}. "
                f"Supported types: {list(cls.PRIMER_CLASSES.keys())}"
            )

        primer_class = cls.PRIMER_CLASSES[tokenizer_type]
        return primer_class(tokenizer)

    @classmethod
    def _detect_tokenizer_type(
        cls,
        tokenizer: "Tokenizer",
    ) -> str:
        """Detect the type of tokenizer.

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            Tokenizer type string ("hsent", "hdt", or "hdtc").
        """
        # Check for tokenizer_type attribute
        if hasattr(tokenizer, "tokenizer_type"):
            return tokenizer.tokenizer_type

        # Fallback: check class name
        class_name = tokenizer.__class__.__name__.lower()

        if "hdtc" in class_name:
            return "hdtc"
        elif "hdt" in class_name:
            return "hdt"
        elif "hsent" in class_name:
            return "hsent"

        # Fallback: check for specific attributes
        if hasattr(tokenizer, "SUPER_START"):
            return "hdtc"
        elif hasattr(tokenizer, "ENTER"):
            return "hdt"
        elif hasattr(tokenizer, "LCOM"):
            return "hsent"

        raise ValueError(
            f"Cannot detect tokenizer type for {tokenizer.__class__.__name__}"
        )

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported tokenizer types.

        Returns:
            List of supported tokenizer type strings.
        """
        return list(cls.PRIMER_CLASSES.keys())

    @classmethod
    def register(cls, tokenizer_type: str, primer_class: type[TokenizerPrimer]) -> None:
        """Register a new primer class for a tokenizer type.

        Args:
            tokenizer_type: Tokenizer type string.
            primer_class: Primer class to register.
        """
        cls.PRIMER_CLASSES[tokenizer_type] = primer_class
