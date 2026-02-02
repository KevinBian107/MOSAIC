"""Tokenizer-specific primer generators.

This submodule provides primer classes for each hierarchical tokenizer type
(HSENT, HDT, HDTC) that convert scaffolds into token sequences for priming.
"""

from src.transfer_learning.primers.base import TokenizerPrimer
from src.transfer_learning.primers.factory import PrimerFactory
from src.transfer_learning.primers.hdt_primer import HDTPrimer
from src.transfer_learning.primers.hdtc_primer import HDTCPrimer
from src.transfer_learning.primers.hsent_primer import HSENTPrimer

__all__ = [
    "TokenizerPrimer",
    "PrimerFactory",
    "HDTPrimer",
    "HDTCPrimer",
    "HSENTPrimer",
]
