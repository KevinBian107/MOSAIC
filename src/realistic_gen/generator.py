"""Unconditional graph generation.

This module provides functionality for generating molecules unconditionally
from a trained model.
"""

from typing import Union

import torch
from torch_geometric.data import Data

from src.models.transformer import GraphGeneratorModule
from src.tokenizers.hdt.tokenizer import HDTTokenizer
from src.tokenizers.sent.tokenizer import SENTTokenizer

# Type alias for supported tokenizers
Tokenizer = Union[HDTTokenizer, SENTTokenizer]


def generate_molecules(
    model: GraphGeneratorModule,
    num_samples: int,
    show_progress: bool = True,
) -> tuple[list[Data], float]:
    """Generate molecules unconditionally.

    Args:
        model: Trained GraphGeneratorModule instance.
        num_samples: Number of molecules to generate.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of (list of generated Data objects, avg time per sample).
    """
    model.eval()
    with torch.no_grad():
        graphs, avg_time, _ = model.generate(
            num_samples=num_samples,
            show_progress=show_progress,
        )
    return graphs, avg_time
