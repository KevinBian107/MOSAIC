"""Adapter for using AutoGraph pretrained models with MOSAIC evaluation.

This module provides a wrapper that makes AutoGraph's SequenceModel compatible
with MOSAIC's GraphGeneratorModule interface, allowing evaluation of AutoGraph
pretrained checkpoints using MOSAIC's test scripts.
"""

import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch

# Add AutoGraph to path
autograph_path = Path(__file__).parent.parent.parent / "tmp" / "AutoGraph"
sys.path.insert(0, str(autograph_path))

from autograph.models.seq_models import SequenceModel

from src.tokenizers.base import Tokenizer


class AutoGraphAdapter(pl.LightningModule):
    """Adapter for AutoGraph pretrained models.

    This class wraps an AutoGraph SequenceModel and provides the same interface
    as MOSAIC's GraphGeneratorModule, allowing seamless evaluation using existing
    test scripts.

    Attributes:
        autograph_model: The loaded AutoGraph SequenceModel.
        tokenizer: MOSAIC tokenizer (for compatibility, not used for generation).
        sampling_batch_size: Batch size for generation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        autograph_checkpoint_path: str,
        sampling_batch_size: int = 32,
        sampling_top_k: int = 10,
        sampling_temperature: float = 1.0,
        sampling_max_length: int = 2048,
    ) -> None:
        """Initialize the adapter with an AutoGraph checkpoint.

        Args:
            tokenizer: MOSAIC tokenizer (for interface compatibility).
            autograph_checkpoint_path: Path to AutoGraph .ckpt file.
            sampling_batch_size: Batch size for generation.
            sampling_top_k: Top-k for sampling.
            sampling_temperature: Temperature for sampling.
            sampling_max_length: Maximum generation length.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.sampling_batch_size = sampling_batch_size
        self.sampling_top_k = sampling_top_k
        self.sampling_temperature = sampling_temperature
        self.sampling_max_length = sampling_max_length

        # Load AutoGraph model
        self.autograph_model = SequenceModel.load_from_checkpoint(
            autograph_checkpoint_path,
            weights_only=False,
        )
        self.autograph_model.eval()

    def generate(
        self,
        num_samples: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> tuple[list, float]:
        """Generate graphs using the AutoGraph model.

        Args:
            num_samples: Number of graphs to generate.
            input_ids: Optional initial tokens (not used for AutoGraph).

        Returns:
            Tuple of (list of graph objects, average time per sample).
        """
        if num_samples is None:
            num_samples = 100

        # Update AutoGraph's sampling config
        original_batch_size = self.autograph_model.cfg.sampling.batch_size
        original_top_k = self.autograph_model.cfg.sampling.top_k
        original_temp = self.autograph_model.cfg.sampling.temperature
        original_max_len = self.autograph_model.cfg.sampling.max_length

        self.autograph_model.cfg.sampling.batch_size = self.sampling_batch_size
        self.autograph_model.cfg.sampling.top_k = self.sampling_top_k
        self.autograph_model.cfg.sampling.temperature = self.sampling_temperature
        self.autograph_model.cfg.sampling.max_length = self.sampling_max_length

        try:
            graphs, avg_time = self.autograph_model.generate(
                num_samples=num_samples,
                input_ids=input_ids,
            )
        finally:
            # Restore original config
            self.autograph_model.cfg.sampling.batch_size = original_batch_size
            self.autograph_model.cfg.sampling.top_k = original_top_k
            self.autograph_model.cfg.sampling.temperature = original_temp
            self.autograph_model.cfg.sampling.max_length = original_max_len

        return graphs, avg_time

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer: Tokenizer,
        **kwargs,
    ) -> "AutoGraphAdapter":
        """Load AutoGraph checkpoint.

        Args:
            checkpoint_path: Path to AutoGraph .ckpt file.
            tokenizer: MOSAIC tokenizer for compatibility.
            **kwargs: Additional adapter arguments.

        Returns:
            Initialized AutoGraphAdapter.
        """
        return cls(
            tokenizer=tokenizer,
            autograph_checkpoint_path=checkpoint_path,
            **kwargs,
        )
