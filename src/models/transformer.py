"""Transformer models for autoregressive graph generation.

This module provides HuggingFace transformer wrappers for training graph
generation models using next-token prediction.
"""

from timeit import default_timer as timer
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import transformers
from torch import nn

from src.tokenizers.base import Tokenizer


class TransformerLM(nn.Module):
    """HuggingFace transformer wrapper for graph generation.

    Supports GPT-2, LLaMA, GPT-NeoX, and Mamba architectures with
    configurable model sizes.

    Attributes:
        tokenizer: The graph tokenizer.
        model: The HuggingFace transformer model.
        model_name: Name of the model architecture and size.
    """

    MODEL_SIZES = {
        "xxs": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4 * 256,
            "n_embd": 256,
            "n_head": 4,
            "n_layer": 4,
        },
        "xs": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 4 * 384,
            "n_embd": 384,
            "n_head": 12,
            "n_layer": 6,
        },
        "s": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 4 * 768,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
        },
        "m": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4 * 1024,
            "n_embd": 1024,
            "n_head": 16,
            "n_layer": 24,
        },
    }

    MODEL_CLASSES = {
        "gpt2": (transformers.GPT2Config, transformers.GPT2LMHeadModel),
        "llama": (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        "llama2": (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        "gpt_neox": (transformers.GPTNeoXConfig, transformers.GPTNeoXForCausalLM),
    }

    def __init__(
        self,
        tokenizer: Tokenizer,
        model_name: str = "gpt2-xs",
        max_length: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Initialize the transformer model.

        Args:
            tokenizer: Graph tokenizer for vocab size and special tokens.
            model_name: Model architecture and size (e.g., 'gpt2-xs', 'llama-s').
            max_length: Maximum sequence length for generation.
            **kwargs: Additional model configuration parameters.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

        name_parts = model_name.split("-")
        arch_name = name_parts[0]
        size_name = name_parts[1] if len(name_parts) > 1 else "xs"

        if arch_name not in self.MODEL_CLASSES:
            available = ", ".join(self.MODEL_CLASSES.keys())
            raise ValueError(f"Unknown model: {arch_name}. Available: {available}")

        if size_name not in self.MODEL_SIZES:
            available = ", ".join(self.MODEL_SIZES.keys())
            raise ValueError(f"Unknown size: {size_name}. Available: {available}")

        vocab_params = {
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.sos,
            "eos_token_id": tokenizer.eos,
            "pad_token_id": tokenizer.pad,
        }

        size_params = self.MODEL_SIZES[size_name]
        config_cls, model_cls = self.MODEL_CLASSES[arch_name]

        self.model = model_cls(
            config_cls(**vocab_params, **size_params, **kwargs)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token indices [batch_size, seq_len].

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size].
        """
        return self.model(input_ids=input_ids).logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
        return_tokens: bool = False,
    ) -> list:
        """Generate graphs autoregressively.

        Args:
            input_ids: Initial token sequence [batch_size, seq_len].
            top_k: Number of highest probability tokens to sample from.
            temperature: Sampling temperature.
            max_length: Maximum generation length.
            return_tokens: If True, return tokens instead of Data objects.

        Returns:
            List of generated Data objects or token sequences.
        """
        batch_size = input_ids.shape[0]
        max_length = max_length or self.max_length

        generated = self.model.generate(
            input_ids,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            max_length=max_length,
        )

        results = []
        for i in range(batch_size):
            if return_tokens:
                results.append(generated[i])
            else:
                try:
                    results.append(self.tokenizer.decode(generated[i]))
                except Exception:
                    # Skip invalid sequences (will be counted as invalid in metrics)
                    pass

        return results


class GraphGeneratorModule(pl.LightningModule):
    """PyTorch Lightning module for training graph generators.

    This module wraps a transformer model and provides training, validation,
    and generation functionality.

    Attributes:
        model: The transformer model.
        tokenizer: The graph tokenizer.
        loss_fn: Cross-entropy loss function.
        learning_rate: Learning rate for optimization.
        sampling_config: Configuration for generation sampling.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        model_name: str = "gpt2-xs",
        learning_rate: float = 6e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        sampling_top_k: int = 10,
        sampling_temperature: float = 1.0,
        sampling_max_length: int = 2048,
        sampling_num_samples: int = 100,
        sampling_batch_size: int = 32,
    ) -> None:
        """Initialize the training module.

        Args:
            tokenizer: Graph tokenizer.
            model_name: Model architecture and size.
            learning_rate: Learning rate.
            weight_decay: Weight decay for AdamW.
            warmup_steps: Number of warmup steps.
            max_steps: Maximum training steps.
            sampling_top_k: Top-k for generation sampling.
            sampling_temperature: Temperature for generation.
            sampling_max_length: Maximum generation length.
            sampling_num_samples: Number of samples for evaluation.
            sampling_batch_size: Batch size for generation.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer
        self.model = TransformerLM(tokenizer, model_name)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.sampling_top_k = sampling_top_k
        self.sampling_temperature = sampling_temperature
        self.sampling_max_length = sampling_max_length
        self.sampling_num_samples = sampling_num_samples
        self.sampling_batch_size = sampling_batch_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(input_ids)

    def _shared_step(self, batch: torch.Tensor, phase: str) -> torch.Tensor:
        """Shared step for train/val/test.

        Args:
            batch: Input batch [batch_size, seq_len].
            phase: One of 'train', 'val', 'test'.

        Returns:
            Loss tensor.
        """
        x, y = batch[:, :-1], batch[:, 1:]
        logits = self.model(x)
        logits = logits.view(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = self.loss_fn(logits, y)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._shared_step(batch, "test")

    def generate(
        self,
        num_samples: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> tuple[list, float]:
        """Generate graphs.

        Args:
            num_samples: Number of graphs to generate.
            input_ids: Optional initial tokens.

        Returns:
            Tuple of (list of Data objects, average time per sample).
        """
        num_samples = num_samples or self.sampling_num_samples
        if input_ids is not None:
            num_samples = input_ids.shape[0]

        graphs = []
        total_time = 0

        for i in range(0, num_samples, self.sampling_batch_size):
            batch_size = min(self.sampling_batch_size, num_samples - i)

            if input_ids is not None:
                init_ids = input_ids[i : i + batch_size].to(self.device)
            else:
                init_ids = torch.full(
                    (batch_size, 1),
                    self.tokenizer.sos,
                    dtype=torch.long,
                    device=self.device,
                )

            tic = timer()
            batch_graphs = self.model.generate(
                init_ids,
                top_k=self.sampling_top_k,
                temperature=self.sampling_temperature,
                max_length=self.sampling_max_length,
            )
            toc = timer()

            graphs.extend(batch_graphs)
            total_time += toc - tic

        avg_time = total_time / num_samples
        return graphs, avg_time

    def configure_optimizers(self) -> dict:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return max(1e-6, step / max(1, self.warmup_steps))
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
