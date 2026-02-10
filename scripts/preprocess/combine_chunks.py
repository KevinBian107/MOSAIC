#!/usr/bin/env python
"""Combine preprocessed chunks into final dataset file.

Usage:
    python scripts/preprocess/combine_chunks.py \
        --tokenizer hsent \
        --chunk_dir data/cache \
        --split train \
        --dataset moses
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from glob import glob

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_cache_filename(
    dataset_name: str,
    split: str,
    tokenizer_type: str,
    num_samples: int,
    tokenizer_config: dict,
) -> str:
    """Generate cache filename with hash (matches preprocess_dataset.py)."""
    config_str = json.dumps(tokenizer_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{dataset_name}_{split}_{tokenizer_type}_{num_samples}_{config_hash}.pt"


def main():
    parser = argparse.ArgumentParser(description="Combine preprocessed chunks")
    parser.add_argument("--tokenizer", choices=["hsent", "hdt"], required=True)
    parser.add_argument(
        "--chunk_dir", type=str, required=True, help="Directory containing chunks"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--dataset", type=str, default="moses", help="Dataset name")
    parser.add_argument(
        "--coarsening-strategy",
        choices=["spectral", "hac"],
        default="spectral",
        help="Coarsening strategy used in chunks (default: spectral)",
    )
    args = parser.parse_args()

    # Find all chunk files (try strategy-prefixed pattern first, fall back to legacy)
    chunk_pattern = (
        f"{args.chunk_dir}/{args.tokenizer}_{args.coarsening_strategy}_chunk_*.pt"
    )
    chunk_files = sorted(glob(chunk_pattern))
    if not chunk_files:
        # Fall back to legacy pattern without coarsening strategy prefix
        chunk_pattern = f"{args.chunk_dir}/{args.tokenizer}_chunk_*.pt"
    chunk_files = sorted(glob(chunk_pattern))

    if not chunk_files:
        log.error(f"No chunk files found matching: {chunk_pattern}")
        sys.exit(1)

    log.info(f"Found {len(chunk_files)} chunk files:")
    for f in chunk_files:
        log.info(f"  {f}")

    # Load and combine chunks
    all_tokens = []
    all_smiles = []
    all_indices = []
    vocab_size = None
    max_num_nodes = None
    tokenizer_type = None
    tokenizer_config = None

    log.info("Loading chunks...")
    for chunk_file in chunk_files:
        log.info(f"  Loading {Path(chunk_file).name}...")
        chunk_data = torch.load(chunk_file)

        all_tokens.extend(chunk_data["tokens"])
        all_smiles.extend(chunk_data["smiles"])
        all_indices.extend(chunk_data["global_indices"])

        # Track maximum values and verify consistency
        if vocab_size is None:
            vocab_size = chunk_data["vocab_size"]
            max_num_nodes = chunk_data["max_num_nodes"]
            tokenizer_type = chunk_data["tokenizer_type"]
            tokenizer_config = chunk_data["tokenizer_config"]
        else:
            # Use maximum vocab_size and max_num_nodes across all chunks
            # (chunks may see different max values depending on molecules)
            if chunk_data["vocab_size"] > vocab_size:
                log.info(
                    f"  Updating vocab_size: {vocab_size} -> {chunk_data['vocab_size']}"
                )
                vocab_size = chunk_data["vocab_size"]
            if chunk_data["max_num_nodes"] > max_num_nodes:
                log.info(
                    f"  Updating max_num_nodes: {max_num_nodes} -> {chunk_data['max_num_nodes']}"
                )
                max_num_nodes = chunk_data["max_num_nodes"]

            # Verify tokenizer settings are consistent
            assert tokenizer_type == chunk_data["tokenizer_type"], (
                "Tokenizer type mismatch!"
            )
            assert tokenizer_config == chunk_data["tokenizer_config"], (
                "Tokenizer config mismatch!"
            )

    # Sort by global indices to ensure correct order
    log.info("Sorting by global indices...")
    sorted_data = sorted(zip(all_indices, all_tokens, all_smiles))
    all_indices, all_tokens, all_smiles = zip(*sorted_data)

    # Create final combined data
    combined_data = {
        "tokens": list(all_tokens),
        "smiles": list(all_smiles),
        "vocab_size": vocab_size,
        "max_num_nodes": max_num_nodes,
        "tokenizer_type": tokenizer_type,
        "tokenizer_config": tokenizer_config,
        "dataset_name": args.dataset,
        "split": args.split,
        "num_samples": len(all_tokens),
        "labeled": True,
    }

    # Generate cache filename with hash
    cache_filename = get_cache_filename(
        args.dataset, args.split, args.tokenizer, len(all_tokens), tokenizer_config
    )
    output_path = Path(args.chunk_dir) / cache_filename

    log.info(f"Saving combined dataset to {output_path}...")
    torch.save(combined_data, output_path)

    log.info(f"\n{'=' * 80}")
    log.info("COMBINATION COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"Total samples: {len(all_tokens)}")
    log.info(f"Vocab size: {vocab_size}")
    log.info(f"Max nodes: {max_num_nodes}")
    log.info(f"Output file: {output_path}")
    log.info(f"Cache filename: {cache_filename}")
    log.info("\nThis file can be used by the training script with:")
    log.info("  data.use_cache=true")


if __name__ == "__main__":
    main()
