#!/usr/bin/env python
"""Preprocess a specific chunk of the dataset for parallel processing.

Usage:
    python scripts/preprocess/preprocess_chunk.py \
        --tokenizer hsent \
        --start 0 \
        --end 100000 \
        --output data/cache/hsent_chunk_0_100000.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.molecular import MolecularDataset, NUM_ATOM_TYPES, NUM_BOND_TYPES
from src.tokenizers import HDTTokenizer, HSENTTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess a chunk of molecules")
    parser.add_argument("--tokenizer", choices=["hsent", "hdt"], required=True)
    parser.add_argument("--start", type=int, required=True, help="Start index")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunk_size = args.end - args.start
    log.info(f"Processing chunk [{args.start}:{args.end}] ({chunk_size} samples)")
    log.info(f"Tokenizer: {args.tokenizer}")
    log.info(f"Output: {args.output}")

    # Load full dataset (MOSES loads everything but we'll only tokenize our chunk)
    log.info("Loading MOSES dataset...")
    mol_dataset = MolecularDataset.from_moses(
        split="train",
        max_molecules=args.end,  # Load up to our end index
        include_hydrogens=False,
        labeled=True,
        seed=args.seed,
    )

    if len(mol_dataset) < args.end:
        log.warning(f"Dataset only has {len(mol_dataset)} samples, adjusting end to {len(mol_dataset)}")
        args.end = len(mol_dataset)
        chunk_size = args.end - args.start

    log.info(f"✓ Loaded {len(mol_dataset)} molecules")

    # Build tokenizer config for cache hash
    tokenizer_config = {
        "type": args.tokenizer,
        "max_length": -1,
        "truncation_length": 2048,
        "labeled_graph": True,
        "node_order": "BFS",
        "min_community_size": 4,
        "motif_aware": False,
        "motif_alpha": 1.0,
        "normalize_by_motif_size": False,
        "undirected": True,
    }

    # Initialize tokenizer
    if args.tokenizer == "hsent":
        tokenizer = HSENTTokenizer(
            max_length=-1,
            truncation_length=2048,
            node_order="BFS",
            min_community_size=4,
            coarsening_strategy="simple_spectral",
            motif_aware=False,
            labeled_graph=True,
            seed=args.seed,
        )
    else:  # hdt
        tokenizer = HDTTokenizer(
            max_length=-1,
            truncation_length=2048,
            node_order="BFS",
            min_community_size=4,
            coarsening_strategy="simple_spectral",
            motif_aware=False,
            labeled_graph=True,
            seed=args.seed,
        )

    tokenizer.set_num_nodes(mol_dataset.max_num_nodes)
    tokenizer.set_num_node_and_edge_types(
        num_node_types=NUM_ATOM_TYPES,
        num_edge_types=NUM_BOND_TYPES,
    )

    log.info(f"Tokenizer config:")
    log.info(f"  Type: {type(tokenizer).__name__}")
    log.info(f"  Coarsener: {type(tokenizer.coarsener).__name__}")
    log.info(f"  n_init: {tokenizer.coarsener.n_init}")
    log.info(f"  k_min_factor: {tokenizer.coarsener.k_min_factor}")
    log.info(f"  k_max_factor: {tokenizer.coarsener.k_max_factor}")

    # Tokenize chunk with timing
    log.info(f"Tokenizing chunk [{args.start}:{args.end}]...")
    tokenized_data = []
    smiles_list = []
    global_indices = []

    start_time = time.time()
    for i in tqdm(range(args.start, args.end), desc="Tokenizing"):
        graph = mol_dataset[i]
        tokens = tokenizer(graph)
        tokenized_data.append(tokens)
        smiles_list.append(mol_dataset.smiles_list[i])
        global_indices.append(i)

    elapsed = time.time() - start_time
    speed = chunk_size / elapsed

    log.info(f"✓ Tokenized {chunk_size} molecules in {elapsed:.2f}s ({speed:.2f} it/s)")

    # Save chunk
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_data = {
        "tokens": tokenized_data,
        "smiles": smiles_list,
        "global_indices": global_indices,
        "vocab_size": tokenizer.vocab_size,
        "max_num_nodes": mol_dataset.max_num_nodes,
        "tokenizer_type": args.tokenizer,
        "tokenizer_config": tokenizer_config,
        "start_idx": args.start,
        "end_idx": args.end,
        "num_samples": len(tokenized_data),
    }

    log.info(f"Saving chunk to {output_path}...")
    torch.save(chunk_data, output_path)
    log.info(f"✓ Saved chunk with {len(tokenized_data)} samples")


if __name__ == "__main__":
    main()
