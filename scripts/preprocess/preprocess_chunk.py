#!/usr/bin/env python
"""Preprocess a specific chunk of the dataset for parallel processing.

Supports MOSES and COCONUT datasets with coarsening strategies that benefit
from precomputation:
- spectral: Optimized spectral clustering (default)
- hac: Hierarchical agglomerative clustering with connectivity constraint

Usage:
    # MOSES (default)
    python scripts/preprocess/preprocess_chunk.py \
        --tokenizer hsent \
        --start 0 \
        --end 100000 \
        --output data/cache/hsent_spectral_chunk_0_100000.pt

    # COCONUT
    python scripts/preprocess/preprocess_chunk.py \
        --tokenizer hsent \
        --dataset coconut \
        --start 0 \
        --end 5000 \
        --output data/cache/hsent_spectral_chunk_0_5000.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.coconut_loader import CoconutLoader
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
    parser.add_argument(
        "--dataset",
        choices=["moses", "coconut"],
        default="moses",
        help="Dataset to preprocess (default: moses)",
    )
    parser.add_argument(
        "--data-file",
        default="data/coconut_complex.smi",
        help="COCONUT SMILES file path (default: data/coconut_complex.smi)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=20,
        help="Min atoms for COCONUT filtering (default: 20)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=100,
        help="Max atoms for COCONUT filtering (default: 100)",
    )
    parser.add_argument(
        "--min-rings",
        type=int,
        default=3,
        help="Min rings for COCONUT filtering (default: 3)",
    )
    parser.add_argument(
        "--coarsening-strategy",
        choices=["spectral", "hac"],
        default="spectral",
        help="Coarsening strategy (default: spectral)",
    )
    # Spectral-only knobs (used for aggressive speed vs quality tradeoffs)
    parser.add_argument(
        "--spectral-n-init",
        type=int,
        default=None,
        help="SpectralClustering n_init (spectral only). If omitted, tokenizer default is used.",
    )
    parser.add_argument(
        "--spectral-k-min-factor",
        type=float,
        default=None,
        help="Spectral k_min_factor (spectral only). If omitted, tokenizer default is used.",
    )
    parser.add_argument(
        "--spectral-k-max-factor",
        type=float,
        default=None,
        help="Spectral k_max_factor (spectral only). If omitted, tokenizer default is used.",
    )
    args = parser.parse_args()

    chunk_size = args.end - args.start
    log.info(f"Processing chunk [{args.start}:{args.end}] ({chunk_size} samples)")
    log.info(f"Tokenizer: {args.tokenizer}")
    log.info(f"Output: {args.output}")

    # Load dataset
    if args.dataset == "moses":
        log.info("Loading MOSES dataset...")
        mol_dataset = MolecularDataset.from_moses(
            split="train",
            max_molecules=args.end,  # Load up to our end index
            include_hydrogens=False,
            labeled=True,
            seed=args.seed,
        )
        # For MOSES, indices into mol_dataset match global indices
        chunk_offset = args.start

        if len(mol_dataset) < args.end:
            log.warning(
                f"Dataset only has {len(mol_dataset)} samples, "
                f"adjusting end to {len(mol_dataset)}"
            )
            args.end = len(mol_dataset)
            chunk_size = args.end - args.start

    elif args.dataset == "coconut":
        log.info("Loading COCONUT dataset...")
        loader = CoconutLoader(
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_rings=args.min_rings,
            data_file=args.data_file,
        )
        all_smiles = loader.load_smiles(seed=args.seed)
        log.info(f"Loaded {len(all_smiles)} filtered COCONUT SMILES")

        if args.end > len(all_smiles):
            log.warning(
                f"Dataset only has {len(all_smiles)} samples, "
                f"adjusting end to {len(all_smiles)}"
            )
            args.end = len(all_smiles)
            chunk_size = args.end - args.start

        chunk_smiles = all_smiles[args.start : args.end]
        mol_dataset = MolecularDataset(
            chunk_smiles,
            dataset_name="coconut_train",
            include_hydrogens=False,
            labeled=True,
        )
        # For COCONUT, mol_dataset is already sliced to the chunk
        # Index 0 in mol_dataset corresponds to global index args.start
        chunk_offset = 0

    log.info(f"Loaded {len(mol_dataset)} molecules")

    # Build tokenizer config for cache hash
    tokenizer_config = {
        "type": args.tokenizer,
        "max_length": -1,
        "truncation_length": 2048,
        "labeled_graph": True,
        "node_order": "BFS",
        "min_community_size": 4,
        "coarsening_strategy": args.coarsening_strategy,
        "motif_aware": False,
        "motif_alpha": 1.0,
        "normalize_by_motif_size": False,
        "undirected": True,
    }
    if args.coarsening_strategy == "spectral":
        # Include spectral parameters in cache hash so precompute matches training config
        tokenizer_config["spectral_n_init"] = args.spectral_n_init
        tokenizer_config["spectral_k_min_factor"] = args.spectral_k_min_factor
        tokenizer_config["spectral_k_max_factor"] = args.spectral_k_max_factor
    # Build common tokenizer kwargs
    tokenizer_kwargs = dict(
        max_length=-1,
        truncation_length=2048,
        node_order="BFS",
        min_community_size=4,
        coarsening_strategy=args.coarsening_strategy,
        motif_aware=False,
        labeled_graph=True,
        seed=args.seed,
    )
    if args.coarsening_strategy == "spectral":
        # Only apply overrides when provided
        if args.spectral_n_init is not None:
            tokenizer_kwargs["n_init"] = args.spectral_n_init
        if args.spectral_k_min_factor is not None:
            tokenizer_kwargs["k_min_factor"] = args.spectral_k_min_factor
        if args.spectral_k_max_factor is not None:
            tokenizer_kwargs["k_max_factor"] = args.spectral_k_max_factor
    # Initialize tokenizer
    if args.tokenizer == "hsent":
        tokenizer = HSENTTokenizer(**tokenizer_kwargs)
    else:  # hdt
        tokenizer = HDTTokenizer(**tokenizer_kwargs)

    tokenizer.set_num_nodes(mol_dataset.max_num_nodes)
    tokenizer.set_num_node_and_edge_types(
        num_node_types=NUM_ATOM_TYPES,
        num_edge_types=NUM_BOND_TYPES,
    )

    log.info("Tokenizer config:")
    log.info(f"  Type: {type(tokenizer).__name__}")
    log.info(f"  Coarsener: {type(tokenizer.coarsener).__name__}")
    if hasattr(tokenizer.coarsener, "n_init"):
        log.info(f"  n_init: {tokenizer.coarsener.n_init}")
    if hasattr(tokenizer.coarsener, "linkage"):
        log.info(f"  linkage: {tokenizer.coarsener.linkage}")
    if hasattr(tokenizer.coarsener, "k_min_factor"):
        log.info(f"  k_min_factor: {tokenizer.coarsener.k_min_factor}")
    if hasattr(tokenizer.coarsener, "k_max_factor"):
        log.info(f"  k_max_factor: {tokenizer.coarsener.k_max_factor}")

    # Tokenize chunk with timing
    log.info(f"Tokenizing chunk [{args.start}:{args.end}]...")
    tokenized_data = []
    smiles_list = []
    global_indices = []

    start_time = time.time()
    local_size = args.end - args.start
    for local_i in tqdm(range(local_size), desc="Tokenizing"):
        dataset_idx = chunk_offset + local_i
        graph = mol_dataset[dataset_idx]
        tokens = tokenizer(graph)
        tokenized_data.append(tokens)
        smiles_list.append(mol_dataset.smiles_list[dataset_idx])
        global_indices.append(args.start + local_i)

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
