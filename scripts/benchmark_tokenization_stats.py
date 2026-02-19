#!/usr/bin/env python
"""Benchmark tokenization methods: compute basic stats (sequence length, etc.).

Runs all tokenization variants (SENT, HSENT, HDT, HDTC with various coarsenings)
on sample molecules from MOSES and reports statistics.

Usage:
    python scripts/benchmark_tokenization_stats.py
    python scripts/benchmark_tokenization_stats.py --num-samples 20
    python scripts/benchmark_tokenization_stats.py --output tokenization_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.molecular import (
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    MolecularDataset,
    graph_to_smiles,
)
from src.tokenizers import (
    HDTCTokenizer,
    HDTTokenizer,
    HSENTTokenizer,
    SENTTokenizer,
)


# ============================================================================
# Tokenizer configurations
# ============================================================================

TOKENIZER_CONFIGS = [
    {
        "name": "SENT",
        "tokenizer_type": "sent",
        "coarsening": None,
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HSENT_MC",
        "tokenizer_type": "hsent",
        "coarsening": "motif_community",
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HSENT_SC",
        "tokenizer_type": "hsent",
        "coarsening": "spectral",
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HSENT_HAC",
        "tokenizer_type": "hsent",
        "coarsening": "hac",
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HDT_MC",
        "tokenizer_type": "hdt",
        "coarsening": "motif_community",
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HDT_SC",
        "tokenizer_type": "hdt",
        "coarsening": "spectral",
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HDT_HAC",
        "tokenizer_type": "hdt",
        "coarsening": "hac",
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
    {
        "name": "HDTC",
        "tokenizer_type": "hdtc",
        "coarsening": None,
        "kwargs": {"max_length": 2048, "labeled_graph": True},
    },
]


def create_tokenizer(config: dict, seed: int = 42, max_num_nodes: int = 100):
    """Create tokenizer from config."""
    tokenizer_type = config["tokenizer_type"]
    kwargs = {**config["kwargs"], "seed": seed}

    if tokenizer_type == "sent":
        tok = SENTTokenizer(**kwargs)
        tok.set_num_nodes(max_num_nodes)
        if kwargs.get("labeled_graph", True):
            tok.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)
        return tok
    elif tokenizer_type == "hsent":
        kwargs["coarsening_strategy"] = config["coarsening"]
        return HSENTTokenizer(**kwargs)
    elif tokenizer_type == "hdt":
        kwargs["coarsening_strategy"] = config["coarsening"]
        return HDTTokenizer(**kwargs)
    elif tokenizer_type == "hdtc":
        return HDTCTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def compute_stats(tokens: list[int], num_nodes: int, num_edges: int) -> dict[str, Any]:
    """Compute tokenization statistics."""
    seq_len = len(tokens)
    return {
        "seq_len": seq_len,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "compression_ratio": seq_len / num_nodes if num_nodes > 0 else 0.0,
        "tokens_per_edge": seq_len / num_edges if num_edges > 0 else 0.0,
    }


def run_benchmark(
    num_samples: int = 10,
    seed: int = 42,
    max_nodes: int = 100,
) -> dict[str, Any]:
    """Run tokenization benchmark on MOSES samples."""
    print(f"Loading {num_samples} molecules from MOSES...")
    try:
        dataset = MolecularDataset.from_moses(
            split="train",
            max_molecules=num_samples * 20,  # Load extra to filter
            include_hydrogens=False,
            labeled=True,
            seed=seed,
        )
    except Exception as e:
        print(f"  Error: {e}")
        print("  Falling back to test molecules...")
        test_smiles = [
            "c1ccccc1",
            "c1ccc2ccccc2c1",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        ]
        dataset = MolecularDataset(test_smiles, dataset_name="test")

    # Collect valid graphs
    graphs = []
    smiles_list = []
    for i in range(len(dataset)):
        if len(graphs) >= num_samples:
            break
        g = dataset[i]
        if g.num_nodes == 0 or g.num_nodes > max_nodes:
            continue
        s = graph_to_smiles(g)
        if s:
            graphs.append(g)
            smiles_list.append(s)

    print(f"  Using {len(graphs)} molecules (nodes: {min(g.num_nodes for g in graphs)}-{max(g.num_nodes for g in graphs)})")
    print()

    results: dict[str, Any] = {
        "num_samples": len(graphs),
        "seed": seed,
        "tokenizers": {},
    }

    max_num_nodes = max(g.num_nodes for g in graphs) + 20 if graphs else 100

    for config in TOKENIZER_CONFIGS:
        name = config["name"]
        print(f"  {name}...", end=" ", flush=True)
        try:
            tokenizer = create_tokenizer(config, seed=seed, max_num_nodes=max_num_nodes)
            seq_lens = []
            stats_per_mol = []

            for graph in graphs:
                try:
                    tokens = tokenizer.tokenize(graph)
                    if tokens.dim() > 1:
                        tokens = tokens.flatten()
                    tok_list = tokens.tolist()
                    seq_lens.append(len(tok_list))
                    stats_per_mol.append(
                        compute_stats(
                            tok_list,
                            graph.num_nodes,
                            graph.edge_index.shape[1] if graph.edge_index is not None else 0,
                        )
                    )
                except Exception as e:
                    seq_lens.append(-1)
                    stats_per_mol.append({"error": str(e)})

            valid_lens = [l for l in seq_lens if l >= 0]
            if valid_lens:
                results["tokenizers"][name] = {
                    "seq_len_mean": sum(valid_lens) / len(valid_lens),
                    "seq_len_min": min(valid_lens),
                    "seq_len_max": max(valid_lens),
                    "seq_len_std": (
                        (sum((x - sum(valid_lens) / len(valid_lens)) ** 2 for x in valid_lens) / len(valid_lens)) ** 0.5
                        if len(valid_lens) > 1
                        else 0.0
                    ),
                    "compression_ratio_mean": (
                        sum(s["compression_ratio"] for s in stats_per_mol if "compression_ratio" in s) / len(valid_lens)
                    ),
                    "num_valid": len(valid_lens),
                    "num_failed": len(seq_lens) - len(valid_lens),
                }
                print(
                    f"seq_len: {results['tokenizers'][name]['seq_len_mean']:.1f} "
                    f"(min={results['tokenizers'][name]['seq_len_min']}, max={results['tokenizers'][name]['seq_len_max']})"
                )
            else:
                results["tokenizers"][name] = {"error": "All samples failed", "num_failed": len(seq_lens)}
                print("FAILED (all samples)")
        except Exception as e:
            results["tokenizers"][name] = {"error": str(e)}
            print(f"FAILED: {e}")

    return results


def print_table(results: dict[str, Any]) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("TOKENIZATION STATISTICS (sequence length)")
    print("=" * 80)
    print(f"{'Tokenizer':<12} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8} {'Compress':>10} {'Valid':>6}")
    print("-" * 80)

    for name, data in results["tokenizers"].items():
        if "error" in data:
            print(f"{name:<12} ERROR: {data['error'][:50]}")
            continue
        mean = data["seq_len_mean"]
        mn = data["seq_len_min"]
        mx = data["seq_len_max"]
        std = data["seq_len_std"]
        comp = data.get("compression_ratio_mean", 0)
        valid = data.get("num_valid", 0)
        print(f"{name:<12} {mean:>8.1f} {mn:>8} {mx:>8} {std:>8.1f} {comp:>10.2f} {valid:>6}")

    print("=" * 80)
    print(f"Compress = seq_len / num_nodes (lower = more compact)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization methods on MOSES samples"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of molecules to benchmark (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100,
        help="Max nodes per molecule (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    results = run_benchmark(
        num_samples=args.num_samples,
        seed=args.seed,
        max_nodes=args.max_nodes,
    )

    print_table(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
