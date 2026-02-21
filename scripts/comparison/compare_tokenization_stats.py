#!/usr/bin/env python
"""Benchmark tokenization methods: compute basic stats (sequence length, etc.).

Runs all tokenization variants (SENT, HSENT, HDT, HDTC with various coarsenings)
on sample molecules from MOSES or COCONUT and reports statistics.

Usage:
    python scripts/comparison/compare_tokenization_stats.py
    python scripts/comparison/compare_tokenization_stats.py --dataset coconut
    python scripts/comparison/compare_tokenization_stats.py --dataset both
    python scripts/comparison/compare_tokenization_stats.py --num-samples 20
    python scripts/comparison/compare_tokenization_stats.py --output tokenization_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.coconut_loader import CoconutLoader
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


def load_dataset(
    dataset_name: str,
    num_samples: int,
    max_nodes: int,
    seed: int,
) -> tuple[list, list]:
    """Load molecules from the specified dataset.

    Returns:
        Tuple of (graphs, smiles_list).
    """
    if dataset_name == "coconut":
        print(f"Loading {num_samples} molecules from COCONUT...")
        loader = CoconutLoader(
            min_atoms=20,
            max_atoms=max_nodes,
            min_rings=3,
            data_file="data/coconut_complex.smi",
        )
        all_smiles = loader.load_smiles(n_samples=num_samples * 5, seed=seed)
        dataset = MolecularDataset(
            all_smiles,
            dataset_name="coconut",
            include_hydrogens=False,
            labeled=True,
        )
    else:
        print(f"Loading {num_samples} molecules from MOSES...")
        try:
            dataset = MolecularDataset.from_moses(
                split="train",
                max_molecules=num_samples * 20,
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

    return graphs, smiles_list


def run_benchmark(
    dataset_name: str = "moses",
    num_samples: int = 10,
    seed: int = 42,
    max_nodes: int = 100,
) -> dict[str, Any]:
    """Run tokenization benchmark on the given dataset."""
    graphs, smiles_list = load_dataset(dataset_name, num_samples, max_nodes, seed)

    print(f"  Using {len(graphs)} molecules (nodes: {min(g.num_nodes for g in graphs)}-{max(g.num_nodes for g in graphs)})")
    print()

    results: dict[str, Any] = {
        "dataset": dataset_name,
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
                sorted_lens = sorted(valid_lens)
                n = len(sorted_lens)
                p95 = sorted_lens[min(int(n * 0.95), n - 1)]
                p99 = sorted_lens[min(int(n * 0.99), n - 1)]
                results["tokenizers"][name] = {
                    "seq_len_mean": sum(valid_lens) / len(valid_lens),
                    "seq_len_min": min(valid_lens),
                    "seq_len_max": max(valid_lens),
                    "seq_len_p95": p95,
                    "seq_len_p99": p99,
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
                    f"(min={results['tokenizers'][name]['seq_len_min']}, max={results['tokenizers'][name]['seq_len_max']}, "
                    f"p95={p95}, p99={p99})"
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
    dataset = results.get("dataset", "unknown").upper()
    print("\n" + "=" * 100)
    print(f"TOKENIZATION STATISTICS — {dataset} ({results['num_samples']} samples)")
    print("=" * 100)
    print(f"{'Tokenizer':<12} {'Mean':>8} {'Min':>8} {'Max':>8} {'P95':>8} {'P99':>8} {'Std':>8} {'Compress':>10} {'Valid':>6}")
    print("-" * 100)

    for name, data in results["tokenizers"].items():
        if "error" in data:
            print(f"{name:<12} ERROR: {data['error'][:50]}")
            continue
        mean = data["seq_len_mean"]
        mn = data["seq_len_min"]
        mx = data["seq_len_max"]
        p95 = data.get("seq_len_p95", 0)
        p99 = data.get("seq_len_p99", 0)
        std = data["seq_len_std"]
        comp = data.get("compression_ratio_mean", 0)
        valid = data.get("num_valid", 0)
        print(f"{name:<12} {mean:>8.1f} {mn:>8} {mx:>8} {p95:>8} {p99:>8} {std:>8.1f} {comp:>10.2f} {valid:>6}")

    print("=" * 100)
    print(f"Compress = seq_len / num_nodes (lower = more compact)")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization methods on MOSES/COCONUT samples"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="moses",
        choices=["moses", "coconut", "both"],
        help="Dataset to benchmark (default: moses)",
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

    datasets = ["moses", "coconut"] if args.dataset == "both" else [args.dataset]
    all_results = {}

    for ds in datasets:
        results = run_benchmark(
            dataset_name=ds,
            num_samples=args.num_samples,
            seed=args.seed,
            max_nodes=args.max_nodes,
        )
        print_table(results)
        all_results[ds] = results

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = all_results if args.dataset == "both" else all_results[datasets[0]]
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
