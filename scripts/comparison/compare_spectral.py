#!/usr/bin/env python
"""Benchmark spectral clustering efficiency and quality.

This script tests different parameter combinations for SpectralCoarsening
to find optimal trade-offs between compute efficiency and modularity quality.

Usage:
    python scripts/comparison/compare_spectral.py
    python scripts/comparison/compare_spectral.py --num-samples 1000
    python scripts/comparison/compare_spectral.py --output results.json
"""

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.molecular import MolecularDataset, smiles_to_graph
from src.tokenizers.coarsening.spectral import SpectralCoarsening

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_test_graphs(num_samples: int = 500, seed: int = 42) -> list[torch.Tensor]:
    """Load diverse molecular graphs for benchmarking.
    
    Args:
        num_samples: Number of graphs to load.
        seed: Random seed for reproducibility.
        
    Returns:
        List of PyG Data objects.
    """
    log.info(f"Loading {num_samples} molecular graphs...")
    
    # Built-in test molecules (always used)
    test_smiles = [
        "c1ccccc1",  # benzene (small, symmetric)
        "c1ccc2ccccc2c1",  # naphthalene (medium, fused rings)
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin (medium, functional groups)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine (medium-large, heterocycles)
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen (medium-large, branched)
    ]
    
    graphs = []
    # Load from MOSES if available
    try:
        dataset = MolecularDataset.from_moses(
            split="train",
            max_molecules=num_samples,
            include_hydrogens=False,
            labeled=True,
            seed=seed,
        )
        for i in range(len(dataset)):
            graphs.append(dataset[i])
        log.info(f"  Loaded {len(graphs)} from MOSES")
    except (ImportError, OSError) as e:
        log.warning(f"  MOSES not available ({e}), using built-in test molecules only")
    
    for smiles in test_smiles:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
    
    log.info(f"Loaded {len(graphs)} graphs")
    log.info(f"  Node counts: min={min(g.num_nodes for g in graphs)}, "
             f"max={max(g.num_nodes for g in graphs)}, "
             f"mean={statistics.mean(g.num_nodes for g in graphs):.1f}")
    
    return graphs


def benchmark_config(
    graphs: list[torch.Tensor],
    k_min_factor: float,
    k_max_factor: float,
    n_init: int,
    assign_labels: str = "kmeans",
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark a specific spectral clustering configuration.
    
    Args:
        graphs: List of molecular graphs to test.
        k_min_factor: Minimum k factor.
        k_max_factor: Maximum k factor.
        n_init: Number of initializations.
        assign_labels: Label assignment method ("kmeans" or "discretize").
        seed: Random seed.
        
    Returns:
        Dictionary with benchmark results.
    """
    # Create coarsener with this configuration
    coarsener = SpectralCoarsening(
        k_min_factor=k_min_factor,
        k_max_factor=k_max_factor,
        n_init=n_init,
        min_community_size=4,
        seed=seed,
    )
    
    # Monkey-patch partition to use custom assign_labels
    # (spectral.py hardcodes "kmeans", so we patch for all cases)
    original_partition = coarsener.partition
    
    def partition_with_labels(data):
        """Wrapper to use custom assign_labels."""
        n = data.num_nodes
        if n <= 1:
            return [set(range(n))]
        
        # Build adjacency matrix manually (avoid torch_geometric dependency)
        adj = torch.zeros((n, n), dtype=torch.float32)
        if data.edge_index.numel() > 0:
            edge_index = data.edge_index
            adj[edge_index[0], edge_index[1]] = 1.0
            adj[edge_index[1], edge_index[0]] = 1.0  # Make symmetric
        adj = adj.numpy()
        
        if adj.sum() == 0:
            return [set(range(n))]
        
        k_min = max(2, int(np.sqrt(n) * k_min_factor))
        k_max = min(n - 1, int(np.sqrt(n) * k_max_factor))
        
        if k_min > k_max:
            k_min = k_max = max(2, min(n - 1, 2))
        
        best_modularity = -float("inf")
        best_partition = None
        
        from sklearn.cluster import SpectralClustering
        
        for K in range(k_min, k_max + 1):
            try:
                sc = SpectralClustering(
                    n_clusters=K,
                    affinity="precomputed",
                    n_init=n_init,
                    random_state=seed,
                    assign_labels=assign_labels,
                )
                labels = sc.fit_predict(adj + np.eye(n) * 1e-6)
                partition = dict(enumerate(labels))
                modularity = coarsener._compute_modularity(adj, partition)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_partition = partition
            except Exception:
                continue
        
        if best_partition is None:
            return [set(range(n))]
        
        communities = {}
        for node, comm in best_partition.items():
            communities.setdefault(comm, set()).add(node)
        return list(communities.values())
    
    # Always patch to ensure we test the exact assign_labels value
    coarsener.partition = partition_with_labels
    
    # Benchmark
    times = []
    modularities = []
    num_communities = []
    failed = 0
    
    for graph in tqdm(graphs, desc=f"k=[{k_min_factor:.1f},{k_max_factor:.1f}], n_init={n_init}, labels={assign_labels}"):
        try:
            start = time.time()
            communities = coarsener.partition(graph)
            elapsed = time.time() - start
            
            # Compute modularity for this partition
            # Build adjacency matrix manually (avoid torch_geometric dependency)
            n = graph.num_nodes
            adj = torch.zeros((n, n), dtype=torch.float32)
            if graph.edge_index.numel() > 0:
                adj[graph.edge_index[0], graph.edge_index[1]] = 1.0
                adj[graph.edge_index[1], graph.edge_index[0]] = 1.0  # Make symmetric
            adj = adj.numpy()
            
            partition_dict = {}
            for comm_id, comm_nodes in enumerate(communities):
                for node in comm_nodes:
                    partition_dict[node] = comm_id
            
            mod = coarsener._compute_modularity(adj, partition_dict)
            
            times.append(elapsed)
            modularities.append(mod)
            num_communities.append(len(communities))
        except Exception as e:
            failed += 1
            log.warning(f"Failed on graph with {graph.num_nodes} nodes: {e}")
    
    if len(times) == 0:
        return {
            "k_min_factor": k_min_factor,
            "k_max_factor": k_max_factor,
            "n_init": n_init,
            "assign_labels": assign_labels,
            "failed": len(graphs),
            "error": "All graphs failed",
        }
    
    return {
        "k_min_factor": k_min_factor,
        "k_max_factor": k_max_factor,
        "n_init": n_init,
        "assign_labels": assign_labels,
        "num_graphs": len(graphs),
        "failed": failed,
        "time_mean": statistics.mean(times),
        "time_median": statistics.median(times),
        "time_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "time_total": sum(times),
        "modularity_mean": statistics.mean(modularities),
        "modularity_median": statistics.median(modularities),
        "modularity_std": statistics.stdev(modularities) if len(modularities) > 1 else 0.0,
        "modularity_min": min(modularities),
        "modularity_max": max(modularities),
        "num_communities_mean": statistics.mean(num_communities),
        "num_communities_median": statistics.median(num_communities),
        "throughput": len(times) / sum(times) if sum(times) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark spectral clustering")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of graphs to test")
    parser.add_argument("--output", type=str, default="spectral_benchmark.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load test graphs
    graphs = load_test_graphs(num_samples=args.num_samples, seed=args.seed)
    
    # Define parameter grid to test
    configs = [
        # Baseline: current defaults
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 10, "assign_labels": "kmeans"},
        
        # Test different n_init values
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 1, "assign_labels": "kmeans"},
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 5, "assign_labels": "kmeans"},
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 20, "assign_labels": "kmeans"},
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 50, "assign_labels": "kmeans"},
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 100, "assign_labels": "kmeans"},
        
        # Test different k ranges (tighter = faster)
        {"k_min_factor": 0.8, "k_max_factor": 1.2, "n_init": 10, "assign_labels": "kmeans"},
        {"k_min_factor": 0.9, "k_max_factor": 1.1, "n_init": 10, "assign_labels": "kmeans"},
        {"k_min_factor": 0.95, "k_max_factor": 1.05, "n_init": 10, "assign_labels": "kmeans"},
        
        # Test discretize vs kmeans
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 10, "assign_labels": "discretize"},
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 5, "assign_labels": "discretize"},
        {"k_min_factor": 0.7, "k_max_factor": 1.3, "n_init": 1, "assign_labels": "discretize"},
        
        # Combined optimizations
        {"k_min_factor": 0.9, "k_max_factor": 1.1, "n_init": 5, "assign_labels": "kmeans"},
        {"k_min_factor": 0.9, "k_max_factor": 1.1, "n_init": 1, "assign_labels": "kmeans"},
        {"k_min_factor": 0.9, "k_max_factor": 1.1, "n_init": 5, "assign_labels": "discretize"},
    ]
    
    log.info(f"Testing {len(configs)} configurations...")
    results = []
    
    for i, config in enumerate(configs):
        log.info(f"\n[{i+1}/{len(configs)}] Testing: {config}")
        result = benchmark_config(graphs, **config, seed=args.seed)
        results.append(result)
        
        log.info(f"  Time: {result.get('time_mean', 0):.4f}s/graph "
                f"(throughput: {result.get('throughput', 0):.2f} graphs/s)")
        log.info(f"  Modularity: {result.get('modularity_mean', 0):.4f} "
                f"(std: {result.get('modularity_std', 0):.4f})")
        log.info(f"  Communities: {result.get('num_communities_mean', 0):.1f}")
    
    # Find baseline (current defaults)
    baseline = next((r for r in results if r["n_init"] == 10 and r["k_min_factor"] == 0.7 
                     and r["k_max_factor"] == 1.3 and r["assign_labels"] == "kmeans"), None)
    
    if baseline:
        baseline_time = baseline["time_mean"]
        baseline_modularity = baseline["modularity_mean"]
        
        log.info("\n" + "=" * 80)
        log.info("RESULTS SUMMARY")
        log.info("=" * 80)
        log.info(f"\nBaseline (current defaults):")
        log.info(f"  Time: {baseline_time:.4f}s/graph")
        log.info(f"  Modularity: {baseline_modularity:.4f}")
        
        log.info(f"\nSpeedup vs baseline:")
        for r in results:
            if r.get("time_mean") and baseline_time > 0:
                speedup = baseline_time / r["time_mean"]
                mod_ratio = r.get("modularity_mean", 0) / baseline_modularity if baseline_modularity > 0 else 0
                log.info(f"  {r['k_min_factor']:.2f}-{r['k_max_factor']:.2f}, "
                        f"n_init={r['n_init']}, {r['assign_labels']}: "
                        f"{speedup:.2f}x speedup, {mod_ratio:.3f}x modularity")
        
        # Find best trade-offs
        log.info(f"\nBest configurations:")
        
        # Fastest with >95% quality
        best_fast = None
        for r in results:
            if r.get("modularity_mean", 0) >= baseline_modularity * 0.95:
                if best_fast is None or r["time_mean"] < best_fast["time_mean"]:
                    best_fast = r
        
        if best_fast:
            speedup = baseline_time / best_fast["time_mean"]
            log.info(f"  Fastest with >95% quality: "
                    f"k=[{best_fast['k_min_factor']:.2f},{best_fast['k_max_factor']:.2f}], "
                    f"n_init={best_fast['n_init']}, {best_fast['assign_labels']}")
            log.info(f"    {speedup:.2f}x speedup, "
                    f"{best_fast['modularity_mean']/baseline_modularity:.3f}x modularity")
        
        # Highest quality
        best_quality = max(results, key=lambda r: r.get("modularity_mean", 0))
        log.info(f"  Highest quality: "
                f"k=[{best_quality['k_min_factor']:.2f},{best_quality['k_max_factor']:.2f}], "
                f"n_init={best_quality['n_init']}, {best_quality['assign_labels']}")
        log.info(f"    Modularity: {best_quality['modularity_mean']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "num_graphs": len(graphs),
            "baseline": baseline,
            "results": results,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
