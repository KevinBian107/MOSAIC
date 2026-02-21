#!/usr/bin/env python
"""Visualize MOSES molecules with Spectral Clustering (discrete vs kmeans).

This script loads sample molecules from MOSES dataset and visualizes them
with their community structure discovered by Spectral Clustering, comparing
discrete vs kmeans label assignment methods.

Reuses visualization functions from compare_community_structure.py.

Usage:
    python scripts/visualization/visualize_moses_sc.py --compare
    python scripts/visualization/visualize_moses_sc.py --compare --num-samples 5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import visualization functions from existing script
from scripts.visualization.compare_community_structure import (  # noqa: E402
    COMMUNITY_COLORS,
    SINGLETON_COLOR,
    compute_rdkit_2d_layout,
    plot_molecule_with_communities,
)

from src.data.molecular import (  # noqa: E402
    MolecularDataset,
    graph_to_smiles,
)
from src.tokenizers.coarsening.spectral import SpectralCoarsening  # noqa: E402
from src.tokenizers.structures import HierarchicalGraph, Partition  # noqa: E402
from torch_geometric.utils import to_dense_adj  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Custom spectral clustering with assign_labels parameter
# ============================================================================


def partition_with_assign_labels(
    data: Data,
    assign_labels: str,
    k_min_factor: float = 0.7,
    k_max_factor: float = 1.3,
    n_init: int = 10,
    seed: int | None = None,
) -> HierarchicalGraph:
    """Partition graph using spectral clustering with custom assign_labels.
    
    Args:
        data: PyG Data object.
        assign_labels: Label assignment method ('kmeans' or 'discretize').
        k_min_factor: Factor for minimum cluster count.
        k_max_factor: Factor for maximum cluster count.
        n_init: Number of initializations (only used for kmeans).
        seed: Random seed.
        
    Returns:
        HierarchicalGraph with partitions.
    """
    n = data.num_nodes
    if n <= 1:
        return HierarchicalGraph(
            partitions=[Partition(part_id=0, global_node_indices=list(range(n)), edge_index=data.edge_index)],
            bipartites=[],
            community_assignment=list(range(n)),
        )
    
    # Build adjacency matrix
    adj = to_dense_adj(data.edge_index, max_num_nodes=n)[0]
    adj = ((adj + adj.t()) / 2).numpy()  # Symmetrize
    
    if adj.sum() == 0:
        return HierarchicalGraph(
            partitions=[Partition(part_id=0, global_node_indices=list(range(n)), edge_index=data.edge_index)],
            bipartites=[],
            community_assignment=list(range(n)),
        )
    
    # Compute k range
    k_min = max(2, int(np.sqrt(n) * k_min_factor))
    k_max = min(n - 1, int(np.sqrt(n) * k_max_factor))
    
    if k_min > k_max:
        k_min = k_max = max(2, min(n - 1, 2))
    
    # Compute modularity helper
    def _compute_modularity(adj: np.ndarray, partition: dict[int, int]) -> float:
        m = adj.sum() / 2
        if m == 0:
            return 0.0
        num_communities = max(partition.values()) + 1
        degrees = adj.sum(axis=1)
        Q = 0.0
        for c in range(num_communities):
            nodes_in_c = [node for node, comm in partition.items() if comm == c]
            if not nodes_in_c:
                continue
            e_c = adj[np.ix_(nodes_in_c, nodes_in_c)].sum() / 2
            d_c = degrees[nodes_in_c].sum()
            Q += e_c / m - (d_c / (2 * m)) ** 2
        return Q
    
    # Search for best K
    best_modularity = -float("inf")
    best_partition: dict[int, int] | None = None
    
    from sklearn.cluster import SpectralClustering
    
    for K in range(k_min, k_max + 1):
        try:
            sc = SpectralClustering(
                n_clusters=K,
                affinity="precomputed",
                n_init=n_init if assign_labels == "kmeans" else 1,
                random_state=seed,
                assign_labels=assign_labels,
            )
            labels = sc.fit_predict(adj + np.eye(n) * 1e-6)
            partition = dict(enumerate(labels))
            modularity = _compute_modularity(adj, partition)
            if modularity > best_modularity:
                best_modularity = modularity
                best_partition = partition
        except Exception:
            continue
    
    if best_partition is None:
        best_partition = {i: 0 for i in range(n)}
    
    # Convert to HierarchicalGraph
    communities: dict[int, set[int]] = {}
    for node, comm in best_partition.items():
        communities.setdefault(comm, set()).add(node)
    
    partitions = []
    for comm_id, nodes in sorted(communities.items()):
        # Get edges within this partition
        node_list = sorted(nodes)
        node_to_local = {node: i for i, node in enumerate(node_list)}
        local_edges = []
        if data.edge_index is not None:
            edge_index = data.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                u, v = int(edge_index[0, i]), int(edge_index[1, i])
                if u in nodes and v in nodes:
                    local_edges.append((node_to_local[u], node_to_local[v]))
        
        if local_edges:
            edge_tensor = torch.tensor(local_edges, dtype=torch.long).t()
        else:
            edge_tensor = torch.zeros((2, 0), dtype=torch.long)
        
        partitions.append(
            Partition(part_id=comm_id, global_node_indices=node_list, edge_index=edge_tensor)
        )
    
    # Build community assignment
    community_assignment = [best_partition[i] for i in range(n)]
    
    return HierarchicalGraph(
        partitions=partitions,
        bipartites=[],  # Single level for comparison
        community_assignment=community_assignment,
    )


# ============================================================================
# Main visualization function
# ============================================================================


def visualize_moses_molecules(
    num_samples: int = 12,
    output_dir: Path | str = "./figures/moses_sc",
    seed: int = 42,
    max_nodes: int = 100,
    k_min_factor: float = 0.7,
    k_max_factor: float = 1.3,
    n_init: int = 10,
    compare: bool = False,
) -> None:
    """Visualize MOSES molecules with Spectral Clustering.

    Args:
        num_samples: Number of molecules to visualize.
        output_dir: Output directory for figures.
        seed: Random seed for sampling.
        max_nodes: Maximum number of nodes per molecule.
        k_min_factor: Factor for minimum cluster count.
        k_max_factor: Factor for maximum cluster count.
        n_init: Number of initializations for spectral clustering.
        compare: If True, compare discrete vs kmeans in 2x5 grid.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if compare:
        num_samples = 5  # Force 5 samples for comparison
        print(f"Comparison mode: Loading {num_samples} molecules from MOSES dataset...")
    else:
        print(f"Loading {num_samples} molecules from MOSES dataset...")
    
    try:
        dataset = MolecularDataset.from_moses(
            split="train",
            max_molecules=num_samples * 10,  # Load more to filter
            include_hydrogens=False,
            labeled=True,
            seed=seed,
        )
        print(f"  Loaded {len(dataset)} molecules from MOSES")
    except Exception as e:
        print(f"  Error loading MOSES: {e}")
        print("  Falling back to test molecules...")
        test_smiles = [
            "c1ccccc1",  # benzene
            "c1ccc2ccccc2c1",  # naphthalene
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        ]
        dataset = MolecularDataset(test_smiles, dataset_name="test")

    if compare:
        # Comparison mode: 2 rows x 5 columns
        print(f"\nComparing Spectral Clustering: discrete vs kmeans")
        print(f"  k_min_factor: {k_min_factor}")
        print(f"  k_max_factor: {k_max_factor}")
        print(f"  n_init: {n_init} (for kmeans)")
        
        # Collect molecules
        molecules = []
        for idx, graph in enumerate(dataset):
            if len(molecules) >= num_samples:
                break
            
            if graph.num_nodes > max_nodes or graph.num_nodes == 0:
                continue
            
            try:
                smiles = graph_to_smiles(graph)
                if not smiles:
                    continue
                pos = compute_rdkit_2d_layout(smiles)
                if pos is None:
                    continue
                molecules.append((graph, smiles, pos))
            except Exception:
                continue
        
        if len(molecules) < num_samples:
            print(f"  Warning: Only found {len(molecules)} valid molecules")
        
        # Create comparison figure: 2 rows x 5 columns
        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for col_idx, (graph, smiles, pos) in enumerate(molecules[:num_samples]):
            # Top row: discrete
            hg_discrete = partition_with_assign_labels(
                graph,
                assign_labels="discretize",
                k_min_factor=k_min_factor,
                k_max_factor=k_max_factor,
                n_init=1,  # Not used for discretize
                seed=seed,
            )
            plot_molecule_with_communities(axes[0, col_idx], graph, smiles, hg_discrete, pos)
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms() if mol else graph.num_nodes
            axes[0, col_idx].set_title(
                f"Discrete\n{num_atoms} atoms, {hg_discrete.num_communities} comm.",
                fontsize=10,
                fontweight="bold",
            )
            
            # Bottom row: kmeans
            hg_kmeans = partition_with_assign_labels(
                graph,
                assign_labels="kmeans",
                k_min_factor=k_min_factor,
                k_max_factor=k_max_factor,
                n_init=n_init,
                seed=seed,
            )
            plot_molecule_with_communities(axes[1, col_idx], graph, smiles, hg_kmeans, pos)
            axes[1, col_idx].set_title(
                f"K-means\n{num_atoms} atoms, {hg_kmeans.num_communities} comm.",
                fontsize=10,
                fontweight="bold",
            )
        
        fig.suptitle(
            "Spectral Clustering Comparison: Discrete vs K-means",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        output_path = output_dir / "moses_sc_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        print(f"\nSaved comparison figure: {output_path}")
        print(f"  Visualized {len(molecules)} molecules")
        return

    # Original single-molecule mode
    coarsener = SpectralCoarsening(
        k_min_factor=k_min_factor,
        k_max_factor=k_max_factor,
        n_init=n_init,
        seed=seed,
    )

    print(f"\nVisualizing molecules with Spectral Clustering (kmeans)...")
    print(f"  k_min_factor: {k_min_factor}")
    print(f"  k_max_factor: {k_max_factor}")
    print(f"  n_init: {n_init}")
    print(f"  assign_labels: kmeans")

    visualized = 0
    skipped = 0

    for idx, graph in enumerate(dataset):
        if visualized >= num_samples:
            break

        # Skip large molecules
        if graph.num_nodes > max_nodes:
            skipped += 1
            continue

        # Skip empty graphs
        if graph.num_nodes == 0:
            skipped += 1
            continue

        try:
            # Get SMILES
            smiles = graph_to_smiles(graph)
            if not smiles:
                skipped += 1
                continue

            # Build hierarchy using spectral clustering
            hg = coarsener.build_hierarchy(graph)

            # Compute layout
            pos = compute_rdkit_2d_layout(smiles)
            if pos is None:
                skipped += 1
                continue

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Plot molecule with communities
            plot_molecule_with_communities(ax, graph, smiles, hg, pos)

            # Add title
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms() if mol else graph.num_nodes
            title = (
                f"MOSES Molecule {visualized + 1} | "
                f"{num_atoms} atoms, {hg.num_communities} communities | "
                f"SC+kmeans"
            )
            fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

            # Save figure
            output_path = output_dir / f"moses_sc_{visualized + 1:04d}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            print(f"  [{visualized + 1}/{num_samples}] Saved: {output_path.name}")
            visualized += 1

        except Exception as e:
            print(f"  Error visualizing molecule {idx}: {e}")
            skipped += 1
            continue

    print(f"\nCompleted!")
    print(f"  Visualized: {visualized} molecules")
    if skipped > 0:
        print(f"  Skipped: {skipped} molecules")
    print(f"  Output directory: {output_dir}")


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize MOSES molecules with Spectral Clustering (discrete vs kmeans)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of molecules to visualize (default: 12, ignored if --compare)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/moses_sc",
        help="Output directory for figures (default: ./figures/moses_sc)",
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
        help="Maximum number of nodes per molecule (default: 100)",
    )
    parser.add_argument(
        "--k-min-factor",
        type=float,
        default=0.7,
        help="Factor for minimum cluster count (default: 0.7)",
    )
    parser.add_argument(
        "--k-max-factor",
        type=float,
        default=1.3,
        help="Factor for maximum cluster count (default: 1.3)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of initializations for spectral clustering (default: 10)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare discrete vs kmeans in 2x5 grid (forces 5 samples)",
    )

    args = parser.parse_args()

    visualize_moses_molecules(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        max_nodes=args.max_nodes,
        k_min_factor=args.k_min_factor,
        k_max_factor=args.k_max_factor,
        n_init=args.n_init,
        compare=args.compare,
    )


if __name__ == "__main__":
    main()
