"""Standard graph generation evaluation metrics.

This module provides metrics for evaluating graph generation quality,
including degree distribution, spectral properties, and clustering.
"""

from typing import Optional

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.evaluation.dist_helper import compute_mmd, gaussian_emd, gaussian_tv


def degree_histogram(G: nx.Graph) -> np.ndarray:
    """Compute degree histogram for a graph.

    Args:
        G: NetworkX graph.

    Returns:
        Numpy array of degree counts.
    """
    return np.array(nx.degree_histogram(G))


def spectral_histogram(G: nx.Graph, n_bins: int = 200) -> np.ndarray:
    """Compute spectral histogram from normalized Laplacian eigenvalues.

    Args:
        G: NetworkX graph.
        n_bins: Number of histogram bins.

    Returns:
        Numpy array of spectral density histogram.
    """
    try:
        L = nx.normalized_laplacian_matrix(G).todense()
        eigs = eigvalsh(L)
    except Exception:
        eigs = np.zeros(G.number_of_nodes())

    hist, _ = np.histogram(eigs, bins=n_bins, range=(-1e-5, 2), density=False)
    return hist / (hist.sum() + 1e-6)


def clustering_histogram(G: nx.Graph, n_bins: int = 100) -> np.ndarray:
    """Compute clustering coefficient histogram.

    Args:
        G: NetworkX graph.
        n_bins: Number of histogram bins.

    Returns:
        Numpy array of clustering coefficient histogram.
    """
    coeffs = list(nx.clustering(G).values())
    hist, _ = np.histogram(coeffs, bins=n_bins, range=(0.0, 1.0), density=False)
    return hist


class GraphMetrics:
    """Evaluator for standard graph generation metrics.

    Computes MMD-based metrics comparing generated graphs to reference
    distributions on degree, spectral, and clustering properties.

    Attributes:
        reference_graphs: List of reference NetworkX graphs.
        compute_emd: Whether to use EMD-based kernel.
        metrics_list: List of metrics to compute.
    """

    def __init__(
        self,
        reference_graphs: list[nx.Graph],
        compute_emd: bool = False,
        metrics_list: Optional[list[str]] = None,
    ) -> None:
        """Initialize the metrics evaluator.

        Args:
            reference_graphs: Reference graphs for comparison.
            compute_emd: Whether to use EMD kernel (slower but more accurate).
            metrics_list: Metrics to compute. Default: degree, spectral, clustering.
        """
        self.reference_graphs = reference_graphs
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list or ["degree", "spectral", "clustering"]

        self._ref_degree: Optional[list[np.ndarray]] = None
        self._ref_spectral: Optional[list[np.ndarray]] = None
        self._ref_clustering: Optional[list[np.ndarray]] = None

    def _precompute_reference(self) -> None:
        """Precompute reference statistics."""
        if "degree" in self.metrics_list and self._ref_degree is None:
            self._ref_degree = [degree_histogram(G) for G in self.reference_graphs]

        if "spectral" in self.metrics_list and self._ref_spectral is None:
            self._ref_spectral = [spectral_histogram(G) for G in self.reference_graphs]

        if "clustering" in self.metrics_list and self._ref_clustering is None:
            self._ref_clustering = [
                clustering_histogram(G) for G in self.reference_graphs
            ]

    def _to_networkx_list(self, graphs: list) -> list[nx.Graph]:
        """Convert a list of graphs to NetworkX format.

        Args:
            graphs: List of Data objects or NetworkX graphs.

        Returns:
            List of NetworkX graphs.
        """
        nx_graphs = []
        for g in graphs:
            if isinstance(g, Data):
                g = to_networkx(g, to_undirected=True, remove_self_loops=True)
            nx_graphs.append(g)
        return nx_graphs

    def compute(self, generated_graphs: list) -> dict[str, float]:
        """Compute all metrics for generated graphs.

        Args:
            generated_graphs: List of generated graphs.

        Returns:
            Dictionary mapping metric names to values.
        """
        self._precompute_reference()

        nx_graphs = self._to_networkx_list(generated_graphs)
        nx_graphs = [G for G in nx_graphs if G.number_of_nodes() > 0]

        kernel = gaussian_emd if self.compute_emd else gaussian_tv
        results = {}

        if "degree" in self.metrics_list:
            pred_degree = [degree_histogram(G) for G in nx_graphs]
            results["degree"] = compute_mmd(self._ref_degree, pred_degree, kernel)

        if "spectral" in self.metrics_list:
            pred_spectral = [spectral_histogram(G) for G in nx_graphs]
            results["spectral"] = compute_mmd(self._ref_spectral, pred_spectral, kernel)

        if "clustering" in self.metrics_list:
            pred_clustering = [clustering_histogram(G) for G in nx_graphs]
            results["clustering"] = compute_mmd(
                self._ref_clustering, pred_clustering, kernel, sigma=0.1
            )

        return results

    def __call__(self, generated_graphs: list) -> dict[str, float]:
        """Compute metrics (callable interface)."""
        return self.compute(generated_graphs)


def compute_validity_metrics(
    generated_graphs: list,
    reference_graphs: list,
) -> dict[str, float]:
    """Compute validity metrics (uniqueness, novelty).

    Args:
        generated_graphs: List of generated graphs.
        reference_graphs: List of reference/training graphs.

    Returns:
        Dictionary with validity metrics.
    """
    nx_gen = []
    for g in generated_graphs:
        if isinstance(g, Data):
            g = to_networkx(g, to_undirected=True, remove_self_loops=True)
        if g.number_of_nodes() > 0:
            nx_gen.append(g)

    nx_ref = []
    for g in reference_graphs:
        if isinstance(g, Data):
            g = to_networkx(g, to_undirected=True, remove_self_loops=True)
        nx_ref.append(g)

    unique_count = 0
    evaluated = []
    for g in nx_gen:
        is_unique = True
        for g_old in evaluated:
            if nx.faster_could_be_isomorphic(g, g_old):
                if nx.is_isomorphic(g, g_old):
                    is_unique = False
                    break
        if is_unique:
            evaluated.append(g)
            unique_count += 1

    novel_count = 0
    for g in evaluated:
        is_novel = True
        for g_ref in nx_ref:
            if nx.faster_could_be_isomorphic(g, g_ref):
                if nx.is_isomorphic(g, g_ref):
                    is_novel = False
                    break
        if is_novel:
            novel_count += 1

    n_total = len(nx_gen) if nx_gen else 1
    return {
        "uniqueness": unique_count / n_total,
        "novelty": novel_count / max(unique_count, 1),
    }
