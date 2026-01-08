"""Motif-specific evaluation metrics for graph generation.

This module provides novel metrics for evaluating how well generated graphs
preserve motif structures compared to reference graphs.
"""

from typing import Optional

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.data.motif import MotifDetector, MotifType
from src.evaluation.dist_helper import compute_mmd, gaussian


class MotifMetrics:
    """Evaluator for motif preservation in graph generation.

    Computes metrics comparing motif distributions between generated
    and reference graphs.

    Attributes:
        reference_graphs: List of reference graphs.
        motif_detector: MotifDetector instance.
        reference_vectors: Cached motif vectors for reference graphs.
    """

    def __init__(
        self,
        reference_graphs: list,
        motif_types: Optional[list[MotifType]] = None,
    ) -> None:
        """Initialize the motif metrics evaluator.

        Args:
            reference_graphs: Reference graphs for comparison.
            motif_types: Motif types to evaluate. Default: all types.
        """
        self.motif_detector = MotifDetector(motif_types)

        self.reference_graphs = []
        for g in reference_graphs:
            if isinstance(g, Data):
                self.reference_graphs.append(g)
            elif isinstance(g, nx.Graph):
                edge_list = list(g.edges())
                if edge_list:
                    import torch

                    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                    if g.is_directed():
                        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                data = Data(edge_index=edge_index, num_nodes=g.number_of_nodes())
                self.reference_graphs.append(data)

        self._ref_vectors: Optional[np.ndarray] = None
        self._ref_counts: Optional[list[dict[str, int]]] = None

    def _precompute_reference(self) -> None:
        """Precompute reference motif statistics."""
        if self._ref_vectors is None:
            self._ref_vectors = np.array(
                [self.motif_detector.get_motif_vector(g) for g in self.reference_graphs]
            )
            self._ref_counts = [
                self.motif_detector.get_motif_counts(g) for g in self.reference_graphs
            ]

    def _to_data_list(self, graphs: list) -> list[Data]:
        """Convert graphs to PyG Data format.

        Args:
            graphs: List of Data or NetworkX graphs.

        Returns:
            List of Data objects.
        """
        data_list = []
        for g in graphs:
            if isinstance(g, Data):
                data_list.append(g)
            elif isinstance(g, nx.Graph):
                if g.number_of_nodes() == 0:
                    continue
                import torch

                edge_list = list(g.edges())
                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                    if g.is_directed():
                        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                data = Data(edge_index=edge_index, num_nodes=g.number_of_nodes())
                data_list.append(data)
        return data_list

    def compute_msc(self, generated_graphs: list) -> float:
        """Compute Motif Signature Comparison (MSC).

        MSC measures the MMD between motif count vectors of generated
        and reference graphs. Lower values indicate better motif preservation.

        Args:
            generated_graphs: List of generated graphs.

        Returns:
            MSC value (MMD between motif signatures).
        """
        self._precompute_reference()

        data_list = self._to_data_list(generated_graphs)
        if not data_list:
            return float("inf")

        gen_vectors = np.array(
            [self.motif_detector.get_motif_vector(g) for g in data_list]
        )

        return compute_mmd(
            list(self._ref_vectors),
            list(gen_vectors),
            kernel=gaussian,
            is_hist=False,
            sigma=1.0,
        )

    def compute_mfd(self, generated_graphs: list) -> dict[str, float]:
        """Compute Motif Frequency Distribution (MFD) per motif type.

        MFD compares the distribution of each motif type separately.

        Args:
            generated_graphs: List of generated graphs.

        Returns:
            Dictionary mapping motif type names to MMD values.
        """
        self._precompute_reference()

        data_list = self._to_data_list(generated_graphs)
        if not data_list:
            return {mt.name.lower(): float("inf") for mt in MotifType}

        gen_counts = [self.motif_detector.get_motif_counts(g) for g in data_list]

        results = {}
        for motif_type in MotifType:
            name = motif_type.name.lower()

            ref_freqs = []
            for i, counts in enumerate(self._ref_counts):
                num_nodes = max(self.reference_graphs[i].num_nodes, 1)
                ref_freqs.append(np.array([counts.get(name, 0) / num_nodes]))

            gen_freqs = []
            for i, counts in enumerate(gen_counts):
                num_nodes = max(data_list[i].num_nodes, 1)
                gen_freqs.append(np.array([counts.get(name, 0) / num_nodes]))

            if ref_freqs and gen_freqs:
                mmd = compute_mmd(ref_freqs, gen_freqs, kernel=gaussian, is_hist=False)
                results[name] = mmd
            else:
                results[name] = 0.0

        return results

    def compute_mpr(
        self,
        generated_graphs: list,
        original_graphs: list,
    ) -> float:
        """Compute Motif Preservation Rate (MPR) for conditional generation.

        MPR measures how well motifs are preserved when regenerating
        graphs from their token representations.

        Args:
            generated_graphs: List of generated graphs.
            original_graphs: List of original graphs (same order).

        Returns:
            Average motif preservation rate [0, 1].
        """
        gen_list = self._to_data_list(generated_graphs)
        orig_list = self._to_data_list(original_graphs)

        if len(gen_list) != len(orig_list):
            raise ValueError("Generated and original lists must have same length")

        preservation_rates = []
        for gen, orig in zip(gen_list, orig_list):
            gen_counts = self.motif_detector.get_motif_counts(gen)
            orig_counts = self.motif_detector.get_motif_counts(orig)

            total_orig = sum(orig_counts.values())
            if total_orig == 0:
                preservation_rates.append(1.0)
                continue

            preserved = 0
            for motif_type, orig_count in orig_counts.items():
                gen_count = gen_counts.get(motif_type, 0)
                preserved += min(gen_count, orig_count)

            preservation_rates.append(preserved / total_orig)

        return np.mean(preservation_rates)

    def compute(self, generated_graphs: list) -> dict[str, float]:
        """Compute all motif metrics.

        Args:
            generated_graphs: List of generated graphs.

        Returns:
            Dictionary with MSC and per-motif MFD values.
        """
        results = {"msc": self.compute_msc(generated_graphs)}

        mfd = self.compute_mfd(generated_graphs)
        for name, value in mfd.items():
            results[f"mfd_{name}"] = value

        return results

    def __call__(self, generated_graphs: list) -> dict[str, float]:
        """Compute metrics (callable interface)."""
        return self.compute(generated_graphs)
