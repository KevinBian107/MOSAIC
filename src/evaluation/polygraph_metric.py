"""PolyGraph Discrepancy metric for graph generation evaluation.

This module provides the PolyGraph Discrepancy (PGD) metric, a classifier-based
metric that measures the quality of generated graphs by training a binary classifier
to distinguish between reference and generated distributions.
"""

from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class PolygraphMetric:
    """PolyGraph Discrepancy metric for evaluating graph generation quality.

    PGD trains a binary classifier to distinguish between reference and generated
    graphs. The classifier's discrimination ability serves as a measure of
    distribution mismatch. Lower scores indicate better generation quality.

    Attributes:
        reference_graphs: List of reference graphs for comparison.
        max_reference_size: Maximum number of reference graphs to use (for memory).
        _pgd_metric: PolyGraph metric instance from polygraph-benchmark library.
        _reference_prepared: Whether reference graphs have been preprocessed.
    """

    def __init__(
        self,
        reference_graphs: list[Data | nx.Graph],
        max_reference_size: int = 10000,
    ) -> None:
        """Initialize PolyGraph Discrepancy metric.

        Args:
            reference_graphs: List of reference graphs (Data objects or NetworkX).
            max_reference_size: Maximum number of reference graphs to use.
                Larger values provide more stable estimates but use more memory.
        """
        # Import here to make it optional dependency
        try:
            from polygraph.metrics import StandardPGD
        except ImportError:
            raise ImportError(
                "PolyGraph benchmark not installed. "
                "Install with: pip install polygraph-benchmark"
            )

        # Limit reference size for memory constraints
        if len(reference_graphs) > max_reference_size:
            reference_graphs = reference_graphs[:max_reference_size]

        self.reference_graphs = reference_graphs
        self.max_reference_size = max_reference_size
        self._pgd_metric: Optional[StandardPGD] = None
        self._reference_prepared = False

    def _to_networkx_list(self, graphs: list[Data | nx.Graph]) -> list[nx.Graph]:
        """Convert graphs to NetworkX format.

        Args:
            graphs: List of PyG Data objects or NetworkX graphs.

        Returns:
            List of NetworkX graphs with invalid graphs filtered out.
        """
        nx_graphs = []
        for g in graphs:
            if isinstance(g, Data):
                # Convert PyG Data to NetworkX
                g = to_networkx(g, to_undirected=True, remove_self_loops=True)
            if isinstance(g, nx.Graph) and g.number_of_nodes() > 0:
                nx_graphs.append(g)
        return nx_graphs

    def _precompute_reference(self) -> None:
        """Prepare reference graphs for PGD computation.

        Converts reference graphs to NetworkX format and initializes
        the PGD metric instance.
        """
        if self._reference_prepared:
            return

        from polygraph.metrics import StandardPGD

        # Convert reference graphs to NetworkX
        self._reference_nx = self._to_networkx_list(self.reference_graphs)

        if len(self._reference_nx) == 0:
            raise ValueError("No valid reference graphs after conversion")

        # Initialize PGD metric with reference graphs
        self._pgd_metric = StandardPGD(reference_graphs=self._reference_nx)
        self._reference_prepared = True

    def compute(self, generated_graphs: list[Data | nx.Graph]) -> dict[str, float]:
        """Compute PolyGraph Discrepancy score.

        Args:
            generated_graphs: List of generated graphs (Data objects or NetworkX).

        Returns:
            Dictionary with single key 'pgd' mapping to score in [0, 1].
            Lower values indicate better generation quality:
            - < 0.1: Excellent (indistinguishable from reference)
            - < 0.3: Good
            - < 0.5: Moderate
            - >= 0.5: Poor (easily distinguishable)
        """
        self._precompute_reference()

        # Convert generated graphs to NetworkX
        generated_nx = self._to_networkx_list(generated_graphs)

        if len(generated_nx) == 0:
            # No valid generated graphs - return worst score
            return {"pgd": 1.0}

        try:
            # PGD requires balanced datasets (same number of ref and gen graphs)
            # Sample reference graphs to match generated graph count
            import random

            if len(self._reference_nx) > len(generated_nx):
                # Sample without replacement to match generated size
                sampled_ref = random.sample(self._reference_nx, len(generated_nx))
            else:
                # Use all reference graphs
                sampled_ref = self._reference_nx

            # Re-initialize PGD with balanced reference set
            from polygraph.metrics import StandardPGD

            balanced_pgd = StandardPGD(reference_graphs=sampled_ref)

            # Compute PGD using polygraph library
            # The metric trains a classifier and returns discrimination score
            pgd_result = balanced_pgd.compute(generated_graphs=generated_nx)

            # Extract the main PGD score from the result
            # PolyGraphDiscrepancyResult has key 'polygraphscore'
            pgd_score = pgd_result.get("polygraphscore", pgd_result.get("pgd", 1.0))

            # PGD should return a value in [0, 1]
            # Ensure it's properly bounded
            pgd_score = float(pgd_score)
            pgd_score = max(0.0, min(1.0, pgd_score))

            return {"pgd": pgd_score}

        except Exception as e:
            # If computation fails, log warning and return invalid score
            import logging

            logging.warning(f"PGD computation failed: {e}")
            return {"pgd": 1.0}

    def __call__(self, generated_graphs: list[Data | nx.Graph]) -> dict[str, float]:
        """Compute PGD score (callable interface).

        Args:
            generated_graphs: List of generated graphs.

        Returns:
            Dictionary with PGD score.
        """
        return self.compute(generated_graphs)
