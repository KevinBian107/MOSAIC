"""Tests for evaluation metrics module."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.dist_helper import compute_mmd, gaussian, gaussian_tv
from src.evaluation.metrics import (
    GraphMetrics,
    compute_validity_metrics,
    degree_histogram,
    spectral_histogram,
    clustering_histogram,
)
from src.evaluation.polygraph_metric import PolygraphMetric
from tests.fixtures.graphs import sample_graph_list, erdos_renyi_graph


class TestDistHelper:
    """Tests for distance helper functions."""

    def test_gaussian_identical(self) -> None:
        """Test Gaussian kernel returns 1 for identical inputs."""
        x = np.array([1.0, 2.0, 3.0])
        result = gaussian(x, x)
        assert np.isclose(result, 1.0)

    def test_gaussian_different(self) -> None:
        """Test Gaussian kernel returns < 1 for different inputs."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        result = gaussian(x, y)
        assert 0 < result < 1

    def test_gaussian_tv_identical(self) -> None:
        """Test Gaussian TV kernel returns 1 for identical inputs."""
        x = np.array([1.0, 2.0, 3.0])
        result = gaussian_tv(x, x)
        assert np.isclose(result, 1.0)

    def test_compute_mmd_identical(self) -> None:
        """Test MMD is 0 for identical distributions."""
        samples = [np.array([1.0, 2.0, 3.0]) for _ in range(5)]
        result = compute_mmd(samples, samples, kernel=gaussian)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_compute_mmd_different(self) -> None:
        """Test MMD is positive for different distributions."""
        samples1 = [np.array([1.0, 0.0]) for _ in range(5)]
        samples2 = [np.array([0.0, 1.0]) for _ in range(5)]
        result = compute_mmd(samples1, samples2, kernel=gaussian)
        assert result > 0


class TestGraphMetrics:
    """Tests for GraphMetrics class."""

    def test_init(self, sample_graph_list: list[Data]) -> None:
        """Test initialization."""
        metrics = GraphMetrics(sample_graph_list)
        assert metrics.reference_graphs == sample_graph_list
        assert "degree" in metrics.metrics_list

    def test_compute_returns_dict(self, sample_graph_list: list[Data]) -> None:
        """Test compute returns dictionary of metrics."""
        metrics = GraphMetrics(sample_graph_list)
        result = metrics.compute(sample_graph_list)

        assert isinstance(result, dict)
        assert "degree" in result
        assert "spectral" in result
        assert "clustering" in result

    def test_compute_self_reference(self, sample_graph_list: list[Data]) -> None:
        """Test MMD is low when comparing to self."""
        metrics = GraphMetrics(sample_graph_list)
        result = metrics.compute(sample_graph_list)

        for value in result.values():
            assert value < 0.1

    def test_callable_interface(self, sample_graph_list: list[Data]) -> None:
        """Test __call__ works same as compute."""
        metrics = GraphMetrics(sample_graph_list)
        result1 = metrics.compute(sample_graph_list)
        result2 = metrics(sample_graph_list)

        assert result1.keys() == result2.keys()


class TestValidityMetrics:
    """Tests for validity metrics."""

    def test_compute_validity_identical(self, sample_graph_list: list[Data]) -> None:
        """Test validity metrics on identical sets."""
        result = compute_validity_metrics(sample_graph_list, sample_graph_list)

        assert "uniqueness" in result
        assert "novelty" in result

    def test_uniqueness_all_different(self) -> None:
        """Test uniqueness is 1 when all graphs are different."""
        import networkx as nx
        from torch_geometric.utils import from_networkx

        graphs = []
        for i in range(3, 6):
            G = nx.cycle_graph(i * 5)
            data = from_networkx(G)
            graphs.append(Data(edge_index=data.edge_index, num_nodes=G.number_of_nodes()))

        result = compute_validity_metrics(graphs, [])
        assert result["uniqueness"] == 1.0


class TestPolygraphMetric:
    """Tests for PolygraphMetric class."""

    def test_init(self, sample_graph_list: list[Data]) -> None:
        """Test PolygraphMetric initialization."""
        try:
            metric = PolygraphMetric(sample_graph_list)
            assert len(metric.reference_graphs) == len(sample_graph_list)
            assert metric.max_reference_size == 100
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_init_with_custom_size(self, sample_graph_list: list[Data]) -> None:
        """Test initialization with custom max_reference_size."""
        try:
            metric = PolygraphMetric(sample_graph_list, max_reference_size=100)
            assert metric.max_reference_size == 100
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_compute_returns_dict(self, sample_graph_list: list[Data]) -> None:
        """Test compute returns dictionary with pgd key."""
        try:
            metric = PolygraphMetric(sample_graph_list)
            result = metric.compute(sample_graph_list)

            assert isinstance(result, dict)
            assert "pgd" in result
            assert isinstance(result["pgd"], float)
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_pgd_bounded(self, sample_graph_list: list[Data]) -> None:
        """Test PGD score is in [0, 1] or -1 (failure sentinel)."""
        try:
            metric = PolygraphMetric(sample_graph_list)
            result = metric.compute(sample_graph_list)

            assert result["pgd"] == -1.0 or (0.0 <= result["pgd"] <= 1.0)
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_self_reference_low_pgd(self) -> None:
        """Test PGD is low when comparing distribution to itself.

        Uses a list of 20 graphs so polygraph's internal CV (n_splits=4) has enough samples.
        """
        try:
            import networkx as nx
            from torch_geometric.utils import from_networkx

            ref_list = [
                Data(
                    edge_index=from_networkx(
                        nx.erdos_renyi_graph(20, 0.3, seed=i)
                    ).edge_index,
                    num_nodes=20,
                )
                for i in range(20)
            ]
            metric = PolygraphMetric(ref_list)
            result = metric.compute(ref_list)

            # When comparing to self, PGD should be relatively low
            assert result["pgd"] < 0.5
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_callable_interface(self, sample_graph_list: list[Data]) -> None:
        """Test __call__ works same as compute."""
        try:
            metric = PolygraphMetric(sample_graph_list)
            result1 = metric.compute(sample_graph_list)
            result2 = metric(sample_graph_list)

            assert result1.keys() == result2.keys()
            assert result1["pgd"] == result2["pgd"]
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_empty_generated(self, sample_graph_list: list[Data]) -> None:
        """Test handling of empty generated graph list."""
        try:
            metric = PolygraphMetric(sample_graph_list)
            result = metric.compute([])

            # Should return failure sentinel for no valid graphs
            assert result["pgd"] == -1.0
        except ImportError:
            pytest.skip("polygraph-benchmark not installed")

    def test_networkx_input(self) -> None:
        """Test PolygraphMetric accepts NetworkX graphs."""
        try:
            import networkx as nx

            # Create reference graphs as NetworkX
            ref_graphs = [nx.erdos_renyi_graph(20, 0.3, seed=i) for i in range(3)]

            metric = PolygraphMetric(ref_graphs)
            result = metric.compute(ref_graphs)

            assert "pgd" in result
            assert result["pgd"] == -1.0 or (0.0 <= result["pgd"] <= 1.0)
        except ImportError:
            pytest.skip("polygraph-benchmark or networkx not installed")

