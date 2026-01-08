"""Tests for motif detection module."""

import pytest
import torch
from torch_geometric.data import Data

from src.data.motif import MotifDetector, MotifType
from tests.fixtures.graphs import (
    triangle_graph,
    square_graph,
    star_graph,
    complete_graph_k4,
    empty_graph,
)


class TestMotifDetector:
    """Tests for MotifDetector class."""

    def test_init_default(self) -> None:
        """Test default initialization with all motif types."""
        detector = MotifDetector()
        assert len(detector.motif_types) == len(MotifType)

    def test_init_specific_types(self) -> None:
        """Test initialization with specific motif types."""
        detector = MotifDetector([MotifType.TRIANGLE, MotifType.STAR])
        assert len(detector.motif_types) == 2
        assert MotifType.TRIANGLE in detector.motif_types

    def test_detect_triangle(self, triangle_graph: Data) -> None:
        """Test triangle detection in a triangle graph."""
        detector = MotifDetector([MotifType.TRIANGLE])
        result = detector.detect(triangle_graph)

        assert "triangle" in result["motif_types"]
        assert result["motif_counts"]["triangle"] == 1

    def test_detect_no_triangle_in_square(self, square_graph: Data) -> None:
        """Test that no triangle is detected in a square graph."""
        detector = MotifDetector([MotifType.TRIANGLE])
        result = detector.detect(square_graph)

        assert result["motif_counts"].get("triangle", 0) == 0

    def test_detect_star(self, star_graph: Data) -> None:
        """Test star detection in a star graph."""
        detector = MotifDetector([MotifType.STAR])
        result = detector.detect(star_graph)

        assert "star" in result["motif_types"]
        assert result["motif_counts"]["star"] >= 1

    def test_detect_four_clique(self, complete_graph_k4: Data) -> None:
        """Test 4-clique detection in K4."""
        detector = MotifDetector([MotifType.FOUR_CLIQUE])
        result = detector.detect(complete_graph_k4)

        assert "four_clique" in result["motif_types"]
        assert result["motif_counts"]["four_clique"] == 1

    def test_detect_empty_graph(self, empty_graph: Data) -> None:
        """Test detection on empty graph returns no motifs."""
        detector = MotifDetector()
        result = detector.detect(empty_graph)

        assert result["num_motifs"] == 0
        assert all(count == 0 for count in result["motif_counts"].values())

    def test_get_motif_vector(self, triangle_graph: Data) -> None:
        """Test motif vector computation."""
        detector = MotifDetector()
        vector = detector.get_motif_vector(triangle_graph)

        assert len(vector) == len(MotifType)
        assert vector.sum() > 0


class TestMotifType:
    """Tests for MotifType enum."""

    def test_all_types_exist(self) -> None:
        """Test that all expected motif types exist."""
        expected = ["TRIANGLE", "FOUR_CYCLE", "FIVE_CYCLE", "FOUR_CLIQUE", "STAR"]
        for name in expected:
            assert hasattr(MotifType, name)

    def test_types_are_unique(self) -> None:
        """Test that all motif type values are unique."""
        values = [mt.value for mt in MotifType]
        assert len(values) == len(set(values))
