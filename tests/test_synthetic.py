"""Tests for synthetic graph generation module."""

import pytest
import torch

from src.data.motif import MotifType
from src.data.synthetic import SyntheticGraphGenerator, create_mixed_dataset


class TestSyntheticGraphGenerator:
    """Tests for SyntheticGraphGenerator class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        gen = SyntheticGraphGenerator()
        assert gen.generator_name == "erdos_renyi"

    def test_init_custom_generator(self) -> None:
        """Test initialization with custom generator."""
        gen = SyntheticGraphGenerator("barabasi_albert", seed=42)
        assert gen.generator_name == "barabasi_albert"

    def test_init_invalid_generator(self) -> None:
        """Test initialization with invalid generator raises error."""
        with pytest.raises(ValueError, match="Unknown generator"):
            SyntheticGraphGenerator("invalid_generator")

    def test_generate_single(self) -> None:
        """Test generating a single graph."""
        gen = SyntheticGraphGenerator("erdos_renyi", seed=42)
        graphs = gen.generate(num_graphs=1, n=20, p=0.3)

        assert len(graphs) == 1
        assert graphs[0].num_nodes == 20
        assert hasattr(graphs[0], "motif_labels")
        assert hasattr(graphs[0], "motif_counts")

    def test_generate_multiple(self) -> None:
        """Test generating multiple graphs."""
        gen = SyntheticGraphGenerator("barabasi_albert", seed=42)
        graphs = gen.generate(num_graphs=5, n=30, m=2)

        assert len(graphs) == 5
        for g in graphs:
            assert g.num_nodes == 30
            assert hasattr(g, "motif_labels")

    def test_generate_deterministic(self) -> None:
        """Test generation is deterministic with same seed."""
        gen1 = SyntheticGraphGenerator("erdos_renyi", seed=42)
        gen2 = SyntheticGraphGenerator("erdos_renyi", seed=42)

        graphs1 = gen1.generate(num_graphs=3, n=20, p=0.3)
        graphs2 = gen2.generate(num_graphs=3, n=20, p=0.3)

        for g1, g2 in zip(graphs1, graphs2):
            assert torch.equal(g1.edge_index, g2.edge_index)

    def test_generate_dataset(self) -> None:
        """Test generating train/val/test splits."""
        gen = SyntheticGraphGenerator("erdos_renyi", seed=42)
        splits = gen.generate_dataset(
            num_train=10, num_val=5, num_test=5, n=20, p=0.3
        )

        assert len(splits["train"]) == 10
        assert len(splits["val"]) == 5
        assert len(splits["test"]) == 5

    def test_motif_labels_shape(self) -> None:
        """Test motif labels have correct shape."""
        gen = SyntheticGraphGenerator("complete", seed=42)
        graphs = gen.generate(num_graphs=1, n=5)

        g = graphs[0]
        assert g.motif_labels.shape[0] == g.num_nodes

    def test_all_generators_work(self) -> None:
        """Test all available generators produce valid output."""
        for name in SyntheticGraphGenerator.GENERATORS.keys():
            gen = SyntheticGraphGenerator(name, seed=42)
            graphs = gen.generate(num_graphs=1)
            assert len(graphs) == 1
            assert graphs[0].num_nodes > 0


class TestCreateMixedDataset:
    """Tests for create_mixed_dataset function."""

    def test_mixed_dataset(self) -> None:
        """Test creating a mixed dataset."""
        generators = ["erdos_renyi", "barabasi_albert"]
        graphs = create_mixed_dataset(
            generators=generators,
            num_per_generator=5,
            seed=42,
        )

        assert len(graphs) == 10

    def test_mixed_dataset_generator_names(self) -> None:
        """Test that graphs have correct generator names."""
        generators = ["erdos_renyi", "star"]
        graphs = create_mixed_dataset(
            generators=generators,
            num_per_generator=2,
            seed=42,
        )

        er_graphs = [g for g in graphs if g.generator_name == "erdos_renyi"]
        star_graphs = [g for g in graphs if g.generator_name == "star"]

        assert len(er_graphs) == 2
        assert len(star_graphs) == 2
