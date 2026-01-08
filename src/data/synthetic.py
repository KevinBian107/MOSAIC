"""Synthetic graph generation with motif labeling.

This module provides generators for synthetic graphs with known motif labels,
enabling evaluation of motif preservation in graph generation models.
"""

from typing import Optional

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from src.data.motif import MotifDetector, MotifType


class SyntheticGraphGenerator:
    """Generator for synthetic graphs with motif labels.

    This class wraps various NetworkX graph generators and automatically
    detects and labels motifs in the generated graphs.

    Attributes:
        generator_name: Name of the graph generator.
        motif_detector: MotifDetector instance for labeling.
        rng: Random number generator.
    """

    GENERATORS: dict[str, dict] = {
        "erdos_renyi": {
            "func": lambda n, p, rng: nx.erdos_renyi_graph(n, p, seed=rng),
            "default_params": {"n": 50, "p": 0.2},
        },
        "barabasi_albert": {
            "func": lambda n, m, rng: nx.barabasi_albert_graph(n, m, seed=rng),
            "default_params": {"n": 50, "m": 3},
        },
        "watts_strogatz": {
            "func": lambda n, k, p, rng: nx.watts_strogatz_graph(n, k, p, seed=rng),
            "default_params": {"n": 50, "k": 4, "p": 0.3},
        },
        "random_regular": {
            "func": lambda d, n, rng: nx.random_regular_graph(d, n, seed=rng),
            "default_params": {"d": 3, "n": 50},
        },
        "caveman": {
            "func": lambda l, k, rng: nx.connected_caveman_graph(l, k),
            "default_params": {"l": 10, "k": 5},
        },
        "grid_2d": {
            "func": lambda m, n, rng: nx.grid_2d_graph(m, n),
            "default_params": {"m": 7, "n": 7},
        },
        "star": {
            "func": lambda n, rng: nx.star_graph(n),
            "default_params": {"n": 20},
        },
        "complete": {
            "func": lambda n, rng: nx.complete_graph(n),
            "default_params": {"n": 10},
        },
        "cycle": {
            "func": lambda n, rng: nx.cycle_graph(n),
            "default_params": {"n": 20},
        },
        "ladder": {
            "func": lambda n, rng: nx.ladder_graph(n),
            "default_params": {"n": 20},
        },
    }

    def __init__(
        self,
        generator_name: str = "erdos_renyi",
        motif_types: Optional[list[MotifType]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the synthetic graph generator.

        Args:
            generator_name: Name of the NetworkX generator to use.
            motif_types: Motif types to detect. If None, detects all types.
            seed: Random seed for reproducibility.
        """
        if generator_name not in self.GENERATORS:
            available = ", ".join(self.GENERATORS.keys())
            raise ValueError(
                f"Unknown generator: {generator_name}. Available: {available}"
            )

        self.generator_name = generator_name
        self.motif_detector = MotifDetector(motif_types)
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        num_graphs: int = 1,
        **kwargs,
    ) -> list[Data]:
        """Generate synthetic graphs with motif labels.

        Args:
            num_graphs: Number of graphs to generate.
            **kwargs: Parameters passed to the graph generator.

        Returns:
            List of PyTorch Geometric Data objects with motif labels.
        """
        gen_info = self.GENERATORS[self.generator_name]
        params = {**gen_info["default_params"], **kwargs}

        graphs = []
        for _ in range(num_graphs):
            G = gen_info["func"](**params, rng=self.rng)

            if isinstance(G.nodes(), nx.classes.reportviews.NodeView):
                mapping = {node: i for i, node in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)

            data = from_networkx(G)
            data = Data(edge_index=data.edge_index, num_nodes=G.number_of_nodes())

            motif_result = self.motif_detector.detect(data)
            data.motif_labels = motif_result["motif_labels"]
            data.motif_types = motif_result["motif_types"]
            data.motif_counts = motif_result["motif_counts"]
            data.num_motifs = motif_result["num_motifs"]
            data.generator_name = self.generator_name

            graphs.append(data)

        return graphs

    def generate_dataset(
        self,
        num_train: int = 1000,
        num_val: int = 100,
        num_test: int = 100,
        **kwargs,
    ) -> dict[str, list[Data]]:
        """Generate train/val/test splits of synthetic graphs.

        Args:
            num_train: Number of training graphs.
            num_val: Number of validation graphs.
            num_test: Number of test graphs.
            **kwargs: Parameters passed to the graph generator.

        Returns:
            Dictionary with 'train', 'val', 'test' keys mapping to graph lists.
        """
        return {
            "train": self.generate(num_train, **kwargs),
            "val": self.generate(num_val, **kwargs),
            "test": self.generate(num_test, **kwargs),
        }


def create_mixed_dataset(
    generators: list[str],
    num_per_generator: int = 100,
    motif_types: Optional[list[MotifType]] = None,
    seed: Optional[int] = None,
) -> list[Data]:
    """Create a dataset mixing multiple graph types.

    Args:
        generators: List of generator names to use.
        num_per_generator: Number of graphs per generator.
        motif_types: Motif types to detect.
        seed: Random seed.

    Returns:
        List of Data objects from all generators.
    """
    all_graphs = []
    for i, gen_name in enumerate(generators):
        gen_seed = seed + i if seed is not None else None
        generator = SyntheticGraphGenerator(gen_name, motif_types, gen_seed)
        graphs = generator.generate(num_per_generator)
        all_graphs.extend(graphs)
    return all_graphs
