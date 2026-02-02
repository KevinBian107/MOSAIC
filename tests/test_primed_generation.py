"""Tests for primed generation integration."""

import pytest
import torch
from unittest.mock import MagicMock
from torch_geometric.data import Data

from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer
from src.transfer_learning.generation.primed_generator import PrimedGenerator
from src.transfer_learning.scaffolds.library import ScaffoldLibrary


@pytest.fixture
def scaffold_library() -> ScaffoldLibrary:
    """Create a scaffold library for testing."""
    return ScaffoldLibrary()


@pytest.fixture
def hdtc_tokenizer() -> HDTCTokenizer:
    """Create HDTC tokenizer."""
    tokenizer = HDTCTokenizer()
    tokenizer.set_num_nodes(50)
    return tokenizer


@pytest.fixture
def mock_model(hdtc_tokenizer: HDTCTokenizer) -> MagicMock:
    """Create a mock GraphGeneratorModule."""
    model = MagicMock()
    model.tokenizer = hdtc_tokenizer

    # Mock generate method to return dummy graphs
    def mock_generate(input_ids=None, **kwargs):
        num_samples = input_ids.size(0) if input_ids is not None else 1
        graphs = []
        for _ in range(num_samples):
            # Create a simple graph (expanded from scaffold)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
            graph = Data(edge_index=edge_index, num_nodes=4)
            graphs.append(graph)
        return graphs, 0.01  # graphs, avg_time

    model.generate = mock_generate
    return model


class TestPrimedGeneratorInit:
    """Tests for PrimedGenerator initialization."""

    def test_init_with_model(
        self, mock_model: MagicMock, scaffold_library: ScaffoldLibrary
    ) -> None:
        """Generator should initialize with model and library."""
        generator = PrimedGenerator(mock_model, scaffold_library)
        assert generator.model == mock_model
        assert generator.scaffold_library == scaffold_library

    def test_init_creates_default_library(self, mock_model: MagicMock) -> None:
        """Generator should create default library if not provided."""
        generator = PrimedGenerator(mock_model)
        assert generator.scaffold_library is not None
        assert len(generator.scaffold_library) > 0

    def test_init_creates_primer(self, mock_model: MagicMock) -> None:
        """Generator should create appropriate primer for tokenizer."""
        generator = PrimedGenerator(mock_model)
        from src.transfer_learning.primers.hdtc_primer import HDTCPrimer

        assert isinstance(generator.primer, HDTCPrimer)


class TestGenerateFromScaffold:
    """Tests for generate_from_scaffold method."""

    def test_generate_from_scaffold_name(self, mock_model: MagicMock) -> None:
        """Generator should accept scaffold name string."""
        generator = PrimedGenerator(mock_model)
        graphs, time = generator.generate_from_scaffold("benzene", num_samples=2)

        assert len(graphs) == 2
        assert all(isinstance(g, Data) for g in graphs)
        assert time > 0

    def test_generate_from_scaffold_object(
        self, mock_model: MagicMock, scaffold_library: ScaffoldLibrary
    ) -> None:
        """Generator should accept Scaffold object."""
        generator = PrimedGenerator(mock_model)
        scaffold = scaffold_library.get_scaffold("naphthalene")
        graphs, time = generator.generate_from_scaffold(scaffold, num_samples=3)

        assert len(graphs) == 3

    def test_generate_from_unknown_scaffold(self, mock_model: MagicMock) -> None:
        """Generator should raise for unknown scaffold name."""
        generator = PrimedGenerator(mock_model)
        with pytest.raises(KeyError):
            generator.generate_from_scaffold("nonexistent")


class TestGenerateFromSmiles:
    """Tests for generate_from_smiles method."""

    def test_generate_from_smiles(self, mock_model: MagicMock) -> None:
        """Generator should accept custom SMILES."""
        generator = PrimedGenerator(mock_model)
        graphs, time = generator.generate_from_smiles("CC(C)C", num_samples=2)

        assert len(graphs) == 2

    def test_generate_from_invalid_smiles(self, mock_model: MagicMock) -> None:
        """Generator should raise for invalid SMILES."""
        generator = PrimedGenerator(mock_model)
        with pytest.raises(ValueError):
            generator.generate_from_smiles("invalid_smiles")


class TestGenerateBatchDiverse:
    """Tests for generate_batch_diverse method."""

    def test_generate_batch_diverse(self, mock_model: MagicMock) -> None:
        """Generator should generate from multiple scaffolds."""
        generator = PrimedGenerator(mock_model)
        scaffolds = ["benzene", "naphthalene", "pyridine"]
        results, time = generator.generate_batch_diverse(
            scaffolds, samples_per_scaffold=2
        )

        assert len(results) == 3
        assert all(len(graphs) == 2 for graphs in results)


class TestGenerateByTier:
    """Tests for generate_by_tier method."""

    def test_generate_by_tier(self, mock_model: MagicMock) -> None:
        """Generator should generate from all scaffolds in a tier."""
        generator = PrimedGenerator(mock_model)
        results, time = generator.generate_by_tier(
            1, samples_per_scaffold=1, max_scaffolds=3
        )

        assert len(results) <= 3
        assert all(isinstance(name, str) for name in results.keys())

    def test_generate_by_tier_max_scaffolds(self, mock_model: MagicMock) -> None:
        """Generator should respect max_scaffolds limit."""
        generator = PrimedGenerator(mock_model)
        results, _ = generator.generate_by_tier(
            1, samples_per_scaffold=1, max_scaffolds=2
        )

        assert len(results) == 2


class TestGetPrimerTokens:
    """Tests for primer token inspection."""

    def test_get_primer_tokens(self, mock_model: MagicMock) -> None:
        """Generator should return primer tokens."""
        generator = PrimedGenerator(mock_model)
        tokens = generator.get_primer_tokens("benzene")

        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 1
        assert len(tokens) > 0

    def test_primer_to_string(self, mock_model: MagicMock) -> None:
        """Generator should return readable primer string."""
        generator = PrimedGenerator(mock_model)
        string = generator.primer_to_string("benzene")

        assert isinstance(string, str)
        assert len(string) > 0


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_list_available_scaffolds(self, mock_model: MagicMock) -> None:
        """Generator should list available scaffolds."""
        generator = PrimedGenerator(mock_model)
        scaffolds = generator.list_available_scaffolds()

        assert isinstance(scaffolds, list)
        assert "benzene" in scaffolds

    def test_get_scaffold_info(self, mock_model: MagicMock) -> None:
        """Generator should return scaffold info."""
        generator = PrimedGenerator(mock_model)
        info = generator.get_scaffold_info("benzene")

        assert "name" in info
        assert "smiles" in info
        assert "tier" in info
        assert info["name"] == "benzene"
        assert info["tier"] == 1


class TestDifferentTokenizers:
    """Tests that PrimedGenerator works with different tokenizers."""

    def test_with_hdt_tokenizer(self) -> None:
        """Generator should work with HDT tokenizer."""
        tokenizer = HDTTokenizer()
        tokenizer.set_num_nodes(50)

        model = MagicMock()
        model.tokenizer = tokenizer
        model.generate = lambda input_ids=None, **kwargs: (
            [Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=1)],
            0.01,
        )

        generator = PrimedGenerator(model)
        from src.transfer_learning.primers.hdt_primer import HDTPrimer

        assert isinstance(generator.primer, HDTPrimer)

        graphs, _ = generator.generate_from_scaffold("benzene")
        assert len(graphs) == 1

    def test_with_hsent_tokenizer(self) -> None:
        """Generator should work with HSENT tokenizer."""
        tokenizer = HSENTTokenizer()
        tokenizer.set_num_nodes(50)

        model = MagicMock()
        model.tokenizer = tokenizer
        model.generate = lambda input_ids=None, **kwargs: (
            [Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=1)],
            0.01,
        )

        generator = PrimedGenerator(model)
        from src.transfer_learning.primers.hsent_primer import HSENTPrimer

        assert isinstance(generator.primer, HSENTPrimer)

        graphs, _ = generator.generate_from_scaffold("benzene")
        assert len(graphs) == 1
