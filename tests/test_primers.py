"""Tests for tokenizer-specific primers."""

import pytest
import torch

from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer
from src.transfer_learning.primers.factory import PrimerFactory
from src.transfer_learning.primers.hdt_primer import HDTPrimer
from src.transfer_learning.primers.hdtc_primer import HDTCPrimer
from src.transfer_learning.primers.hsent_primer import HSENTPrimer
from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary


@pytest.fixture
def scaffold_library() -> ScaffoldLibrary:
    """Create a scaffold library for testing."""
    return ScaffoldLibrary()


@pytest.fixture
def benzene_scaffold(scaffold_library: ScaffoldLibrary) -> Scaffold:
    """Get benzene scaffold."""
    return scaffold_library.get_scaffold("benzene")


@pytest.fixture
def naphthalene_scaffold(scaffold_library: ScaffoldLibrary) -> Scaffold:
    """Get naphthalene scaffold."""
    return scaffold_library.get_scaffold("naphthalene")


class TestHDTCPrimer:
    """Tests for HDTCPrimer."""

    @pytest.fixture
    def tokenizer(self) -> HDTCTokenizer:
        """Create HDTC tokenizer."""
        tokenizer = HDTCTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HDTCTokenizer) -> HDTCPrimer:
        """Create HDTC primer."""
        return HDTCPrimer(tokenizer)

    def test_create_primer_starts_with_sos(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should start with SOS token."""
        tokens = primer.create_primer(benzene_scaffold)
        assert tokens[0].item() == primer._tokenizer.SOS

    def test_create_primer_does_not_end_with_eos(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should NOT end with EOS token."""
        tokens = primer.create_primer(benzene_scaffold)
        assert tokens[-1].item() != primer._tokenizer.EOS

    def test_validate_primer(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Valid primers should pass validation."""
        tokens = primer.create_primer(benzene_scaffold)
        assert primer.validate_primer(tokens)

    def test_validate_primer_empty(self, primer: HDTCPrimer) -> None:
        """Empty primer should fail validation."""
        empty = torch.tensor([], dtype=torch.long)
        assert not primer.validate_primer(empty)

    def test_validate_primer_with_eos(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer ending with EOS should fail validation."""
        tokens = primer.create_primer(benzene_scaffold)
        tokens_with_eos = torch.cat(
            [
                tokens,
                torch.tensor([primer._tokenizer.EOS], dtype=torch.long),
            ]
        )
        assert not primer.validate_primer(tokens_with_eos)

    def test_primer_length_reasonable(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should have reasonable length."""
        tokens = primer.create_primer(benzene_scaffold)
        # At minimum: SOS + community structure
        assert len(tokens) > 1
        # Should not be excessively long for small molecule
        assert len(tokens) < 100

    def test_get_special_tokens(self, primer: HDTCPrimer) -> None:
        """Primer should return special token mappings."""
        tokens = primer.get_special_tokens()
        assert "sos" in tokens
        assert "eos" in tokens
        assert "pad" in tokens
        assert "comm_start" in tokens


class TestHDTPrimer:
    """Tests for HDTPrimer."""

    @pytest.fixture
    def tokenizer(self) -> HDTTokenizer:
        """Create HDT tokenizer."""
        tokenizer = HDTTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HDTTokenizer) -> HDTPrimer:
        """Create HDT primer."""
        return HDTPrimer(tokenizer)

    def test_create_primer_starts_with_sos(
        self, primer: HDTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should start with SOS token."""
        tokens = primer.create_primer(benzene_scaffold)
        assert tokens[0].item() == primer._tokenizer.SOS

    def test_create_primer_does_not_end_with_eos(
        self, primer: HDTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should NOT end with EOS token."""
        tokens = primer.create_primer(benzene_scaffold)
        assert tokens[-1].item() != primer._tokenizer.EOS

    def test_validate_primer(
        self, primer: HDTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Valid primers should pass validation."""
        tokens = primer.create_primer(benzene_scaffold)
        assert primer.validate_primer(tokens)

    def test_validate_primer_empty(self, primer: HDTPrimer) -> None:
        """Empty primer should fail validation."""
        empty = torch.tensor([], dtype=torch.long)
        assert not primer.validate_primer(empty)

    def test_get_special_tokens(self, primer: HDTPrimer) -> None:
        """Primer should return special token mappings."""
        tokens = primer.get_special_tokens()
        assert "sos" in tokens
        assert "eos" in tokens
        assert "enter" in tokens
        assert "exit" in tokens


class TestHSENTPrimer:
    """Tests for HSENTPrimer."""

    @pytest.fixture
    def tokenizer(self) -> HSENTTokenizer:
        """Create HSENT tokenizer."""
        tokenizer = HSENTTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HSENTTokenizer) -> HSENTPrimer:
        """Create HSENT primer."""
        return HSENTPrimer(tokenizer)

    def test_create_primer_starts_with_sos(
        self, primer: HSENTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should start with SOS token."""
        tokens = primer.create_primer(benzene_scaffold)
        assert tokens[0].item() == primer._tokenizer.SOS

    def test_create_primer_does_not_end_with_eos(
        self, primer: HSENTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Primer should NOT end with EOS token."""
        tokens = primer.create_primer(benzene_scaffold)
        assert tokens[-1].item() != primer._tokenizer.EOS

    def test_validate_primer(
        self, primer: HSENTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Valid primers should pass validation."""
        tokens = primer.create_primer(benzene_scaffold)
        assert primer.validate_primer(tokens)

    def test_validate_primer_empty(self, primer: HSENTPrimer) -> None:
        """Empty primer should fail validation."""
        empty = torch.tensor([], dtype=torch.long)
        assert not primer.validate_primer(empty)

    def test_get_special_tokens(self, primer: HSENTPrimer) -> None:
        """Primer should return special token mappings."""
        tokens = primer.get_special_tokens()
        assert "sos" in tokens
        assert "eos" in tokens
        assert "lcom" in tokens
        assert "rcom" in tokens


class TestPrimerFactory:
    """Tests for PrimerFactory."""

    def test_create_hdtc_primer(self) -> None:
        """Factory should create HDTCPrimer for HDTCTokenizer."""
        tokenizer = HDTCTokenizer()
        tokenizer.set_num_nodes(50)
        primer = PrimerFactory.create(tokenizer)
        assert isinstance(primer, HDTCPrimer)

    def test_create_hdt_primer(self) -> None:
        """Factory should create HDTPrimer for HDTTokenizer."""
        tokenizer = HDTTokenizer()
        tokenizer.set_num_nodes(50)
        primer = PrimerFactory.create(tokenizer)
        assert isinstance(primer, HDTPrimer)

    def test_create_hsent_primer(self) -> None:
        """Factory should create HSENTPrimer for HSENTTokenizer."""
        tokenizer = HSENTTokenizer()
        tokenizer.set_num_nodes(50)
        primer = PrimerFactory.create(tokenizer)
        assert isinstance(primer, HSENTPrimer)

    def test_get_supported_types(self) -> None:
        """Factory should return supported types."""
        types = PrimerFactory.get_supported_types()
        assert "hdtc" in types
        assert "hdt" in types
        assert "hsent" in types


class TestBatchPrimers:
    """Tests for batch primer creation."""

    @pytest.fixture
    def tokenizer(self) -> HDTCTokenizer:
        """Create HDTC tokenizer."""
        tokenizer = HDTCTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HDTCTokenizer) -> HDTCPrimer:
        """Create HDTC primer."""
        return HDTCPrimer(tokenizer)

    def test_batch_primers(
        self, primer: HDTCPrimer, scaffold_library: ScaffoldLibrary
    ) -> None:
        """Batch primers should be padded to same length."""
        scaffolds = [
            scaffold_library.get_scaffold("benzene"),
            scaffold_library.get_scaffold("naphthalene"),
        ]
        batch = primer.batch_primers(scaffolds)

        assert batch.ndim == 2
        assert batch.size(0) == 2
        # Both should be same length (padded)
        assert batch.size(1) > 0

    def test_batch_primers_padding(
        self, primer: HDTCPrimer, scaffold_library: ScaffoldLibrary
    ) -> None:
        """Shorter primers should be padded."""
        scaffolds = [
            scaffold_library.get_scaffold("benzene"),
            scaffold_library.get_scaffold("naphthalene"),
        ]
        batch = primer.batch_primers(scaffolds)

        # Check that padding token exists in shorter sequence
        pad_token = primer._tokenizer.PAD
        # At least one row should have padding (unless both same length)
        individual_lengths = [len(primer.create_primer(s)) for s in scaffolds]
        if individual_lengths[0] != individual_lengths[1]:
            shorter_idx = 0 if individual_lengths[0] < individual_lengths[1] else 1
            assert pad_token in batch[shorter_idx].tolist()


class TestPrimerRoundTrip:
    """Tests for primer round-trip decoding."""

    @pytest.fixture
    def tokenizer(self) -> HDTCTokenizer:
        """Create HDTC tokenizer."""
        tokenizer = HDTCTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HDTCTokenizer) -> HDTCPrimer:
        """Create HDTC primer."""
        return HDTCPrimer(tokenizer)

    def test_decode_primer_preserves_structure(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Decoded primer should preserve scaffold structure."""
        tokens = primer.create_primer(benzene_scaffold)
        decoded = primer.decode_primer_to_graph(tokens)

        # Should have same number of atoms
        assert decoded.num_atoms == benzene_scaffold.num_atoms


class TestHDTCCutPoints:
    """Tests for HDTC find_valid_cut_points."""

    @pytest.fixture
    def tokenizer(self) -> HDTCTokenizer:
        """Create HDTC tokenizer."""
        tokenizer = HDTCTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HDTCTokenizer) -> HDTCPrimer:
        """Create HDTC primer."""
        return HDTCPrimer(tokenizer)

    def test_find_cut_points_returns_list(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """find_valid_cut_points should return a list of indices."""
        graph = benzene_scaffold.get_graph(labeled=True)
        tokens = primer._tokenizer.tokenize(graph)
        cut_points = primer.find_valid_cut_points(tokens)

        assert isinstance(cut_points, list)

    def test_cut_points_are_after_comm_end(
        self, primer: HDTCPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Cut points should be at COMM_END token positions."""
        graph = benzene_scaffold.get_graph(labeled=True)
        tokens = primer._tokenizer.tokenize(graph)
        cut_points = primer.find_valid_cut_points(tokens)

        # Each cut point should be at a COMM_END token
        for idx in cut_points:
            assert tokens[idx].item() == primer._tokenizer.COMM_END

    def test_create_primer_at_level(
        self, primer: HDTCPrimer, naphthalene_scaffold: Scaffold
    ) -> None:
        """create_primer_at_level should cut at valid boundaries."""
        # Naphthalene should have at least one community
        primer_tokens = primer.create_primer_at_level(
            naphthalene_scaffold, cut_level=-1
        )

        # Should start with SOS
        assert primer_tokens[0].item() == primer._tokenizer.SOS
        # Should not end with EOS
        assert primer_tokens[-1].item() != primer._tokenizer.EOS
        # Should be valid primer
        assert primer.validate_primer(primer_tokens)


class TestHDTCutPoints:
    """Tests for HDT find_valid_cut_points."""

    @pytest.fixture
    def tokenizer(self) -> HDTTokenizer:
        """Create HDT tokenizer."""
        tokenizer = HDTTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HDTTokenizer) -> HDTPrimer:
        """Create HDT primer."""
        return HDTPrimer(tokenizer)

    def test_find_cut_points_returns_list(
        self, primer: HDTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """find_valid_cut_points should return a list of indices."""
        graph = benzene_scaffold.get_graph(labeled=True)
        tokens = primer._tokenizer.tokenize(graph)
        cut_points = primer.find_valid_cut_points(tokens)

        assert isinstance(cut_points, list)

    def test_cut_points_are_after_exit_at_root(
        self, primer: HDTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Cut points should be at EXIT tokens when depth returns to 0."""
        graph = benzene_scaffold.get_graph(labeled=True)
        tokens = primer._tokenizer.tokenize(graph)
        cut_points = primer.find_valid_cut_points(tokens)

        # Each cut point should be at an EXIT token
        for idx in cut_points:
            assert tokens[idx].item() == primer._tokenizer.EXIT

    def test_create_primer_at_level(
        self, primer: HDTPrimer, naphthalene_scaffold: Scaffold
    ) -> None:
        """create_primer_at_level should cut at valid boundaries."""
        primer_tokens = primer.create_primer_at_level(
            naphthalene_scaffold, cut_level=-1
        )

        # Should start with SOS
        assert primer_tokens[0].item() == primer._tokenizer.SOS
        # Should not end with EOS
        assert primer_tokens[-1].item() != primer._tokenizer.EOS
        # Should be valid primer
        assert primer.validate_primer(primer_tokens)


class TestHSENTCutPoints:
    """Tests for HSENT find_valid_cut_points."""

    @pytest.fixture
    def tokenizer(self) -> HSENTTokenizer:
        """Create HSENT tokenizer."""
        tokenizer = HSENTTokenizer()
        tokenizer.set_num_nodes(50)
        return tokenizer

    @pytest.fixture
    def primer(self, tokenizer: HSENTTokenizer) -> HSENTPrimer:
        """Create HSENT primer."""
        return HSENTPrimer(tokenizer)

    def test_find_cut_points_returns_list(
        self, primer: HSENTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """find_valid_cut_points should return a list of indices."""
        graph = benzene_scaffold.get_graph(labeled=True)
        tokens = primer._tokenizer.tokenize(graph)
        cut_points = primer.find_valid_cut_points(tokens)

        assert isinstance(cut_points, list)

    def test_cut_points_are_after_rcom(
        self, primer: HSENTPrimer, benzene_scaffold: Scaffold
    ) -> None:
        """Cut points should be at RCOM token positions."""
        graph = benzene_scaffold.get_graph(labeled=True)
        tokens = primer._tokenizer.tokenize(graph)
        cut_points = primer.find_valid_cut_points(tokens)

        # Each cut point should be at an RCOM token
        for idx in cut_points:
            assert tokens[idx].item() == primer._tokenizer.RCOM

    def test_create_primer_at_level(
        self, primer: HSENTPrimer, naphthalene_scaffold: Scaffold
    ) -> None:
        """create_primer_at_level should cut at valid boundaries."""
        primer_tokens = primer.create_primer_at_level(
            naphthalene_scaffold, cut_level=-1
        )

        # Should start with SOS
        assert primer_tokens[0].item() == primer._tokenizer.SOS
        # Should not end with EOS
        assert primer_tokens[-1].item() != primer._tokenizer.EOS
        # Should be valid primer
        assert primer.validate_primer(primer_tokens)
