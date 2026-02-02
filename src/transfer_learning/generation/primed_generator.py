"""Primed generation for scaffold-based molecule generation.

This module provides the PrimedGenerator class for generating molecules
from scaffold primes using trained models.
"""

from typing import TYPE_CHECKING, Optional, Union

from torch import Tensor
from torch_geometric.data import Data

from src.transfer_learning.primers.factory import PrimerFactory
from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary

if TYPE_CHECKING:
    from src.models.transformer import GraphGeneratorModule
    from src.transfer_learning.primers.base import TokenizerPrimer


class PrimedGenerator:
    """Generator for scaffold-primed molecule generation.

    This class wraps a trained GraphGeneratorModule and provides methods
    for generating molecules from scaffold primes.

    The generator works by:
    1. Converting scaffolds to primer tokens (using tokenizer-specific primers)
    2. Passing primer tokens as `input_ids` to model.generate()
    3. Returning completed molecular graphs

    Attributes:
        model: Trained GraphGeneratorModule.
        scaffold_library: Library of scaffold patterns.
        primer: Tokenizer-specific primer for creating token sequences.
    """

    def __init__(
        self,
        model: "GraphGeneratorModule",
        scaffold_library: Optional[ScaffoldLibrary] = None,
    ) -> None:
        """Initialize the primed generator.

        Args:
            model: Trained GraphGeneratorModule with tokenizer.
            scaffold_library: Optional scaffold library. If not provided,
                uses default library.
        """
        self.model = model
        self.scaffold_library = scaffold_library or ScaffoldLibrary()

        # Create tokenizer-specific primer
        self.primer: "TokenizerPrimer" = PrimerFactory.create(model.tokenizer)

    def generate_from_scaffold(
        self,
        scaffold: Union[str, Scaffold],
        num_samples: int = 1,
        priming_level: str = "scaffold_only",
        **kwargs,
    ) -> tuple[list[Data], float]:
        """Generate molecules from a scaffold.

        Args:
            scaffold: Scaffold name (string) or Scaffold object.
            num_samples: Number of molecules to generate.
            priming_level: Level of priming detail.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (list of generated Data objects, average time per sample).
        """
        # Resolve scaffold
        if isinstance(scaffold, str):
            scaffold_obj = self.scaffold_library.get_scaffold(scaffold)
        else:
            scaffold_obj = scaffold

        # Create primer tokens
        primer_tokens = self.primer.create_primer(scaffold_obj, priming_level)

        # Validate primer
        if not self.primer.validate_primer(primer_tokens):
            raise ValueError(
                f"Invalid primer created for scaffold: {scaffold_obj.name}"
            )

        # Expand primer for batch
        primer_batch = primer_tokens.unsqueeze(0).expand(num_samples, -1)

        # Generate using model
        graphs, avg_time = self.model.generate(
            input_ids=primer_batch,
            **kwargs,
        )

        return graphs, avg_time

    def generate_from_smiles(
        self,
        smiles: str,
        num_samples: int = 1,
        priming_level: str = "scaffold_only",
        **kwargs,
    ) -> tuple[list[Data], float]:
        """Generate molecules from a custom SMILES scaffold.

        Args:
            smiles: SMILES string representing the scaffold.
            num_samples: Number of molecules to generate.
            priming_level: Level of priming detail.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (list of generated Data objects, average time per sample).
        """
        # Create scaffold from SMILES
        scaffold = self.scaffold_library.from_smiles(smiles)

        return self.generate_from_scaffold(
            scaffold,
            num_samples=num_samples,
            priming_level=priming_level,
            **kwargs,
        )

    def generate_batch_diverse(
        self,
        scaffolds: list[Union[str, Scaffold]],
        samples_per_scaffold: int = 1,
        priming_level: str = "scaffold_only",
        **kwargs,
    ) -> tuple[list[list[Data]], float]:
        """Generate molecules from multiple scaffolds.

        Args:
            scaffolds: List of scaffold names or Scaffold objects.
            samples_per_scaffold: Number of samples per scaffold.
            priming_level: Level of priming detail.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (list of lists of generated Data objects, total time).
        """
        all_graphs: list[list[Data]] = []
        total_time = 0.0

        for scaffold in scaffolds:
            graphs, avg_time = self.generate_from_scaffold(
                scaffold,
                num_samples=samples_per_scaffold,
                priming_level=priming_level,
                **kwargs,
            )
            all_graphs.append(graphs)
            total_time += avg_time * samples_per_scaffold

        return all_graphs, total_time

    def generate_by_tier(
        self,
        tier: int,
        samples_per_scaffold: int = 1,
        max_scaffolds: Optional[int] = None,
        **kwargs,
    ) -> tuple[dict[str, list[Data]], float]:
        """Generate molecules using scaffolds from a specific tier.

        Args:
            tier: Complexity tier (1, 2, or 3).
            samples_per_scaffold: Number of samples per scaffold.
            max_scaffolds: Maximum number of scaffolds to use (None = all).
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (dict mapping scaffold names to generated graphs, total time).
        """
        scaffolds = self.scaffold_library.get_scaffolds_by_tier(tier)

        if max_scaffolds is not None:
            scaffolds = scaffolds[:max_scaffolds]

        results: dict[str, list[Data]] = {}
        total_time = 0.0

        for scaffold in scaffolds:
            graphs, avg_time = self.generate_from_scaffold(
                scaffold,
                num_samples=samples_per_scaffold,
                **kwargs,
            )
            results[scaffold.name] = graphs
            total_time += avg_time * samples_per_scaffold

        return results, total_time

    def get_primer_tokens(
        self,
        scaffold: Union[str, Scaffold],
        priming_level: str = "scaffold_only",
    ) -> Tensor:
        """Get primer tokens for a scaffold without generating.

        Useful for debugging and inspection.

        Args:
            scaffold: Scaffold name or Scaffold object.
            priming_level: Level of priming detail.

        Returns:
            1D tensor of primer tokens.
        """
        if isinstance(scaffold, str):
            scaffold_obj = self.scaffold_library.get_scaffold(scaffold)
        else:
            scaffold_obj = scaffold

        return self.primer.create_primer(scaffold_obj, priming_level)

    def primer_to_string(
        self,
        scaffold: Union[str, Scaffold],
        priming_level: str = "scaffold_only",
    ) -> str:
        """Get human-readable primer string for a scaffold.

        Args:
            scaffold: Scaffold name or Scaffold object.
            priming_level: Level of priming detail.

        Returns:
            Human-readable string representation of primer tokens.
        """
        primer_tokens = self.get_primer_tokens(scaffold, priming_level)

        if hasattr(self.model.tokenizer, "tokens_to_string"):
            return self.model.tokenizer.tokens_to_string(primer_tokens)

        return str(primer_tokens.tolist())

    def list_available_scaffolds(self) -> list[str]:
        """List all available scaffold names.

        Returns:
            List of scaffold names.
        """
        return self.scaffold_library.list_scaffolds()

    def get_scaffold_info(self, scaffold_name: str) -> dict:
        """Get information about a scaffold.

        Args:
            scaffold_name: Name of the scaffold.

        Returns:
            Dictionary with scaffold information.
        """
        scaffold = self.scaffold_library.get_scaffold(scaffold_name)
        return {
            "name": scaffold.name,
            "smiles": scaffold.smiles,
            "tier": scaffold.tier,
            "category": scaffold.category,
            "num_atoms": scaffold.num_atoms,
        }
