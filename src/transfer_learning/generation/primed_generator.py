"""Primed generation for scaffold-based molecule generation.

This module provides the PrimedGenerator class for generating molecules
from scaffold primes using trained models.
"""

from typing import TYPE_CHECKING, Optional, Union

from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data

from src.transfer_learning.primers.factory import PrimerFactory
from src.transfer_learning.scaffolds.library import Scaffold, ScaffoldLibrary

if TYPE_CHECKING:
    from src.models.transformer import GraphGeneratorModule
    from src.transfer_learning.datasets.complex_molecule_dataset import (
        ComplexMoleculeSample,
    )
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
        min_new_tokens: Optional[int] = None,
        primer_fraction: Optional[float] = None,
        **kwargs,
    ) -> tuple[list[Data], float]:
        """Generate molecules from a scaffold.

        Args:
            scaffold: Scaffold name (string) or Scaffold object.
            num_samples: Number of molecules to generate.
            priming_level: Level of priming detail.
            min_new_tokens: Minimum number of new tokens to generate beyond primer.
                If set, EOS token is suppressed until this many tokens are generated.
                Useful to encourage longer molecule completions.
            primer_fraction: Fraction of scaffold structure to use as primer (0.0-1.0).
                - 1.0 (default/None): Use full scaffold (original behavior)
                - 0.5: Use first half of scaffold's structural units
                - 0.0: Use minimal primer (just SOS)
                Lower values give model more "room" to generate beyond scaffold.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (list of generated Data objects, average time per sample).
        """
        # Resolve scaffold
        if isinstance(scaffold, str):
            scaffold_obj = self.scaffold_library.get_scaffold(scaffold)
        else:
            scaffold_obj = scaffold

        # Create primer tokens based on primer_fraction
        if primer_fraction is not None and primer_fraction < 1.0:
            primer_tokens = self._create_fractional_primer(scaffold_obj, primer_fraction)
        else:
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
            min_new_tokens=min_new_tokens,
            **kwargs,
        )

        return graphs, avg_time

    def _create_fractional_primer(
        self,
        scaffold: Scaffold,
        fraction: float,
    ) -> Tensor:
        """Create a primer using only a fraction of the scaffold's structure.

        Args:
            scaffold: Scaffold object.
            fraction: Fraction of structure to include (0.0-1.0).

        Returns:
            Primer tokens tensor.
        """
        import torch

        # Get full tokenization to find cut points
        graph = scaffold.get_graph(labeled=True)
        if graph is None:
            # Fall back to just SOS
            return torch.tensor([self.primer.tokenizer.sos], dtype=torch.long)

        # Ensure max_num_nodes is set
        tokenizer = self.primer.tokenizer
        if tokenizer.max_num_nodes is None:
            tokenizer.set_num_nodes(graph.num_nodes)
        elif graph.num_nodes > tokenizer.max_num_nodes:
            tokenizer.set_num_nodes(graph.num_nodes)

        tokens = tokenizer.tokenize(graph)
        cut_points = self.primer.find_valid_cut_points(tokens)

        if not cut_points:
            # No valid cut points, return just SOS
            return torch.tensor([tokenizer.sos], dtype=torch.long)

        # Calculate which cut point to use based on fraction
        # fraction=0.0 -> cut_level=0 (first cut point, minimal primer)
        # fraction=1.0 -> cut_level=-1 (last cut point, full scaffold)
        n_cuts = len(cut_points)
        cut_level = int(fraction * (n_cuts - 1))
        cut_level = max(0, min(cut_level, n_cuts - 1))

        cut_idx = cut_points[cut_level]
        return tokens[: cut_idx + 1]

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

    def generate_from_complex_molecule(
        self,
        target_mol: Chem.Mol,
        num_samples: int = 10,
        cut_level: int = -1,
        **kwargs,
    ) -> tuple[list[Data], "ComplexMoleculeSample"]:
        """Generate completions for a complex molecule's scaffold.

        Extracts the Murcko scaffold from the target molecule and generates
        completions, returning both the generated graphs and the sample
        information for evaluation.

        Args:
            target_mol: RDKit Mol object of the target molecule.
            num_samples: Number of molecules to generate.
            cut_level: Cut level for primer (see create_primer_at_level).
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (list of generated Data objects, ComplexMoleculeSample).

        Raises:
            ValueError: If scaffold extraction fails.
        """
        from src.transfer_learning.datasets.complex_molecule_dataset import (
            ComplexMoleculeSample,
        )
        from src.transfer_learning.scaffolds.murcko_extractor import MurckoExtractor

        # Extract scaffold
        extractor = MurckoExtractor()
        scaffold_mol = extractor.extract_scaffold(target_mol)
        if scaffold_mol is None:
            raise ValueError("Failed to extract scaffold from molecule")

        mol_smiles = Chem.MolToSmiles(target_mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol)

        # Create scaffold object
        scaffold = Scaffold(
            name="complex_mol",
            smiles=scaffold_smiles,
            tier=0,
            category="coconut",
            num_atoms=scaffold_mol.GetNumAtoms(),
        )

        # Create sample for evaluation
        sample = ComplexMoleculeSample(
            molecule=target_mol,
            molecule_smiles=mol_smiles,
            scaffold=scaffold_mol,
            scaffold_smiles=scaffold_smiles,
        )

        # Generate using appropriate method based on cut_level
        if cut_level == -1:
            # Use regular primer (strips EOS at end)
            graphs, avg_time = self.generate_from_scaffold(
                scaffold,
                num_samples=num_samples,
                **kwargs,
            )
        else:
            # Use level-based primer
            primer_tokens = self.primer.create_primer_at_level(scaffold, cut_level)

            if not self.primer.validate_primer(primer_tokens):
                raise ValueError("Invalid primer created for scaffold")

            primer_batch = primer_tokens.unsqueeze(0).expand(num_samples, -1)

            graphs, avg_time = self.model.generate(
                input_ids=primer_batch,
                **kwargs,
            )

        return graphs, sample

    def generate_from_mol_smiles(
        self,
        mol_smiles: str,
        num_samples: int = 10,
        cut_level: int = -1,
        **kwargs,
    ) -> tuple[list[Data], "ComplexMoleculeSample"]:
        """Generate completions for a molecule given as SMILES.

        Convenience method that parses SMILES and calls
        generate_from_complex_molecule.

        Args:
            mol_smiles: SMILES string of the target molecule.
            num_samples: Number of molecules to generate.
            cut_level: Cut level for primer.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Tuple of (list of generated Data objects, ComplexMoleculeSample).

        Raises:
            ValueError: If SMILES is invalid or scaffold extraction fails.
        """
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_smiles}")

        return self.generate_from_complex_molecule(
            mol,
            num_samples=num_samples,
            cut_level=cut_level,
            **kwargs,
        )
