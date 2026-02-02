"""Priming evaluator for scaffold-based generation.

This module provides the PrimingEvaluator class for evaluating scaffold
priming against ground truth complex molecules.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from torch_geometric.data import Data

from src.data.molecular import graph_to_smiles
from src.transfer_learning.datasets.complex_molecule_dataset import (
    ComplexMoleculeDataset,
    ComplexMoleculeSample,
)

if TYPE_CHECKING:
    from src.transfer_learning.generation.primed_generator import PrimedGenerator


class PrimingEvaluator:
    """Evaluate scaffold priming against ground truth.

    This evaluator compares generated molecules against target molecules
    to assess how well the model can complete scaffolds to match
    complex molecules.

    Attributes:
        fingerprint_radius: Morgan fingerprint radius.
        fingerprint_nbits: Number of bits in fingerprint.
    """

    def __init__(
        self,
        fingerprint_radius: int = 2,
        fingerprint_nbits: int = 2048,
    ) -> None:
        """Initialize the evaluator.

        Args:
            fingerprint_radius: Radius for Morgan fingerprint.
            fingerprint_nbits: Number of bits in Morgan fingerprint.
        """
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_nbits = fingerprint_nbits

    def evaluate_sample(
        self,
        target: ComplexMoleculeSample,
        generated_graphs: list[Data],
    ) -> dict:
        """Evaluate generated molecules against a target.

        Args:
            target: Target complex molecule sample.
            generated_graphs: List of generated PyG Data objects.

        Returns:
            Dictionary of evaluation metrics:
                - scaffold_preservation: Fraction containing the scaffold
                - tanimoto_similarities: List of Tanimoto similarities
                - tanimoto_mean: Mean Tanimoto similarity
                - tanimoto_max: Max Tanimoto similarity
                - atom_count_ratios: Ratio of generated/target atoms
                - valid_rate: Fraction of valid SMILES
                - n_generated: Number of graphs generated
                - n_valid: Number of valid SMILES
        """
        results = {
            "scaffold_preservation": [],
            "tanimoto_similarities": [],
            "atom_count_ratios": [],
            "valid_smiles": [],
            "n_generated": len(generated_graphs),
        }

        target_fp = self._get_fingerprint(target.molecule)
        target_atoms = target.molecule.GetNumAtoms()

        for graph in generated_graphs:
            smiles = graph_to_smiles(graph)
            if smiles is None:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            results["valid_smiles"].append(smiles)

            # Check scaffold preservation
            has_scaffold = self._contains_scaffold(mol, target.scaffold)
            results["scaffold_preservation"].append(has_scaffold)

            # Calculate Tanimoto similarity
            gen_fp = self._get_fingerprint(mol)
            if target_fp is not None and gen_fp is not None:
                similarity = DataStructs.TanimotoSimilarity(target_fp, gen_fp)
                results["tanimoto_similarities"].append(similarity)

            # Calculate atom count ratio
            ratio = mol.GetNumAtoms() / target_atoms
            results["atom_count_ratios"].append(ratio)

        # Compute summary statistics
        n_valid = len(results["valid_smiles"])
        results["n_valid"] = n_valid
        results["valid_rate"] = (
            n_valid / len(generated_graphs) if generated_graphs else 0
        )

        if results["scaffold_preservation"]:
            results["scaffold_preservation_rate"] = np.mean(
                results["scaffold_preservation"]
            )
        else:
            results["scaffold_preservation_rate"] = 0.0

        if results["tanimoto_similarities"]:
            results["tanimoto_mean"] = np.mean(results["tanimoto_similarities"])
            results["tanimoto_max"] = np.max(results["tanimoto_similarities"])
            results["tanimoto_std"] = np.std(results["tanimoto_similarities"])
        else:
            results["tanimoto_mean"] = 0.0
            results["tanimoto_max"] = 0.0
            results["tanimoto_std"] = 0.0

        if results["atom_count_ratios"]:
            results["atom_count_ratio_mean"] = np.mean(results["atom_count_ratios"])
        else:
            results["atom_count_ratio_mean"] = 0.0

        return results

    def evaluate_dataset(
        self,
        dataset: ComplexMoleculeDataset,
        generator: "PrimedGenerator",
        samples_per_scaffold: int = 10,
        max_molecules: Optional[int] = None,
        verbose: bool = True,
        return_per_sample: bool = False,
        min_new_tokens: Optional[int] = None,
        primer_fraction: Optional[float] = None,
    ) -> dict:
        """Full evaluation over a dataset.

        Args:
            dataset: ComplexMoleculeDataset to evaluate.
            generator: PrimedGenerator instance.
            samples_per_scaffold: Number of samples to generate per scaffold.
            max_molecules: Maximum molecules to evaluate (None = all).
            verbose: Whether to print progress.
            return_per_sample: If True, include per-sample results for visualization.
            min_new_tokens: Minimum number of new tokens to generate beyond primer.
                If set, encourages longer molecule generation.
            primer_fraction: Fraction of scaffold to use as primer (0.0-1.0).
                Lower values give the model more room to generate beyond the scaffold.

        Returns:
            Dictionary of aggregated evaluation metrics. If return_per_sample is True,
            also includes "per_sample_results" key with list of per-sample results.
        """
        from src.transfer_learning.scaffolds.library import Scaffold

        all_results = []
        samples = list(dataset)[:max_molecules] if max_molecules else list(dataset)

        for i, sample in enumerate(samples):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluating {i + 1}/{len(samples)}...")

            # Create scaffold object for the generator
            scaffold = Scaffold(
                name=f"sample_{i}",
                smiles=sample.scaffold_smiles,
                tier=0,
                category="coconut",
                num_atoms=sample.scaffold.GetNumAtoms(),
            )

            try:
                graphs, _ = generator.generate_from_scaffold(
                    scaffold,
                    num_samples=samples_per_scaffold,
                    min_new_tokens=min_new_tokens,
                    primer_fraction=primer_fraction,
                )
                results = self.evaluate_sample(sample, graphs)
                results["scaffold_smiles"] = sample.scaffold_smiles
                results["target_smiles"] = sample.molecule_smiles
                all_results.append(results)
            except Exception as e:
                if verbose:
                    print(f"  Error on sample {i}: {e}")
                continue

        if not all_results:
            return {"error": "No successful evaluations"}

        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        aggregated["n_molecules"] = len(all_results)
        aggregated["samples_per_scaffold"] = samples_per_scaffold

        # Include per-sample results for visualization if requested
        if return_per_sample:
            aggregated["per_sample_results"] = all_results

        return aggregated

    def _get_fingerprint(self, mol: Chem.Mol):
        """Get Morgan fingerprint for a molecule."""
        try:
            return AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.fingerprint_radius,
                nBits=self.fingerprint_nbits,
            )
        except Exception:
            return None

    def _contains_scaffold(self, mol: Chem.Mol, scaffold: Chem.Mol) -> bool:
        """Check if molecule contains the scaffold structure.

        Uses Maximum Common Substructure (MCS) to check if the scaffold
        is fully contained within the generated molecule.

        Args:
            mol: Generated molecule.
            scaffold: Target scaffold.

        Returns:
            True if scaffold is contained in mol.
        """
        try:
            # Try direct substructure match first (faster)
            if mol.HasSubstructMatch(scaffold):
                return True

            # Fall back to MCS for more robust matching
            mcs = rdFMCS.FindMCS(
                [mol, scaffold],
                timeout=1,
                matchValences=False,
                ringMatchesRingOnly=True,
            )
            if mcs.numAtoms >= scaffold.GetNumAtoms():
                return True

            return False
        except Exception:
            return False

    def _aggregate_results(self, results_list: list[dict]) -> dict:
        """Aggregate results across multiple samples.

        Args:
            results_list: List of per-sample result dictionaries.

        Returns:
            Aggregated statistics.
        """
        # Collect all values
        scaffold_rates = [r["scaffold_preservation_rate"] for r in results_list]
        tanimoto_means = [r["tanimoto_mean"] for r in results_list]
        tanimoto_maxes = [r["tanimoto_max"] for r in results_list]
        valid_rates = [r["valid_rate"] for r in results_list]
        atom_ratios = [r["atom_count_ratio_mean"] for r in results_list]

        return {
            "scaffold_preservation_mean": np.mean(scaffold_rates),
            "scaffold_preservation_std": np.std(scaffold_rates),
            "tanimoto_mean": np.mean(tanimoto_means),
            "tanimoto_std": np.std(tanimoto_means),
            "tanimoto_max_mean": np.mean(tanimoto_maxes),
            "tanimoto_max_max": np.max(tanimoto_maxes),
            "valid_rate_mean": np.mean(valid_rates),
            "valid_rate_std": np.std(valid_rates),
            "atom_ratio_mean": np.mean(atom_ratios),
            "atom_ratio_std": np.std(atom_ratios),
        }

    def compare_to_target(
        self,
        generated_smiles: str,
        target_smiles: str,
    ) -> dict:
        """Compare a single generated molecule to target.

        Convenience method for comparing individual molecules.

        Args:
            generated_smiles: SMILES of generated molecule.
            target_smiles: SMILES of target molecule.

        Returns:
            Dictionary with comparison metrics.
        """
        gen_mol = Chem.MolFromSmiles(generated_smiles)
        target_mol = Chem.MolFromSmiles(target_smiles)

        if gen_mol is None or target_mol is None:
            return {"valid": False}

        gen_fp = self._get_fingerprint(gen_mol)
        target_fp = self._get_fingerprint(target_mol)

        similarity = 0.0
        if gen_fp is not None and target_fp is not None:
            similarity = DataStructs.TanimotoSimilarity(target_fp, gen_fp)

        return {
            "valid": True,
            "tanimoto_similarity": similarity,
            "gen_atoms": gen_mol.GetNumAtoms(),
            "target_atoms": target_mol.GetNumAtoms(),
            "atom_ratio": gen_mol.GetNumAtoms() / target_mol.GetNumAtoms(),
            "exact_match": generated_smiles == target_smiles,
        }
