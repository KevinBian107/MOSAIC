#!/usr/bin/env python
"""Prepare complex molecule data from COCONUT for priming evaluation.

This script downloads and filters complex natural products from the COCONUT
database (COlleCtion of Open Natural prodUcTs) for scaffold priming evaluation.

Downloads the CSV lite format (~191MB) and extracts SMILES.

Usage:
    # Download and filter COCONUT data (default)
    python scripts/preprocess/prepare_coconut_data.py

    # Custom output and settings
    python scripts/preprocess/prepare_coconut_data.py \
        --output data/coconut_complex.smi \
        --n-molecules 10000 \
        --min-atoms 25 \
        --min-rings 3
"""

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress RDKit warnings
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# COCONUT download URL (CSV lite format with SMILES)
COCONUT_URL = "https://coconut.s3.uni-jena.de/prod/downloads/2026-02/coconut_csv_lite-02-2026.zip"


def download_coconut(cache_dir: Path) -> Path:
    """Download COCONUT CSV and extract SMILES.

    Args:
        cache_dir: Directory to cache the downloaded file.

    Returns:
        Path to the extracted SMILES file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "coconut_smiles.smi"

    if cached_file.exists():
        print(f"Using cached COCONUT data: {cached_file}")
        return cached_file

    zip_file = cache_dir / "coconut_csv_lite.zip"
    expected_min_size = 180_000_000  # ~180MB minimum expected size

    # Download if zip doesn't exist or is incomplete
    if not zip_file.exists() or zip_file.stat().st_size < expected_min_size:
        if zip_file.exists():
            print(f"  Removing incomplete download ({zip_file.stat().st_size} bytes)")
            zip_file.unlink()

        print("Downloading COCONUT dataset...")

        try:
            # Get file size for progress bar
            with urllib.request.urlopen(COCONUT_URL) as response:
                total_size = int(response.headers.get("Content-Length", 0))

            # Download with progress bar
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="  Downloading",
            )

            def reporthook(block_num: int, block_size: int, total_size: int) -> None:
                if progress_bar.total != total_size and total_size > 0:
                    progress_bar.total = total_size
                progress_bar.update(block_size)

            urllib.request.urlretrieve(COCONUT_URL, zip_file, reporthook=reporthook)
            progress_bar.close()

            # Verify download
            if zip_file.stat().st_size < expected_min_size:
                raise RuntimeError(
                    f"Download incomplete: {zip_file.stat().st_size} bytes "
                    f"(expected ~195MB). Try: curl -L -o {zip_file} '{COCONUT_URL}'"
                )
            print(f"  Downloaded to {zip_file}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print(
                "  You can manually download with curl:\n"
                f"    curl -L -o {zip_file} '{COCONUT_URL}'"
            )
            raise

    # Extract SMILES from CSV
    print("  Extracting SMILES from CSV...")
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            # Find the CSV file in the zip
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV file found in zip archive")

            csv_name = csv_names[0]
            print(f"  Processing {csv_name}...")

            with zf.open(csv_name) as csv_file:
                # Read as text
                text_file = io.TextIOWrapper(csv_file, encoding="utf-8")
                reader = csv.DictReader(text_file)

                # Find SMILES column (could be 'smiles', 'SMILES', 'canonical_smiles', etc.)
                smiles_col = None
                for col in reader.fieldnames or []:
                    if "smiles" in col.lower():
                        smiles_col = col
                        break

                if smiles_col is None:
                    raise ValueError(
                        f"No SMILES column found. Available columns: {reader.fieldnames}"
                    )

                print(f"  Using column: {smiles_col}")

                # Extract SMILES
                smiles_list = []
                for row in reader:
                    smiles = row.get(smiles_col, "").strip()
                    if smiles:
                        smiles_list.append(smiles)

        # Write SMILES file
        print(f"  Writing {len(smiles_list)} SMILES to {cached_file}")
        with open(cached_file, "w") as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")

        print(f"  Extracted {len(smiles_list)} molecules")
        return cached_file

    except Exception as e:
        print(f"  Extraction failed: {e}")
        raise


def filter_complex_molecule(
    mol: Chem.Mol,
    min_atoms: int = 25,
    max_atoms: int = 100,
    min_rings: int = 3,
    min_scaffold_atoms: int = 12,
) -> bool:
    """Check if molecule meets complexity requirements."""
    if mol is None:
        return False

    num_atoms = mol.GetNumAtoms()
    if num_atoms < min_atoms or num_atoms > max_atoms:
        return False

    num_rings = rdMolDescriptors.CalcNumRings(mol)
    if num_rings < min_rings:
        return False

    # Check scaffold size
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() < min_scaffold_atoms:
            return False
    except Exception:
        return False

    return True


def extract_complex_molecules_from_coconut(
    coconut_file: Path,
    n_molecules: int = 10000,
    min_atoms: int = 25,
    max_atoms: int = 100,
    min_rings: int = 3,
    min_scaffold_atoms: int = 12,
    seed: int = 42,
) -> list[str]:
    """Extract complex molecules from COCONUT dataset."""
    import random

    print(f"Loading COCONUT data from {coconut_file}...")

    all_smiles = []
    with open(coconut_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("smiles"):
                continue
            # COCONUT format: SMILES\tID or just SMILES
            parts = line.split("\t")
            smiles = parts[0].split()[0]  # Handle space-separated too
            all_smiles.append(smiles)

    print(f"  Loaded {len(all_smiles)} SMILES from COCONUT")

    # Shuffle for random sampling
    random.seed(seed)
    random.shuffle(all_smiles)

    print("Filtering for complex molecules...")
    complex_smiles = []

    for i, smiles in enumerate(all_smiles):
        if i % 50000 == 0 and i > 0:
            print(
                f"  Processed {i}/{len(all_smiles)}, found {len(complex_smiles)} complex..."
            )

        mol = Chem.MolFromSmiles(smiles)
        if filter_complex_molecule(
            mol,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_rings=min_rings,
            min_scaffold_atoms=min_scaffold_atoms,
        ):
            # Canonicalize
            canonical = Chem.MolToSmiles(mol)
            complex_smiles.append(canonical)

            if len(complex_smiles) >= n_molecules:
                break

    print(f"  Found {len(complex_smiles)} complex molecules")
    return complex_smiles


def main():
    parser = argparse.ArgumentParser(
        description="Prepare complex molecule data from COCONUT for priming evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/coconut_complex.smi",
        help="Output SMILES file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory to cache downloaded COCONUT data",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=10000,
        help="Number of molecules to extract",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=25,
        help="Minimum number of atoms",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=100,
        help="Maximum number of atoms",
    )
    parser.add_argument(
        "--min-rings",
        type=int,
        default=3,
        help="Minimum number of rings",
    )
    parser.add_argument(
        "--min-scaffold-atoms",
        type=int,
        default=12,
        help="Minimum scaffold atoms",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    # Download COCONUT if needed
    coconut_file = download_coconut(cache_dir)

    # Extract complex molecules
    molecules = extract_complex_molecules_from_coconut(
        coconut_file,
        n_molecules=args.n_molecules,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        min_rings=args.min_rings,
        min_scaffold_atoms=args.min_scaffold_atoms,
        seed=args.seed,
    )

    if not molecules:
        print("ERROR: No molecules found meeting the criteria!")
        print("Try lowering --min-atoms or --min-rings")
        sys.exit(1)

    # Write to file
    print(f"\nWriting {len(molecules)} molecules to {output_path}...")
    with open(output_path, "w") as f:
        f.write("# Complex natural products from COCONUT for scaffold priming\n")
        f.write(f"# min_atoms={args.min_atoms}, min_rings={args.min_rings}\n")
        for smiles in molecules:
            f.write(f"{smiles}\n")

    print(f"Done! Created {output_path}")

    # Print statistics
    print("\nDataset statistics:")
    atom_counts = []
    ring_counts = []
    scaffold_counts = []
    for smiles in molecules[:100]:  # Sample first 100
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            atom_counts.append(mol.GetNumAtoms())
            ring_counts.append(rdMolDescriptors.CalcNumRings(mol))
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_counts.append(scaffold.GetNumAtoms())
            except Exception:
                pass

    if atom_counts:
        print(
            f"  Atoms: {min(atom_counts)}-{max(atom_counts)} "
            f"(mean: {sum(atom_counts)/len(atom_counts):.1f})"
        )
        print(
            f"  Rings: {min(ring_counts)}-{max(ring_counts)} "
            f"(mean: {sum(ring_counts)/len(ring_counts):.1f})"
        )
    if scaffold_counts:
        print(
            f"  Scaffold atoms: {min(scaffold_counts)}-{max(scaffold_counts)} "
            f"(mean: {sum(scaffold_counts)/len(scaffold_counts):.1f})"
        )


if __name__ == "__main__":
    main()
