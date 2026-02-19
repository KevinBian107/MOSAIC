#!/usr/bin/env python
"""Export MOSES train and test SMILES to plain text files (one per line).

Run once to create data/moses_smiles/train_smiles.txt and test_smiles.txt.
Then set data.use_precomputed_smiles=true so the datamodule loads from these
files instead of re-reading CSV and converting to graphs every time.

Usage:
    python scripts/export_moses_smiles.py
    python scripts/export_moses_smiles.py --output-dir data/moses_smiles
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.molecular import load_moses_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MOSES SMILES to .txt files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/moses_smiles",
        help="Output directory for train_smiles.txt and test_smiles.txt",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MOSES train split (full)...")
    train_smiles = load_moses_dataset("train", max_molecules=None)
    print(f"  Loaded {len(train_smiles)} train SMILES")

    print("Loading MOSES test split (full)...")
    test_smiles = load_moses_dataset("test", max_molecules=None)
    print(f"  Loaded {len(test_smiles)} test SMILES")

    train_path = out_dir / "train_smiles.txt"
    test_path = out_dir / "test_smiles.txt"

    with open(train_path, "w") as f:
        for s in train_smiles:
            f.write(s.strip() + "\n")
    print(f"Wrote {train_path}")

    with open(test_path, "w") as f:
        for s in test_smiles:
            f.write(s.strip() + "\n")
    print(f"Wrote {test_path}")

    print("Done. Use data.use_precomputed_smiles=true and data.precomputed_smiles_dir=<path> to load from these files.")


if __name__ == "__main__":
    main()
