#!/usr/bin/env python
"""Export all MOSES SMILES to a single file (train then test, with a header for the split).

Run once to create data/moses_smiles/moses_smiles.txt. Then set
data.use_precomputed_smiles=true (and optionally data.precomputed_smiles_dir) so the
datamodule and precompute_benchmarks.sh chunks load from it instead of re-reading CSV.
See docs/commands_reference.md and docs/setup_training.md.

Usage:
    python scripts/preprocess/export_moses_smiles.py
    python scripts/preprocess/export_moses_smiles.py --output-dir data/moses_smiles
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.molecular import load_moses_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all MOSES SMILES to one .txt file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/moses_smiles",
        help="Output directory for moses_smiles.txt",
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

    out_path = out_dir / "moses_smiles.txt"
    with open(out_path, "w") as f:
        # First line: number of train SMILES (so loader knows where train ends and test begins)
        f.write(f"{len(train_smiles)}\n")
        for s in train_smiles:
            f.write(s.strip() + "\n")
        for s in test_smiles:
            f.write(s.strip() + "\n")
    print(f"Wrote {out_path} ({len(train_smiles) + len(test_smiles)} SMILES, train_count={len(train_smiles)})")

    print("Done. Use data.use_precomputed_smiles=true and data.precomputed_smiles_dir=<path> to load.")


if __name__ == "__main__":
    main()
