#!/usr/bin/env python
"""Rename cache files to match new hash format.

This script renames existing cache files to use the new consistent hash format
that includes spectral parameters with the correct key names (n_init, k_min_factor,
k_max_factor instead of spectral_n_init, etc.).

Usage:
    python scripts/preprocess/rename_cache_files.py --cache-dir data/cache
    python scripts/preprocess/rename_cache_files.py --cache-dir data/cache --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datamodule import get_cache_filename

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def normalize_tokenizer_config(config: dict) -> dict:
    """Normalize tokenizer config to new format.

    Converts old format (spectral_n_init, etc.) to new format (n_init, etc.)
    and ensures all spectral parameters are included with defaults if missing.

    Args:
        config: Tokenizer config dict (may be in old or new format).

    Returns:
        Normalized config dict in new format.
    """
    normalized = config.copy()

    # Convert spectral_* keys to new format
    if "spectral_n_init" in normalized:
        normalized["n_init"] = normalized.pop("spectral_n_init")
    if "spectral_k_min_factor" in normalized:
        normalized["k_min_factor"] = normalized.pop("spectral_k_min_factor")
    if "spectral_k_max_factor" in normalized:
        normalized["k_max_factor"] = normalized.pop("spectral_k_max_factor")

    # Add default spectral parameters if missing (for spectral coarsening)
    if normalized.get("coarsening_strategy") == "spectral":
        if "n_init" not in normalized:
            normalized["n_init"] = 10
        if "k_min_factor" not in normalized:
            normalized["k_min_factor"] = 0.7
        if "k_max_factor" not in normalized:
            normalized["k_max_factor"] = 1.3

    return normalized


def get_new_cache_filename(
    old_cache_path: Path, cache_data: dict
) -> tuple[str, dict]:
    """Compute new cache filename based on cache data.

    Args:
        old_cache_path: Path to existing cache file.
        cache_data: Loaded cache data dict.

    Returns:
        Tuple of (new_filename, normalized_config).
    """
    # Extract info from old filename: {dataset}_{split}_{tokenizer}_{num_samples}_{hash}.pt
    parts = old_cache_path.stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected cache filename format: {old_cache_path.name}")

    # Reconstruct from cache data if available, otherwise parse filename
    dataset_name = cache_data.get("dataset_name", parts[0])
    split = cache_data.get("split", parts[1])
    tokenizer_type = cache_data.get("tokenizer_type", parts[2])
    num_samples = cache_data.get("num_samples", int(parts[3]))

    # Get tokenizer config and normalize it
    tokenizer_config = cache_data.get("tokenizer_config", {})
    normalized_config = normalize_tokenizer_config(tokenizer_config)

    # Compute new hash
    new_filename = get_cache_filename(
        dataset_name, split, tokenizer_type, num_samples, normalized_config
    )

    return new_filename, normalized_config


def main():
    parser = argparse.ArgumentParser(
        description="Rename cache files to match new hash format"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory containing cache files (default: data/cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="File pattern to match (default: *.pt)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        log.error(f"Cache directory does not exist: {cache_dir}")
        sys.exit(1)

    # Find all cache files
    cache_files = list(cache_dir.glob(args.pattern))
    if not cache_files:
        log.warning(f"No cache files found matching {args.pattern} in {cache_dir}")
        return

    log.info(f"Found {len(cache_files)} cache file(s)")

    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for cache_file in sorted(cache_files):
        try:
            # Load cache file to get config
            log.info(f"\nProcessing: {cache_file.name}")
            cache_data = torch.load(cache_file, weights_only=False)

            # Compute new filename
            new_filename, normalized_config = get_new_cache_filename(cache_file, cache_data)
            new_path = cache_dir / new_filename

            # Check if already correct
            if cache_file.name == new_filename:
                log.info(f"  ✓ Already correct: {new_filename}")
                skipped_count += 1
                continue

            # Check if target already exists
            if new_path.exists():
                log.warning(
                    f"  ⚠ Target already exists: {new_filename} (skipping rename)"
                )
                skipped_count += 1
                continue

            # Show what would change
            log.info(f"  Old: {cache_file.name}")
            log.info(f"  New: {new_filename}")

            # Show config changes if any
            old_config = cache_data.get("tokenizer_config", {})
            if old_config != normalized_config:
                log.info("  Config changes:")
                for key in set(list(old_config.keys()) + list(normalized_config.keys())):
                    old_val = old_config.get(key, "<missing>")
                    new_val = normalized_config.get(key, "<missing>")
                    if old_val != new_val:
                        log.info(f"    {key}: {old_val} -> {new_val}")

            if args.dry_run:
                log.info("  [DRY RUN] Would rename")
            else:
                cache_file.rename(new_path)
                log.info(f"  ✓ Renamed to {new_filename}")
                renamed_count += 1

        except Exception as e:
            log.error(f"  ✗ Error processing {cache_file.name}: {e}")
            error_count += 1

    log.info("\n" + "=" * 60)
    log.info("Summary:")
    log.info(f"  Renamed: {renamed_count}")
    log.info(f"  Skipped: {skipped_count}")
    log.info(f"  Errors: {error_count}")
    log.info("=" * 60)

    if args.dry_run:
        log.info("\nRun without --dry-run to actually rename files")


if __name__ == "__main__":
    main()
