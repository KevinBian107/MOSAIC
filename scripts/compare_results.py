#!/usr/bin/env python
"""Compare test results across multiple runs.

This script reads results from test output directories and displays
them side-by-side in a table for easy comparison.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --filter "moses_hdt"
    python scripts/compare_results.py --filter "hsent" --metrics validity,uniqueness,fcd
    python scripts/compare_results.py --latest 5
    python scripts/compare_results.py --output results_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


# Default metrics to display (in order)
DEFAULT_METRICS = [
    "validity",
    "uniqueness",
    "novelty",
    "fcd",
    "snn",
    "frag_similarity",
    "scaff_similarity",
    "internal_diversity",
    "motif_fg_mmd",
    "motif_smarts_mmd",
    "motif_ring_mmd",
    "motif_brics_mmd",
    "generation_time",
]

# Metrics where lower is better (for highlighting)
LOWER_IS_BETTER = {
    "fcd",
    "motif_fg_mmd",
    "motif_smarts_mmd",
    "motif_ring_mmd",
    "motif_brics_mmd",
    "generation_time",
}


def load_run_data(run_dir: Path) -> dict | None:
    """Load results and config from a test run directory.

    Args:
        run_dir: Path to the test run directory.

    Returns:
        Dictionary with run data or None if loading fails.
    """
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.yaml"

    if not results_path.exists():
        return None

    try:
        with open(results_path) as f:
            results = json.load(f)

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        return {
            "name": run_dir.name,
            "path": str(run_dir),
            "results": results,
            "config": config,
        }
    except Exception as e:
        print(f"Warning: Failed to load {run_dir.name}: {e}", file=sys.stderr)
        return None


def get_run_info(run_data: dict) -> dict:
    """Extract key info from run config for display.

    Args:
        run_data: Dictionary with run data.

    Returns:
        Dictionary with extracted info.
    """
    config = run_data.get("config", {})
    tokenizer = config.get("tokenizer", {})
    model = config.get("model", {})
    sampling = config.get("sampling", {})

    return {
        "tokenizer": tokenizer.get("type", "?"),
        "coarsening": tokenizer.get("coarsening_strategy", "-"),
        "num_samples": sampling.get("num_samples", "?"),
        "checkpoint": Path(model.get("checkpoint_path", "?")).name[:30]
        if model.get("checkpoint_path")
        else "?",
    }


def format_value(value: float | int | None, metric: str) -> str:
    """Format a metric value for display.

    Args:
        value: The metric value.
        metric: The metric name.

    Returns:
        Formatted string.
    """
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def print_table(
    runs: list[dict],
    metrics: list[str],
    show_info: bool = True,
) -> None:
    """Print comparison table to stdout.

    Args:
        runs: List of run data dictionaries.
        metrics: List of metric names to display.
        show_info: Whether to show run info (tokenizer, etc.).
    """
    if not runs:
        print("No runs found.")
        return

    # Prepare column headers (run names, truncated)
    max_name_len = 35
    headers = ["Metric"] + [
        r["name"][:max_name_len] + ("..." if len(r["name"]) > max_name_len else "")
        for r in runs
    ]

    # Calculate column widths
    col_widths = [max(15, len(h)) for h in headers]
    col_widths[0] = max(20, len(headers[0]))  # Metric column wider

    # Print header separator
    def print_row(values: list[str], widths: list[int]) -> None:
        row = " | ".join(v.ljust(w) for v, w in zip(values, widths))
        print(row)

    def print_separator(widths: list[int]) -> None:
        print("-+-".join("-" * w for w in widths))

    # Print run info section
    if show_info:
        print("\n" + "=" * 80)
        print("RUN CONFIGURATION")
        print("=" * 80)

        info_rows = [
            ("Tokenizer", [get_run_info(r)["tokenizer"] for r in runs]),
            ("Coarsening", [get_run_info(r)["coarsening"] for r in runs]),
            ("Num Samples", [str(get_run_info(r)["num_samples"]) for r in runs]),
        ]

        print_row(headers, col_widths)
        print_separator(col_widths)
        for label, values in info_rows:
            print_row([label] + values, col_widths)

    # Print metrics section
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)

    print_row(headers, col_widths)
    print_separator(col_widths)

    for metric in metrics:
        values = []
        for run in runs:
            value = run["results"].get(metric)
            values.append(format_value(value, metric))

        # Find best value for highlighting (optional: could add ANSI colors)
        print_row([metric] + values, col_widths)

    # Print valid/total samples
    print_separator(col_widths)
    valid_row = ["num_valid_smiles"] + [
        str(r["results"].get("num_valid_smiles", "?")) for r in runs
    ]
    print_row(valid_row, col_widths)


def save_csv(
    runs: list[dict],
    metrics: list[str],
    output_path: Path,
) -> None:
    """Save comparison to CSV file.

    Args:
        runs: List of run data dictionaries.
        metrics: List of metric names to include.
        output_path: Path to save CSV.
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["metric"] + [r["name"] for r in runs]
        writer.writerow(header)

        # Info rows
        writer.writerow(["tokenizer"] + [get_run_info(r)["tokenizer"] for r in runs])
        writer.writerow(
            ["coarsening"] + [get_run_info(r)["coarsening"] for r in runs]
        )
        writer.writerow(
            ["num_samples"] + [str(get_run_info(r)["num_samples"]) for r in runs]
        )
        writer.writerow([])  # Empty row

        # Metrics
        for metric in metrics:
            row = [metric] + [
                format_value(r["results"].get(metric), metric) for r in runs
            ]
            writer.writerow(row)

    print(f"Results saved to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare test results across multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_results.py
  python scripts/compare_results.py --filter "hdt"
  python scripts/compare_results.py --latest 3
  python scripts/compare_results.py --metrics validity,fcd,snn
  python scripts/compare_results.py --output comparison.csv
        """,
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/test"),
        help="Directory containing test outputs (default: outputs/test)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by name pattern (e.g., 'moses_hdt', 'hsent')",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Only show the N most recent runs",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics to display",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Show all available metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Don't show run configuration info",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="Specific run directory names to compare",
    )

    args = parser.parse_args()

    # Find test run directories
    test_dir = args.test_dir
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)

    # Get all run directories
    if args.runs:
        # Specific runs requested
        run_dirs = [test_dir / name for name in args.runs]
        run_dirs = [d for d in run_dirs if d.exists()]
    else:
        run_dirs = sorted(
            [d for d in test_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
        )

    # Apply filter
    if args.filter:
        run_dirs = [d for d in run_dirs if args.filter in d.name]

    # Apply latest limit
    if args.latest:
        run_dirs = run_dirs[-args.latest :]

    if not run_dirs:
        print("No matching runs found.", file=sys.stderr)
        sys.exit(1)

    # Load run data
    runs = []
    for run_dir in run_dirs:
        data = load_run_data(run_dir)
        if data:
            runs.append(data)

    if not runs:
        print("No valid runs found (missing results.json).", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} runs to compare")

    # Determine metrics to show
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]
    elif args.all_metrics:
        # Collect all metrics from all runs
        all_metrics = set()
        for run in runs:
            for key, value in run["results"].items():
                if isinstance(value, (int, float)) and key != "num_samples":
                    all_metrics.add(key)
        metrics = sorted(all_metrics)
    else:
        metrics = DEFAULT_METRICS

    # Print table
    print_table(runs, metrics, show_info=not args.no_info)

    # Save to CSV if requested
    if args.output:
        save_csv(runs, metrics, args.output)


if __name__ == "__main__":
    main()
