#!/usr/bin/env python
"""Compare test results across multiple runs.

This script reads results from test output directories and creates
a table image comparing metrics across tokenization schemes.
It also incorporates realistic generation metrics when available,
matching them to test runs by checkpoint path.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --filter "moses"
    python scripts/compare_results.py --output comparison.png
    python scripts/compare_results.py --all  # Show all runs, not just best per tokenizer
    python scripts/compare_results.py --test-only  # Exclude realistic gen metrics
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


# Default metrics to display (in order)
# Test metrics
DEFAULT_TEST_METRICS = [
    "validity",
    "uniqueness",
    "novelty",
    "pgd",
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

# Realistic generation metrics
DEFAULT_REALISTIC_METRICS = [
    "validity_rate",
    "motif_rate",
    "substitution_tv",
    "substitution_kl",
    "functional_group_tv",
    "functional_group_kl",
]

# Combined default metrics
DEFAULT_METRICS = DEFAULT_TEST_METRICS + DEFAULT_REALISTIC_METRICS

# Metrics where lower is better (for highlighting)
LOWER_IS_BETTER = {
    "pgd",
    "fcd",
    "motif_fg_mmd",
    "motif_smarts_mmd",
    "motif_ring_mmd",
    "motif_brics_mmd",
    "generation_time",
    # Realistic gen metrics (distribution distances - lower is better)
    "substitution_tv",
    "substitution_kl",
    "functional_group_tv",
    "functional_group_kl",
}

# Display names for metrics
METRIC_DISPLAY_NAMES = {
    # Test metrics
    "validity": "Validity",
    "uniqueness": "Uniqueness",
    "novelty": "Novelty",
    "pgd": "PGD",
    "fcd": "FCD",
    "snn": "SNN",
    "frag_similarity": "Frag Sim",
    "scaff_similarity": "Scaff Sim",
    "internal_diversity": "Int Div",
    "motif_fg_mmd": "FG MMD",
    "motif_smarts_mmd": "SMARTS MMD",
    "motif_ring_mmd": "Ring MMD",
    "motif_brics_mmd": "BRICS MMD",
    "generation_time": "Gen Time (s)",
    "num_valid_smiles": "Valid SMILES",
    # Realistic generation metrics
    "validity_rate": "Valid Rate (RG)",
    "motif_rate": "Motif Rate",
    "substitution_tv": "Subst TV",
    "substitution_kl": "Subst KL",
    "functional_group_tv": "FG TV",
    "functional_group_kl": "FG KL",
}

# Display names for tokenizers
TOKENIZER_DISPLAY_NAMES = {
    "sent": "SENT",
    "hsent": "H-SENT",
    "hdt": "HDT",
    "hdtc": "HDTC",
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


def load_realistic_gen_runs(realistic_gen_dir: Path) -> list[dict]:
    """Load all realistic generation runs.

    Args:
        realistic_gen_dir: Path to the realistic_gen directory.

    Returns:
        List of run data dictionaries.
    """
    if not realistic_gen_dir.exists():
        return []

    runs = []
    for run_dir in realistic_gen_dir.iterdir():
        if run_dir.is_dir():
            data = load_run_data(run_dir)
            if data:
                runs.append(data)

    return runs


def get_checkpoint_key(config: dict) -> str | None:
    """Extract a comparable checkpoint key from config.

    Args:
        config: Configuration dictionary.

    Returns:
        Normalized checkpoint path key or None.
    """
    checkpoint_path = config.get("model", {}).get("checkpoint_path")
    if checkpoint_path:
        # Normalize to just the checkpoint directory name
        # e.g., "outputs/train/moses_hdtc_n100000_20260126-204311/best.ckpt"
        # becomes "moses_hdtc_n100000_20260126-204311"
        return Path(checkpoint_path).parent.name
    return None


def match_realistic_gen_to_test(
    test_runs: list[dict],
    realistic_runs: list[dict],
) -> dict[str, dict]:
    """Match realistic generation runs to test runs by checkpoint.

    Args:
        test_runs: List of test run data dictionaries.
        realistic_runs: List of realistic gen run data dictionaries.

    Returns:
        Dictionary mapping test run names to their matched realistic gen data.
    """
    # Build index of realistic runs by checkpoint key
    realistic_by_checkpoint: dict[str, list[dict]] = {}
    for run in realistic_runs:
        checkpoint_key = get_checkpoint_key(run.get("config", {}))
        if checkpoint_key:
            if checkpoint_key not in realistic_by_checkpoint:
                realistic_by_checkpoint[checkpoint_key] = []
            realistic_by_checkpoint[checkpoint_key].append(run)

    # Match test runs to realistic gen runs
    matches = {}
    for test_run in test_runs:
        checkpoint_key = get_checkpoint_key(test_run.get("config", {}))
        if checkpoint_key and checkpoint_key in realistic_by_checkpoint:
            # Get the most recent realistic gen run for this checkpoint
            matched_runs = realistic_by_checkpoint[checkpoint_key]
            # Sort by directory modification time (newest first)
            matched_runs.sort(
                key=lambda r: Path(r["path"]).stat().st_mtime,
                reverse=True,
            )
            matches[test_run["name"]] = matched_runs[0]

    return matches


def merge_realistic_gen_results(
    test_runs: list[dict],
    realistic_matches: dict[str, dict],
) -> list[dict]:
    """Merge realistic generation results into test run data.

    Args:
        test_runs: List of test run data dictionaries.
        realistic_matches: Dictionary mapping test run names to realistic gen data.

    Returns:
        List of test runs with merged realistic gen results.
    """
    merged_runs = []
    for test_run in test_runs:
        run_copy = {
            "name": test_run["name"],
            "path": test_run["path"],
            "config": test_run["config"],
            "results": dict(test_run["results"]),  # Copy results
        }

        # Merge realistic gen results if available
        realistic_run = realistic_matches.get(test_run["name"])
        if realistic_run:
            realistic_results = realistic_run.get("results", {})
            for metric in DEFAULT_REALISTIC_METRICS:
                if metric in realistic_results:
                    run_copy["results"][metric] = realistic_results[metric]
            run_copy["realistic_gen_path"] = realistic_run["path"]

        merged_runs.append(run_copy)

    return merged_runs


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

    tokenizer_type = tokenizer.get("type", "?").lower()

    return {
        "tokenizer": tokenizer_type,
        "tokenizer_display": TOKENIZER_DISPLAY_NAMES.get(
            tokenizer_type, tokenizer_type.upper()
        ),
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
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def select_best_per_tokenizer(runs: list[dict]) -> list[dict]:
    """Select the best run for each tokenizer type based on PGD score.

    Args:
        runs: List of run data dictionaries.

    Returns:
        List with only the best run per tokenizer (lowest PGD).
    """
    tokenizer_runs: dict[str, list[dict]] = {}

    for run in runs:
        tokenizer_type = get_run_info(run)["tokenizer"]
        if tokenizer_type not in tokenizer_runs:
            tokenizer_runs[tokenizer_type] = []
        tokenizer_runs[tokenizer_type].append(run)

    best_runs = []
    for tokenizer_type, type_runs in tokenizer_runs.items():
        # Sort by PGD (lower is better), treating None as infinity
        def get_pgd(r: dict) -> float:
            pgd = r["results"].get("pgd")
            return pgd if pgd is not None else float("inf")

        sorted_runs = sorted(type_runs, key=get_pgd)
        best_runs.append(sorted_runs[0])

    # Sort by tokenizer type for consistent ordering
    tokenizer_order = ["sent", "hsent", "hdt", "hdtc"]
    best_runs.sort(
        key=lambda r: (
            tokenizer_order.index(get_run_info(r)["tokenizer"])
            if get_run_info(r)["tokenizer"] in tokenizer_order
            else 999
        )
    )

    return best_runs


def create_table_image(
    runs: list[dict],
    metrics: list[str],
    output_path: Path,
    title: str = "Model Comparison Results",
) -> None:
    """Create a table image comparing runs.

    Args:
        runs: List of run data dictionaries.
        metrics: List of metric names to display.
        output_path: Path to save the image.
        title: Title for the table.
    """
    if not runs:
        print("No runs to display.")
        return

    # Prepare data for table
    col_labels = [get_run_info(r)["tokenizer_display"] for r in runs]
    row_labels = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]

    # Build cell data and track best values per metric
    cell_data = []
    cell_colors = []

    for metric in metrics:
        row_values = []
        raw_values = []

        for run in runs:
            value = run["results"].get(metric)
            row_values.append(format_value(value, metric))
            raw_values.append(
                value
                if value is not None
                else float("inf")
                if metric in LOWER_IS_BETTER
                else float("-inf")
            )

        cell_data.append(row_values)

        # Determine best value for highlighting
        row_colors = []
        if metric in LOWER_IS_BETTER:
            best_val = min(v for v in raw_values if v != float("inf"))
        else:
            best_val = max(v for v in raw_values if v != float("-inf"))

        for val in raw_values:
            if val == best_val and val not in (float("inf"), float("-inf")):
                row_colors.append("#90EE90")  # Light green for best
            else:
                row_colors.append("white")

        cell_colors.append(row_colors)

    # Create figure
    fig_width = max(8, 2 + len(runs) * 1.5)
    fig_height = max(6, 1 + len(metrics) * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * len(row_labels),
        colColours=["#4472C4"] * len(col_labels),
        cellLoc="center",
        loc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header cells
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        cell.set_text_props(weight="bold")

    # Add title
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    # Add footnote
    footnote = (
        "Best values per metric highlighted in green. PGD/FCD/MMD: lower is better."
    )
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=8, style="italic")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Table image saved to {output_path}")


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
        header = ["metric"] + [get_run_info(r)["tokenizer_display"] for r in runs]
        writer.writerow(header)

        # Metrics
        for metric in metrics:
            row = [METRIC_DISPLAY_NAMES.get(metric, metric)] + [
                format_value(r["results"].get(metric), metric) for r in runs
            ]
            writer.writerow(row)

    print(f"CSV saved to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare test results across tokenization schemes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_results.py
  python scripts/compare_results.py --filter "moses"
  python scripts/compare_results.py --output comparison.png
  python scripts/compare_results.py --all  # Show all runs, not just best
  python scripts/compare_results.py --csv results.csv
  python scripts/compare_results.py --test-only  # Exclude realistic gen metrics
        """,
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/test"),
        help="Directory containing test outputs (default: outputs/test)",
    )
    parser.add_argument(
        "--realistic-gen-dir",
        type=Path,
        default=Path("outputs/realistic_gen"),
        help="Directory containing realistic gen outputs (default: outputs/realistic_gen)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only show test metrics, exclude realistic generation metrics",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by name pattern (e.g., 'moses', 'qm9')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/test/comparison.png"),
        help="Output path for table image (default: outputs/test/comparison.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also save results to CSV file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all runs instead of best per tokenizer",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics to display",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Model Comparison Results (Best per Tokenizer by PGD)",
        help="Title for the table",
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

    print(f"Found {len(runs)} total test runs")

    # Load and merge realistic generation results unless --test-only
    if not args.test_only:
        realistic_runs = load_realistic_gen_runs(args.realistic_gen_dir)
        if realistic_runs:
            print(f"Found {len(realistic_runs)} realistic generation runs")
            realistic_matches = match_realistic_gen_to_test(runs, realistic_runs)
            print(
                f"Matched {len(realistic_matches)} test runs with realistic gen results"
            )
            runs = merge_realistic_gen_results(runs, realistic_matches)

            # Print matching details
            for test_name, realistic_data in realistic_matches.items():
                print(f"  {test_name} -> {Path(realistic_data['path']).name}")
        else:
            print("No realistic generation runs found")

    # Select best per tokenizer unless --all is specified
    if not args.all:
        runs = select_best_per_tokenizer(runs)
        print(f"Selected {len(runs)} best runs (one per tokenizer by PGD)")
        title = args.title
    else:
        title = args.title.replace("(Best per Tokenizer by PGD)", "(All Runs)")

    # Determine metrics to show
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]
    elif args.test_only:
        metrics = DEFAULT_TEST_METRICS
    else:
        metrics = DEFAULT_METRICS

    # Filter metrics to only those with data
    available_metrics = []
    for metric in metrics:
        has_data = any(run["results"].get(metric) is not None for run in runs)
        if has_data:
            available_metrics.append(metric)

    if not available_metrics:
        print("No metrics with data found.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create table image
    create_table_image(runs, available_metrics, args.output, title=title)

    # Save CSV if requested
    if args.csv:
        save_csv(runs, available_metrics, args.csv)


if __name__ == "__main__":
    main()
