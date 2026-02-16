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
    python scripts/compare_results.py --all  # Show all runs, not just best per tokenizer+coarsening
    python scripts/compare_results.py --test-only  # Exclude realistic gen metrics
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


# Default metrics to display (in order), organized by section
# Each section is a tuple of (section_name, list_of_metrics)
# Use None as section_name for metrics without a section header
METRIC_SECTIONS = [
    (
        "Training Info",
        [
            "train_data_size",
            "train_max_steps",
            "coarsening_strategy",
            "generation_time",
        ],
    ),
    (
        "Core Generation Quality",
        [
            "validity",
            "uniqueness",
            "novelty",
        ],
    ),
    (
        "Distribution Matching",
        [
            "pgd",
            "fcd",
            "snn",
        ],
    ),
    (
        "Structural Similarity",
        [
            "frag_similarity",
            "scaff_similarity",
            "internal_diversity",
        ],
    ),
    (
        "Motif MMD",
        [
            "motif_fg_mmd",
            "motif_smarts_mmd",
            "motif_ring_mmd",
            "motif_brics_mmd",
        ],
    ),
    (
        "Realistic Generation",
        [
            "motif_rate",
            "substitution_tv",
            "substitution_kl",
            "functional_group_tv",
            "functional_group_kl",
        ],
    ),
    (
        "Test Loss",
        [
            "test_loss",
            "perplexity",
        ],
    ),
]

# Flat list of all metrics (for backward compatibility)
TRAINING_INFO_METRICS = ["train_data_size", "train_max_steps", "coarsening_strategy"]

# Categorical metrics that should not have best/second-best highlighting
CATEGORICAL_METRICS = {"coarsening_strategy"}
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
DEFAULT_REALISTIC_METRICS = [
    "motif_rate",
    "substitution_tv",
    "substitution_kl",
    "functional_group_tv",
    "functional_group_kl",
]
DEFAULT_TEST_LOSS_METRICS = [
    "test_loss",
    "perplexity",
]

# Combined default metrics
DEFAULT_METRICS = (
    TRAINING_INFO_METRICS
    + DEFAULT_TEST_METRICS
    + DEFAULT_REALISTIC_METRICS
    + DEFAULT_TEST_LOSS_METRICS
)

# Metrics where lower is better (for highlighting)
LOWER_IS_BETTER = {
    # Training info (smaller/more efficient is better)
    "train_data_size",
    "train_max_steps",
    # Test metrics
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
    # Test loss metrics (lower is better)
    "test_loss",
    "perplexity",
}

# Display names for metrics
METRIC_DISPLAY_NAMES = {
    # Training info
    "train_data_size": "Train Data Size",
    "train_max_steps": "Train Steps",
    "coarsening_strategy": "Coarsening",
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
    "motif_brics_mmd": "BRICS L2",
    "generation_time": "Gen Time (s)",
    "num_valid_smiles": "Valid SMILES",
    # Realistic generation metrics
    "motif_rate": "Motif Rate",
    "substitution_tv": "Subst TV",
    "substitution_kl": "Subst KL",
    "functional_group_tv": "FG TV",
    "functional_group_kl": "FG KL",
    # Test loss metrics
    "test_loss": "Test Loss",
    "perplexity": "Perplexity",
}

# Display names for tokenizers
TOKENIZER_DISPLAY_NAMES = {
    "sent": "SENT",
    "hsent": "H-SENT (ours)",
    "hdt": "HDT (ours)",
    "hdtc": "HDTC (ours)",
}

# Display names for coarsening strategies
COARSENING_DISPLAY_NAMES = {
    "N/A": "N/A",
    "motif": "Motif",
    "spectral": "Spectral",
    "motif_aware_spectral": "MAS",
    "motif_community": "MC",
    "motif_fg_community": "MFC",
    "functional_group": "FG",
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

        # Extract training info and add to results
        training_info = extract_training_info(config)
        results.update(training_info)

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


def load_training_config(config: dict) -> dict | None:
    """Load training config from the checkpoint's training directory.

    Args:
        config: Test run configuration dictionary.

    Returns:
        Training configuration dictionary or None if not found.
    """
    checkpoint_path = config.get("model", {}).get("checkpoint_path")
    if not checkpoint_path:
        return None

    # Get the training directory from checkpoint path
    train_dir = Path(checkpoint_path).parent
    train_config_path = train_dir / "config.yaml"

    if not train_config_path.exists():
        return None

    try:
        with open(train_config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def get_checkpoint_global_step(checkpoint_path: str | None) -> int | None:
    """Extract global_step from a PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The global step count or None if not available.
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    try:
        import torch

        # Load only the metadata, not the full state dict
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ckpt.get("global_step")
    except Exception:
        return None


def extract_training_info(config: dict) -> dict:
    """Extract training info (data size, actual steps, coarsening) from config.

    Extracts the actual global_step from the checkpoint file to show how many
    steps the model was trained for. Falls back to max_steps from config.
    For data size, attempts to load from training config or parse directory name.
    For coarsening strategy, extracts from tokenizer config (N/A for SENT).

    Args:
        config: Test run configuration dictionary.

    Returns:
        Dictionary with train_data_size, train_max_steps, and coarsening_strategy.
    """
    info = {
        "train_data_size": None,
        "train_max_steps": None,
        "coarsening_strategy": None,
    }

    # Extract coarsening strategy from tokenizer config
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type", "").lower()
    if tokenizer_type == "sent":
        # SENT tokenizer doesn't use coarsening
        info["coarsening_strategy"] = "N/A"
    elif tokenizer_type == "hdtc":
        # HDTC uses fixed FunctionalHierarchyBuilder (motif + functional group communities)
        info["coarsening_strategy"] = "motif_fg_community"
    else:
        # For hierarchical tokenizers (hsent, hdt), get the strategy from config
        coarsening = tokenizer_config.get("coarsening_strategy")
        if coarsening:
            info["coarsening_strategy"] = coarsening

    checkpoint_path = config.get("model", {}).get("checkpoint_path")
    dir_name = Path(checkpoint_path).parent.name if checkpoint_path else None

    # Get actual training steps from checkpoint
    actual_steps = get_checkpoint_global_step(checkpoint_path)
    if actual_steps is not None:
        info["train_max_steps"] = actual_steps

    # Try to load training config for data size (and fallback steps)
    train_config = load_training_config(config)
    if train_config:
        num_train = train_config.get("data", {}).get("num_train")

        # Use config max_steps as fallback if checkpoint didn't have it
        if info["train_max_steps"] is None:
            info["train_max_steps"] = train_config.get("trainer", {}).get("max_steps")

        # If num_train is -1 (full dataset), try to parse from directory name
        if num_train is not None and num_train != -1:
            info["train_data_size"] = num_train
        elif dir_name:
            # Fallback: parse from directory name
            match = re.search(r"_n(\d+)_", dir_name)
            if match:
                info["train_data_size"] = int(match.group(1))

        return info

    # Fallback: parse checkpoint directory name
    # e.g., "moses_hdtc_n100000_20260126-204311" -> n=100000
    if dir_name:
        # Look for pattern like _n100000_ or _n1000000_
        match = re.search(r"_n(\d+)_", dir_name)
        if match:
            info["train_data_size"] = int(match.group(1))

    return info


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
    # Use display names for coarsening strategy
    if metric == "coarsening_strategy" and isinstance(value, str):
        return COARSENING_DISPLAY_NAMES.get(value, value)
    if isinstance(value, int):
        # Use thousands separator for large integers (training info metrics)
        if metric in TRAINING_INFO_METRICS:
            return f"{value:,}"
        return str(value)
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def select_best_per_tokenizer_and_coarsening(runs: list[dict]) -> list[dict]:
    """Select the best run for each tokenizer + coarsening strategy combination.

    Groups runs by (tokenizer_type, coarsening_strategy) and selects the best
    run from each group based on PGD score (lower is better).

    Args:
        runs: List of run data dictionaries.

    Returns:
        List with only the best run per (tokenizer, coarsening) combination.
    """
    # Group runs by (tokenizer, coarsening_strategy)
    grouped_runs: dict[tuple[str, str], list[dict]] = {}

    for run in runs:
        tokenizer_type = get_run_info(run)["tokenizer"]
        coarsening = run["results"].get("coarsening_strategy", "N/A") or "N/A"
        key = (tokenizer_type, coarsening)
        if key not in grouped_runs:
            grouped_runs[key] = []
        grouped_runs[key].append(run)

    best_runs = []
    for (tokenizer_type, coarsening), group_runs in grouped_runs.items():
        # Sort by PGD (lower is better), treating None as infinity
        def get_pgd(r: dict) -> float:
            pgd = r["results"].get("pgd")
            return pgd if pgd is not None else float("inf")

        sorted_runs = sorted(group_runs, key=get_pgd)
        best_runs.append(sorted_runs[0])

    # Sort by tokenizer type, then coarsening strategy for consistent ordering
    tokenizer_order = ["sent", "hsent", "hdt", "hdtc"]
    coarsening_order = [
        "N/A",
        "motif",
        "spectral",
        "functional_group",
        "motif_fg_community",
    ]

    def sort_key(r: dict) -> tuple[int, int]:
        tokenizer = get_run_info(r)["tokenizer"]
        coarsening = r["results"].get("coarsening_strategy", "N/A") or "N/A"
        tok_idx = (
            tokenizer_order.index(tokenizer) if tokenizer in tokenizer_order else 999
        )
        coarse_idx = (
            coarsening_order.index(coarsening)
            if coarsening in coarsening_order
            else 999
        )
        return (tok_idx, coarse_idx)

    best_runs.sort(key=sort_key)

    return best_runs


def create_table_image(
    runs: list[dict],
    metrics: list[str],
    output_path: Path,
    title: str = "Model Comparison Results",
    col_labels: list[str] | None = None,
) -> None:
    """Create a table image comparing runs with section separators.

    Args:
        runs: List of run data dictionaries.
        metrics: List of metric names to display.
        output_path: Path to save the image.
        title: Title for the table.
        col_labels: Optional list of column labels (one per run). If None, uses tokenizer_display.
    """
    if not runs:
        print("No runs to display.")
        return

    num_cols = len(runs)
    if col_labels is not None and len(col_labels) == num_cols:
        col_labels = list(col_labels)
    else:
        col_labels = [get_run_info(r)["tokenizer_display"] for r in runs]

    # Build table data with section headers
    cell_data = []
    cell_colors = []
    row_labels = []
    section_rows = []  # Track which rows are section headers

    # Build a set of available metrics for filtering
    available_metrics_set = set(metrics)

    row_idx = 0
    for section_name, section_metrics in METRIC_SECTIONS:
        # Filter to only metrics that are in the available list
        section_metrics_filtered = [
            m for m in section_metrics if m in available_metrics_set
        ]
        if not section_metrics_filtered:
            continue

        # Add section header row
        row_labels.append(section_name)
        cell_data.append([""] * num_cols)
        cell_colors.append(["#D9E2F3"] * num_cols)  # Light blue for section headers
        section_rows.append(row_idx)
        row_idx += 1

        # Add metric rows for this section
        for metric in section_metrics_filtered:
            row_labels.append(METRIC_DISPLAY_NAMES.get(metric, metric))

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

            # Determine gradient colors based on value ranking
            row_colors = []

            # Skip highlighting for categorical metrics
            if metric in CATEGORICAL_METRICS:
                row_colors = ["white"] * len(raw_values)
                cell_colors.append(row_colors)
                row_idx += 1
                continue

            valid_values = [
                v for v in raw_values if v not in (float("inf"), float("-inf"))
            ]

            if not valid_values:
                row_colors = ["white"] * len(raw_values)
            else:
                # Sort values to find best and second best
                if metric in LOWER_IS_BETTER:
                    sorted_vals = sorted(set(valid_values))
                else:
                    sorted_vals = sorted(set(valid_values), reverse=True)

                best_val = sorted_vals[0] if len(sorted_vals) >= 1 else None
                second_val = sorted_vals[1] if len(sorted_vals) >= 2 else None

                # Count how many have the best value (for tie detection)
                best_count = sum(1 for v in raw_values if v == best_val)
                is_tie = best_count > 1

                for val in raw_values:
                    if val in (float("inf"), float("-inf")):
                        row_colors.append("white")
                    elif val == best_val:
                        if is_tie:
                            row_colors.append("#FFFF99")  # Yellow for ties
                        else:
                            row_colors.append("#90EE90")  # Light green for best
                    elif val == second_val:
                        row_colors.append("#ADD8E6")  # Light blue for second
                    else:
                        row_colors.append("white")

            cell_colors.append(row_colors)
            row_idx += 1

    # Create figure
    fig_width = max(8, 2 + num_cols * 1.5)
    fig_height = max(6, 1 + len(row_labels) * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * len(row_labels),
        colColours=["#4472C4"] * num_cols,
        cellLoc="center",
        loc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header cells
    for j in range(num_cols):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold", color="white")

    # Style row labels and section headers
    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        if i in section_rows:
            # Section header styling
            cell.set_text_props(weight="bold", style="italic")
            cell.set_facecolor("#D9E2F3")
        else:
            cell.set_text_props(weight="bold")

    # Add title
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    # Add footnote
    footnote = (
        "Green = best, Yellow = tie, Blue = second best. "
        "Train size/steps, PGD/FCD/MMD, Gen Time: lower is better."
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
    col_labels: list[str] | None = None,
) -> None:
    """Save comparison to CSV file.

    Args:
        runs: List of run data dictionaries.
        metrics: List of metric names to include.
        output_path: Path to save CSV.
        col_labels: Optional list of column labels (one per run). If None, uses tokenizer_display.
    """
    import csv

    if col_labels is not None and len(col_labels) == len(runs):
        headers = col_labels
    else:
        headers = [get_run_info(r)["tokenizer_display"] for r in runs]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["metric"] + headers
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
        "--test-loss-dir",
        type=Path,
        default=Path("outputs/test_loss"),
        help="Directory containing test loss outputs (default: outputs/test_loss)",
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
        help="Show all runs instead of best per tokenizer+coarsening",
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
        default="Model Comparison Results (Best per Tokenizer+Coarsening by PGD)",
        help="Title for the table",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="Specific run directory names to compare",
    )
    parser.add_argument(
        "--column-labels",
        type=str,
        default=None,
        help="Comma-separated column labels (one per run, same order as --runs)",
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

    # Load and merge test loss results
    if not args.test_only:
        test_loss_runs = load_realistic_gen_runs(args.test_loss_dir)
        if test_loss_runs:
            print(f"Found {len(test_loss_runs)} test loss runs")
            test_loss_matches = match_realistic_gen_to_test(runs, test_loss_runs)
            print(f"Matched {len(test_loss_matches)} test runs with test loss results")
            for test_run in runs:
                test_loss_run = test_loss_matches.get(test_run["name"])
                if test_loss_run:
                    test_loss_results = test_loss_run.get("results", {})
                    for metric in DEFAULT_TEST_LOSS_METRICS:
                        if metric in test_loss_results:
                            test_run["results"][metric] = test_loss_results[metric]
        else:
            print("No test loss runs found")

    # Select best per tokenizer+coarsening unless --all is specified
    if not args.all:
        runs = select_best_per_tokenizer_and_coarsening(runs)
        print(
            f"Selected {len(runs)} best runs "
            "(one per tokenizer+coarsening combination by PGD)"
        )
        title = args.title
    else:
        title = args.title.replace(
            "(Best per Tokenizer+Coarsening by PGD)", "(All Runs)"
        )

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

    # Parse optional column labels (comma-separated, same order as runs)
    col_labels = None
    if args.column_labels:
        col_labels = [s.strip() for s in args.column_labels.split(",")]

    # Create table image
    create_table_image(
        runs, available_metrics, args.output, title=title, col_labels=col_labels
    )

    # Save CSV if requested
    if args.csv:
        save_csv(runs, available_metrics, args.csv, col_labels=col_labels)


if __name__ == "__main__":
    main()
