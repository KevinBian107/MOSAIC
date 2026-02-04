#!/usr/bin/env python
"""Compare fine-tuning results across different tokenizers.

This script reads evaluation results from fine-tuned model outputs and creates
a comparison table showing transfer learning performance across tokenizers.

Usage:
    # After running finetune_benchmarks.sh, compare the results:
    python scripts/compare_finetune_results.py

    # Compare from a specific directory:
    python scripts/compare_finetune_results.py --finetune-dir outputs/finetune

    # Filter by pattern:
    python scripts/compare_finetune_results.py --filter "hdtc"

    # Save as CSV:
    python scripts/compare_finetune_results.py --csv results.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


# Metrics to display in the table (organized by section)
METRIC_SECTIONS = [
    (
        "Training Info",
        [
            "tokenizer",
            "coarsening",
            "pretrained_from",
        ],
    ),
    (
        "Generation Quality",
        [
            "validity",
            "uniqueness",
            "novelty",
        ],
    ),
    (
        "Distribution Matching",
        [
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
        "Motif Metrics",
        [
            "motif_mmd",
            "motif_hist_mean",
        ],
    ),
    (
        "Performance",
        [
            "generation_time_per_sample",
        ],
    ),
]

# Metrics where lower is better
LOWER_IS_BETTER = {
    "fcd",
    "motif_mmd",
    "motif_hist_mean",
    "motif_hist_max",
    "generation_time_per_sample",
}

# Display names
METRIC_DISPLAY_NAMES = {
    "tokenizer": "Tokenizer",
    "coarsening": "Coarsening",
    "pretrained_from": "Pretrained From",
    "validity": "Validity",
    "uniqueness": "Uniqueness",
    "novelty": "Novelty",
    "fcd": "FCD",
    "snn": "SNN",
    "frag_similarity": "Frag Sim",
    "scaff_similarity": "Scaff Sim",
    "internal_diversity": "Int Div",
    "motif_mmd": "Motif MMD",
    "motif_hist_mean": "Hist KL",
    "motif_hist_max": "Hist Max KL",
    "generation_time_per_sample": "Gen Time (s)",
    "num_valid": "Valid Count",
    "num_generated": "Generated",
}

TOKENIZER_DISPLAY_NAMES = {
    "sent": "SENT",
    "hsent": "H-SENT",
    "hdt": "HDT",
    "hdtc": "HDTC",
}

COARSENING_DISPLAY_NAMES = {
    "N/A": "N/A",
    "spectral": "SC",
    "motif_community": "MC",
    "motif_aware_spectral": "MAS",
}


def extract_run_info(run_dir: Path) -> dict:
    """Extract tokenizer and coarsening info from run directory name.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dictionary with tokenizer and coarsening info.
    """
    dir_name = run_dir.name

    # Extract tokenizer type
    tokenizer = "unknown"
    if "_hsent_" in dir_name or dir_name.endswith("_hsent"):
        tokenizer = "hsent"
    elif "_hdtc_" in dir_name or dir_name.endswith("_hdtc") or "_hdtc" in dir_name:
        tokenizer = "hdtc"
    elif "_hdt_" in dir_name or dir_name.endswith("_hdt"):
        tokenizer = "hdt"
    elif "_sent_" in dir_name or dir_name.endswith("_sent"):
        tokenizer = "sent"

    # Extract coarsening strategy
    coarsening = "N/A"
    if tokenizer in ["hsent", "hdt"]:
        if "_mc_" in dir_name or "_mc" in dir_name:
            coarsening = "motif_community"
        elif "_sc_" in dir_name or "_sc" in dir_name:
            coarsening = "spectral"
        elif "_mas_" in dir_name:
            coarsening = "motif_aware_spectral"
        else:
            coarsening = "spectral"  # Default

    return {
        "tokenizer": tokenizer,
        "coarsening": coarsening,
        "dir_name": dir_name,
    }


def load_finetune_run(run_dir: Path) -> dict | None:
    """Load results from a fine-tuning run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dictionary with run data or None if loading fails.
    """
    # Check for evaluation results file
    results_path = run_dir / "evaluation_results.json"
    config_path = run_dir / "config.yaml"

    # If no evaluation_results.json, check subdirectories
    if not results_path.exists():
        for subdir in run_dir.iterdir():
            if subdir.is_dir():
                sub_results = subdir / "evaluation_results.json"
                if sub_results.exists():
                    results_path = sub_results
                    config_path = subdir / "config.yaml"
                    break

    if not results_path.exists():
        return None

    try:
        with open(results_path) as f:
            results = json.load(f)

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        # Extract run info
        run_info = extract_run_info(run_dir)

        # Try to get tokenizer from config if available
        if config:
            tok_config = config.get("tokenizer", {})
            if tok_config.get("type"):
                run_info["tokenizer"] = tok_config["type"]
            if tok_config.get("coarsening_strategy"):
                run_info["coarsening"] = tok_config["coarsening_strategy"]

        # Add run info to results
        results["tokenizer"] = run_info["tokenizer"]
        results["coarsening"] = run_info["coarsening"]

        # Extract pretrained source from checkpoint path
        if config:
            ckpt_path = config.get("model", {}).get("checkpoint_path", "")
            if ckpt_path:
                pretrained_dir = Path(ckpt_path).parent.name
                # Shorten it
                match = re.search(r"moses_(\w+)_", pretrained_dir)
                if match:
                    results["pretrained_from"] = f"MOSES-{match.group(1).upper()}"
                else:
                    results["pretrained_from"] = pretrained_dir[:20]
            else:
                results["pretrained_from"] = "-"
        else:
            results["pretrained_from"] = "-"

        return {
            "name": run_dir.name,
            "path": str(run_dir),
            "results": results,
            "config": config,
        }
    except Exception as e:
        print(f"Warning: Failed to load {run_dir.name}: {e}", file=sys.stderr)
        return None


def format_value(value, metric: str) -> str:
    """Format a metric value for display.

    Args:
        value: The metric value.
        metric: The metric name.

    Returns:
        Formatted string.
    """
    if value is None:
        return "-"
    if metric == "tokenizer":
        return TOKENIZER_DISPLAY_NAMES.get(value, str(value).upper())
    if metric == "coarsening":
        return COARSENING_DISPLAY_NAMES.get(value, str(value))
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def create_table_image(
    runs: list[dict],
    output_path: Path,
    title: str = "Fine-tuning Results Comparison",
) -> None:
    """Create a comparison table image.

    Args:
        runs: List of run data dictionaries.
        output_path: Path to save the image.
        title: Title for the table.
    """
    if not runs:
        print("No runs to display.")
        return

    # Collect all available metrics
    all_metrics = set()
    for run in runs:
        all_metrics.update(run["results"].keys())

    num_cols = len(runs)

    # Create column labels with tokenizer + coarsening
    col_labels = []
    for r in runs:
        tok = TOKENIZER_DISPLAY_NAMES.get(
            r["results"].get("tokenizer", "?"), r["results"].get("tokenizer", "?")
        )
        coarse = COARSENING_DISPLAY_NAMES.get(
            r["results"].get("coarsening", "N/A"), r["results"].get("coarsening", "-")
        )
        if coarse != "N/A":
            col_labels.append(f"{tok}\n({coarse})")
        else:
            col_labels.append(tok)

    # Build table data
    cell_data = []
    cell_colors = []
    row_labels = []
    section_rows = []

    row_idx = 0
    for section_name, section_metrics in METRIC_SECTIONS:
        # Filter to metrics with data
        available_section_metrics = [
            m for m in section_metrics if m in all_metrics or m in ["tokenizer", "coarsening", "pretrained_from"]
        ]
        if not available_section_metrics:
            continue

        # Section header
        row_labels.append(section_name)
        cell_data.append([""] * num_cols)
        cell_colors.append(["#D9E2F3"] * num_cols)
        section_rows.append(row_idx)
        row_idx += 1

        # Metric rows
        for metric in available_section_metrics:
            row_labels.append(METRIC_DISPLAY_NAMES.get(metric, metric))

            row_values = []
            raw_values = []

            for run in runs:
                value = run["results"].get(metric)
                row_values.append(format_value(value, metric))
                if isinstance(value, (int, float)):
                    raw_values.append(value)
                else:
                    raw_values.append(None)

            cell_data.append(row_values)

            # Determine colors for numeric metrics
            if metric in ["tokenizer", "coarsening", "pretrained_from"]:
                row_colors = ["white"] * num_cols
            else:
                valid_values = [v for v in raw_values if v is not None]
                if not valid_values:
                    row_colors = ["white"] * num_cols
                else:
                    if metric in LOWER_IS_BETTER:
                        sorted_vals = sorted(set(valid_values))
                    else:
                        sorted_vals = sorted(set(valid_values), reverse=True)

                    best_val = sorted_vals[0] if sorted_vals else None
                    second_val = sorted_vals[1] if len(sorted_vals) > 1 else None

                    best_count = sum(1 for v in raw_values if v == best_val)
                    is_tie = best_count > 1

                    row_colors = []
                    for val in raw_values:
                        if val is None:
                            row_colors.append("white")
                        elif val == best_val:
                            if is_tie:
                                row_colors.append("#FFFF99")  # Yellow
                            else:
                                row_colors.append("#90EE90")  # Green
                        elif val == second_val:
                            row_colors.append("#ADD8E6")  # Blue
                        else:
                            row_colors.append("white")

            cell_colors.append(row_colors)
            row_idx += 1

    # Create figure
    fig_width = max(8, 2 + num_cols * 1.8)
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

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style headers
    for j in range(num_cols):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold", color="white", fontsize=8)

    # Style row labels
    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        if i in section_rows:
            cell.set_text_props(weight="bold", style="italic")
            cell.set_facecolor("#D9E2F3")
        else:
            cell.set_text_props(weight="bold")

    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    footnote = "Green = best, Yellow = tie, Blue = second best. FCD/MMD/Gen Time: lower is better."
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Table saved to {output_path}")


def save_csv(runs: list[dict], output_path: Path) -> None:
    """Save results to CSV.

    Args:
        runs: List of run data.
        output_path: Path to save CSV.
    """
    import csv

    # Collect all metrics
    all_metrics = []
    for section_name, section_metrics in METRIC_SECTIONS:
        all_metrics.extend(section_metrics)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["Metric"] + [r["results"].get("tokenizer", "?") for r in runs]
        writer.writerow(header)

        for metric in all_metrics:
            row = [METRIC_DISPLAY_NAMES.get(metric, metric)]
            for run in runs:
                row.append(format_value(run["results"].get(metric), metric))
            writer.writerow(row)

    print(f"CSV saved to {output_path}")


def print_summary(runs: list[dict]) -> None:
    """Print a text summary table to console.

    Args:
        runs: List of run data.
    """
    print("\n" + "=" * 80)
    print("FINE-TUNING RESULTS COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'Metric':<20}"
    for run in runs:
        tok = run["results"].get("tokenizer", "?").upper()
        coarse = COARSENING_DISPLAY_NAMES.get(
            run["results"].get("coarsening", "N/A"), ""
        )
        if coarse and coarse != "N/A":
            label = f"{tok}-{coarse}"
        else:
            label = tok
        header += f"{label:>12}"
    print(header)
    print("-" * 80)

    # Key metrics
    key_metrics = [
        "validity",
        "uniqueness",
        "novelty",
        "fcd",
        "motif_mmd",
        "motif_hist_mean",
    ]

    for metric in key_metrics:
        row = f"{METRIC_DISPLAY_NAMES.get(metric, metric):<20}"
        for run in runs:
            value = run["results"].get(metric)
            row += f"{format_value(value, metric):>12}"
        print(row)

    print("=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare fine-tuning results across tokenizers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_finetune_results.py
  python scripts/compare_finetune_results.py --finetune-dir outputs/finetune
  python scripts/compare_finetune_results.py --filter "hdtc"
  python scripts/compare_finetune_results.py --csv results.csv
        """,
    )
    parser.add_argument(
        "--finetune-dir",
        type=Path,
        default=Path("outputs/finetune"),
        help="Directory containing fine-tuning outputs (default: outputs/finetune)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by name pattern",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/finetune/comparison.png"),
        help="Output path for table image",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also save to CSV file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Transfer Learning Results (COCONUT Fine-tuning)",
        help="Title for the table",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Don't generate image, only print summary",
    )

    args = parser.parse_args()

    if not args.finetune_dir.exists():
        print(f"Error: Directory not found: {args.finetune_dir}", file=sys.stderr)
        sys.exit(1)

    # Find run directories
    run_dirs = []
    for item in args.finetune_dir.iterdir():
        if item.is_dir():
            run_dirs.append(item)

    if args.filter:
        run_dirs = [d for d in run_dirs if args.filter in d.name]

    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    # Load runs
    runs = []
    for run_dir in sorted(run_dirs):
        data = load_finetune_run(run_dir)
        if data:
            runs.append(data)

    if not runs:
        print("No valid runs found (missing evaluation_results.json).", file=sys.stderr)
        print("\nTo generate evaluation results, run eval_finetune.py on each checkpoint:")
        print("  python scripts/eval_finetune.py model.checkpoint_path=outputs/finetune/XXX/best.ckpt")
        sys.exit(1)

    print(f"Found {len(runs)} fine-tuning runs with results")

    # Sort by tokenizer type
    tokenizer_order = ["sent", "hsent", "hdt", "hdtc"]
    runs.sort(
        key=lambda r: (
            tokenizer_order.index(r["results"].get("tokenizer", "?"))
            if r["results"].get("tokenizer", "?") in tokenizer_order
            else 999,
            r["results"].get("coarsening", ""),
        )
    )

    # Print summary
    print_summary(runs)

    # Create table image
    if not args.no_image:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        create_table_image(runs, args.output, title=args.title)

    # Save CSV
    if args.csv:
        save_csv(runs, args.csv)


if __name__ == "__main__":
    main()
