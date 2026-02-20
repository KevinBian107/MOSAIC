#!/usr/bin/env python
"""Generate the comparison table image from stored JSON data only.

Use this when you have already run eval_benchmarks_auto.sh (or compare_results.py)
and have a comparison_data.json file. This script only loads the JSON and draws
the table—no loading of results.json/config.yaml or checkpoint dirs.

Usage:
    python scripts/visualize_comparison.py outputs/eval_my_run/comparison.json
    python scripts/visualize_comparison.py outputs/eval_my_run/comparison.json -o comparison_updated.png
    python scripts/visualize_comparison.py outputs/eval_my_run/comparison.png   # infers .json same base path
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def render_table_from_data(table_data: dict, output_path: Path) -> None:
    """Render the comparison table image from a table_data dict.

    Expects table_data to have: title, col_labels, row_labels, cell_data,
    cell_colors, section_rows (as produced by compare_results.py and saved to JSON).
    """
    row_labels = table_data["row_labels"]
    col_labels = table_data["col_labels"]
    cell_data = table_data["cell_data"]
    cell_colors = table_data["cell_colors"]
    section_rows = set(table_data.get("section_rows", []))
    title = table_data.get("title", "Model Comparison Results")

    num_cols = len(col_labels)
    if not row_labels or not cell_data:
        print("Error: Table data has no rows.", file=sys.stderr)
        sys.exit(1)

    fig_width = max(8, 2 + num_cols * 1.5)
    fig_height = max(6, 1 + len(row_labels) * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

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

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(num_cols):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold", color="white")

    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        if i in section_rows:
            cell.set_text_props(weight="bold", style="italic")
            cell.set_facecolor("#D9E2F3")
        else:
            cell.set_text_props(weight="bold")

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    footnote = (
        "Green = best, Yellow = tie, Blue = second best. "
        "Train size/steps, PGD/FCD/MMD, Gen Time: lower is better."
    )
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Table image saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render comparison table from stored JSON (no eval or config loading).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to comparison JSON (e.g. comparison.json or comparison.png to use same-dir comparison.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: same as data path with .png extension)",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if data_path.suffix.lower() == ".png":
        # User passed comparison.png; look for comparison.json alongside
        data_path = data_path.with_suffix(".json")
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path) as f:
        table_data = json.load(f)

    for key in ("row_labels", "col_labels", "cell_data", "cell_colors"):
        if key not in table_data:
            print(f"Error: Missing key '{key}' in {data_path}", file=sys.stderr)
            sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = data_path.with_suffix(".png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_table_from_data(table_data, output_path)


if __name__ == "__main__":
    main()
