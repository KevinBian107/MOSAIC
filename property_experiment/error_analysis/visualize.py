"""Visualization for valence violation error analysis.

Creates a 3-panel figure showing validity rates, error location distributions,
and boundary error ratios across models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Color scheme: SC=red tones, HAC=orange tones, HDTC=blue
MODEL_COLORS = {
    "H-SENT SC": "#c0392b",
    "HDT SC": "#e74c3c",
    "H-SENT HAC": "#e67e22",
    "HDT HAC": "#f39c12",
    "HDTC": "#2980b9",
}

ROLE_COLORS = {
    "ring_interior": "#3498db",
    "ring_boundary": "#e74c3c",
    "chain_boundary": "#e67e22",
    "chain_interior": "#95a5a6",
}

ROLE_LABELS = {
    "ring_interior": "Ring Interior",
    "ring_boundary": "Ring Boundary",
    "chain_boundary": "Chain Boundary",
    "chain_interior": "Chain Interior",
}


def create_figure(all_results: dict[str, dict], output_dir: str) -> Path:
    """Create the 3-panel error analysis figure.

    Args:
        all_results: Dict mapping model name -> analyze_batch() output.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Valence Violation Analysis: Where Do Errors Occur?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # === Panel A: Validity Rate ===
    ax = axes[0]
    validity_rates = [all_results[m]["validity_rate"] for m in model_names]
    colors = [MODEL_COLORS.get(m, "#7f8c8d") for m in model_names]
    bars = ax.bar(range(n_models), validity_rates, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, rate) in enumerate(zip(bars, validity_rates)):
        n_valid = all_results[model_names[i]]["num_valid"]
        n_total = all_results[model_names[i]]["total"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}\n({n_valid}/{n_total})",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Validity Rate")
    ax.set_title("A. Validity Rate", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # === Panel B: Error Location Distribution (stacked bar) ===
    ax = axes[1]
    roles = ["ring_interior", "ring_boundary", "chain_boundary", "chain_interior"]

    # Filter to models with violations
    models_with_violations = [
        m for m in model_names if all_results[m]["total_violations"] > 0
    ]

    if models_with_violations:
        x_pos = range(len(models_with_violations))
        bottom = np.zeros(len(models_with_violations))

        for role in roles:
            fractions = []
            for m in models_with_violations:
                total_v = all_results[m]["total_violations"]
                count = all_results[m]["role_counts"][role]
                fractions.append(count / total_v if total_v > 0 else 0)

            bars = ax.bar(
                x_pos,
                fractions,
                bottom=bottom,
                color=ROLE_COLORS[role],
                label=ROLE_LABELS[role],
                edgecolor="white",
                linewidth=0.5,
            )

            # Annotate counts on segments that are large enough
            for i, (frac, b) in enumerate(zip(fractions, bottom)):
                if frac > 0.06:
                    m = models_with_violations[i]
                    count = all_results[m]["role_counts"][role]
                    ax.text(
                        i,
                        b + frac / 2,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        fontweight="bold",
                    )

            bottom += np.array(fractions)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models_with_violations, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

        # Add total violation counts above bars
        for i, m in enumerate(models_with_violations):
            total_v = all_results[m]["total_violations"]
            ax.text(
                i, 1.02, f"n={total_v}", ha="center", va="bottom", fontsize=7
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No violations found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of Violations")
    ax.set_title("B. Error Location Distribution", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # === Panel C: Boundary Error Ratio ===
    ax = axes[2]

    if models_with_violations:
        boundary_ratios = [
            all_results[m]["boundary_ratio"] for m in models_with_violations
        ]
        bar_colors = [
            MODEL_COLORS.get(m, "#7f8c8d") for m in models_with_violations
        ]
        bars = ax.bar(
            range(len(models_with_violations)),
            boundary_ratios,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.5,
        )

        for i, (bar, ratio) in enumerate(zip(bars, boundary_ratios)):
            m = models_with_violations[i]
            boundary_count = (
                all_results[m]["role_counts"]["ring_boundary"]
                + all_results[m]["role_counts"]["chain_boundary"]
            )
            total_v = all_results[m]["total_violations"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{ratio:.1%}\n({boundary_count}/{total_v})",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_xticks(range(len(models_with_violations)))
        ax.set_xticklabels(
            models_with_violations, rotation=30, ha="right", fontsize=8
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No violations found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Boundary Violation Ratio")
    ax.set_title(
        "C. Boundary Error Ratio\n(ring_boundary + chain_boundary) / total",
        fontsize=11,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = output_path / "valence_violation_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to {fig_path}")
    return fig_path
