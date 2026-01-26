"""Analysis tools for realistic generation evaluation.

This module provides functions to analyze benzene substitution patterns
and functional group co-occurrence, comparing generated molecules to
training data distributions.

A good generative model should produce molecules with the same structural
characteristics as the training data.
"""

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw


# Common functional groups attached to benzene (as SMARTS patterns)
# The [#6] at the end matches the benzene carbon attachment point
FUNCTIONAL_GROUPS = {
    "Hydroxyl (-OH)": "[OX2H1][c]",
    "Amino (-NH2)": "[NX3H2][c]",
    "Carboxyl (-COOH)": "[CX3](=O)[OX2H1][c]",
    "Methyl (-CH3)": "[CH3][c]",
    "Methoxy (-OCH3)": "[OX2][CH3].[OX2][c]",
    "Fluoro (-F)": "[F][c]",
    "Chloro (-Cl)": "[Cl][c]",
    "Bromo (-Br)": "[Br][c]",
    "Nitro (-NO2)": "[NX3+](=O)[O-].[NX3+][c]",
    "Cyano (-CN)": "[CX2]#[NX1].[CX2][c]",
    "Aldehyde (-CHO)": "[CX3H1](=O)[c]",
    "Ketone (C=O)": "[CX3](=O)[#6].[CX3][c]",
    "Ester (-COOR)": "[CX3](=O)[OX2][#6].[CX3][c]",
    "Amide (-CONH)": "[CX3](=O)[NX3].[CX3][c]",
    "Sulfonyl (-SO2)": "[SX4](=O)(=O)[c]",
}

# Simpler patterns that directly match substituents on aromatic carbons
SUBSTITUENT_SMARTS = {
    "Hydroxyl (-OH)": "cO",
    "Amino (-NH2)": "cN",
    "Methyl (-CH3)": "c[CH3]",
    "Halogen (-X)": "[c;$([cF]),$([cCl]),$([cBr]),$([cI])]",
    "Fluoro (-F)": "cF",
    "Chloro (-Cl)": "cCl",
    "Bromo (-Br)": "cBr",
    "Carbonyl (C=O)": "cC=O",
    "Carboxyl (-COOH)": "cC(=O)O",
    "Nitro (-NO2)": "c[N+](=O)[O-]",
    "Cyano (-CN)": "cC#N",
    "Ether (-OR)": "cOC",
    "Alkyl chain": "c[CH2]",
}


def get_benzene_substitution_count(mol: Chem.Mol) -> int:
    """Count the number of substituents on benzene ring(s).

    Args:
        mol: RDKit molecule object.

    Returns:
        Number of non-hydrogen substituents on benzene carbons.
    """
    if mol is None:
        return 0

    # Find benzene ring pattern
    benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
    matches = mol.GetSubstructMatches(benzene_pattern)

    if not matches:
        return 0

    # For the first benzene ring found, count substituents
    benzene_atoms = set(matches[0])
    substituent_count = 0

    for atom_idx in benzene_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in benzene_atoms:
                # This neighbor is a substituent (not part of benzene ring)
                substituent_count += 1

    return substituent_count


def classify_disubstitution(mol: Chem.Mol) -> Optional[str]:
    """Classify di-substituted benzene as ortho, meta, or para.

    Args:
        mol: RDKit molecule with exactly 2 substituents on benzene.

    Returns:
        "ortho", "meta", "para", or None if not applicable.
    """
    if mol is None:
        return None

    benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
    matches = mol.GetSubstructMatches(benzene_pattern)

    if not matches:
        return None

    benzene_atoms = list(matches[0])
    substituent_positions = []

    for i, atom_idx in enumerate(benzene_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in benzene_atoms:
                substituent_positions.append(i)
                break

    if len(substituent_positions) != 2:
        return None

    # Calculate distance around the ring (6-membered)
    pos1, pos2 = substituent_positions
    distance = min(abs(pos1 - pos2), 6 - abs(pos1 - pos2))

    if distance == 1:
        return "ortho"
    elif distance == 2:
        return "meta"
    elif distance == 3:
        return "para"

    return None


def analyze_benzene_substitution(
    smiles_list: list[str],
) -> dict[str, Counter]:
    """Analyze benzene substitution patterns in a list of molecules.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Dictionary with:
            - "substitution_count": Counter of mono/di/tri/poly substitution
            - "disubstitution_pattern": Counter of ortho/meta/para
    """
    substitution_counts = Counter()
    disubstitution_patterns = Counter()

    for smiles in smiles_list:
        if smiles is None:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Check if molecule contains benzene
        benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
        if not mol.HasSubstructMatch(benzene_pattern):
            continue

        count = get_benzene_substitution_count(mol)

        if count == 1:
            substitution_counts["mono"] += 1
        elif count == 2:
            substitution_counts["di"] += 1
            pattern = classify_disubstitution(mol)
            if pattern:
                disubstitution_patterns[pattern] += 1
        elif count == 3:
            substitution_counts["tri"] += 1
        elif count >= 4:
            substitution_counts["poly"] += 1
        else:
            substitution_counts["unsubstituted"] += 1

    return {
        "substitution_count": substitution_counts,
        "disubstitution_pattern": disubstitution_patterns,
    }


def analyze_functional_groups(
    smiles_list: list[str],
) -> Counter:
    """Analyze functional groups attached to benzene rings.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Counter of functional group occurrences.
    """
    group_counts = Counter()

    for smiles in smiles_list:
        if smiles is None:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Check if molecule contains benzene
        benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
        if not mol.HasSubstructMatch(benzene_pattern):
            continue

        # Check for each functional group
        for group_name, smarts in SUBSTITUENT_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                group_counts[group_name] += 1

    return group_counts


def compare_distributions(
    train_counts: Counter,
    gen_counts: Counter,
) -> dict[str, float]:
    """Compare two distributions and compute similarity metrics.

    Args:
        train_counts: Counter from training data.
        gen_counts: Counter from generated data.

    Returns:
        Dictionary with comparison metrics.
    """
    # Get all keys
    all_keys = set(train_counts.keys()) | set(gen_counts.keys())

    if not all_keys:
        return {"kl_divergence": 0.0, "total_variation": 0.0}

    # Normalize to distributions
    train_total = sum(train_counts.values()) or 1
    gen_total = sum(gen_counts.values()) or 1

    train_dist = {k: train_counts.get(k, 0) / train_total for k in all_keys}
    gen_dist = {k: gen_counts.get(k, 0) / gen_total for k in all_keys}

    # KL divergence (with smoothing to avoid log(0))
    epsilon = 1e-10
    kl_div = 0.0
    for k in all_keys:
        p = train_dist[k] + epsilon
        q = gen_dist[k] + epsilon
        kl_div += p * np.log(p / q)

    # Total variation distance
    tv_dist = 0.5 * sum(abs(train_dist[k] - gen_dist[k]) for k in all_keys)

    return {
        "kl_divergence": kl_div,
        "total_variation": tv_dist,
    }


def plot_substitution_comparison(
    train_results: dict,
    gen_results: dict,
    output_path: Optional[str] = None,
    title: str = "Benzene Substitution Patterns",
) -> plt.Figure:
    """Plot comparison of substitution patterns.

    Args:
        train_results: Results from analyze_benzene_substitution on training data.
        gen_results: Results from analyze_benzene_substitution on generated data.
        output_path: Optional path to save the figure.
        title: Plot title.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Substitution count distribution
    ax1 = axes[0]
    categories = ["unsubstituted", "mono", "di", "tri", "poly"]

    train_counts = train_results["substitution_count"]
    gen_counts = gen_results["substitution_count"]

    train_total = sum(train_counts.values()) or 1
    gen_total = sum(gen_counts.values()) or 1

    train_pcts = [100 * train_counts.get(c, 0) / train_total for c in categories]
    gen_pcts = [100 * gen_counts.get(c, 0) / gen_total for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        train_pcts,
        width,
        label="Training Data",
        color="#2ecc71",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2, gen_pcts, width, label="Generated", color="#3498db", alpha=0.8
    )

    ax1.set_xlabel("Substitution Pattern", fontsize=12)
    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_title("Substitution Count Distribution", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 2: Di-substitution pattern (ortho/meta/para)
    ax2 = axes[1]
    patterns = ["ortho", "meta", "para"]

    train_di = train_results["disubstitution_pattern"]
    gen_di = gen_results["disubstitution_pattern"]

    train_di_total = sum(train_di.values()) or 1
    gen_di_total = sum(gen_di.values()) or 1

    train_di_pcts = [100 * train_di.get(p, 0) / train_di_total for p in patterns]
    gen_di_pcts = [100 * gen_di.get(p, 0) / gen_di_total for p in patterns]

    x2 = np.arange(len(patterns))

    bars3 = ax2.bar(
        x2 - width / 2,
        train_di_pcts,
        width,
        label="Training Data",
        color="#2ecc71",
        alpha=0.8,
    )
    bars4 = ax2.bar(
        x2 + width / 2,
        gen_di_pcts,
        width,
        label="Generated",
        color="#3498db",
        alpha=0.8,
    )

    ax2.set_xlabel("Position Pattern", fontsize=12)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.set_title("Di-substitution Patterns (ortho/meta/para)", fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(patterns)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_functional_group_comparison(
    train_counts: Counter,
    gen_counts: Counter,
    output_path: Optional[str] = None,
    title: str = "Functional Groups Attached to Benzene",
    top_n: int = 10,
) -> plt.Figure:
    """Plot comparison of functional group distributions.

    Args:
        train_counts: Counter from training data.
        gen_counts: Counter from generated data.
        output_path: Optional path to save the figure.
        title: Plot title.
        top_n: Number of top groups to show.

    Returns:
        Matplotlib figure object.
    """
    # Get top groups by combined frequency
    combined = train_counts + gen_counts
    top_groups = [g for g, _ in combined.most_common(top_n)]

    if not top_groups:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No functional groups detected", ha="center", va="center")
        return fig

    train_total = sum(train_counts.values()) or 1
    gen_total = sum(gen_counts.values()) or 1

    train_pcts = [100 * train_counts.get(g, 0) / train_total for g in top_groups]
    gen_pcts = [100 * gen_counts.get(g, 0) / gen_total for g in top_groups]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(top_groups))
    width = 0.35

    ax.bar(
        x - width / 2,
        train_pcts,
        width,
        label="Training Data",
        color="#2ecc71",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2, gen_pcts, width, label="Generated", color="#3498db", alpha=0.8
    )

    ax.set_xlabel("Functional Group", fontsize=12)
    ax.set_ylabel("Percentage of Molecules (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top_groups, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_combined_analysis(
    train_smiles: list[str],
    gen_smiles: list[str],
    output_path: Optional[str] = None,
    title_prefix: str = "Generation",
) -> plt.Figure:
    """Create a combined analysis figure comparing training and generated data.

    Args:
        train_smiles: List of training SMILES (filtered to contain motif).
        gen_smiles: List of generated SMILES.
        output_path: Optional path to save the figure.
        title_prefix: Prefix for the title (e.g., tokenizer name).

    Returns:
        Matplotlib figure object.
    """
    # Analyze both datasets
    train_sub = analyze_benzene_substitution(train_smiles)
    gen_sub = analyze_benzene_substitution(gen_smiles)

    train_fg = analyze_functional_groups(train_smiles)
    gen_fg = analyze_functional_groups(gen_smiles)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Substitution count (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    categories = ["unsubstituted", "mono", "di", "tri", "poly"]
    train_total = sum(train_sub["substitution_count"].values()) or 1
    gen_total = sum(gen_sub["substitution_count"].values()) or 1

    train_pcts = [
        100 * train_sub["substitution_count"].get(c, 0) / train_total
        for c in categories
    ]
    gen_pcts = [
        100 * gen_sub["substitution_count"].get(c, 0) / gen_total for c in categories
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(
        x - width / 2, train_pcts, width, label="Training", color="#2ecc71", alpha=0.8
    )
    ax1.bar(
        x + width / 2, gen_pcts, width, label="Generated", color="#3498db", alpha=0.8
    )
    ax1.set_xlabel("Substitution Type")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Substitution Count Distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Di-substitution pattern (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    patterns = ["ortho", "meta", "para"]
    train_di_total = sum(train_sub["disubstitution_pattern"].values()) or 1
    gen_di_total = sum(gen_sub["disubstitution_pattern"].values()) or 1

    train_di_pcts = [
        100 * train_sub["disubstitution_pattern"].get(p, 0) / train_di_total
        for p in patterns
    ]
    gen_di_pcts = [
        100 * gen_sub["disubstitution_pattern"].get(p, 0) / gen_di_total
        for p in patterns
    ]

    ax2.bar(
        x[:3] - width / 2,
        train_di_pcts,
        width,
        label="Training",
        color="#2ecc71",
        alpha=0.8,
    )
    ax2.bar(
        x[:3] + width / 2,
        gen_di_pcts,
        width,
        label="Generated",
        color="#3498db",
        alpha=0.8,
    )
    ax2.set_xlabel("Position Pattern")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Di-substitution Patterns")
    ax2.set_xticks(x[:3])
    ax2.set_xticklabels(patterns)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Functional groups (bottom, spanning both columns)
    ax3 = fig.add_subplot(2, 1, 2)
    combined = train_fg + gen_fg
    top_groups = [g for g, _ in combined.most_common(10)]

    if top_groups:
        train_fg_total = sum(train_fg.values()) or 1
        gen_fg_total = sum(gen_fg.values()) or 1

        train_fg_pcts = [100 * train_fg.get(g, 0) / train_fg_total for g in top_groups]
        gen_fg_pcts = [100 * gen_fg.get(g, 0) / gen_fg_total for g in top_groups]

        x3 = np.arange(len(top_groups))
        ax3.bar(
            x3 - width / 2,
            train_fg_pcts,
            width,
            label="Training",
            color="#2ecc71",
            alpha=0.8,
        )
        ax3.bar(
            x3 + width / 2,
            gen_fg_pcts,
            width,
            label="Generated",
            color="#3498db",
            alpha=0.8,
        )
        ax3.set_xlabel("Functional Group")
        ax3.set_ylabel("Percentage (%)")
        ax3.set_title("Functional Groups Attached to Benzene")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(top_groups, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

    # Compute metrics
    sub_metrics = compare_distributions(
        train_sub["substitution_count"], gen_sub["substitution_count"]
    )
    fg_metrics = compare_distributions(train_fg, gen_fg)

    fig.suptitle(
        f"{title_prefix} Analysis\n"
        f"Substitution TV: {sub_metrics['total_variation']:.3f} | "
        f"Functional Group TV: {fg_metrics['total_variation']:.3f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ===========================================================================
# Molecule Visualization Functions
# ===========================================================================


def _categorize_by_substitution(
    smiles_list: list[str],
) -> dict[str, list[str]]:
    """Categorize molecules by benzene substitution pattern.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Dictionary mapping category to list of SMILES.
    """
    categories = defaultdict(list)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
        if not mol.HasSubstructMatch(benzene_pattern):
            continue

        count = get_benzene_substitution_count(mol)

        if count == 0:
            categories["unsubstituted"].append(smiles)
        elif count == 1:
            categories["mono"].append(smiles)
        elif count == 2:
            pattern = classify_disubstitution(mol)
            if pattern == "ortho":
                categories["di-ortho"].append(smiles)
            elif pattern == "meta":
                categories["di-meta"].append(smiles)
            elif pattern == "para":
                categories["di-para"].append(smiles)
            else:
                categories["di-other"].append(smiles)
        elif count == 3:
            categories["tri"].append(smiles)
        else:
            categories["poly"].append(smiles)

    return categories


def _categorize_by_functional_group(
    smiles_list: list[str],
) -> dict[str, list[str]]:
    """Categorize molecules by functional group attached to benzene.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Dictionary mapping functional group to list of SMILES.
    """
    categories = defaultdict(list)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
        if not mol.HasSubstructMatch(benzene_pattern):
            continue

        for group_name, smarts in SUBSTITUENT_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                categories[group_name].append(smiles)

    return categories


def _draw_comparison_grid(
    train_smiles_dict: dict[str, list[str]],
    gen_smiles_dict: dict[str, list[str]],
    output_path: str,
    categories_to_show: Optional[list[str]] = None,
    n_per_category: int = 3,
    mol_size: tuple[int, int] = (220, 180),
    seed: int = 42,
) -> None:
    """Draw side-by-side comparison of training vs generated molecules.

    Args:
        train_smiles_dict: Training data categorized by pattern.
        gen_smiles_dict: Generated data categorized by pattern.
        output_path: Path to save the image.
        categories_to_show: Categories to include (None = all common categories).
        n_per_category: Number of molecules per category per source.
        mol_size: Size of each molecule image.
        seed: Random seed for sampling.
    """
    random.seed(seed)

    # Determine categories to show
    if categories_to_show is None:
        # Use categories present in both
        common = set(train_smiles_dict.keys()) & set(gen_smiles_dict.keys())
        categories_to_show = sorted(common)

    if not categories_to_show:
        return

    # Build grid: each row is a category, columns alternate train/gen
    mols = []
    legends = []

    for cat_name in categories_to_show:
        train_list = train_smiles_dict.get(cat_name, [])
        gen_list = gen_smiles_dict.get(cat_name, [])

        # Sample from training
        train_sampled = random.sample(train_list, min(n_per_category, len(train_list)))
        # Sample from generated
        gen_sampled = random.sample(gen_list, min(n_per_category, len(gen_list)))

        # Add training molecules
        for i, smi in enumerate(train_sampled):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                if i == 0:
                    legends.append(f"Train: {cat_name}")
                else:
                    legends.append("Train")

        # Pad if needed
        while len(train_sampled) < n_per_category:
            mols.append(Chem.MolFromSmiles(""))
            legends.append("")
            train_sampled.append(None)

        # Add generated molecules
        for i, smi in enumerate(gen_sampled):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                if i == 0:
                    legends.append(f"Gen: {cat_name}")
                else:
                    legends.append("Gen")

        # Pad if needed
        while len(gen_sampled) < n_per_category:
            mols.append(Chem.MolFromSmiles(""))
            legends.append("")
            gen_sampled.append(None)

    if not any(m is not None and m.GetNumAtoms() > 0 for m in mols):
        return

    # Calculate grid: n_per_category * 2 columns (train + gen)
    n_cols = n_per_category * 2

    # Draw grid
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=legends,
        returnPNG=False,
    )

    # Save
    img.save(output_path)


def draw_molecule_comparison(
    train_smiles: list[str],
    gen_smiles: list[str],
    output_dir: str,
    tokenizer_name: str,
    motif_smiles: str = "c1ccccc1",
    seed: int = 42,
) -> dict[str, str]:
    """Draw molecule comparison visualizations.

    Creates two visualization files:
    1. Substitution pattern comparison
    2. Functional group comparison

    Args:
        train_smiles: List of training SMILES.
        gen_smiles: List of generated SMILES.
        output_dir: Output directory for images.
        tokenizer_name: Name of tokenizer (for filename).
        motif_smiles: SMILES of motif to filter by (default: benzene).
        seed: Random seed for sampling.

    Returns:
        Dictionary mapping visualization type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by motif
    motif = Chem.MolFromSmiles(motif_smiles)
    if motif is None:
        return {}

    train_filtered = [
        s for s in train_smiles
        if Chem.MolFromSmiles(s) is not None
        and Chem.MolFromSmiles(s).HasSubstructMatch(motif)
    ]
    gen_filtered = [
        s for s in gen_smiles
        if Chem.MolFromSmiles(s) is not None
        and Chem.MolFromSmiles(s).HasSubstructMatch(motif)
    ]

    if not train_filtered or not gen_filtered:
        return {}

    output_files = {}

    # 1. Substitution pattern visualization
    train_sub_cats = _categorize_by_substitution(train_filtered)
    gen_sub_cats = _categorize_by_substitution(gen_filtered)

    sub_categories = ["mono", "di-ortho", "di-meta", "di-para", "tri", "poly"]
    sub_path = output_dir / f"substitution_molecules_{tokenizer_name}.png"

    _draw_comparison_grid(
        train_sub_cats,
        gen_sub_cats,
        str(sub_path),
        categories_to_show=sub_categories,
        n_per_category=3,
        seed=seed,
    )
    output_files["substitution"] = str(sub_path)

    # 2. Functional group visualization
    train_fg_cats = _categorize_by_functional_group(train_filtered)
    gen_fg_cats = _categorize_by_functional_group(gen_filtered)

    # Get top functional groups
    combined_counts = {
        k: len(train_fg_cats.get(k, [])) + len(gen_fg_cats.get(k, []))
        for k in set(train_fg_cats.keys()) | set(gen_fg_cats.keys())
    }
    top_groups = sorted(
        combined_counts.keys(), key=lambda x: combined_counts[x], reverse=True
    )[:6]

    fg_path = output_dir / f"functional_group_molecules_{tokenizer_name}.png"

    _draw_comparison_grid(
        train_fg_cats,
        gen_fg_cats,
        str(fg_path),
        categories_to_show=top_groups,
        n_per_category=3,
        seed=seed,
    )
    output_files["functional_group"] = str(fg_path)

    return output_files
