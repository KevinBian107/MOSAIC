"""Visualization utilities for scaffold priming evaluation.

This module provides functions for visualizing scaffold priming results,
showing the primer (scaffold), generated molecules, and target molecules
side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from rdkit import Chem
from rdkit.Chem import Draw

if TYPE_CHECKING:
    from PIL import Image


def visualize_priming_comparison(
    scaffold_smiles: str,
    generated_smiles: list[str],
    target_smiles: str,
    output_path: Optional[Union[str, Path]] = None,
    n_generated: int = 3,
    img_size: tuple[int, int] = (300, 300),
    title: Optional[str] = None,
) -> Optional[Image.Image]:
    """Create a comparison image showing scaffold, generated, and target molecules.

    Creates a grid image with:
    - Top row: Scaffold (primer) | Target molecule
    - Bottom row(s): Generated molecules

    Args:
        scaffold_smiles: SMILES of the scaffold used for priming.
        generated_smiles: List of SMILES of generated molecules.
        target_smiles: SMILES of the target complex molecule.
        output_path: Path to save the image. If None, returns the image.
        n_generated: Maximum number of generated molecules to show.
        img_size: Size of each molecule image (width, height).
        title: Optional title for the image.

    Returns:
        PIL Image if output_path is None, otherwise None.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "PIL is required for visualization. Install with: pip install pillow"
        )

    # Parse molecules
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    target_mol = Chem.MolFromSmiles(target_smiles)

    generated_mols = []
    for smi in generated_smiles[:n_generated]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            generated_mols.append(mol)

    if not generated_mols:
        generated_mols = [None]  # Placeholder for empty

    # Create molecule images
    scaffold_img = (
        Draw.MolToImage(scaffold_mol, size=img_size)
        if scaffold_mol
        else _create_placeholder(img_size)
    )
    target_img = (
        Draw.MolToImage(target_mol, size=img_size)
        if target_mol
        else _create_placeholder(img_size)
    )

    generated_imgs = []
    for mol in generated_mols:
        if mol is not None:
            generated_imgs.append(Draw.MolToImage(mol, size=img_size))
        else:
            generated_imgs.append(_create_placeholder(img_size))

    # Calculate layout
    label_height = 30
    padding = 10
    n_cols = max(2, len(generated_imgs))
    n_rows = 2  # Top row: scaffold + target, Bottom row: generated

    total_width = n_cols * img_size[0] + (n_cols + 1) * padding
    total_height = n_rows * (img_size[1] + label_height) + (n_rows + 1) * padding

    if title:
        total_height += 40

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
        title_font = font

    y_offset = padding

    # Draw title if provided
    if title:
        draw.text((padding, y_offset), title, fill="black", font=title_font)
        y_offset += 40

    # Top row: Scaffold and Target
    # Scaffold
    x = padding
    draw.text((x, y_offset), "Scaffold (Primer)", fill="blue", font=font)
    canvas.paste(scaffold_img, (x, y_offset + label_height))

    # Target
    x = padding + img_size[0] + padding
    draw.text((x, y_offset), "Target Molecule", fill="green", font=font)
    canvas.paste(target_img, (x, y_offset + label_height))

    y_offset += img_size[1] + label_height + padding

    # Bottom row: Generated molecules
    for i, gen_img in enumerate(generated_imgs):
        x = padding + i * (img_size[0] + padding)
        label = f"Generated #{i + 1}"
        draw.text((x, y_offset), label, fill="purple", font=font)
        canvas.paste(gen_img, (x, y_offset + label_height))

    # Save or return
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return None
    else:
        return canvas


def _create_placeholder(size: tuple[int, int]) -> Image.Image:
    """Create a placeholder image for invalid molecules."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    img = PILImage.new("RGB", size, "lightgray")
    draw = ImageDraw.Draw(img)
    draw.text((size[0] // 4, size[1] // 2), "Invalid", fill="red")
    return img


def visualize_evaluation_results(
    results: list[dict],
    output_dir: Union[str, Path],
    max_samples: int = 10,
    n_generated_per_sample: int = 3,
) -> list[Path]:
    """Visualize multiple evaluation results.

    Args:
        results: List of evaluation result dictionaries, each containing:
            - scaffold_smiles: Scaffold SMILES
            - target_smiles: Target molecule SMILES
            - valid_smiles: List of valid generated SMILES
            - tanimoto_mean: Mean Tanimoto similarity
            - scaffold_preservation_rate: Scaffold preservation rate
        output_dir: Directory to save images.
        max_samples: Maximum number of samples to visualize.
        n_generated_per_sample: Number of generated molecules per sample.

    Returns:
        List of paths to saved images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, result in enumerate(results[:max_samples]):
        scaffold_smiles = result.get("scaffold_smiles", "")
        target_smiles = result.get("target_smiles", "")
        generated_smiles = result.get("valid_smiles", [])

        if not scaffold_smiles or not target_smiles:
            continue

        # Create title with metrics
        tanimoto = result.get("tanimoto_mean", 0)
        preservation = result.get("scaffold_preservation_rate", 0)
        title = f"Sample {i + 1}: Tanimoto={tanimoto:.3f}, Scaffold Preserved={preservation:.1%}"

        output_path = output_dir / f"sample_{i + 1:03d}.png"

        visualize_priming_comparison(
            scaffold_smiles=scaffold_smiles,
            generated_smiles=generated_smiles,
            target_smiles=target_smiles,
            output_path=output_path,
            n_generated=n_generated_per_sample,
            title=title,
        )

        saved_paths.append(output_path)

    return saved_paths


def create_summary_grid(
    results: list[dict],
    output_path: Union[str, Path],
    n_samples: int = 6,
    img_size: tuple[int, int] = (200, 200),
) -> None:
    """Create a summary grid showing scaffold -> best generated for multiple samples.

    Args:
        results: List of evaluation result dictionaries.
        output_path: Path to save the summary image.
        n_samples: Number of samples to include in grid.
        img_size: Size of each molecule image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "PIL is required for visualization. Install with: pip install pillow"
        )

    # Filter valid results
    valid_results = [
        r
        for r in results
        if r.get("scaffold_smiles") and r.get("target_smiles") and r.get("valid_smiles")
    ][:n_samples]

    if not valid_results:
        return

    # Layout: 3 columns (scaffold, best generated, target) x n_samples rows
    n_cols = 3
    n_rows = len(valid_results)
    label_height = 25
    padding = 5
    row_label_width = 80

    total_width = row_label_width + n_cols * img_size[0] + (n_cols + 1) * padding
    total_height = label_height + n_rows * (img_size[1] + padding) + padding

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        header_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
        header_font = font

    # Draw column headers
    headers = ["Scaffold", "Best Generated", "Target"]
    colors = ["blue", "purple", "green"]
    for i, (header, color) in enumerate(zip(headers, colors)):
        x = row_label_width + padding + i * (img_size[0] + padding)
        draw.text((x, 5), header, fill=color, font=header_font)

    # Draw each row
    y = label_height
    for row_idx, result in enumerate(valid_results):
        scaffold_mol = Chem.MolFromSmiles(result["scaffold_smiles"])
        target_mol = Chem.MolFromSmiles(result["target_smiles"])

        # Find best generated (highest Tanimoto if available)
        best_gen_smiles = result["valid_smiles"][0] if result["valid_smiles"] else None
        best_gen_mol = Chem.MolFromSmiles(best_gen_smiles) if best_gen_smiles else None

        # Row label with metrics
        tanimoto = result.get("tanimoto_max", result.get("tanimoto_mean", 0))
        row_label = f"#{row_idx + 1}\nT={tanimoto:.2f}"
        draw.text((5, y + img_size[1] // 3), row_label, fill="black", font=font)

        # Draw molecules
        mols = [scaffold_mol, best_gen_mol, target_mol]
        for col_idx, mol in enumerate(mols):
            x = row_label_width + padding + col_idx * (img_size[0] + padding)
            if mol is not None:
                mol_img = Draw.MolToImage(mol, size=img_size)
                canvas.paste(mol_img, (x, y))
            else:
                placeholder = _create_placeholder(img_size)
                canvas.paste(placeholder, (x, y))

        y += img_size[1] + padding

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
