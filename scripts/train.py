#!/usr/bin/env python
"""Training script for molecular graph generation models.

This script trains a transformer model on molecular graph data using
either SENT (flat) or H-SENT (hierarchical) tokenization.

Usage:
    python scripts/train.py
    python scripts/train.py data.dataset_name=qm9
    python scripts/train.py tokenizer.type=hsent  # Use hierarchical tokenization
    python scripts/train.py model.model_name=llama-s trainer.max_steps=200000
    python scripts/train.py wandb.enabled=true wandb.project=my-project

DDP multi-GPU (non-MIG):
    python scripts/train.py trainer.devices=4 trainer.strategy=ddp ...

DDP with MIG (use train_benchmarks.sh --ddp, or manually):
    CUDA_VISIBLE_DEVICES=<MIG-UUID> MASTER_ADDR=localhost MASTER_PORT=29500 \\
        WORLD_SIZE=4 RANK=0 LOCAL_RANK=0 GROUP_RANK=0 LOCAL_WORLD_SIZE=1 \\
        python scripts/train.py trainer.devices=1 trainer.num_nodes=4 trainer.strategy=ddp ...
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# MIG-aware CUDA device assignment for DDP with torchrun.
# Must run BEFORE importing torch so that CUDA_VISIBLE_DEVICES is set
# before the CUDA runtime initializes.  With MIG, each process can only
# access one MIG instance.  When torchrun passes a comma-separated list
# of MIG UUIDs via CUDA_VISIBLE_DEVICES, we slice it by LOCAL_RANK so
# each process sees exactly one device.
# NOTE: Only triggers for MIG UUIDs (MIG-xxx format), not regular GPU
# indices, to avoid breaking non-MIG DDP where PL manages devices.
_local_rank_env = os.environ.get("LOCAL_RANK")
if _local_rank_env is not None:
    _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _cuda_vis:
        _device_list = [d.strip() for d in _cuda_vis.split(",")]
        _is_mig = any(d.startswith("MIG-") for d in _device_list)
        _lr = int(_local_rank_env)
        if _is_mig and len(_device_list) > 1 and _lr < len(_device_list):
            os.environ["CUDA_VISIBLE_DEVICES"] = _device_list[_lr]
            # Each process now sees 1 device; LOCAL_RANK must be 0
            os.environ["LOCAL_RANK"] = "0"

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_lightning.callbacks import Callback

from src.data.datamodule import MolecularDataModule
from src.data.molecular import graph_to_smiles
from src.evaluation.motif_distribution import (
    MOLECULAR_MOTIFS,
    get_motif_counts,
)
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import HDTCTokenizer, HDTTokenizer, HSENTTokenizer, SENTTokenizer

log = logging.getLogger(__name__)

# Categorical colormap for motif types (organized by category)
MOTIF_COLORS = {
    # Aromatic rings - blues/purples
    "benzene": (0.2, 0.4, 0.8),
    "pyridine": (0.3, 0.3, 0.9),
    "pyrrole": (0.4, 0.2, 0.8),
    "furan": (0.5, 0.3, 0.7),
    "thiophene": (0.3, 0.5, 0.8),
    "imidazole": (0.4, 0.4, 0.9),
    "pyrimidine": (0.2, 0.5, 0.7),
    "naphthalene": (0.1, 0.3, 0.9),
    # Functional groups - greens/yellows
    "hydroxyl": (0.2, 0.8, 0.3),
    "carboxyl": (0.8, 0.6, 0.2),
    "carbonyl": (0.9, 0.7, 0.1),
    "aldehyde": (0.7, 0.8, 0.2),
    "ester": (0.6, 0.7, 0.3),
    "amide": (0.5, 0.8, 0.4),
    "amine_primary": (0.3, 0.9, 0.5),
    "amine_secondary": (0.4, 0.85, 0.5),
    "amine_tertiary": (0.5, 0.8, 0.5),
    "nitro": (0.9, 0.2, 0.2),
    "nitrile": (0.7, 0.3, 0.5),
    # Halogens - oranges/reds
    "halogen": (1.0, 0.5, 0.0),
    "fluorine": (0.9, 0.6, 0.1),
    "chlorine": (0.8, 0.5, 0.2),
    "bromine": (0.7, 0.4, 0.1),
    "iodine": (0.6, 0.3, 0.2),
    # Others - teals/cyans
    "ether": (0.2, 0.7, 0.7),
    "thioether": (0.3, 0.6, 0.6),
    "sulfone": (0.4, 0.5, 0.7),
    "sulfonamide": (0.5, 0.6, 0.8),
    "phosphate": (0.6, 0.4, 0.6),
}


def setup_wandb_logger(cfg: DictConfig) -> Optional[pl.loggers.WandbLogger]:
    """Set up Weights & Biases logger with full configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        WandbLogger instance or None if disabled.
    """
    if not cfg.wandb.enabled:
        return None

    run_name = cfg.wandb.name
    if run_name is None:
        run_name = f"{cfg.model.model_name}_{cfg.data.dataset_name}_s{cfg.seed}"

    tags = list(cfg.wandb.tags) if cfg.wandb.tags else []
    tags.append(cfg.model.model_name.split("-")[0])
    tags.append(cfg.data.dataset_name)

    wandb_logger = pl.loggers.WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=tags,
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.logs.path,
        log_model=cfg.wandb.log_model,
    )

    return wandb_logger


def log_generated_molecules_to_wandb(
    wandb_logger: pl.loggers.WandbLogger,
    smiles_list: list[str],
    prefix: str = "generated",
    max_molecules: int = 9,
) -> None:
    """Log generated molecule visualizations to WandB.

    Args:
        wandb_logger: WandB logger instance.
        smiles_list: List of generated SMILES strings.
        prefix: Prefix for the logged image name.
        max_molecules: Maximum number of molecules to visualize.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb
        from rdkit import Chem
        from rdkit.Chem import Draw

        valid_mols = []
        valid_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and len(valid_mols) < max_molecules:
                valid_mols.append(mol)
                valid_smiles.append(smiles)

        if not valid_mols:
            return

        # Draw molecules in a grid
        n_mols = len(valid_mols)
        n_cols = 3
        n_rows = (n_mols + n_cols - 1) // n_cols

        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=n_cols,
            subImgSize=(300, 300),
            legends=valid_smiles[:n_mols],
        )

        wandb_logger.experiment.log({f"{prefix}/molecules": wandb.Image(img)})

    except ImportError as e:
        log.warning(f"Could not log molecules to WandB: {e}")
    except Exception as e:
        log.warning(f"Error logging molecules to WandB: {e}")


def log_molecules_with_motifs_to_wandb(
    wandb_logger: pl.loggers.WandbLogger,
    smiles_list: list[str],
    prefix: str = "generated",
    max_molecules: int = 12,
) -> None:
    """Log generated molecules with color-coded motif highlighting to WandB.

    Creates:
    1. Grid image with per-motif-type colored highlights
    2. WandB Table with per-molecule metadata
    3. Color legend for motif types

    Args:
        wandb_logger: WandB logger instance.
        smiles_list: List of generated SMILES strings.
        prefix: Prefix for logged items.
        max_molecules: Maximum molecules to visualize.
    """
    if wandb_logger is None:
        return

    try:
        import wandb
        from rdkit import Chem
        from rdkit.Chem import Draw

        valid_data = []
        for smiles in smiles_list:
            if not smiles or smiles in ["INVALID", ""]:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and len(valid_data) < max_molecules:
                motifs = get_motif_counts(smiles)
                valid_data.append(
                    {
                        "smiles": smiles,
                        "mol": mol,
                        "motifs": motifs,
                    }
                )

        if not valid_data:
            return

        # --- 1. Create Grid with Color-Coded Motif Highlights ---
        mols = [d["mol"] for d in valid_data]
        legends = []
        highlight_atoms_list = []
        highlight_atom_colors_list = []

        for d in valid_data:
            mol = d["mol"]
            motifs = d["motifs"]

            # Collect atoms with per-motif colors
            highlight_atoms = []
            atom_colors = {}

            for motif_name in motifs:
                if motif_name in MOLECULAR_MOTIFS:
                    pattern = Chem.MolFromSmarts(MOLECULAR_MOTIFS[motif_name])
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        color = MOTIF_COLORS.get(motif_name, (0.5, 0.5, 0.5))
                        for match in matches:
                            for atom_idx in match:
                                if atom_idx not in atom_colors:
                                    highlight_atoms.append(atom_idx)
                                    atom_colors[atom_idx] = color

            highlight_atoms_list.append(highlight_atoms)
            highlight_atom_colors_list.append(atom_colors)

            # Create legend with top motifs
            top_motifs = sorted(motifs.items(), key=lambda x: -x[1])[:3]
            motif_str = ", ".join(f"{k}:{v}" for k, v in top_motifs)
            legends.append(f"{d['smiles'][:30]}\n{motif_str}")

        # Draw grid with color-coded highlights
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(350, 350),
            legends=legends,
            highlightAtomLists=highlight_atoms_list,
            highlightAtomColors=highlight_atom_colors_list,
        )

        wandb_logger.experiment.log(
            {f"{prefix}/molecules_with_motifs": wandb.Image(img)}
        )

        # --- 2. Create WandB Table with Metadata ---
        columns = ["smiles", "num_atoms", "num_motifs", "motif_list"]
        table_data = []

        for d in valid_data:
            mol = d["mol"]
            motifs = d["motifs"]

            table_data.append(
                [
                    d["smiles"],
                    mol.GetNumAtoms(),
                    sum(motifs.values()),
                    ", ".join(f"{k}({v})" for k, v in motifs.items()),
                ]
            )

        table = wandb.Table(columns=columns, data=table_data)
        wandb_logger.experiment.log({f"{prefix}/molecule_details": table})

        # --- 3. Log Color Legend ---
        motifs_found = set()
        for d in valid_data:
            motifs_found.update(d["motifs"].keys())
        _log_motif_color_legend(wandb_logger, motifs_found, prefix)

    except ImportError as e:
        log.warning(f"Could not log molecules with motifs to WandB: {e}")
    except Exception as e:
        log.warning(f"Error logging molecules with motifs to WandB: {e}")


def _log_motif_color_legend(
    wandb_logger: pl.loggers.WandbLogger,
    motifs_found: set[str],
    prefix: str,
) -> None:
    """Log a color legend for the motif types found.

    Args:
        wandb_logger: WandB logger instance.
        motifs_found: Set of motif names that were found.
        prefix: Prefix for logged items.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import wandb

        # Only show motifs that were actually found
        legend_items = []
        for motif_name in sorted(motifs_found):
            if motif_name in MOTIF_COLORS:
                color = MOTIF_COLORS[motif_name]
                patch = mpatches.Patch(color=color, label=motif_name)
                legend_items.append(patch)

        if not legend_items:
            return

        fig, ax = plt.subplots(figsize=(4, max(2, len(legend_items) * 0.3)))
        ax.axis("off")
        ax.legend(handles=legend_items, loc="center", frameon=False, fontsize=8)
        plt.tight_layout()

        wandb_logger.experiment.log({f"{prefix}/motif_color_legend": wandb.Image(fig)})
        plt.close(fig)

    except Exception as e:
        log.warning(f"Could not log motif color legend: {e}")


def log_final_metrics_to_wandb(
    wandb_logger: pl.loggers.WandbLogger,
    metrics: dict,
    prefix: str = "final",
) -> None:
    """Log final evaluation metrics to WandB.

    Args:
        wandb_logger: WandB logger instance.
        metrics: Dictionary of metric names to values.
        prefix: Prefix for metric names.
    """
    if wandb_logger is None:
        return

    log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb_logger.experiment.log(log_dict)


def save_model_artifact(
    wandb_logger: pl.loggers.WandbLogger,
    checkpoint_path: str,
    artifact_name: str = "model",
    artifact_type: str = "model",
) -> None:
    """Save model checkpoint as WandB artifact.

    Args:
        wandb_logger: WandB logger instance.
        checkpoint_path: Path to the checkpoint file.
        artifact_name: Name for the artifact.
        artifact_type: Type of artifact.
    """
    if wandb_logger is None:
        return

    try:
        import wandb

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description="Trained molecular graph generation model",
        )
        artifact.add_file(checkpoint_path)
        wandb_logger.experiment.log_artifact(artifact)
        log.info(f"Saved model artifact: {artifact_name}")
    except Exception as e:
        log.warning(f"Could not save model artifact: {e}")


class TokenizationVisualizationCallback(Callback):
    """Callback to log tokenization visualization at the start of training.

    Logs a single example from the training set with visualization of:
    - Molecule structure with motif highlighting
    - Tokenization structure (communities for hierarchical, walk for SENT)
    - Token sequence

    Attributes:
        datamodule: Data module with training dataset.
        tokenizer: Tokenizer instance.
        seed: Random seed for layout.
        _logged: Whether the visualization has been logged.
    """

    def __init__(
        self, datamodule, tokenizer, seed: int = 42, num_examples: int = 3
    ) -> None:
        """Initialize the callback.

        Args:
            datamodule: Data module with training dataset.
            tokenizer: Tokenizer instance (SENT, H-SENT, or HDT).
            seed: Random seed for consistent layout.
            num_examples: Number of examples to log (default 3).
        """
        super().__init__()
        self.datamodule = datamodule
        self.tokenizer = tokenizer
        self.seed = seed
        self.num_examples = num_examples
        self._logged = False

    def on_train_start(self, trainer, pl_module) -> None:
        """Log tokenization examples at the start of training."""
        if self._logged:
            return
        self._logged = True

        # Find WandB logger
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                wandb_logger = logger
                break

        if wandb_logger is None:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
            import torch
            import wandb
            from matplotlib.gridspec import GridSpec
            from rdkit import Chem
            from torch_geometric.data import Data
            from torch_geometric.utils import to_networkx

            if (
                not hasattr(self.datamodule, "train_smiles")
                or len(self.datamodule.train_smiles) == 0
            ):
                log.warning(
                    "No training SMILES available for tokenization visualization"
                )
                return

            tokenizer_type = type(self.tokenizer).__name__
            num_to_log = min(self.num_examples, len(self.datamodule.train_smiles))

            for idx in range(num_to_log):
                smiles = self.datamodule.train_smiles[idx]
                log.info(f"Tokenization viz [{idx}]: SMILES={smiles[:50]}...")

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    log.warning(f"Could not parse SMILES: {smiles}")
                    continue

                # Build graph from SMILES
                edges = []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edges.append([i, j])
                    edges.append([j, i])

                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)

                data = Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())
                data.smiles = smiles

                # Get graph layout
                try:
                    G = to_networkx(data, to_undirected=True)
                except Exception:
                    G = nx.Graph()
                    G.add_nodes_from(range(data.num_nodes))
                    if edge_index.numel() > 0:
                        ei = edge_index.numpy()
                        for i in range(ei.shape[1]):
                            G.add_edge(int(ei[0, i]), int(ei[1, i]))
                pos = nx.spring_layout(G, seed=self.seed + idx, k=1.5, iterations=100)

                # Create figure
                fig = plt.figure(figsize=(16, 8))
                gs = GridSpec(
                    2, 2, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.3
                )

                # Panel 1: Molecule with motifs
                ax1 = fig.add_subplot(gs[0, 0])
                _plot_molecule_with_motifs(ax1, data, smiles, pos)

                # Panel 2: Tokenization structure
                ax2 = fig.add_subplot(gs[0, 1])
                tokens = self.tokenizer.tokenize(data).tolist()

                if tokenizer_type in ("HSENTTokenizer", "HDTTokenizer"):
                    hg = self.tokenizer.coarsener.build_hierarchy(data)
                    log.info(
                        f"Tokenization viz [{idx}]: {hg.num_communities} communities, {len(tokens)} tokens"
                    )
                    _plot_hierarchical_structure(ax2, data, hg, tokenizer_type, pos)
                else:
                    log.info(f"Tokenization viz [{idx}]: {len(tokens)} tokens")
                    _plot_sent_walk(ax2, data, tokens, self.tokenizer, pos)

                # Panel 3: Token sequence
                ax3 = fig.add_subplot(gs[1, :])
                _plot_token_sequence(ax3, tokens, self.tokenizer, tokenizer_type)

                # Title
                fig.suptitle(
                    f"Tokenization Example {idx + 1}: {tokenizer_type}\n"
                    f"SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''} "
                    f"({mol.GetNumAtoms()} atoms, {len(tokens)} tokens)",
                    fontsize=12,
                    fontweight="bold",
                )

                plt.tight_layout()
                wandb_logger.experiment.log(
                    {f"tokenization/example_{idx}": wandb.Image(fig)}
                )
                plt.close(fig)

            log.info(f"Logged {num_to_log} tokenization examples to WandB")

        except ImportError as e:
            log.warning(f"Could not log tokenization examples to WandB: {e}")
        except Exception as e:
            log.warning(f"Error logging tokenization examples to WandB: {e}")


def _plot_molecule_with_motifs(
    ax: "plt.Axes",
    data,
    smiles: str,
    pos: dict,
) -> None:
    """Plot molecule structure with motif highlighting."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from rdkit import Chem

    from src.tokenizers.motif import CLUSTERING_MOTIFS

    # Use same patterns as the coarsener for consistency
    motif_patterns = CLUSTERING_MOTIFS

    # Colors for each motif type
    motif_colors = {
        # Aromatic 6-membered rings - reds/oranges
        "benzene": "#FF6B6B",
        "pyridine": "#FF8C42",
        "pyrimidine": "#FFB347",
        "pyrazine": "#FFC87C",
        # Aromatic 5-membered rings - blues/teals
        "pyrrole": "#45B7D1",
        "furan": "#4ECDC4",
        "thiophene": "#2EC4B6",
        "imidazole": "#3D9970",
        "oxazole": "#20C997",
        "thiazole": "#17A2B8",
        # Fused ring systems - purples
        "naphthalene": "#9B59B6",
        "indole": "#8E44AD",
        "quinoline": "#6C5CE7",
        "benzofuran": "#A55EEA",
        "benzothiophene": "#7C3AED",
        # Saturated rings - greens
        "cyclopropane": "#96CEB4",
        "cyclobutane": "#88D8B0",
        "cyclopentane": "#56AB91",
        "cyclohexane": "#2D6A4F",
        # Partially unsaturated - yellows
        "cyclohexene": "#FFEAA7",
        "cyclopentene": "#FDCB6E",
    }

    mol = Chem.MolFromSmiles(smiles)
    motifs = {}
    if mol:
        for name, pattern in motif_patterns.items():
            query = Chem.MolFromSmarts(pattern)
            if query:
                matches = mol.GetSubstructMatches(query)
                if matches:
                    motifs[name] = list(matches)

    # Node colors
    node_colors = ["#E8E8E8"] * data.num_nodes
    for motif_name, matches in motifs.items():
        color = motif_colors.get(motif_name, "#CCCCCC")
        for match in matches:
            for atom_idx in match:
                if atom_idx < data.num_nodes:
                    node_colors[atom_idx] = color

    # Draw edges
    edge_index = data.edge_index
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.numpy()
        # Ensure edge_index is 2D with shape (2, num_edges)
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        drawn_edges = set()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u not in pos or v not in pos:
                continue
            edge_key = (min(u, v), max(u, v))
            if edge_key not in drawn_edges:
                drawn_edges.add(edge_key)
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    color="#555555",
                    linewidth=2,
                    zorder=1,
                )

    # Draw nodes
    for node in range(data.num_nodes):
        x, y = pos[node]
        circle = plt.Circle(
            (x, y),
            0.08,
            facecolor=node_colors[node],
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    # Legend
    legend_patches = []
    for motif_name in motifs.keys():
        if motif_name in motif_colors:
            patch = mpatches.Patch(
                color=motif_colors[motif_name], label=motif_name.capitalize()
            )
            legend_patches.append(patch)
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper left", fontsize=7, framealpha=0.9)

    ax.set_title("Molecule Structure with Motifs", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def _plot_hierarchical_structure(
    ax: "plt.Axes",
    data,
    hg,
    tokenizer_type: str,
    pos: dict,
) -> None:
    """Plot hierarchical structure (communities) for H-SENT/HDT."""
    import matplotlib.pyplot as plt

    comm_colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#FF8C00",
        "#9B59B6",
    ]

    # Draw edges
    edge_index = data.edge_index
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.numpy()
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        drawn_edges = set()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u not in pos or v not in pos:
                continue
            edge_key = (min(u, v), max(u, v))
            if edge_key not in drawn_edges:
                drawn_edges.add(edge_key)
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    color="#AAAAAA",
                    linewidth=1.5,
                    zorder=1,
                )

    # Draw nodes colored by community
    for part_idx, part in enumerate(hg.partitions):
        comm_color = comm_colors[part_idx % len(comm_colors)]
        for global_idx in part.global_node_indices:
            if global_idx in pos:
                x, y = pos[global_idx]
                circle = plt.Circle(
                    (x, y),
                    0.1,
                    facecolor=comm_color,
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=3,
                )
                ax.add_patch(circle)
                ax.text(
                    x,
                    y,
                    str(global_idx),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    zorder=4,
                )

    title = f"{tokenizer_type}: {hg.num_communities} communities"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def _plot_sent_walk(
    ax: "plt.Axes",
    data,
    tokens: list[int],
    tokenizer,
    pos: dict,
) -> None:
    """Plot SENT walk visualization."""
    import matplotlib.pyplot as plt

    # Draw edges
    edge_index = data.edge_index
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.numpy()
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        drawn_edges = set()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u not in pos or v not in pos:
                continue
            edge_key = (min(u, v), max(u, v))
            if edge_key not in drawn_edges:
                drawn_edges.add(edge_key)
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    color="#AAAAAA",
                    linewidth=1.5,
                    zorder=1,
                )

    # Extract visit order from tokens
    idx_offset = getattr(tokenizer, "idx_offset", getattr(tokenizer, "IDX_OFFSET", 10))
    visit_order = {}
    order = 0
    for tok in tokens:
        if tok >= idx_offset:
            node_id = tok - idx_offset
            if node_id not in visit_order:
                visit_order[node_id] = order
                order += 1

    # Draw nodes with visit order
    colors = plt.cm.viridis(
        [i / max(1, len(visit_order)) for i in range(len(visit_order))]
    )
    for node in range(data.num_nodes):
        x, y = pos[node]
        if node in visit_order:
            color = colors[visit_order[node]]
        else:
            color = "#E8E8E8"
        circle = plt.Circle(
            (x, y), 0.1, facecolor=color, edgecolor="black", linewidth=1.5, zorder=3
        )
        ax.add_patch(circle)
        label = str(visit_order.get(node, "?"))
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )

    ax.set_title("SENT: Walk Order", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def _plot_token_sequence(
    ax: "plt.Axes",
    tokens: list[int],
    tokenizer,
    tokenizer_type: str,
    max_tokens: int = 100,
) -> None:
    """Plot token sequence as text."""
    # Build token name mappings based on tokenizer type
    if tokenizer_type == "HDTTokenizer":
        token_names = {
            tokenizer.SOS: "[SOS]",
            tokenizer.EOS: "[EOS]",
            tokenizer.PAD: "[PAD]",
            tokenizer.ENTER: "↓",
            tokenizer.EXIT: "↑",
            tokenizer.LEDGE: "[",
            tokenizer.REDGE: "]",
        }
        idx_offset = tokenizer.IDX_OFFSET
    elif tokenizer_type == "HDTCTokenizer":
        token_names = {
            tokenizer.SOS: "[SOS]",
            tokenizer.EOS: "[EOS]",
            tokenizer.PAD: "[PAD]",
            tokenizer.COMM_START: "{",
            tokenizer.COMM_END: "}",
            tokenizer.LEDGE: "[",
            tokenizer.REDGE: "]",
            tokenizer.SUPER_START: "<S",
            tokenizer.SUPER_END: "S>",
            tokenizer.TYPE_RING: "R",
            tokenizer.TYPE_FUNC: "F",
            tokenizer.TYPE_SINGLETON: "1",
        }
        idx_offset = tokenizer.IDX_OFFSET
    elif tokenizer_type == "HSENTTokenizer":
        token_names = {
            tokenizer.SOS: "[SOS]",
            tokenizer.EOS: "[EOS]",
            tokenizer.PAD: "[PAD]",
            tokenizer.RESET: "[R]",
            tokenizer.LADJ: "[",
            tokenizer.RADJ: "]",
            tokenizer.LCOM: "{",
            tokenizer.RCOM: "}",
            tokenizer.LBIP: "<",
            tokenizer.RBIP: ">",
            tokenizer.SEP: "|",
        }
        idx_offset = tokenizer.IDX_OFFSET
    else:  # SENTTokenizer
        token_names = {
            tokenizer.sos: "[SOS]",
            tokenizer.eos: "[EOS]",
            tokenizer.pad: "[PAD]",
            tokenizer.reset: "[R]",
            tokenizer.ladj: "[",
            tokenizer.radj: "]",
        }
        idx_offset = tokenizer.idx_offset

    # Convert tokens to string symbols
    display_tokens = tokens[:max_tokens]
    truncated = len(tokens) > max_tokens

    token_strs = []
    for tok in display_tokens:
        if tok in token_names:
            token_strs.append(token_names[tok])
        elif tok >= idx_offset:
            token_strs.append(str(tok - idx_offset))
        else:
            token_strs.append(f"?{tok}")

    # Join tokens with line breaks
    tokens_per_line = 25
    lines = []
    for i in range(0, len(token_strs), tokens_per_line):
        lines.append(" ".join(token_strs[i : i + tokens_per_line]))
    token_text = "\n".join(lines)
    if truncated:
        token_text += f" ... ({len(tokens)} total)"

    ax.text(
        0.5,
        0.5,
        token_text,
        ha="center",
        va="center",
        fontsize=8,
        fontfamily="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dee2e6"),
    )

    ax.set_title(
        f"Token Sequence ({len(tokens)} tokens)", fontsize=10, fontweight="bold"
    )
    ax.axis("off")


class IntermediateEvaluationCallback(Callback):
    """Callback to run generation and log visualizations during training.

    Triggers every `eval_every_n_val` validation epochs to:
    1. Generate sample molecules
    2. Log validity fraction and generation time
    3. Log molecules with color-coded motif highlights to WandB

    Heavy metric computation (MolecularMetrics, MotifDistribution, etc.)
    is deferred to the test script.

    Attributes:
        tokenizer: Tokenizer for decoding generated tokens.
        eval_every_n_val: Run evaluation every N validation epochs.
        num_samples: Number of molecules to generate.
        max_logged_molecules: Maximum molecules to visualize.
        wandb_logger: WandB logger instance.
    """

    def __init__(
        self,
        tokenizer,
        eval_every_n_val: int = 5,
        num_samples: int = 50,
        max_logged_molecules: int = 12,
        wandb_logger: Optional[pl.loggers.WandbLogger] = None,
    ) -> None:
        """Initialize the intermediate evaluation callback.

        Args:
            tokenizer: Tokenizer for decoding generated tokens.
            eval_every_n_val: Run evaluation every N validation epochs.
            num_samples: Number of molecules to generate per evaluation.
            max_logged_molecules: Maximum molecules to visualize in WandB.
            wandb_logger: WandB logger instance (optional).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_every_n_val = eval_every_n_val
        self.num_samples = num_samples
        self.max_logged_molecules = max_logged_molecules
        self.wandb_logger = wandb_logger

        self._val_epoch_count = 0

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Run intermediate evaluation after validation epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: The Lightning module being trained.
        """
        self._val_epoch_count += 1

        # Skip if not at evaluation interval
        if self._val_epoch_count % self.eval_every_n_val != 0:
            return

        # Skip if no WandB logger
        if self.wandb_logger is None:
            return

        # Skip on non-rank-0 processes (generation is not DDP-synchronized)
        if trainer.global_rank != 0:
            return

        log.info(
            f"Running intermediate evaluation (val epoch {self._val_epoch_count})..."
        )

        # Generate samples
        pl_module.eval()
        generated_graphs, gen_time, _ = pl_module.generate(
            num_samples=self.num_samples, show_progress=True
        )

        # Convert to SMILES
        INVALID = "INVALID"
        generated_smiles = []
        for g in tqdm(generated_graphs, desc="Converting to SMILES"):
            smiles = graph_to_smiles(g)
            generated_smiles.append(smiles if smiles else INVALID)

        valid_count = sum(1 for s in generated_smiles if s != INVALID)
        log.info(f"  Generated {valid_count}/{len(generated_smiles)} valid molecules")

        # Log validity and generation time
        step = trainer.global_step
        self.wandb_logger.experiment.log(
            {
                "intermediate/generation_time": gen_time,
                "intermediate/valid_fraction": valid_count
                / max(len(generated_smiles), 1),
            },
            step=step,
        )

        # Log molecule visualizations with motif highlighting
        log_molecules_with_motifs_to_wandb(
            self.wandb_logger,
            generated_smiles,
            prefix="intermediate",
            max_molecules=self.max_logged_molecules,
        )

        log.info(f"  Logged intermediate evaluation at step {step}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration.
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Enable Tensor Core matmul precision for better performance on A5000/A100
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(cfg.seed, workers=True)

    # Create output directory and save configuration
    output_dir = Path(cfg.logs.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")

    # Check for checkpoint BEFORE loading data (saves time if resuming)
    ckpt_path = None
    if cfg.get("resume", True):
        output_dir = cfg.logs.path
        last_ckpt = os.path.join(output_dir, "last.ckpt")
        best_ckpt = os.path.join(output_dir, "best.ckpt")

        if os.path.exists(last_ckpt):
            log.info(f"✓ Found checkpoint: {last_ckpt}")
            log.info("Will resume training after setup...")
            ckpt_path = last_ckpt
        elif os.path.exists(best_ckpt):
            log.info(f"✓ Found checkpoint: {best_ckpt}")
            log.info("Will resume training after setup...")
            ckpt_path = best_ckpt
        else:
            log.info("No checkpoint found. Starting fresh training.")
    else:
        log.info("Resume disabled (resume=false). Starting fresh training.")

    # Now load data (expensive operation)
    log.info("Setting up dataset and tokenizer...")

    # Select tokenizer based on config
    tokenizer_type = cfg.tokenizer.get("type", "sent").lower()
    if tokenizer_type == "hdt":
        coarsening_strategy = cfg.tokenizer.get("coarsening_strategy", "spectral")
        log.info(
            f"Using hierarchical HDT tokenizer with {coarsening_strategy} coarsening"
        )
        if coarsening_strategy in ("motif_aware_spectral", "motif_community"):
            log.info(f"  motif_alpha: {cfg.tokenizer.get('motif_alpha', 1.0)}")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  min_community_size: {cfg.tokenizer.get('min_community_size', 4)}")

        tokenizer = HDTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=coarsening_strategy,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hdtc":
        log.info("Using HDTC (compositional) tokenizer with functional hierarchy")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  include_rings: {cfg.tokenizer.get('include_rings', True)}")

        tokenizer = HDTCTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            include_rings=cfg.tokenizer.get("include_rings", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    elif tokenizer_type == "hsent":
        coarsening_strategy = cfg.tokenizer.get("coarsening_strategy", "spectral")
        log.info(
            f"Using hierarchical H-SENT tokenizer with {coarsening_strategy} coarsening"
        )
        if coarsening_strategy in ("motif_aware_spectral", "motif_community"):
            log.info(f"  motif_alpha: {cfg.tokenizer.get('motif_alpha', 1.0)}")
        log.info(f"  node_order: {cfg.tokenizer.get('node_order', 'BFS')}")
        log.info(f"  min_community_size: {cfg.tokenizer.get('min_community_size', 4)}")

        tokenizer = HSENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            node_order=cfg.tokenizer.get("node_order", "BFS"),
            min_community_size=cfg.tokenizer.get("min_community_size", 4),
            coarsening_strategy=coarsening_strategy,
            motif_alpha=cfg.tokenizer.get("motif_alpha", 1.0),
            normalize_by_motif_size=cfg.tokenizer.get("normalize_by_motif_size", False),
            labeled_graph=cfg.tokenizer.get("labeled_graph", True),
            seed=cfg.seed,
        )
    else:
        log.info("Using flat SENT tokenizer")
        tokenizer = SENTTokenizer(
            max_length=cfg.tokenizer.max_length,
            truncation_length=cfg.tokenizer.truncation_length,
            undirected=cfg.tokenizer.get("undirected", True),
            labeled_graph=cfg.tokenizer.get("labeled_graph", False),
            seed=cfg.seed,
        )

    datamodule = MolecularDataModule(
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        test_num_workers=cfg.data.get("test_num_workers", 0),
        num_train=cfg.data.num_train,
        num_val=cfg.data.num_val,
        num_test=cfg.data.num_test,
        include_hydrogens=cfg.data.get("include_hydrogens", False),
        seed=cfg.seed,
        data_root=cfg.data.get("data_root", "data"),
        use_cache=cfg.data.get("use_cache", False),
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
        # COCONUT-specific parameters
        data_file=cfg.data.get("data_file"),
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
    )

    datamodule.setup()

    # Determine training duration (epochs vs steps) - define early for use throughout
    max_epochs = cfg.trainer.get("max_epochs")
    max_steps = cfg.trainer.max_steps if max_epochs is None else -1

    # DDP world size: prefer WORLD_SIZE env (set by torchrun) over config devices
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        devices_cfg = cfg.trainer.devices
        world_size = (
            devices_cfg if isinstance(devices_cfg, int) and devices_cfg > 1 else 1
        )

    # Calculate steps per epoch and adjust val_check_interval if needed
    # Skip this if validation is disabled (limit_val_batches=0)
    limit_val_batches = cfg.trainer.get("limit_val_batches", 1.0)

    if limit_val_batches == 0:
        # Validation disabled - set val_check_interval to None to avoid validation setup
        val_check_interval = None
        log.info("Validation disabled (limit_val_batches=0)")
    else:
        train_dataset_size = len(datamodule.train_dataset)
        # In DDP each GPU gets 1/world_size of the data via DistributedSampler
        steps_per_epoch = train_dataset_size // (cfg.data.batch_size * world_size)
        val_check_interval = cfg.trainer.val_check_interval

        # If val_check_interval exceeds steps per epoch, use steps per epoch (1 eval per epoch)
        if val_check_interval > steps_per_epoch:
            log.warning(
                f"val_check_interval ({val_check_interval}) exceeds steps per epoch ({steps_per_epoch}). "
                f"Setting val_check_interval={steps_per_epoch} (1 validation per epoch)"
            )
            val_check_interval = steps_per_epoch

        log.info(f"Training dataset size: {train_dataset_size:,} samples")
        if world_size > 1:
            log.info(f"DDP world size: {world_size} (steps per epoch adjusted)")
        log.info(f"Steps per epoch: {steps_per_epoch:,}")
        log.info(f"Validation check interval: {val_check_interval:,} steps")

    # Ensure model position embeddings can handle any sequence the tokenizer produces
    model_max_length = max(
        cfg.sampling.max_length,
        cfg.tokenizer.get("truncation_length", cfg.sampling.max_length),
    )

    # Calculate total training steps for lr scheduler
    # If using max_epochs, estimate total steps; if using max_steps, use that directly
    if max_epochs is not None:
        train_dataset_size = len(datamodule.train_dataset)
        # Account for DDP: each GPU sees a fraction of the data
        steps_per_epoch = train_dataset_size // (cfg.data.batch_size * world_size)
        total_steps = steps_per_epoch * max_epochs
        log.info(f"Estimated total steps for {max_epochs} epochs: {total_steps:,}")
    else:
        total_steps = cfg.trainer.max_steps

    model = GraphGeneratorModule(
        tokenizer=tokenizer,
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=total_steps,
        sampling_top_k=cfg.sampling.top_k,
        sampling_temperature=cfg.sampling.temperature,
        sampling_max_length=model_max_length,
        sampling_num_samples=cfg.sampling.num_samples,
        sampling_batch_size=cfg.sampling.batch_size,
    )

    # Load pretrained weights if specified (for transfer learning / fine-tuning)
    pretrained_path = cfg.model.get("pretrained_path")
    if pretrained_path:
        log.info(f"Loading pretrained weights from {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location="cpu")
        # Handle both Lightning checkpoint format and raw state dict
        if "state_dict" in pretrained:
            state_dict = pretrained["state_dict"]
        else:
            state_dict = pretrained
        # Load with strict=False to allow for vocab size differences
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            log.warning(
                f"Unexpected keys when loading pretrained weights: {unexpected}"
            )
        log.info("Pretrained weights loaded successfully")

    loggers = []
    loggers.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    wandb_logger = setup_wandb_logger(cfg)
    if wandb_logger is not None:
        loggers.append(wandb_logger)
        log.info(f"WandB logging enabled: {cfg.wandb.project}")

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=cfg.logs.path,
            filename="best",  # Simple name for best checkpoint
            save_last=True,  # Saves last.ckpt automatically
            mode="min",
        ),
    ]

    # Add tokenization visualization callback if WandB enabled
    if wandb_logger is not None:
        callbacks.append(
            TokenizationVisualizationCallback(
                datamodule=datamodule,
                tokenizer=tokenizer,
                seed=cfg.seed,
            )
        )
        log.info("Tokenization visualization will be logged at training start")

    # Add intermediate evaluation callback if WandB enabled and configured
    eval_every_n_val = cfg.wandb.get("eval_every_n_val", 0)
    if wandb_logger is not None and eval_every_n_val > 0:
        eval_callback = IntermediateEvaluationCallback(
            tokenizer=tokenizer,
            eval_every_n_val=eval_every_n_val,
            num_samples=cfg.wandb.get("eval_num_samples", 50),
            max_logged_molecules=cfg.wandb.get("max_logged_molecules", 12),
            wandb_logger=wandb_logger,
        )
        callbacks.append(eval_callback)
        log.info(
            f"Intermediate evaluation enabled every {eval_every_n_val} validation epochs"
        )

    # Print DDP speedup estimation
    num_gpus = world_size
    current_batch_size = cfg.data.batch_size
    effective_batch_size = current_batch_size * num_gpus

    # Baseline: 1 GPU, batch_size=32, 20 epochs
    baseline_batch = 32
    baseline_gpus = 1
    baseline_epochs = 20
    baseline_effective_batch = baseline_batch * baseline_gpus

    if max_epochs is not None:
        current_epochs = max_epochs

        # Calculate relative training time
        # Time is proportional to: (epochs * data_size) / (effective_batch_size * num_gpus * gpu_efficiency)
        # Speedup from batch size scaling
        batch_speedup = effective_batch_size / baseline_effective_batch
        # Speedup from epoch reduction
        epoch_speedup = baseline_epochs / current_epochs
        # DDP efficiency (typically 0.85-0.95 for 2-4 GPUs, accounting for communication overhead)
        ddp_efficiency = (
            1.0
            if num_gpus == 1
            else (0.95 if num_gpus == 2 else 0.90 if num_gpus <= 4 else 0.85)
        )

        # Total expected speedup
        total_speedup = batch_speedup * epoch_speedup * ddp_efficiency

        log.info("=" * 80)
        log.info("DDP SPEEDUP ESTIMATION")
        log.info("=" * 80)
        log.info(
            f"Baseline:  {baseline_gpus} GPU  × batch={baseline_batch}  × {baseline_epochs} epochs"
        )
        log.info(
            f"Current:   {num_gpus} GPU{'s' if num_gpus > 1 else ''}  × batch={current_batch_size}  × {current_epochs} epochs"
        )
        log.info(
            f"Effective batch size: {baseline_effective_batch} → {effective_batch_size} ({effective_batch_size / baseline_effective_batch:.1f}x)"
        )
        log.info(f"DDP efficiency factor: {ddp_efficiency:.2f}")
        log.info(f"Expected speedup: {total_speedup:.2f}x faster than baseline")
        if total_speedup > 1:
            log.info(f"Estimated time reduction: {(1 - 1 / total_speedup) * 100:.1f}%")
        log.info("=" * 80)

    trainer = pl.Trainer(
        max_epochs=max_epochs if max_epochs is not None else 1000,
        max_steps=max_steps,
        val_check_interval=val_check_interval
        if val_check_interval is not None
        else 1.0,
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch"),
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.get("num_nodes", 1),
        strategy=cfg.trainer.get("strategy", "auto"),
        logger=loggers,
        callbacks=callbacks,
    )

    # Checkpoint detection was already done earlier to save time
    if ckpt_path:
        log.info(f"Resuming training from: {ckpt_path}")
    else:
        log.info("Starting fresh training...")

    log.info("Starting training...")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # Explicitly save last.ckpt with final model weights.
    # PL 2.x only updates last.ckpt when best.ckpt also updates (i.e., when
    # val loss improves). If val loss plateaus early, last.ckpt gets stuck at
    # the best checkpoint step instead of the final training step.
    last_ckpt_path = os.path.join(cfg.logs.path, "last.ckpt")
    trainer.save_checkpoint(last_ckpt_path)
    log.info(f"Saved final checkpoint to {last_ckpt_path} (step {trainer.global_step})")

    # Skip test evaluation and generation if num_samples is 0 (for pure training performance testing)
    if cfg.sampling.num_samples > 0:
        log.info("Running evaluation on test set...")
        trainer.test(model, datamodule)

        # Generation is autoregressive and not DDP-synchronized.
        # Only rank 0 generates and logs to avoid deadlocks.
        if trainer.global_rank == 0:
            log.info("Generating samples for evaluation...")
            model.eval()
            generated_graphs, gen_time, *_ = model.generate(
                num_samples=cfg.sampling.num_samples, show_progress=True
            )
            log.info(
                f"Generated {len(generated_graphs)} graphs in {gen_time:.4f}s per sample"
            )

            # Convert generated graphs to SMILES for molecular metrics
            # Use sentinel value for failed conversions to compute accurate metrics
            INVALID_SMILES_SENTINEL = "INVALID"
            generated_smiles = []
            for g in tqdm(generated_graphs, desc="Converting to SMILES"):
                smiles = graph_to_smiles(g)
                generated_smiles.append(smiles if smiles else INVALID_SMILES_SENTINEL)

            valid_count = sum(
                1 for s in generated_smiles if s != INVALID_SMILES_SENTINEL
            )
            log.info(
                f"Successfully converted {valid_count}/{len(generated_smiles)} graphs to SMILES"
            )

            if wandb_logger is not None and cfg.wandb.log_graphs:
                log.info("Logging generated molecules to WandB...")
                # Simple molecule grid
                log_generated_molecules_to_wandb(
                    wandb_logger, generated_smiles, prefix="final"
                )
                # Enhanced visualization with color-coded motif highlighting
                log_molecules_with_motifs_to_wandb(
                    wandb_logger,
                    generated_smiles,
                    prefix="final",
                    max_molecules=cfg.wandb.get("max_logged_molecules", 12),
                )

            # Log final validity and generation time
            # Full metric computation is handled by the test script (scripts/test.py)
            log_final_metrics_to_wandb(
                wandb_logger,
                {
                    "valid_fraction": valid_count / max(len(generated_smiles), 1),
                    "generation_time": gen_time,
                },
            )
    else:
        log.info("Skipping test evaluation and generation (num_samples=0)")

    if wandb_logger is not None:
        import wandb

        wandb.finish()
        log.info("WandB run finished")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
