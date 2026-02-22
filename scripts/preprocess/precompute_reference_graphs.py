#!/usr/bin/env python
"""Precompute reference SMILES -> graphs for PGD and save to disk.

Used by eval_benchmarks_auto.sh so that each checkpoint eval can load the same
reference graphs instead of reconverting SMILES every time. Set
metrics.reference_graphs_path to the printed path in test.yaml or pass it to test.py.
See docs/commands_reference.md.

Usage:
    python scripts/preprocess/precompute_reference_graphs.py experiment=moses reference_graphs.output_dir=outputs/eval_run
    # Script prints the path to the saved file (e.g. outputs/eval_run/reference_graphs/reference_graphs_moses_test_100.pt)
"""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import MolecularDataModule
from src.data.molecular import smiles_to_graph
from src.tokenizers import SENTTokenizer


def main_with_overrides(overrides: list[str]) -> str:
    """Run with explicit Hydra overrides (for callers that pass overrides)."""
    project_root = Path(__file__).resolve().parent.parent
    config_dir = str(project_root / "configs")
    overrides = list(overrides) + ["tokenizer=sent"]  # minimal tokenizer, we only need SMILES

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="test", overrides=overrides)

    output_dir = OmegaConf.select(cfg, "reference_graphs.output_dir")
    if not output_dir:
        raise ValueError("reference_graphs.output_dir is required (e.g. reference_graphs.output_dir=outputs/eval_run)")
    output_dir = Path(output_dir)

    ref_split = cfg.metrics.get("reference_split", "test")
    pgd_size = cfg.metrics.get("pgd_reference_size", 100)
    dataset_name = cfg.data.dataset_name

    out_subdir = output_dir / "reference_graphs"
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"reference_graphs_{dataset_name}_{ref_split}_{pgd_size}.pt"
    if out_path.exists():
        return str(out_path)

    include_h = cfg.data.get("include_hydrogens", False)

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
        include_hydrogens=include_h,
        seed=cfg.seed,
        data_root=cfg.data.get("data_root", "data"),
        use_cache=cfg.data.get("use_cache", False),
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
        data_file=cfg.data.get("data_file", None),
        min_atoms=cfg.data.get("min_atoms", 20),
        max_atoms=cfg.data.get("max_atoms", 100),
        min_rings=cfg.data.get("min_rings", 3),
        use_precomputed_smiles=cfg.data.get("use_precomputed_smiles", False),
        precomputed_smiles_dir=cfg.data.get("precomputed_smiles_dir", None),
    )
    datamodule.setup(stage="test")

    ref_size = cfg.metrics.get("reference_size", 100000)
    train_smiles = list(datamodule.train_smiles)
    if ref_split == "full" and train_smiles:
        import random
        combined = train_smiles + list(datamodule.test_smiles)
        random.Random(cfg.seed).shuffle(combined)
        reference_smiles = combined[:ref_size]
    else:
        reference_smiles = list(datamodule.test_smiles[:ref_size])
    pgd_smiles = reference_smiles[:pgd_size]

    reference_graphs = []
    for smi in tqdm(pgd_smiles, desc="Converting reference SMILES to graphs"):
        try:
            g = smiles_to_graph(smi, include_hydrogens=include_h)
            if g is not None and g.num_nodes > 0:
                reference_graphs.append(g)
        except Exception:
            continue

    torch.save(reference_graphs, out_path)
    return str(out_path)


if __name__ == "__main__":
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        overrides = sys.argv[1:idx]
    else:
        overrides = sys.argv[1:]
    if not overrides or "reference_graphs.output_dir" not in " ".join(overrides):
        print("Usage: python scripts/preprocess/precompute_reference_graphs.py experiment=NAME reference_graphs.output_dir=OUTPUT_DIR", file=sys.stderr)
        sys.exit(1)
    out_path = main_with_overrides(overrides)
    print(out_path)
