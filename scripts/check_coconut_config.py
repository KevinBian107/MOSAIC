#!/usr/bin/env python
"""Check resolved config for experiment=coconut (no training, no Hydra).

Merges configs in the same order as train.yaml defaults:
  1. tokenizer/hdtc
  2. _self_ (train.yaml)
  3. experiment/coconut  (last = wins for trainer.target_samples_seen)

Run from project root: python scripts/check_coconut_config.py
"""
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    # Drop Hydra-only keys so we only have content
    data.pop("defaults", None)
    data.pop("# @package _global_", None)
    return data or {}


def main():
    root = Path(__file__).resolve().parent.parent
    config_dir = root / "configs"

    # Load in Hydra defaults order: tokenizer, _self_, experiment
    tokenizer = load_yaml(config_dir / "tokenizer" / "hdtc.yaml")
    train = load_yaml(config_dir / "train.yaml")
    coconut = load_yaml(config_dir / "experiment" / "coconut.yaml")

    # Merge: later overrides earlier (same order as Hydra defaults in train.yaml)
    cfg = deep_merge(tokenizer, train)
    cfg = deep_merge(cfg, coconut)

    # Simulate CLI override: data.batch_size=48
    if "data" not in cfg:
        cfg["data"] = {}
    cfg["data"]["batch_size"] = 48

    trainer = cfg.get("trainer", {})
    target = trainer.get("target_samples_seen")
    batch_size = cfg.get("data", {}).get("batch_size", 32)
    devices = trainer.get("devices", 1)
    if isinstance(devices, list):
        devices = len(devices) or 1
    accum = trainer.get("accumulate_grad_batches", 1)
    effective_batch = batch_size * devices * accum
    derived_steps = (target + effective_batch - 1) // effective_batch if target else None

    print("Resolved config (experiment=coconut, data.batch_size=48):")
    print("  data.batch_size:", batch_size)
    print("  trainer.target_samples_seen:", target)
    print("  trainer.max_steps (fallback):", trainer.get("max_steps"))
    print("  effective_batch_size:", effective_batch)
    print("  derived max_steps (ceil(target/effective_batch)):", derived_steps)
    print()

    # Test: coconut should override to 1.6M
    assert target == 1_600_000, (
        f"Expected trainer.target_samples_seen=1600000 (coconut override), got {target}"
    )
    print("OK: trainer.target_samples_seen is 1_600_000 (coconut override applied).")


if __name__ == "__main__":
    main()
