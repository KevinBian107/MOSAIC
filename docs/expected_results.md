# Expected Results

This document describes what each benchmark script produces and the expected output ranges. These are **placeholder targets** — actual numbers depend on hardware, random seeds, and training duration.

## Training Scripts

### `bash_scripts/train/train_benchmarks.sh`

Trains 8 tokenizer variants on MOSES (default) or COCONUT (`--coconut`).

**Variants trained:** SENT, HSENT+MC, HSENT+SC, HSENT+HAC, HDT+MC, HDT+SC, HDT+HAC, HDTC

**Output directory:** `outputs/benchmark/` (MOSES) or `outputs/benchmark_coconut/` (COCONUT)

**Expected output per variant:**
```
outputs/benchmark/<variant>_<timestamp>/
├── best.ckpt          # Best checkpoint (by validation loss)
├── last.ckpt          # Final checkpoint
├── config.yaml        # Resolved Hydra config
└── wandb/             # WandB run logs
```

**Expected training time (single GPU, MOSES):**

| Variant | Approx. wall time | Notes |
|---------|-------------------|-------|
| SENT | ~8–12 h | Fastest tokenizer (no coarsening) |
| HSENT+MC | ~10–14 h | Motif community coarsening |
| HSENT+SC | ~12–18 h | Spectral coarsening (slower tokenization) |
| HSENT+HAC | ~12–18 h | HAC coarsening |
| HDT+MC | ~10–14 h | Shorter sequences than HSENT |
| HDT+SC | ~12–18 h | Spectral coarsening |
| HDT+HAC | ~12–18 h | HAC coarsening |
| HDTC | ~10–14 h | Functional hierarchy (deterministic) |

**COCONUT training** is faster (~2–4 h per variant) due to smaller dataset (5K molecules, 50K steps).

---

### `bash_scripts/train/train_lr_sweep.sh`

Runs learning rate grid search via task parallelism on MIG GPUs.

**Output directory:** `outputs/lr_sweep/`

**Expected output:** One checkpoint directory per (tokenizer, LR) combination.

---

### `bash_scripts/train/precompute_benchmarks.sh`

Pre-tokenizes datasets for spectral/HAC variants to speed up training.

**Output directory:** `data/cache/`

**Expected output:** `.pt` cache files (~100 MB–1 GB each depending on dataset size and tokenizer).

---

### `bash_scripts/train/finetune_benchmarks.sh`

Fine-tunes MOSES-pretrained models on COCONUT.

**Output directory:** `outputs/finetune_coconut/`

**Expected output:** Same structure as `train_benchmarks.sh` but starting from pretrained checkpoints.

---

## Evaluation Scripts

### `bash_scripts/eval/eval_benchmarks.sh`

Full evaluation of all trained models using `scripts/test.py`.

**Output directory:** `outputs/test/` (MOSES) or `outputs/test_coconut/` (COCONUT)

**Expected output per variant:**
```
outputs/test/<variant>_<timestamp>/
├── results.json           # All computed metrics
├── generated_smiles.txt   # Generated SMILES strings
├── generated_metadata.json
├── generated_graphs.pt    # For PGD computation
└── visualization/         # Molecule grid images
```

**Expected metric ranges (MOSES, 1000 samples):**

| Metric | SENT | HSENT+MC | HDT+MC | HDTC | Notes |
|--------|------|----------|--------|------|-------|
| Validity (%) | 85–95 | 80–92 | 75–90 | 80–92 | Fraction of valid SMILES |
| Uniqueness (%) | 95–100 | 95–100 | 95–100 | 95–100 | Among valid molecules |
| Novelty (%) | 95–100 | 95–100 | 95–100 | 95–100 | Not in training set |
| FCD | 0.5–3.0 | 0.5–4.0 | 0.5–4.0 | 0.5–3.5 | Lower is better |

**Expected metric ranges (COCONUT, 1000 samples):**

| Metric | SENT | HSENT+MC | HDT+MC | HDTC | Notes |
|--------|------|----------|--------|------|-------|
| Validity (%) | 40–70 | 30–60 | 20–50 | 30–60 | Harder dataset |
| Uniqueness (%) | 90–100 | 90–100 | 90–100 | 90–100 | Among valid molecules |
| Novelty (%) | 95–100 | 95–100 | 95–100 | 95–100 | Not in training set |
| FCD | 2.0–8.0 | 2.0–10.0 | 2.0–10.0 | 2.0–8.0 | Lower is better |

> **Note:** COCONUT validity is significantly lower than MOSES due to larger, more complex molecules with stricter valence constraints.

---

### `bash_scripts/eval/eval_benchmarks_2phase.sh`

Two-phase evaluation separating GPU and CPU work for faster wall time.

**Phase 1 (GPU, sequential):** Core metrics + FCD + PGD. Saves `generated_smiles.txt`, `generated_metadata.json`, `generated_graphs.pt`.

**Phase 2 (CPU, parallel):** Motif distribution metrics (MSC, MFD) in parallel screen sessions.

**Phase 3:** Realistic generation analysis + comparison chart via `compare_results.py`.

**Expected output:**
```
outputs/test/<variant>/
├── results.json               # Phase 1 metrics
├── results_motif.json         # Phase 2 motif metrics
├── generated_smiles.txt
├── generated_metadata.json
└── generated_graphs.pt
outputs/realistic_gen/<variant>/
├── results.json               # Realistic generation analysis
└── figures/                   # Distribution comparison plots
```

---

### `bash_scripts/eval/eval_loss_benchmarks.sh`

Computes cross-entropy test loss for all trained models.

**Output:** JSON files with per-model test loss.

**Expected loss ranges:**

| Variant | MOSES test loss | COCONUT test loss | Notes |
|---------|----------------|-------------------|-------|
| SENT | 1.0–2.0 | 1.5–3.0 | Flat tokenization (larger vocab) |
| HSENT | 0.8–1.8 | 1.2–2.5 | Hierarchical |
| HDT | 0.8–1.8 | 1.2–2.5 | Compact hierarchy |
| HDTC | 0.8–1.8 | 1.2–2.5 | Compositional |

> **Caution:** Do not compare loss across tokenizers — vocabulary size and token distribution differ. Use generation metrics instead.

---

### `bash_scripts/eval/eval_finetune_benchmarks.sh`

Evaluates fine-tuned (MOSES to COCONUT) models. Same output structure and metrics as `eval_benchmarks.sh`.

---

## Comparison & Visualization

### `scripts/comparison/compare_results.py`

Aggregates `results.json` files from all variants and produces a comparison table.

**Expected output:** Console table + optional JSON/LaTeX summary comparing all tokenizers side by side.

### `scripts/visualization/generation_gallery.py`

Generates paper-quality molecule grids for each tokenizer.

**Expected output:** PNG images in `figures/` showing representative generated molecules.

---

## Interpreting Results

- **Validity** is the primary quality indicator. Higher is better.
- **Uniqueness** close to 100% means the model isn't mode-collapsing.
- **Novelty** close to 100% means the model isn't memorizing training data.
- **FCD** measures distributional similarity to the reference set. Values below 1.0 are excellent for MOSES; below 5.0 is reasonable for COCONUT.
- **PGD** (PolyGraph Discrepancy) is a classifier-based metric. Scores near 0.5 mean generated graphs are indistinguishable from real ones; scores near 1.0 mean they're easily distinguishable.
- **Motif metrics** (MSC, MFD) measure preservation of functional group distributions. Lower divergence is better.
