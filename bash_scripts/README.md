# Bash Scripts

Batch automation for training and evaluation. All scripts support `--dry-run`, `--force`, and `--help`.

## Directory Structure

```
bash_scripts/
├── train/
│   ├── train_benchmarks.sh          # Train all tokenizer variants
│   ├── train_lr_sweep.sh            # LR sweep (task parallelism on MIG GPUs)
│   ├── precompute_benchmarks.sh     # Precompute tokenized cache (optional)
│   └── finetune_benchmarks.sh       # Fine-tune on COCONUT
└── eval/
    ├── eval_benchmarks.sh           # Evaluate pretrained models
    ├── eval_benchmarks_auto.sh      # Evaluate from a mapping file
    ├── eval_finetune_benchmarks.sh  # Evaluate fine-tuned models
    └── eval_loss_benchmarks.sh      # Compute test loss
```

## Fair Benchmarking Across Tokenizers

We benchmark 8 tokenizer variants on the same model architecture (GPT-xs). Different tokenizers produce **different sequence lengths** for the same molecules — this is the variable we're studying. Everything else must be identical for a fair comparison.

> For full details on the math, see [Reproducibility Across GPUs](../docs/designs/reproducibility_across_gpus.md).

### What must stay constant across all runs

| Parameter | Value (MOSES) | Value (COCONUT) | Why |
|-----------|---------------|-----------------|-----|
| Effective batch size ($B_{\text{eff}}$) | 32 | 32 | Gradient variance must be identical |
| Max steps | 500,000 | 50,000 | Same number of optimizer updates |
| Peak learning rate | 8.49e-4 | 6e-4 | Same optimization trajectory |
| Warmup steps | 1,414 | 1,000 | Same LR ramp (absolute, not %) |
| Seed | 42 | 42 | Reproducible init + data order |

### Using DDP to speed up training

DDP replicates the model to each GPU. Each GPU processes its own batch of $B$ samples, so the effective batch size becomes $B \times \text{GPUs} \times \text{accum}$.

To keep the comparison fair when adding GPUs, we must **hold $B_{\text{eff}}$ constant** by compensating with `accumulate_grad_batches`:

| Setup | batch/GPU | GPUs | accum | $B_{\text{eff}}$ | Wall time |
|-------|-----------|------|-------|-------------------|-----------|
| 1 GPU (baseline) | 32 | 1 | 1 | **32** | 1x |
| 2 GPUs | 16 | 2 | 1 | **32** | ~0.5x |
| 4 GPUs | 8 | 4 | 1 | **32** | ~0.25x |

The key insight: **LR, warmup, and max_steps do NOT change with GPUs.** Only `batch_size` per GPU is adjusted so the product stays constant. This means the optimization trajectory (gradient noise, LR schedule shape, total weight updates) is mathematically identical regardless of hardware.

> **Note:** DDP introduces microscopic floating-point non-determinism from asynchronous gradient reductions. This is negligible for benchmarking — downstream metric differences will be far larger.

### What NOT to use for comparison

**Do not compare validation loss across tokenizers.** Cross-entropy loss depends on vocabulary size and token distribution. A tokenizer with vocab 50 has fundamentally different loss dynamics than one with vocab 150. Always evaluate using downstream generation metrics: Validity, Uniqueness, Novelty, FCD, etc.

### Task Parallelism (MIG)

Runs independent training jobs on separate MIG GPU instances. Used by `train_lr_sweep.sh` to train multiple LR configurations simultaneously (4 LRs on 4 MIG GPUs). Each job is a single-GPU run — no DDP math involved.

## Quick Reference

### Training

```bash
# Train all 8 tokenizer variants on MOSES
./bash_scripts/train/train_benchmarks.sh

# Train on COCONUT
./bash_scripts/train/train_benchmarks.sh --coconut

# Multi-GPU DDP (keeps effective batch size constant)
./bash_scripts/train/train_benchmarks.sh --ddp              # 4 GPUs (default)
./bash_scripts/train/train_benchmarks.sh --devices=2         # 2 GPUs

# Only MC variants (skip SC/HAC)
./bash_scripts/train/train_benchmarks.sh --skip-sc-hac

# LR sweep (task parallelism on 4 MIG GPUs)
./bash_scripts/train/train_lr_sweep.sh --sc --coconut

# Precompute tokenized cache (optional, speeds up SC/HAC training)
./bash_scripts/train/precompute_benchmarks.sh
```

### Evaluation

Use the evaluation environment for all eval scripts:

```bash
conda activate mosaic-eval
```

```bash
# Evaluate all trained models
./bash_scripts/eval/eval_benchmarks.sh
./bash_scripts/eval/eval_benchmarks.sh --coconut

# Evaluate specific checkpoints from a mapping file
./bash_scripts/eval/eval_benchmarks_auto.sh MAPPING.txt outputs/eval

# Core metrics only (fast: validity, uniqueness, novelty)
./bash_scripts/eval/eval_benchmarks_auto.sh MAPPING.txt outputs/eval --core-only

# Two-phase evaluation with parallel motif metrics
./bash_scripts/eval/eval_benchmarks_2phase.sh            # MOSES
./bash_scripts/eval/eval_benchmarks_2phase.sh --coconut  # COCONUT
```

#### `eval/eval_benchmarks_2phase.sh`

Two-phase evaluation script that separates GPU-bound work from CPU-bound motif metrics, and preserves the original comparison chart behavior:
The primary goal is to **speed up end-to-end evaluation wall time** when you have one GPU but many CPU cores.

- **Phase 1 (GPU, sequential)**:
  - Runs `scripts/test.py` *without motif metrics* for each checkpoint.
  - Computes core metrics, FCD, PGD, and saves:
    - `generated_smiles.txt`
    - `generated_metadata.json` (attempted + valid counts)
    - `generated_graphs.pt` (for PGD)
- **Phase 2 (CPU, parallel)**:
  - Runs `scripts/test.py` in motif-only + `metrics_only` mode for each checkpoint.
  - Each run is launched in its own detached `/usr/bin/screen` session.
- **Phase 3 (Realistic generation + chart)**:
  - Optionally runs `scripts/realistic_gen.py` for each checkpoint.
  - Calls `scripts/comparison/compare_results.py` to regenerate the comparison image.

Common flags:

- `--coconut`: switch to COCONUT benchmarks (`outputs/benchmark_coconut`, `test_coconut*` trees).
- `--full-ref`: full-reference metrics (`reference_split=full`).
- `--force`: recompute even when `results.json` already exists.
- `--reuse-generated`:
  - Phase 1 uses existing `generated_smiles.txt` (and `generated_metadata.json`) in `metrics_only` mode.
  - Phase 3 reuses the same SMILES for `realistic_gen.py` (no generation).
- `--phase1`, `--phase2`, `--phase3`:
  - Restrict which phases to run; can be combined (e.g., `--phase2 --phase3`).
- `--test-only`, `--gen-only`: behave like `eval_benchmarks.sh` (skip realistic_gen or test, respectively).

> **Note:** Reuse modes require the corresponding artifacts (`generated_smiles.txt`, `generated_metadata.json`, and ideally `generated_graphs.pt`) to be present in the test output directory for each run.

### Tokenizer Variants

8 variants are trained by default:

| Variant | Tokenizer | Coarsening |
|---------|-----------|------------|
| SENT | sent | none (baseline) |
| HSENT+MC | hsent | motif_community |
| HSENT+SC | hsent | spectral |
| HSENT+HAC | hsent | hac |
| HDT+MC | hdt | motif_community |
| HDT+SC | hdt | spectral |
| HDT+HAC | hdt | hac |
| HDTC | hdtc | none (functional hierarchy) |

With `--skip-sc-hac`: only SENT, HSENT+MC, HDT+MC, HDTC.

### Output Structure

```
outputs/
├── benchmark/              # MOSES training checkpoints
├── benchmark_coconut/      # COCONUT training checkpoints
├── test/                   # test.py evaluation results
├── realistic_gen/          # realistic_gen.py results
└── lr_sweep/               # LR sweep checkpoints + eval
```
