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

## Multi-GPU Parallelism

### Data Parallelism (DDP)

Splits batches across GPUs. The training budget is controlled by `target_samples_seen` in the experiment config (e.g., 1.6M for COCONUT, 16M for MOSES). With more GPUs, the effective batch size increases, so fewer steps are needed to reach the same `target_samples_seen`:

```
target_samples_seen = effective_batch_size × max_steps
effective_batch_size = batch_size_per_gpu × num_gpus × accumulate_grad_batches
```

When you pass `--ddp` or `--devices=N`, `train_benchmarks.sh` auto-adjusts:

1. **Batch size** → increased to 64/GPU (more VRAM available vs MIG)
2. **LR** → scaled by sqrt(effective_batch / base_batch)
3. **Warmup** → scaled by sqrt(effective_batch / base_batch)
4. **Steps** → NOT set by the script; `train.py` derives it automatically from `target_samples_seen / effective_batch_size`

| Setting | 1 GPU | 4 GPUs (`--devices=4`) |
|---------|-------|------------------------|
| Batch/GPU | 32 | 64 (auto) |
| Effective batch | 32 | 256 (auto) |
| LR | 6e-4 | 1.70e-3 (auto) |
| Warmup | 1000 | 2828 (auto) |
| Steps | 50,000 | 6,250 (auto, from train.py) |
| Samples seen | 1.6M | 1.6M (same) |

`train.py` validates that `target_samples_seen` can be reached and warns if `max_steps` would be too low.

```bash
# Auto-scales everything for 4 GPUs
./bash_scripts/train/train_benchmarks.sh --ddp --devices=4
```

### Task Parallelism (MIG)

Runs independent training jobs on separate MIG GPU instances. Used by `train_lr_sweep.sh` to train multiple LR configurations simultaneously (4 LRs on 4 MIG GPUs). Each job is a single-GPU run.

## Quick Reference

### Training

```bash
# Train all 8 tokenizer variants on MOSES
./bash_scripts/train/train_benchmarks.sh

# Train on COCONUT
./bash_scripts/train/train_benchmarks.sh --coconut

# Multi-GPU DDP
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

```bash
# Evaluate all trained models
./bash_scripts/eval/eval_benchmarks.sh
./bash_scripts/eval/eval_benchmarks.sh --coconut

# Evaluate specific checkpoints from a mapping file
./bash_scripts/eval/eval_benchmarks_auto.sh MAPPING.txt outputs/eval

# Core metrics only (fast: validity, uniqueness, novelty)
./bash_scripts/eval/eval_benchmarks_auto.sh MAPPING.txt outputs/eval --core-only
```

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
