# Bash Scripts

Utility scripts for running batch operations.

## Workflow

```
0. precompute_benchmarks.sh       → Precompute tokenized cache (optional, speeds up training)
    stop_precompute_benchmarks.sh → Stop precompute screen sessions (cancel running jobs)
1. train_benchmarks.sh            → Pretrain on MOSES (or COCONUT)
    train_lr_sweep.sh             → LR sweep for SC/HAC coarsening (4 LRs × 2 tokenizers, MIG parallel)
2. eval_benchmarks.sh             → Evaluate pretrained models (discovers checkpoints automatically)
    eval_benchmarks_auto.sh       → Evaluate checkpoints from a mapping file (with optional caching, precomputed SMILES, reference graphs)
3. finetune_benchmarks.sh         → Fine-tune on COCONUT (transfer learning)
4. eval_finetune_benchmarks.sh    → Evaluate fine-tuned models
```

See also [docs/commands_reference.md](../docs/commands_reference.md) for GCP, DSMLP, setup, precompute, and eval command examples.

---

## precompute_benchmarks.sh

Precomputes tokenized cache files for hierarchical tokenizers (H-SENT, HDT) with SC and HAC coarsening. This is **optional** but speeds up training startup significantly for large datasets.

Only SC (spectral clustering) and HAC (hierarchical agglomerative clustering) benefit from precomputation. Other coarsening methods (MC, MAS) are fast enough to tokenize on-the-fly during training.

### What it does

1. For each tokenizer + coarsening combo, runs `preprocess_chunk.py` to tokenize molecules
2. For MOSES (1M samples): launches parallel **screen** sessions for chunked processing
3. For COCONUT (5K samples): runs directly in foreground (takes ~4 minutes per combo)
4. Combines chunks into a single cache file per combo
5. Cache files are used by training with `data.use_cache=true`
6. Optionally uses **precomputed SMILES** (`--use-precomputed-smiles`) so chunks read from `data/moses_smiles/moses_smiles.txt` instead of CSV (run `python scripts/preprocess/export_moses_smiles.py` first)

### Usage

```bash
# Precompute all combos for MOSES (default, uses screen sessions)
./bash_scripts/precompute_benchmarks.sh

# Precompute all combos for COCONUT (runs directly, ~16 min total)
./bash_scripts/precompute_benchmarks.sh --coconut

# Precompute both MOSES and COCONUT
./bash_scripts/precompute_benchmarks.sh --all

# Single tokenizer + coarsening combo
./bash_scripts/precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc

# Custom train/val sample counts and chunks
./bash_scripts/precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc --train-samples=50000 --val-samples=1000 --chunks=4

# Use precomputed SMILES file (faster; run scripts/preprocess/export_moses_smiles.py first)
./bash_scripts/precompute_benchmarks.sh --use-precomputed-smiles --tokenizer=hsent --coarsening=sc

# Dry run (show commands without executing)
./bash_scripts/precompute_benchmarks.sh --dry-run

# Re-run even if cache files exist
./bash_scripts/precompute_benchmarks.sh --force

# Show help
./bash_scripts/precompute_benchmarks.sh --help
```

### Options

| Option | Description |
|--------|-------------|
| `--coconut` | Precompute COCONUT instead of MOSES |
| `--all` | Precompute both MOSES and COCONUT |
| `--coarsening=STRATEGY` | Filter: `sc`, `hac`, or `all` (default: `all`) |
| `--tokenizer=TYPE` | Filter: `hsent`, `hdt`, or `all` (default: `all`) |
| `--chunks=N` | Number of parallel chunks for MOSES (default: 8) |
| `--train-samples=N` | MOSES training samples (default: 1000000) |
| `--val-samples=N` | MOSES validation samples (default: 0) |
| `--spectral-n-init=N` | Spectral clustering n_init for SC only (default: 1 for faster precompute) |
| `--spectral-k-min-factor=F` | Spectral k_min_factor for SC (default: 0.9) |
| `--spectral-k-max-factor=F` | Spectral k_max_factor for SC (default: 1.1) |
| `--use-precomputed-smiles` | Read SMILES from `data/moses_smiles/moses_smiles.txt` instead of CSV |
| `--precomputed-smiles-dir=PATH` | Directory with moses_smiles.txt (default: data/moses_smiles) |
| `--output-dir=PATH` | Cache directory (default: `data/cache`) |
| `--dry-run` | Show commands without executing |
| `--force` | Re-run even if cache files exist |

### Combos precomputed (default: all 4)

```
hsent:sc   hsent:hac
hdt:sc     hdt:hac
```

### Output

Cache files are saved to `data/cache/` with the naming pattern:
```
{dataset}_{split}_{tokenizer}_{num_samples}_{config_hash}.pt
```

The config hash includes spectral parameters (`n_init`, `k_min_factor`, `k_max_factor`) so different spectral settings get different cache files. Use in training: `python scripts/train.py ... data.use_cache=true`

---

## stop_precompute_benchmarks.sh

Stops (kills) all **screen** sessions started by `precompute_benchmarks.sh`. Use this to cancel running precompute jobs.

### Usage

```bash
# Stop H-SENT + spectral precompute screens (default 8 chunks)
./bash_scripts/stop_precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc

# Stop HDT + spectral with custom chunks
./bash_scripts/stop_precompute_benchmarks.sh --tokenizer=hdt --coarsening=sc --chunks=4

# Stop all MOSES + COCONUT precompute screens
./bash_scripts/stop_precompute_benchmarks.sh --all

./bash_scripts/stop_precompute_benchmarks.sh --help
```

Options mirror the relevant subset of `precompute_benchmarks.sh` (`--tokenizer`, `--coarsening`, `--chunks`, `--coconut`, `--all`).

---

## eval_benchmarks_auto.sh

Evaluates a **list of checkpoints** from a **mapping file**, with optional result caching, precomputed SMILES, and precomputed PGD reference graphs. Use this when you want to compare specific runs in a fixed column order and avoid re-running test/gen when results already exist.

### Mapping file format

One line per checkpoint; order of lines = column order in the comparison table. Optional second column (space-separated) is the display label.

```
directory_name
directory_name    Display Label
# comments and empty lines are skipped
```

Checkpoints are looked for under `BENCHMARK_DIR` (default `outputs/benchmark`) as `BENCHMARK_DIR/<directory_name>/best.ckpt` or `last.ckpt`.

### What it does

1. Reads mapping file and finds each checkpoint under `BENCHMARK_DIR`
2. Optionally runs `scripts/preprocess/precompute_reference_graphs.py` once and passes the `.pt` path to every `test.py` so PGD reference graphs are not reconverted for each run
3. For each checkpoint: runs `test.py` (and optionally `realistic_gen.py`) unless results already exist (skip if `results.json` and `generated_smiles.txt` exist)
4. Runs `scripts/comparison/compare_results.py` to produce a comparison table image

### Usage

```bash
./bash_scripts/eval_benchmarks_auto.sh MAPPING_FILE OUTPUT_PATH [OPTIONS]

# Example: evaluate checkpoints listed in compare_ckpts.txt, write to outputs/eval, use last.ckpt, core metrics only, precomputed SMILES
./bash_scripts/eval_benchmarks_auto.sh \
    outputs/benchmark/compare_ckpts.txt \
    outputs/eval \
    --last \
    --core-only \
    --use-precomputed-smiles
```

### Options

| Option | Description |
|--------|-------------|
| `--last` | Use `last.ckpt` instead of `best.ckpt` |
| `--best` | Use `best.ckpt` (default) |
| `--benchmark-dir DIR` | Directory to search for checkpoints (default: outputs/benchmark) |
| `--dataset NAME` | moses or coconut (default: moses) |
| `--test-only` | Only run test.py (skip realistic_gen.py) |
| `--gen-only` | Only run realistic_gen.py (skip test.py) |
| `--core-only` | Only core metrics: validity, uniqueness, novelty (no FCD, PGD, motif, realistic_gen) |
| `--use-precomputed-smiles` | Pass data.use_precomputed_smiles=true and use precomputed SMILES when precomputing reference graphs |
| `--recompute DIR[,DIR2,...]` | Force re-run test and/or gen for these directory names (still included in table) |
| `-h`, `--help` | Show help |

### Output

- Test outputs: `OUTPUT_PATH/test/<directory_name>/results.json`, `generated_smiles.txt`
- Realistic gen: `OUTPUT_PATH/realistic_gen/<directory_name>/...`
- Comparison table: `OUTPUT_PATH/comparison.png`

---

## train_lr_sweep.sh

Finds good learning rates for SC (spectral clustering) and HAC coarsening strategies before committing to long training runs. Trains HSENT and HDT at 4 learning rates (6e-4, 1e-3, 2e-3, 4e-3) using **task parallelism** on 4 MIG GPUs, then evaluates and generates comparison tables.

### What it does

1. Detects MIG GPUs (4 required for parallel execution, falls back to sequential)
2. Precomputes tokenized data caches for needed tokenizer+coarsening combos
3. For each coarsening (SC, HAC, or both):
   a. Trains HSENT at 4 LRs on 4 MIG GPUs in parallel
   b. Trains HDT at 4 LRs on 4 MIG GPUs in parallel
   c. Evaluates all 8 models sequentially
   d. Generates a comparison table PNG
4. Prints summary of all runs

Each model trains with `accumulate_grad_batches=2` and `batch_size=32`, giving an effective batch of 64. With 25K optimizer steps, this equals 50K equivalent base steps.

### Usage

```bash
# Both SC+HAC on COCONUT (default)
./bash_scripts/train_lr_sweep.sh

# SC only on COCONUT
./bash_scripts/train_lr_sweep.sh --sc --coconut

# HAC only on MOSES
./bash_scripts/train_lr_sweep.sh --hac --moses

# Dry run (show commands without executing)
./bash_scripts/train_lr_sweep.sh --sc --coconut --dry-run

# Skip training, only evaluate existing checkpoints
./bash_scripts/train_lr_sweep.sh --eval-only

# Train without evaluation
./bash_scripts/train_lr_sweep.sh --sc --no-eval

# Re-run even if checkpoints exist
./bash_scripts/train_lr_sweep.sh --force

# Custom step count
./bash_scripts/train_lr_sweep.sh --steps=50000
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sc` | | Only SC coarsening |
| `--hac` | | Only HAC coarsening |
| `--both` | default | Both SC and HAC |
| `--coconut` | default | Train on COCONUT |
| `--moses` | | Train on MOSES |
| `--dry-run` | | Show commands without executing |
| `--no-wandb` | | Disable WandB logging |
| `--eval-only` | | Skip training, only evaluate |
| `--no-eval` | | Skip evaluation |
| `--force` | | Re-run even if checkpoints exist |
| `--steps=N` | 25000 | Optimizer steps (with accum=2, equivalent to 2N base steps) |

### Sweep grid

2 tokenizers (HSENT, HDT) x 4 LRs (6e-4, 1e-3, 2e-3, 4e-3) = 8 models per coarsening type.

### Output

```
outputs/lr_sweep/
    coconut_hsent_sc_lr6e-4_20260222-.../best.ckpt
    coconut_hsent_sc_lr1e-3_20260222-.../best.ckpt
    ...
outputs/lr_sweep_eval/
    coconut_sc/
        coconut_hsent_sc_lr6e-4/results.json
        ...
        comparison_sc.png
    coconut_hac/
        coconut_hsent_hac_lr6e-4/results.json
        ...
        comparison_hac.png
```

---

## train_benchmarks.sh

Trains all tokenizer variants from scratch on MOSES or COCONUT dataset.

### What it does

1. Trains all tokenizer configurations (SENT, H-SENT, HDT, HDTC) with all coarsening variants
2. By default trains all 8 variants: MC, SC, and HAC coarsening for hierarchical tokenizers
3. Optionally skips SC and HAC variants with `--skip-sc-hac` (MC only)
4. Saves checkpoints to `outputs/benchmark/` (MOSES) or `outputs/benchmark_coconut/` (COCONUT)

### Usage

```bash
# Train all tokenizers on MOSES (default, 500K steps)
./bash_scripts/train_benchmarks.sh

# Train on COCONUT instead (50K steps)
./bash_scripts/train_benchmarks.sh --coconut

# Only train MC variants (skip SC and HAC, faster)
./bash_scripts/train_benchmarks.sh --skip-sc-hac

# Dry run (show commands without executing)
./bash_scripts/train_benchmarks.sh --dry-run

# Skip already completed models
./bash_scripts/train_benchmarks.sh --force  # Re-run even if exists

# Custom training steps
./bash_scripts/train_benchmarks.sh --steps=100000

# Disable WandB logging
./bash_scripts/train_benchmarks.sh --no-wandb

# Show help
./bash_scripts/train_benchmarks.sh --help
```

### Tokenizers trained

**Default (all 8 variants):**
- `SENT` - Flat sequential tokenizer (baseline)
- `H-SENT + MC` - Hierarchical SENT with motif community coarsening
- `H-SENT + SC` - Hierarchical SENT with spectral clustering
- `H-SENT + HAC` - Hierarchical SENT with hierarchical agglomerative clustering
- `HDT + MC` - Hierarchical DFS with motif community coarsening
- `HDT + SC` - Hierarchical DFS with spectral clustering
- `HDT + HAC` - Hierarchical DFS with hierarchical agglomerative clustering
- `HDTC` - Compositional (uses functional hierarchy, no coarsening needed)

**With `--skip-sc-hac` (4 variants):**
- `SENT`, `H-SENT + MC`, `HDT + MC`, `HDTC`

### Per-tokenizer max sequence length

The script automatically sets `sampling.max_length` per tokenizer and dataset to size the position embedding table appropriately. Values were derived from tokenization stats on 1000 samples with ~15% buffer, rounded to multiples of 128.

**MOSES** (molecules have 10-26 nodes):

| Tokenizer | Observed Max | Recommended `max_length` |
|-----------|-------------|--------------------------|
| SENT      | 121         | 128                      |
| HSENT_MC  | 358         | 384                      |
| HSENT_SC  | 375         | 384                      |
| HSENT_HAC | 474         | 512                      |
| HDT_MC    | 232         | 256                      |
| HDT_SC    | 234         | 256                      |
| HDT_HAC   | 272         | 384                      |
| HDTC      | 308         | 384                      |

**COCONUT** (molecules have 20-100 nodes):

| Tokenizer | Observed Max | Recommended `max_length` |
|-----------|-------------|--------------------------|
| SENT      | 433         | 512                      |
| HSENT_MC  | 1337        | 1536                     |
| HSENT_SC  | 1413        | 1536                     |
| HSENT_HAC | 1736        | 2048                     |
| HDT_MC    | 868         | 1024                     |
| HDT_SC    | 840         | 1024                     |
| HDT_HAC   | 1010        | 1280                     |
| HDTC      | 1180        | 1536                     |

To regenerate these stats: `python scripts/comparison/compare_tokenization_stats.py --dataset both --num-samples 1000`

### Output

**MOSES training:**
- `outputs/benchmark/moses_{tokenizer}_{coarsening}_{timestamp}/best.ckpt`
- `outputs/benchmark/moses_{tokenizer}_{coarsening}_{timestamp}/last.ckpt`

**COCONUT training:**
- `outputs/benchmark_coconut/coconut_{tokenizer}_{coarsening}_{timestamp}/best.ckpt`
- `outputs/benchmark_coconut/coconut_{tokenizer}_{coarsening}_{timestamp}/last.ckpt`

---

## finetune_benchmarks.sh

Fine-tunes all pretrained models in `outputs/benchmark/` on the COCONUT complex natural products dataset for transfer learning evaluation.

### What it does

1. Finds all `best.ckpt` files in `outputs/benchmark/`
2. Extracts tokenizer type and coarsening strategy from directory names
3. Runs `scripts/finetune.py` on each checkpoint with appropriate settings
4. Saves fine-tuned models to `outputs/finetune/` (or `outputs/finetune_fewshot/` in few-shot mode)

### Usage

```bash
# Full fine-tuning (5000 training samples, 50000 steps)
./bash_scripts/finetune_benchmarks.sh

# Few-shot fine-tuning (200 training samples, 10000 steps)
./bash_scripts/finetune_benchmarks.sh --few-shot

# Few-shot with custom sample size
./bash_scripts/finetune_benchmarks.sh --few-shot=500

# Dry run (show commands without executing)
./bash_scripts/finetune_benchmarks.sh --dry-run

# Skip already completed checkpoints
./bash_scripts/finetune_benchmarks.sh --force  # Re-run even if exists

# Skip spectral clustering and HAC models
./bash_scripts/finetune_benchmarks.sh --skip-sc-hac

# Disable WandB logging
./bash_scripts/finetune_benchmarks.sh --no-wandb

# Custom number of training steps (default: 50000 full, 10000 few-shot)
./bash_scripts/finetune_benchmarks.sh --steps=100000

# Show help
./bash_scripts/finetune_benchmarks.sh --help
```

### Output

**Full mode** - Fine-tuned models are saved to:
- `outputs/finetune/coconut_{tokenizer}_{coarsening}/best.ckpt`
- `outputs/finetune/coconut_{tokenizer}_{coarsening}/last.ckpt`
- `outputs/finetune/coconut_{tokenizer}_{coarsening}/config.yaml`

**Few-shot mode** - Fine-tuned models are saved to:
- `outputs/finetune_fewshot/coconut_{tokenizer}_{coarsening}/best.ckpt`
- `outputs/finetune_fewshot/coconut_{tokenizer}_{coarsening}/last.ckpt`
- `outputs/finetune_fewshot/coconut_{tokenizer}_{coarsening}/config.yaml`

### Few-shot Transfer Learning

Few-shot mode tests model retention under data scarcity. With limited fine-tuning data:
- Flat tokenizers (SENT) tend to overfit and lose pretrained knowledge
- Hierarchical tokenizers (H-SENT, HDT, HDTC) retain more MOSES patterns

This demonstrates the memory retention benefits of hierarchical representations.

### Prerequisites

Before running, ensure you have prepared the COCONUT dataset:

```bash
# Download and prepare COCONUT data (~191MB download)
python scripts/preprocess/prepare_coconut_data.py
```

This creates `data/coconut_complex.smi` with ~10,000 complex natural products.

---

## eval_benchmarks.sh

Evaluates all checkpoints in `outputs/benchmark/` (or `outputs/benchmark_coconut/`) and generates a comparison table.

### What it does

1. Finds all `last.ckpt` files in the benchmark directory
2. Runs `scripts/test.py` on each checkpoint (computes validity, uniqueness, novelty, FCD, PGD, etc.)
3. Runs `scripts/realistic_gen.py` on each checkpoint (computes motif rate, substitution patterns, etc.)
4. Runs `scripts/comparison/compare_results.py` to generate a comparison table PNG

### Usage

```bash
# Evaluate MOSES benchmarks (default)
./bash_scripts/eval_benchmarks.sh

# Evaluate COCONUT benchmarks
./bash_scripts/eval_benchmarks.sh --coconut

# Only run test.py (skip realistic generation analysis)
./bash_scripts/eval_benchmarks.sh --test-only

# Only run realistic_gen.py (skip standard test metrics)
./bash_scripts/eval_benchmarks.sh --gen-only

# Show help
./bash_scripts/eval_benchmarks.sh --help
```

### Output

**MOSES (default):**
- `outputs/test/{run_name}/results.json` - Test metrics
- `outputs/realistic_gen/{run_name}/results.json` - Realistic generation metrics
- `outputs/test/comparison.png` - Comparison table image

**COCONUT (`--coconut`):**
- `outputs/test_coconut/{run_name}/results.json` - Test metrics
- `outputs/realistic_gen_coconut/{run_name}/results.json` - Realistic generation metrics
- `outputs/test_coconut/comparison.png` - Comparison table image

### Adding checkpoints

Place your trained checkpoints in `outputs/benchmark/`. The script looks for `best.ckpt` files:

```
outputs/benchmark/
├── my_model_1/
│   └── best.ckpt
├── my_model_2/
│   └── best.ckpt
└── ...
```

### Naming convention

The script automatically extracts tokenizer type and coarsening strategy from directory names.

**Directory name pattern:** `{dataset}_{tokenizer}_{coarsening}_{other}_...`

**Tokenizer types:**
- `hsent` - Hierarchical SENT (supports coarsening)
- `hdt` - Hierarchical DFS-based Tokenizer (supports coarsening)
- `hdtc` - HDT Compositional (no coarsening, uses FunctionalHierarchyBuilder)
- `sent` - Flat SENT (no coarsening)

**Coarsening strategies** (only for `hsent` and `hdt`):
- `mc` → `motif_community` - Direct motif-based community assignment
- `sc` → `spectral` - Standard spectral clustering (default)
- `hac` → `hac` - Hierarchical agglomerative clustering
- `mas` → `motif_aware_spectral` - Spectral clustering with motif preservation

**Examples:**
```
moses_hsent_mc_n100000_20260126/   → tokenizer=hsent, coarsening=motif_community
moses_hdt_sc_n100000_20260126/    → tokenizer=hdt, coarsening=spectral
moses_hsent_hac_n100000_20260126/ → tokenizer=hsent, coarsening=hac
moses_hsent_mas_n100000_20260126/ → tokenizer=hsent, coarsening=motif_aware_spectral
moses_hdtc_n100000_20260126/      → tokenizer=hdtc (no coarsening)
moses_sent_n100000_20260126/      → tokenizer=sent (no coarsening)
```

---

## eval_finetune_benchmarks.sh

Evaluates fine-tuned checkpoints and generates a comparison table showing transfer learning performance.

### What it does

1. Finds all `best.ckpt` files in `outputs/finetune/` (or `outputs/finetune_fewshot/`)
2. Runs `scripts/eval_finetune.py` on each checkpoint
3. Computes transfer performance metrics (FCD, SNN, fragment/scaffold similarity)
4. Computes adaptation metrics (vs COCONUT distribution)
5. Computes retention metrics (vs MOSES distribution)
6. Generates comparison table using `scripts/comparison/compare_finetune_results.py`

### Usage

```bash
# Evaluate full fine-tuned models
./bash_scripts/eval_finetune_benchmarks.sh

# Evaluate few-shot fine-tuned models
./bash_scripts/eval_finetune_benchmarks.sh --few-shot

# Dry run (show commands without executing)
./bash_scripts/eval_finetune_benchmarks.sh --dry-run

# Just regenerate comparison table (skip evaluation)
./bash_scripts/eval_finetune_benchmarks.sh --compare-only
./bash_scripts/eval_finetune_benchmarks.sh --few-shot --compare-only

# Custom generation settings
./bash_scripts/eval_finetune_benchmarks.sh --samples=500 --reference=500

# Show help
./bash_scripts/eval_finetune_benchmarks.sh --help
```

### Output

**Full mode:**
- `outputs/eval_finetune/*/evaluation_results.json` - Evaluation metrics
- `outputs/finetune/comparison.png` - Comparison table image

**Few-shot mode:**
- `outputs/eval_finetune_fewshot/*/evaluation_results.json` - Evaluation metrics
- `outputs/finetune_fewshot/comparison.png` - Comparison table image

### Metrics

The comparison table includes:

**Training Info:**
- Tokenizer type
- Coarsening strategy
- Training steps

**Transfer Performance:**
- Validity, FCD, SNN
- Fragment similarity, Scaffold similarity
- Internal diversity

**Adaptation (COCONUT):**
- Motif histogram KL divergence
- Ring count KL, Scaffold retention
- Atom type KL, Functional group KL

**Retention (MOSES):**
- Same metrics computed against MOSES distribution
- Lower values = better retention of pretrained knowledge
