# Bash Scripts

Utility scripts for running batch operations.

## Workflow

```
0. precompute_benchmarks.sh       → Precompute tokenized cache (optional, speeds up training)
1. train_benchmarks.sh            → Pretrain on MOSES (or COCONUT)
2. eval_benchmarks.sh             → Evaluate pretrained models
3. finetune_benchmarks.sh         → Fine-tune on COCONUT (transfer learning)
4. eval_finetune_benchmarks.sh    → Evaluate fine-tuned models
```

---

## precompute_benchmarks.sh

Precomputes tokenized cache files for hierarchical tokenizers (H-SENT, HDT) with all coarsening strategies. This is **optional** but speeds up training startup significantly for large datasets.

### What it does

1. For each tokenizer + coarsening combo, runs `preprocess_chunk.py` to tokenize molecules
2. For MOSES (1M samples): launches parallel screen sessions for chunked processing
3. For COCONUT (5K samples): runs directly in foreground (takes ~4 minutes per combo)
4. Combines chunks into a single cache file per combo
5. Cache files are used by training with `data.use_cache=true`

### Usage

```bash
# Precompute all combos for MOSES (default, uses screen sessions)
./bash_scripts/precompute_benchmarks.sh

# Precompute all combos for COCONUT (runs directly, ~24 min total)
./bash_scripts/precompute_benchmarks.sh --coconut

# Precompute both MOSES and COCONUT
./bash_scripts/precompute_benchmarks.sh --all

# Only MC variants (skip SC and HAC)
./bash_scripts/precompute_benchmarks.sh --skip-sc-hac

# Single tokenizer + coarsening combo
./bash_scripts/precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc

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
| `--coarsening=STRATEGY` | Filter: `sc`, `hac`, `mc`, or `all` (default: `all`) |
| `--tokenizer=TYPE` | Filter: `hsent`, `hdt`, or `all` (default: `all`) |
| `--chunks=N` | Number of parallel chunks for MOSES (default: 8) |
| `--output-dir=PATH` | Cache directory (default: `data/cache`) |
| `--skip-sc-hac` | Only precompute MC variants |
| `--dry-run` | Show commands without executing |
| `--force` | Re-run even if cache files exist |

### Combos precomputed (default: all 6)

```
hsent:sc   hsent:hac   hsent:mc
hdt:sc     hdt:hac     hdt:mc
```

### Output

Cache files are saved to `data/cache/` with the naming pattern:
```
{dataset}_train_{tokenizer}_{num_samples}_{config_hash}.pt
```

Use in training: `python scripts/train.py ... data.use_cache=true`

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
python scripts/prepare_coconut_data.py
```

This creates `data/coconut_complex.smi` with ~10,000 complex natural products.

---

## eval_benchmarks.sh

Evaluates all checkpoints in `outputs/benchmark/` (or `outputs/benchmark_coconut/`) and generates a comparison table.

### What it does

1. Finds all `last.ckpt` files in the benchmark directory
2. Runs `scripts/test.py` on each checkpoint (computes validity, uniqueness, novelty, FCD, PGD, etc.)
3. Runs `scripts/realistic_gen.py` on each checkpoint (computes motif rate, substitution patterns, etc.)
4. Runs `scripts/compare_results.py` to generate a comparison table PNG

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
6. Generates comparison table using `scripts/compare_finetune_results.py`

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
