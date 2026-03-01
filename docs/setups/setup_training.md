# Setup and Training Guide

This guide covers environment setup, configuration, and training workflows for MOSAIC.

- **UCSD DSMLP / GCP:** [Server Quick Reference](server_setup.md), [GCP Setup](setup_gcp.md)

---

## Environment Setup

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate mosaic
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
```

### Step 3: Fix OpenMP Conflicts (if needed)

```bash
export LD_PRELOAD=/path/to/miniconda3/envs/mosaic/lib/libgomp.so.1
```

---

## Data Preparation

MOSAIC automatically downloads datasets on first use.

| Dataset | Molecules | Description |
|---------|-----------|-------------|
| **MOSES** | 1.58M | Drug-like molecules (default) |
| **QM9** | 134K | Small organic molecules |
| **ZINC250K** | 250K | Drug-like molecules |

Custom data directory:
```bash
mkdir -p data/moses
cd data/moses
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv
cd ~/MOSAIC 
```

### Optional: Precomputed SMILES (faster loading)

To avoid re-reading CSV and converting SMILES to graphs on every run, export MOSES once to a single file:

```bash
python scripts/preprocess/export_moses_smiles.py
# Creates data/moses_smiles/moses_smiles.txt
```

Then use `data.use_precomputed_smiles=true` in training/test, or `--use-precomputed-smiles` with `eval_benchmarks_auto.sh`.

### Optional: Precomputed PGD reference graphs (faster evaluation)

When evaluating many checkpoints, you can precompute reference SMILES→graphs once and reuse:

```bash
python scripts/preprocess/precompute_reference_graphs.py experiment=moses reference_graphs.output_dir=outputs/eval_run
# Then pass metrics.reference_graphs_path=<printed path> to test.py, or use eval_benchmarks_auto.sh (it does this automatically).
```

---

## Configuration Reference

### Model Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `model.model_name` | `gpt2-xs` | `gpt2-xxs` (2.7M), `gpt2-xs` (11M), `gpt2-s` (50M), `gpt2-m` (100M) |
| `model.learning_rate` | `8.49e-4` | float |
| `model.weight_decay` | `0.1` | float |
| `model.warmup_steps` | `1414` | int |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment` | `moses` | Dataset config: `moses`, `qm9`, `coconut` |
| `data.batch_size` | `32` | Training batch size |
| `data.num_workers` | `8` | DataLoader workers |
| `data.num_train` | (from experiment) | Training samples (override with CLI) |
| `data.num_val` | (from experiment) | Validation samples |
| `data.num_test` | (from experiment) | Test samples |
| `data.use_cache` | `false` | Use precomputed tokenized cache (run `precompute_benchmarks.sh` or `preprocess_dataset.py` first) |
| `data.use_precomputed_smiles` | `false` | Load train/test from single SMILES file (run `scripts/preprocess/export_moses_smiles.py` first) |
| `data.precomputed_smiles_dir` | `data/moses_smiles` | Directory containing `moses_smiles.txt` |

**Note**: Dataset name, data file, and sample counts are set by the experiment config. Use `experiment=moses`, `experiment=qm9`, or `experiment=coconut`.

### Tokenizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokenizer.type` | `sent` | `sent`, `hsent`, `hdt` |
| `tokenizer.max_length` | `2048` | Max sequence length |
| `tokenizer.labeled_graph` | `true` | Encode atom/bond types |
| `tokenizer.undirected` | `true` | Treat graphs as undirected |

**Important**: Always use `tokenizer.labeled_graph=true` for molecular generation.

### Trainer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.target_samples_seen` | `16000000` | Primary training budget (hardware-agnostic) |
| `trainer.max_steps` | `250000` | Fallback when `target_samples_seen=null` |
| `trainer.val_checks_per_epoch` | `5` | Preferred validation cadence |
| `trainer.validate_every_n_epochs` | `1` | Validate every N epochs |
| `trainer.precision` | `32` | `32`, `16-mixed`, `bf16-mixed` |
| `trainer.gradient_clip_val` | `1.0` | Gradient clipping |

Preferred validation controls are epoch-based (`val_checks_per_epoch`, `validate_every_n_epochs`).
Legacy step-based controls (`val_check_interval`, `check_val_every_n_epoch`) remain supported for old runs.

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling.num_samples` | `1000` | Samples per evaluation |
| `sampling.temperature` | `1.0` | Sampling temperature |
| `sampling.top_k` | `10` | Top-k sampling |
| `sampling.batch_size` | `32` | Generation batch size |

### Logging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wandb.enabled` | `false` | Enable W&B logging |
| `wandb.project` | `molecular-graph-gen` | W&B project name |
| `wandb.name` | `null` | Run name |
| `logs.path` | `outputs` | Output directory |

---

## Training Examples

**Important**: Always specify an experiment config (`experiment=moses`, `experiment=coconut`, or `experiment=qm9`) to set the dataset.

### Quick Test Run

```bash
python scripts/train.py \
    experiment=moses \
    model.model_name=gpt2-xxs \
    data.num_train=1000 \
    trainer.target_samples_seen=32000 \
    wandb.enabled=false
```

### SENT Tokenizer (Baseline)

```bash
python scripts/train.py \
    experiment=moses \
    tokenizer=sent \
    model.model_name=gpt2-xs \
    trainer.target_samples_seen=16000000 \
    wandb.enabled=true \
    wandb.project=mosaic-sent
```

### H-SENT Tokenizer

```bash
python scripts/train.py \
    experiment=moses \
    tokenizer=hsent \
    model.model_name=gpt2-xs \
    data.use_cache=true \
    trainer.max_steps=100000 \
    wandb.enabled=true \
    wandb.project=mosaic-hsent
```

### HDT Tokenizer

```bash
python scripts/train.py \
    experiment=moses \
    tokenizer=hdt \
    model.model_name=gpt2-xs \
    data.use_cache=true \
    trainer.max_steps=100000 \
    wandb.enabled=true \
    wandb.project=mosaic-hdt
```

### Dataset Experiments

```bash
# MOSES - Drug-like molecules (1.6M)
python scripts/train.py experiment=moses

# QM9 - Small organic molecules (134K)
python scripts/train.py experiment=qm9

# COCONUT - Complex natural products (5K filtered)
python scripts/train.py experiment=coconut
```

---

## Output Structure

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/         # Hydra config logs
        ├── csv_logs/       # Training metrics
        ├── train.log       # Full training log
        ├── best.ckpt       # Best checkpoint (val loss)
        └── last.ckpt       # Final checkpoint
```

---

## Evaluation

```bash
# Basic evaluation
python scripts/test.py \
    model.checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/best.ckpt \
    sampling.num_samples=1000

# With specific tokenizer
python scripts/test.py \
    tokenizer=hsent \
    model.checkpoint_path=<path>/best.ckpt \
    sampling.num_samples=1000
```

Results are saved to:
- `outputs/test/YYYY-MM-DD/HH-MM-SS/results.json`
- `outputs/test/YYYY-MM-DD/HH-MM-SS/generated_smiles.txt`

For evaluating multiple checkpoints with a mapping file, precomputed SMILES, and cached reference graphs, use `./bash_scripts/eval/eval_benchmarks_auto.sh`; see [bash_scripts/README.md](../../bash_scripts/README.md).

---

## Claude Code Integration

We provide a `.claude` configuration for AI-assisted development.

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --dangerously-skip-permissions
```

Use the `investigation` skill for structured code exploration with logging to a `scratch/` folder (gitignored).
