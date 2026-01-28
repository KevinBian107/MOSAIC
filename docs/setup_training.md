# Setup and Training Guide

This guide covers environment setup, configuration, and training workflows for MOSAIC.

For UCSD DSMLP cluster-specific commands, see [Server Quick Reference](server_setup.md).

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

---

## Configuration Reference

### Model Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `model.model_name` | `gpt2-xs` | `gpt2-xxs` (2.7M), `gpt2-xs` (11M), `gpt2-s` (50M), `gpt2-m` (100M) |
| `model.learning_rate` | `6e-4` | float |
| `model.weight_decay` | `0.01` | float |
| `model.warmup_steps` | `1000` | int |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.dataset_name` | `moses` | `moses`, `qm9`, `zinc250k` |
| `data.batch_size` | `32` | Training batch size |
| `data.num_workers` | `4` | DataLoader workers |
| `data.num_train` | `10000` | Training samples (-1 for all) |
| `data.num_val` | `1000` | Validation samples |
| `data.num_test` | `1000` | Test samples |
| `data.use_cache` | `false` | Cache tokenized data |

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
| `trainer.max_steps` | `100000` | Total training steps |
| `trainer.val_check_interval` | `1000` | Validation frequency |
| `trainer.precision` | `bf16-mixed` | `32`, `16-mixed`, `bf16-mixed` |
| `trainer.gradient_clip_val` | `1.0` | Gradient clipping |

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

### Quick Test Run

```bash
python scripts/train.py \
    model.model_name=gpt2-xxs \
    data.num_train=1000 \
    trainer.max_steps=100 \
    wandb.enabled=false
```

### SENT Tokenizer (Baseline)

```bash
python scripts/train.py \
    tokenizer=sent \
    model.model_name=gpt2-xs \
    trainer.max_steps=500000 \
    wandb.enabled=true \
    wandb.project=mosaic-sent
```

### H-SENT Tokenizer

```bash
python scripts/train.py \
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
    tokenizer=hdt \
    model.model_name=gpt2-xs \
    data.use_cache=true \
    trainer.max_steps=100000 \
    wandb.enabled=true \
    wandb.project=mosaic-hdt
```

### Using Experiment Configs

```bash
# QM9 experiment
python scripts/train.py experiment=qm9

# MOSES experiment
python scripts/train.py experiment=moses
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

---

## Claude Code Integration

We provide a `.claude` configuration for AI-assisted development.

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --dangerously-skip-permissions
```

Use the `investigation` skill for structured code exploration with logging to a `scratch/` folder (gitignored).
