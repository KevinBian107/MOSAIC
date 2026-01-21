# Setup and Training Guide

This guide walks you through setting up the MOSAIC environment and running your first training experiment. Always check on the [datahub.ucsd.edu/services/disk-quota-service/](https://datahub.ucsd.edu/services/disk-quota-service/) to see the quota limit we have first before installing anything. This is an instruction for creating the environment needed for this project, we will establish our environment first similar to [AutoGraph](https://github.com/BorgwardtLab/AutoGraph), then add in additional dependencies if it's needed. All the environment instantiation was tested on an A30 of UCSD's DSMLP system.

---

## Environment Setup

### Step 1: Create Conda Environment

```bash
# Create environment
conda create -n mosaic python=3.9 -y
conda activate mosaic
```

### Step 2: Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning: {pl.__version__}')"
```

### Step 3: Set Environment Variables (Optional)

If you encounter OpenMP library conflicts:

```bash
export LD_PRELOAD=/path/to/miniconda3/envs/mosaic/lib/libgomp.so.1
```

---

## Data Preparation

### Automatic Data Download

MOSAIC automatically downloads datasets on first use:

```python
# datasets are downloaded to data/ directory by default
# Supported datasets: moses, qm9, zinc250k
```

### Supported Datasets

| Dataset | Molecules | Description | Use Case |
|---------|-----------|-------------|----------|
| **MOSES** | 1.58M | Drug-like molecules | Large-scale training |
| **QM9** | 134K | Small organic molecules | Quick experiments |
| **ZINC250K** | 250K | Drug-like molecules | Medium-scale training |

### Custom Dataset Location

```bash
# Specify custom data directory
python scripts/train.py data.data_root=/path/to/data/
```

---

# Setup Claude Code
We have provided the `.claude` file for helping code developement and understanding of the codebase, install claude code as the following to automatically used the codebase instructions we have provided.

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --dangerously-skip-permissions
```

To use the skills we have created, just call the keyword `investigation` and claude should be conducting reasoning and coding in the structured manner that we have defined, with logging of each step into a \scratch folder that's gitignored.

---

## Training Parameters

### Core Parameters

#### Model Selection

```bash
# Choose model size (gpt2-xs, gpt2-s, gpt2-m, gpt2-l)
python scripts/train.py model.model_name=gpt2-s
```

**Model Sizes:**
- `gpt2-xxs`: 2.7M params, 4 layers, 256 hidden, 4 heads (ultra-fast, debugging)
- `gpt2-xs`: 11.1M params, 6 layers, 384 hidden, 12 heads (fast, baseline)
- `gpt2-s`: 50M params, 12 layers, 768 hidden, 12 heads (balanced)
- `gpt2-m`: 100M params, 24 layers, 1024 hidden, 16 heads (high capacity)

#### Dataset Configuration

```bash
# Select dataset
python scripts/train.py data.dataset_name=qm9

# Control dataset splits
python scripts/train.py \
    data.num_train=10000 \
    data.num_val=1000 \
    data.num_test=1000

# Use full dataset (-1 means all available)
python scripts/train.py \
    data.num_train=-1 \
    data.num_val=5000 \
    data.num_test=5000
```

#### Batch Size and Workers

```bash
# Adjust batch size (default: 64)
python scripts/train.py data.batch_size=128

# Set number of data loading workers (default: 4)
python scripts/train.py data.num_workers=8
```


#### Mixed Precision Training

```bash
# Use bfloat16 mixed precision (recommended for A100/H100)
python scripts/train.py trainer.precision=bf16-mixed

# Use float16 mixed precision (for older GPUs)
python scripts/train.py trainer.precision=16-mixed

# Full float32 precision
python scripts/train.py trainer.precision=32
```

#### Tokenization Strategy

```bash
# Flat SENT tokenization (default)
python scripts/train.py tokenizer.type=sent

# Hierarchical H-SENT tokenization
python scripts/train.py tokenizer.type=hsent

# Labeled SENT (encode atom/bond types - RECOMMENDED for molecules)
python scripts/train.py tokenizer.labeled_graph=true

# Tokenizer parameters
python scripts/train.py \
    tokenizer.max_length=2048 \
    tokenizer.truncation_length=512 \
    tokenizer.undirected=true \
    tokenizer.labeled_graph=true
```

**Important:** For molecular generation, **always use `tokenizer.labeled_graph=true`**. This encodes chemical information (atom types, bond types) in addition to graph topology. Without labeled graphs, models only learn connectivity and generate chemically invalid molecules.

#### Sampling/Generation Parameters

```bash
# Number of samples to generate during evaluation
python scripts/train.py sampling.num_samples=1000

# Sampling temperature (higher = more diverse)
python scripts/train.py sampling.temperature=1.0

# Top-k sampling (lower = more conservative)
python scripts/train.py sampling.top_k=10

# Max sequence length during generation
python scripts/train.py sampling.max_length=2048

# Batch size for generation
python scripts/train.py sampling.batch_size=32
```

### Complete Parameter Reference

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Model** | `model.model_name` | `gpt2-xs` | Model architecture size |
| | `model.learning_rate` | `6e-4` | AdamW learning rate |
| | `model.weight_decay` | `0.01` | L2 regularization |
| | `model.warmup_steps` | `1000` | LR warmup duration |
| | `model.max_steps` | `100000` | Total training steps |
| **Data** | `data.dataset_name` | `moses` | Dataset to use |
| | `data.batch_size` | `32` | Training batch size |
| | `data.num_workers` | `4` | DataLoader workers |
| | `data.num_train` | `10000` | Training samples (-1 for all) |
| | `data.num_val` | `1000` | Validation samples |
| | `data.num_test` | `1000` | Test samples |
| | `data.include_hydrogens` | `false` | Include H atoms |
| | `data.data_root` | `data` | Data directory |
| **Trainer** | `trainer.max_steps` | `100000` | Max training steps |
| | `trainer.val_check_interval` | `1000` | Validation frequency |
| | `trainer.precision` | `bf16-mixed` | Training precision |
| | `trainer.gradient_clip_val` | `1.0` | Gradient clipping |
| | `trainer.accelerator` | `gpu` | Hardware accelerator |
| | `trainer.devices` | `1` | Number of devices |
| **Tokenizer** | `tokenizer.type` | `sent` | Tokenization scheme |
| | `tokenizer.max_length` | `2048` | Max sequence length |
| | `tokenizer.truncation_length` | `512` | Truncation length |
| | `tokenizer.undirected` | `true` | Treat graphs as undirected |
| | `tokenizer.labeled_graph` | `true` | Encode atom/bond types |
| **Sampling** | `sampling.num_samples` | `1000` | Samples per evaluation |
| | `sampling.temperature` | `1.0` | Sampling temperature |
| | `sampling.top_k` | `10` | Top-k sampling |
| | `sampling.max_length` | `2048` | Max generation length |
| | `sampling.batch_size` | `32` | Generation batch size |
| **Logging** | `logs.path` | `outputs` | Output directory |
| | `seed` | `42` | Random seed |
| **Wandb** | `wandb.enabled` | `false` | Enable wandb logging |
| | `wandb.project` | `molecular-graph-gen` | Wandb project name |
| | `wandb.entity` | `null` | Wandb username/team |
| | `wandb.name` | `null` | Run name (auto-generated) |
| | `wandb.log_model` | `true` | Log model checkpoints |
| | `wandb.log_graphs` | `true` | Log molecule images |

---

## Running Training

### Quick Start Examples

#### Example 1: Small QM9 Experiment (5 minutes)
```bash
python scripts/train.py \
    experiment=qm9 \
    data.num_train=1000 \
    data.num_val=100 \
    data.num_test=100 \
    trainer.max_steps=1000 \
    trainer.val_check_interval=250
```

#### Example 2: MOSES Training with Labeled SENT (500K steps)
```bash
# GPT2-XS (11.1M params) - Baseline
python scripts/train.py \
    model.model_name=gpt2-xs \
    trainer.max_steps=500000 \
    trainer.val_check_interval=1500 \
    sampling.num_samples=1000 \
    wandb.enabled=true \
    wandb.project=mosaic-labeled-sent \
    wandb.name=gpt2-xs-500k-labeled

# GPT2-XXS (2.7M params) - Smaller/faster
python scripts/train.py \
    model.model_name=gpt2-xxs \
    trainer.max_steps=500000 \
    trainer.val_check_interval=1500 \
    sampling.num_samples=1000 \
    wandb.enabled=true \
    wandb.project=mosaic-labeled-sent \
    wandb.name=gpt2-xxs-500k-labeled
```

**Note:** These commands use labeled SENT (enabled by default in `configs/train.yaml`). Expected validity after 500K steps: 80-95%.

#### Example 3: With Wandb Logging
```bash
python scripts/train.py \
    model.model_name=gpt2-m \
    trainer.max_steps=200000 \
    wandb.enabled=true \
    wandb.project=mosaic-experiments \
    wandb.name=gpt2m-moses-baseline \
    wandb.tags=[baseline,gpt2-m,moses]
```

#### Example 4: Hierarchical Tokenization
```bash
python scripts/train.py \
    tokenizer.type=hsent \
    model.model_name=gpt2-s \
    trainer.max_steps=100000
```

### Using Pre-configured Experiments

```bash
# Use experiment config (combines multiple settings)
python scripts/train.py experiment=qm9

# Override experiment parameters
python scripts/train.py experiment=qm9 model.model_name=gpt2-m
```

### Training Output Structure

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/           # Timestamp of run
        ├── .hydra/         # Hydra configuration logs
        ├── csv_logs/       # CSV training logs
        │   └── version_0/
        │       └── metrics.csv
        ├── train.log       # Full training log
        ├── gpt2-xs.ckpt   # Best checkpoint (based on val loss)
        └── gpt2-xs-last.ckpt  # Final checkpoint
```

---

## Testing Your Trained Model

After training completes, evaluate your model:

```bash
# Test with your checkpoint
python scripts/test.py \
    model.checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/gpt2-xs.ckpt \
    sampling.num_samples=1000

# Results saved to:
# - outputs/test/YYYY-MM-DD/HH-MM-SS/results.json
# - outputs/test/YYYY-MM-DD/HH-MM-SS/generated_smiles.txt
```