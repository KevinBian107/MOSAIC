# Setup and Training Guide

This guide walks you through setting up the MOSAIC environment and running your first training experiment.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Configuration System](#configuration-system)
4. [Training Parameters](#training-parameters)
5. [Running Training](#running-training)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Common Issues](#common-issues)

---

## Environment Setup

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU support)
- conda or miniconda

### Step 1: Create Conda Environment

```bash
# Create environment
conda create -n mosaic python=3.9 -y
conda activate mosaic
```

### Step 2: Install PyTorch

```bash
# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Step 3: Install Dependencies

```bash
# Install PyTorch Geometric
pip install torch-geometric

# Install additional PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install core dependencies
pip install pytorch-lightning hydra-core omegaconf
pip install transformers accelerate

# Install molecular chemistry libraries
pip install rdkit
pip install "pandas<2.0"  # Required for rdkit compatibility

# Install logging and visualization
pip install wandb matplotlib

# Optional: Install FCD metric (requires TensorFlow)
# pip install fcd_torch
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning: {pl.__version__}')"
```

### Step 5: Set Environment Variables (Optional)

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

## Configuration System

MOSAIC uses Hydra for hierarchical configuration management.

### Configuration Structure

```
configs/
├── train.yaml              # Main training config
├── test.yaml               # Testing config
├── data/
│   ├── moses.yaml         # MOSES dataset config
│   ├── qm9.yaml           # QM9 dataset config
│   └── zinc250k.yaml      # ZINC250K dataset config
├── model/
│   ├── gpt2-xs.yaml       # Extra-small model (20M params)
│   ├── gpt2-s.yaml        # Small model (50M params)
│   ├── gpt2-m.yaml        # Medium model (100M params)
│   └── gpt2-l.yaml        # Large model (200M params)
├── tokenizer/
│   ├── sent.yaml          # Flat SENT tokenization
│   └── hsent.yaml         # Hierarchical H-SENT tokenization
└── experiment/
    └── qm9.yaml           # Pre-configured QM9 experiment
```

### Viewing Current Configuration

```bash
# Print full configuration
python scripts/train.py --cfg job

# Print specific config group
python scripts/train.py --cfg job data=qm9
```

---

## Training Parameters

### Core Parameters

#### Model Selection

```bash
# Choose model size (gpt2-xs, gpt2-s, gpt2-m, gpt2-l)
python scripts/train.py model.model_name=gpt2-s
```

**Model Sizes:**
- `gpt2-xs`: 20M params, 6 layers, 384 hidden, 6 heads (fastest, baseline)
- `gpt2-s`: 50M params, 8 layers, 512 hidden, 8 heads (balanced)
- `gpt2-m`: 100M params, 12 layers, 768 hidden, 12 heads (high capacity)
- `gpt2-l`: 200M params, 16 layers, 1024 hidden, 16 heads (research)

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

#### Training Duration

```bash
# Set maximum training steps
python scripts/train.py trainer.max_steps=100000

# Set validation frequency (steps)
python scripts/train.py trainer.val_check_interval=1000
```

#### Batch Size and Workers

```bash
# Adjust batch size (default: 64)
python scripts/train.py data.batch_size=128

# Set number of data loading workers (default: 4)
python scripts/train.py data.num_workers=8
```

#### Learning Rate and Optimization

```bash
# Learning rate (default: 3e-4)
python scripts/train.py model.learning_rate=5e-4

# Weight decay (default: 0.01)
python scripts/train.py model.weight_decay=0.02

# Warmup steps (default: 1000)
python scripts/train.py model.warmup_steps=2000

# Max steps for LR schedule
python scripts/train.py model.max_steps=100000
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

# Tokenizer parameters
python scripts/train.py \
    tokenizer.max_length=2048 \
    tokenizer.truncation_length=512 \
    tokenizer.undirected=true
```

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
| | `model.learning_rate` | `3e-4` | AdamW learning rate |
| | `model.weight_decay` | `0.01` | L2 regularization |
| | `model.warmup_steps` | `1000` | LR warmup duration |
| | `model.max_steps` | `100000` | Total training steps |
| **Data** | `data.dataset_name` | `moses` | Dataset to use |
| | `data.batch_size` | `64` | Training batch size |
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

#### Example 2: Full MOSES Training (GPU required)
```bash
python scripts/train.py \
    model.model_name=gpt2-s \
    data.dataset_name=moses \
    data.num_train=-1 \
    data.num_val=5000 \
    data.num_test=5000 \
    trainer.max_steps=500000 \
    trainer.val_check_interval=5000 \
    trainer.precision=bf16-mixed \
    data.batch_size=128 \
    data.num_workers=8
```

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

## Monitoring and Logging

### Local Monitoring

#### View Training Logs (Live)
```bash
# Follow training log in real-time
tail -f outputs/YYYY-MM-DD/HH-MM-SS/train.log

# View last 100 lines
tail -n 100 outputs/YYYY-MM-DD/HH-MM-SS/train.log
```

#### Analyze Metrics CSV
```python
import pandas as pd

# Load training metrics
metrics = pd.read_csv("outputs/YYYY-MM-DD/HH-MM-SS/csv_logs/version_0/metrics.csv")

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(metrics['step'], metrics['train_loss'], label='Train')
plt.plot(metrics['step'].dropna(), metrics['val_loss'].dropna(), label='Val')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')

plt.subplot(1, 2, 2)
plt.plot(metrics['step'].dropna(), metrics['lr'].dropna())
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Wandb Monitoring

#### Setup Wandb
```bash
# Login to wandb (one-time setup)
wandb login
```

#### Enable Wandb Logging
```bash
python scripts/train.py \
    wandb.enabled=true \
    wandb.project=my-project \
    wandb.name=my-experiment
```

#### Wandb Features
- **Metrics**: Automatic logging of loss, learning rate, generation metrics
- **System**: GPU utilization, memory usage, CPU usage
- **Artifacts**: Model checkpoints stored as versioned artifacts
- **Molecules**: Visualizations of generated molecules (2D structures)
- **Config**: Full Hydra configuration for reproducibility

---

## Common Issues

### Issue 1: OpenMP Library Conflict

**Error:**
```
Error: libgomp.so.1: cannot allocate memory in static TLS block
```

**Solution:**
```bash
export LD_PRELOAD=/path/to/miniconda3/envs/mosaic/lib/libgomp.so.1
python scripts/train.py ...
```

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python scripts/train.py data.batch_size=32

# Use gradient accumulation (effective batch size = batch_size * accumulate)
python scripts/train.py \
    data.batch_size=16 \
    trainer.accumulate_grad_batches=4

# Use smaller model
python scripts/train.py model.model_name=gpt2-xs

# Enable mixed precision
python scripts/train.py trainer.precision=bf16-mixed
```

### Issue 3: Slow Data Loading

**Symptoms:**
- Training loop waits for data
- Low GPU utilization

**Solutions:**
```bash
# Increase num_workers (typically 2-8 per GPU)
python scripts/train.py data.num_workers=8

# Reduce dataset size for debugging
python scripts/train.py data.num_train=10000
```

### Issue 4: Checkpoint Loading Failure

**Error:**
```
RuntimeError: size mismatch for model.transformer.wte.weight
```

**Cause:** Vocabulary size mismatch between checkpoint and current tokenizer

**Solution:** The test script now automatically extracts vocab size from checkpoints. If you encounter this during training, ensure consistent dataset splits.

### Issue 5: RDKit Import Error

**Error:**
```
ModuleNotFoundError: No module named 'rdkit.six'
```

**Solution:**
```bash
# Downgrade pandas
pip install "pandas<2.0"

# Or reinstall rdkit
conda install -c conda-forge rdkit -y
```

### Issue 6: Wandb Authentication

**Error:**
```
wandb.errors.UsageError: api_key not configured
```

**Solution:**
```bash
# Login to wandb
wandb login

# Or set API key as environment variable
export WANDB_API_KEY=your_api_key_here
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

### Evaluation Metrics

**Molecular Metrics:**
- **Validity**: Percentage of valid molecules (parseable SMILES)
- **Uniqueness**: Percentage of unique molecules
- **Novelty**: Percentage not in training set
- **SNN**: Structural similarity to training set
- **Fragment Similarity**: Similarity based on molecular fragments
- **Scaffold Similarity**: Similarity based on molecular scaffolds
- **Internal Diversity**: Diversity within generated set
- **FCD**: Fréchet ChemNet Distance (requires additional package)

**Motif Distribution Metrics:**
- **Functional Groups MMD**: Maximum Mean Discrepancy for functional groups
- **SMARTS Motifs MMD**: MMD for SMARTS pattern frequencies
- **Ring Systems MMD**: MMD for ring system distributions
- **BRICS Fragments MMD**: MMD for BRICS fragment distributions

---

## Next Steps

1. **Experiment Tracking**: Set up wandb for systematic experiment tracking
2. **Hyperparameter Tuning**: Try different model sizes, learning rates, and batch sizes
3. **Advanced Tokenization**: Experiment with H-SENT hierarchical tokenization
4. **Custom Datasets**: Add your own molecular datasets to `src/data/`
5. **Evaluation**: Analyze generated molecules and compare motif distributions

For more information, see:
- `README.md` - Project overview
- `src/` - Source code with detailed docstrings
- `configs/` - All available configuration options
