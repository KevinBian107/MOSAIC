# Server Setup Guide

This guide provides instructions for setting up and running MOSAIC on remote GPU servers, specifically tailored for UCSD DSMLP cluster environments.

## SSH Access

```bash
# Connect to DSMLP login node
ssh any012@dsmlp-login.ucsd.edu
```

## Pod Management

### Creating a Pod

```bash
# Create pod: 1 GPU, 8 CPU, 64GB Memory, using A30, run in background
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 1 -c 8 -m 64 -n 31 -b
```

### kubectl Commands

```bash
# List all your pods
kubectl get pods

# Shell into a pod
kubesh any012-2705327

# Delete a pod
kubectl delete pod any012-2705327

# View pod logs
kubectl logs any012-2705327
```

## Environment Setup

```bash
# Navigate to project directory
cd MOSAIC

# Create conda environment from server-specific configuration
conda env create -f environment_server.yaml
conda activate mosaic

# Verify installation
python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print('All packages installed successfully')"
```

## Dataset Setup

### Download MOSES Dataset

```bash
# Create data directory structure
mkdir -p data/moses

# Navigate to data directory
cd data/moses

# Download train, test, and validation sets from official MOSES repository
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv

# Return to project root
cd ../..
```

## Training

### Quick Test Run

Test the setup with a small subset before running full training:

```bash
python scripts/train.py \
    model.model_name=gpt2-xxs \
    data.num_train=1000 \
    data.num_val=100 \
    trainer.max_steps=10 \
    wandb.enabled=false
```

### Full Training with Entire Dataset

**SENT Tokenizer (Baseline):**

```bash
python scripts/train.py \
    model.model_name=gpt2-xs \
    data.num_train=-1 \
    data.num_val=-1 \
    data.num_test=-1 \
    data.num_workers=2 \
    trainer.max_steps=700000 \
    trainer.val_check_interval=10000 \
    sampling.num_samples=1000 \
    wandb.enabled=true \
    wandb.project=mosaic-labeled-sent \
    wandb.name=gpt2-xs-500k-full-moses \
    logs.path=outputs/2026-01-16/00-52-54 &
```

### Training with H-SENT and HDT Tokenizers

**H-SENT Small Subset Test:**

```bash
python scripts/train.py \
    tokenizer=hsent \
    data.num_train=50000 \
    trainer.max_steps=1000 \
    resume=false 2>&1 | tee hsent_training.log
```

**HDT Small Subset Test:**

```bash
python scripts/train.py \
    tokenizer=hdt \
    data.num_train=50000 \
    trainer.max_steps=1000 \
    resume=false 2>&1 | tee hdt_training.log
```

**H-SENT Full Training with Caching:**

```bash
nohup python scripts/train.py \
    tokenizer=hsent \
    data.use_cache=true \
    data.num_train=50000 \
    data.num_val=5000 \
    data.num_test=5000 \
    data.batch_size=32 \
    data.num_workers=2 \
    trainer.max_steps=100000 \
    trainer.val_check_interval=2000 \
    trainer.gradient_clip_val=1.0 \
    model.learning_rate=0.0006 \
    model.weight_decay=0.1 \
    model.warmup_steps=1000 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hsent-50k-cached \
    wandb.tags=[hsent,cached,50k] \
    wandb.log_model=true \
    wandb.log_graphs=true > hsent_training_full.log 2>&1 &
```

**HDT Full Training with Caching:**

```bash
nohup python scripts/train.py \
    tokenizer=hdt \
    data.use_cache=true \
    data.num_train=50000 \
    data.num_val=5000 \
    data.num_test=5000 \
    data.batch_size=32 \
    data.num_workers=2 \
    trainer.max_steps=100000 \
    trainer.val_check_interval=2000 \
    trainer.gradient_clip_val=1.0 \
    model.learning_rate=0.0006 \
    model.weight_decay=0.1 \
    model.warmup_steps=1000 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hdt-50k-cached \
    wandb.tags=[hdt,cached,50k] \
    wandb.log_model=true \
    wandb.log_graphs=true > hdt_training_full.log 2>&1 &
```

### Checkpoint Test Training

For testing checkpoint saving and resuming:

```bash
python scripts/train.py \
    model.model_name=gpt2-xs \
    data.num_train=500 \
    data.num_val=50 \
    data.num_test=50 \
    data.num_workers=2 \
    trainer.max_steps=1000 \
    trainer.val_check_interval=50 \
    sampling.num_samples=10 \
    wandb.enabled=true \
    wandb.project=mosaic-labeled-sent \
    wandb.name=gpt2-xs-500k-ckpt-test &
```

## Evaluation

### Finding Checkpoints

```bash
# Find all checkpoints in a specific output directory
find outputs/2026-01-16 -name "*.ckpt"
```

### Running Tests

**Small Test (10 samples):**

```bash
python scripts/test.py \
    model.checkpoint_path=outputs/2026-01-16/00-52-54/last.ckpt \
    data.dataset_name=moses \
    sampling.num_samples=10
```

**Full Test (1000 samples):**

```bash
python scripts/test.py \
    model.checkpoint_path=outputs/2026-01-16/00-52-54/last.ckpt \
    sampling.num_samples=1000 \
    data.num_train=-1
```

**H-SENT Model Evaluation:**

```bash
nohup python scripts/test.py \
    tokenizer=hsent \
    data.num_train=50000 \
    data.num_val=5000 \
    data.num_test=5000 \
    model.checkpoint_path=outputs/2026-01-21/05-02-52/best.ckpt \
    sampling.num_samples=1000 \
    sampling.batch_size=32 \
    > hsent_test_results.log 2>&1 &
```

**HDT Model Evaluation:**

```bash
nohup python scripts/test.py \
    tokenizer=hdt \
    data.num_train=50000 \
    data.num_val=5000 \
    data.num_test=5000 \
    model.checkpoint_path=outputs/2026-01-21/05-03-55/best.ckpt \
    sampling.num_samples=1000 \
    sampling.batch_size=32 \
    > hdt_test_results.log 2>&1 &
```

## Monitoring Jobs

### Viewing Logs

```bash
# Monitor training logs in real-time
tail -f hsent_training_full.log

# View last 50 lines of a log
tail -n 50 hdt_training_full.log

# Search for specific patterns
grep "validation" hsent_training_full.log
```

### Checking Running Jobs

```bash
# List all background jobs
jobs

# Check GPU usage
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi
```

## Tips and Best Practices

1. **Use `nohup` and `&`**: For long-running jobs, always use `nohup ... &` to run in background and prevent termination on disconnect
2. **Redirect output**: Use `> logfile.log 2>&1` to capture both stdout and stderr
3. **Monitor resources**: Regularly check `nvidia-smi` to ensure GPU utilization
4. **Use caching**: Enable `data.use_cache=true` for faster subsequent runs with the same tokenizer
5. **Checkpoint management**: Keep `best.ckpt` and `last.ckpt` for evaluation and resuming
6. **WandB logging**: Enable for better experiment tracking and visualization
7. **Descriptive naming**: Use meaningful names for `wandb.name` and `logs.path`

## Troubleshooting

### Pod Issues

```bash
# Pod not starting - check events
kubectl describe pod <pod-name>

# Out of resources - delete old pods
kubectl get pods
kubectl delete pod <old-pod-name>
```

### Training Issues

```bash
# CUDA out of memory - reduce batch size
data.batch_size=16  # Instead of 32

# Slow data loading - increase workers
data.num_workers=4  # Instead of 2

# Resume from checkpoint - set resume flag
resume=true
```

### Log File Issues

```bash
# Log file too large - compress old logs
gzip old_training.log

# Monitor disk usage
df -h
du -sh outputs/
```
