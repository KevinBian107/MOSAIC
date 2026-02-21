# DSMLP Server Quick Reference

Quick reference commands for running MOSAIC on UCSD DSMLP cluster. See also [commands_reference.md](commands_reference.md) for a combined GCP + DSMLP + setup + training cheat sheet.

## SSH Access

```bash
ssh any012@dsmlp-login.ucsd.edu
```

## Pod Management

```bash
# Create pod: 1 GPU, 8 CPU, 64 GB Memory, A30, run in background
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 1 -c 8 -m 64 -n 31 -b

# Create pod: 2 GPUs, low priority, A5000
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 2 -p low -c 8 -m 64 -n 30 -b

# Create pod: 4 GPUs, low priority, A5000
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 4 -p low -c 8 -m 64 -n 30 -b

# List pods
kubectl get pods

# Shell into pod
kubesh any012-2279099

# Delete pod
kubectl delete pod any012-2279099

# View pod logs
kubectl logs any012-2279099

# Check pod events (debugging)
kubectl describe pod <pod-name>
```

## Environment Setup

```bash
cd MOSAIC
conda env create -f environment_server.yaml
conda activate mosaic

# Verify installation
python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print('OK')"
```

## Dataset Download (MOSES)

```bash
mkdir -p data/moses && cd data/moses
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv
cd ../..
```

## Training Commands

```bash
# Quick test
python scripts/train.py model.model_name=gpt2-xxs data.num_train=1000 trainer.max_steps=10

# H-SENT training (background)
nohup python scripts/train.py tokenizer=hsent data.use_cache=true \
    trainer.max_steps=100000 wandb.enabled=true > hsent.log 2>&1 &

# HDT training (background)
nohup python scripts/train.py tokenizer=hdt data.use_cache=true \
    trainer.max_steps=100000 wandb.enabled=true > hdt.log 2>&1 &
```

## Multi-GPU DDP Training Commands

### 4 GPUs × batch_size=96 (effective batch=384)

```bash
python scripts/train.py \
  model.model_name=gpt2-xs \
  tokenizer=hsent \
  data.num_train=500000 \
  data.num_val=100 \
  data.use_cache=true \
  trainer.devices=4 \
  trainer.strategy=ddp \
  trainer.max_epochs=20 \
  trainer.max_steps=-1 \
  trainer.check_val_every_n_epoch=1 \
  data.batch_size=96 \
  model.learning_rate=1.2e-3 \
  model.warmup_steps=2000 \
  sampling.num_samples=0 \
  wandb.enabled=true \
  wandb.project=molecular-graph-gen \
  wandb.name=hsent-500k-ddp-4gpu-bs96-20epochs \
  wandb.log_model=true \
  wandb.log_graphs=true \
  wandb.eval_every_n_val=0
```

**Expected speedup: ~10.8x faster than baseline (1 GPU, batch=32, 20 epochs)**

### 4 GPUs × batch_size=64 (effective batch=256)

```bash
python scripts/train.py \
  model.model_name=gpt2-xs \
  tokenizer=hsent \
  data.num_train=500000 \
  data.num_val=100 \
  data.use_cache=true \
  trainer.devices=4 \
  trainer.strategy=ddp \
  trainer.max_epochs=20 \
  trainer.max_steps=-1 \
  trainer.check_val_every_n_epoch=1 \
  data.batch_size=64 \
  model.learning_rate=1.7e-3 \
  model.warmup_steps=2830 \
  sampling.num_samples=0 \
  wandb.enabled=true \
  wandb.project=molecular-graph-gen \
  wandb.name=hsent-500k-ddp-4gpu-bs64-20epochs \
  wandb.log_model=true \
  wandb.log_graphs=true \
  wandb.eval_every_n_val=0
```

**Expected speedup: ~7.2x faster than baseline (1 GPU, batch=32, 20 epochs)**

### Notes on DDP Training

- **Effective batch size** = `batch_size × num_GPUs`
- **Learning rate scaling**: Use √(effective_batch / baseline_batch) scaling
  - Baseline: 1 GPU × batch=32 → LR=6e-4
  - 4 GPUs × batch=96 → effective=384 → LR=6e-4 × √12 ≈ 1.2e-3
  - 4 GPUs × batch=64 → effective=256 → LR=6e-4 × √8 ≈ 1.7e-3
- **Warmup steps**: Scale proportionally with batch size increase
- **DDP efficiency**: Expect ~90% efficiency with 4 GPUs (communication overhead)
- **Tensor Cores**: Set via `torch.set_float32_matmul_precision('medium')` for A5000 speedup
- **Validation**: Set to once per epoch to minimize overhead during training

## Evaluation Commands

```bash
# Find checkpoints
find outputs/ -name "*.ckpt"

# Run evaluation
python scripts/test.py model.checkpoint_path=<path>/best.ckpt sampling.num_samples=1000
```

## Monitoring

```bash
# Monitor logs
tail -f training.log

# GPU usage
watch -n 1 nvidia-smi

# Background jobs
jobs

# Disk usage
df -h && du -sh outputs/
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `data.batch_size` |
| Slow data loading | Increase `data.num_workers` |
| Pod not starting | `kubectl describe pod <name>` |
| Resume training | Set `resume=true` |
