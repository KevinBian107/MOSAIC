# DSMLP Server Quick Reference

Quick reference commands for running MOSAIC on UCSD DSMLP cluster.

## SSH Access

```bash
ssh <username>@dsmlp-login.ucsd.edu
```

## Pod Management

```bash
# Create pod: 1 GPU, 8 CPU, 64GB Memory, A30
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 1 -c 8 -m 64 -n 31 -b

# Or if you want a pod with 4 * A5000s
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 4 -p low -c 8 -m 64 -n 30 -b

# List pods
kubectl get pods

# Shell into pod
kubesh <pod-name>

# Delete pod
kubectl delete pod <pod-name>

# View pod logs
kubectl logs <pod-name>

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
