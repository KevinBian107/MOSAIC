# Command reference

Quick reference for GCP, UCSD DSMLP, environment setup, precomputation, training, and evaluation. For full guides see [setup_gcp.md](setup_gcp.md), [server_setup.md](server_setup.md), and [setup_training.md](setup_training.md).

---

## GCP (Google Cloud)

### SSH into a VM

```bash
# Basic SSH (uses default user)
gcloud compute ssh mosaic-v100 --zone=us-central1-a

# SSH as a specific user
gcloud compute ssh andrewyang@mosaic-v100 --zone=us-central1-a
```

After SSH, activate the project and environment:

```bash
cd andrew/MOSAIC && conda activate mosaic
```

See [setup_gcp.md](setup_gcp.md) for VM creation, GPU types, and cost tips.

---

## UCSD DSMLP cluster

### SSH and pod creation

```bash
# SSH to DSMLP login node
ssh any012@dsmlp-login.ucsd.edu
```

Create a pod (run in background with `-b`):

```bash
# 1 GPU, 8 CPU, 64 GB memory, A30, background
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 1 -c 8 -m 64 -n 31 -b

# 2 GPUs, low priority, A5000
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 2 -p low -c 8 -m 64 -n 30 -b

# 4 GPUs, low priority, A5000
launch-scipy-ml.sh -W DSC180A_FA25_A00 -g 4 -p low -c 8 -m 64 -n 30 -b
```

### kubectl

```bash
kubectl get pods
kubesh any012-2279099          # shell into pod
kubectl delete pod any012-2279099
kubectl logs any012-2279099
```

See [server_setup.md](server_setup.md) for more DSMLP details.

---

## Setup (on remote server or local)

### Environment and data

```bash
cd MOSAIC
conda env create -f environment_server.yaml
conda activate mosaic

# Create data directory and download MOSES CSVs
mkdir -p data/moses
cd data/moses
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv
cd ../..
```

### Verify stack

```bash
python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print('All installed')"
```

### Optional: precomputed SMILES (faster data loading)

To avoid re-reading CSV and converting SMILES to graphs on every run, export MOSES to a single file once, then use it for training and evaluation:

```bash
python scripts/export_moses_smiles.py
# Creates data/moses_smiles/moses_smiles.txt
```

Then pass `data.use_precomputed_smiles=true` to `train.py` / `test.py`, or use `--use-precomputed-smiles` with `eval_benchmarks_auto.sh`.

---

## Quick test and full training

### Quick sanity check (small subset)

```bash
python scripts/train.py \
    model.model_name=gpt2-xxs \
    data.num_train=1000 \
    data.num_val=100 \
    trainer.max_steps=10 \
    wandb.enabled=false
```

### Full training (entire dataset, background)

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

### Find checkpoints and run test

```bash
find outputs/ -name "*.ckpt"

python scripts/test.py \
    model.checkpoint_path=outputs/2026-01-16/00-52-54/last.ckpt \
    data.dataset_name=moses \
    sampling.num_samples=10

python scripts/test.py \
    model.checkpoint_path=outputs/2026-01-16/00-52-54/last.ckpt \
    sampling.num_samples=1000 \
    data.num_train=-1
```

---

## H-SENT and HDT training

### Short runs (no cache)

```bash
# H-SENT
python scripts/train.py tokenizer=hsent data.num_train=50000 trainer.max_steps=1000 resume=false 2>&1 | tee hsent_training.log

# HDT
python scripts/train.py tokenizer=hdt data.num_train=50000 trainer.max_steps=1000 resume=false 2>&1 | tee hdt_training.log
```

### With cached tokenized data (recommended for large runs)

Precompute cache first (see **Precomputation** below), then:

```bash
# H-SENT with cache, background
nohup python scripts/train.py tokenizer=hsent data.use_cache=true data.num_train=50000 data.num_val=5000 data.num_test=5000 \
    trainer.max_steps=100000 wandb.enabled=true wandb.name=hsent-50k-cached wandb.tags=[hsent,cached,50k] > hsent_training_full.log 2>&1 &

# HDT with cache, background
nohup python scripts/train.py tokenizer=hdt data.use_cache=true data.num_train=50000 data.num_val=5000 data.num_test=5000 \
    trainer.max_steps=100000 wandb.enabled=true wandb.name=hdt-50k-cached wandb.tags=[hdt,cached,50k] > hdt_training_full.log 2>&1 &
```

### H-SENT / HDT with full training options

```bash
# H-SENT
python scripts/train.py \
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
    wandb.log_graphs=true &

# HDT
python scripts/train.py \
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
    wandb.log_graphs=true &
```

---

## Large-scale and multi-GPU training

### 500K samples, single run

```bash
python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hsent \
    data.num_train=500000 \
    data.num_val=1000 \
    data.num_test=1000 \
    data.use_cache=true \
    trainer.max_steps=300000 \
    trainer.val_check_interval=10000000 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hsent-500k-cached-multi \
    wandb.log_model=true \
    wandb.log_graphs=true

python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hdt \
    data.num_train=500000 \
    data.num_val=1000 \
    data.num_test=1000 \
    data.use_cache=true \
    trainer.max_steps=300000 \
    trainer.val_check_interval=10000000 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hdt-500k-cached-multi \
    wandb.log_model=true \
    wandb.log_graphs=true
```

### Preprocess dataset (alternative to precompute_benchmarks.sh for smaller runs)

```bash
python scripts/preprocess/preprocess_dataset.py tokenizer=hsent data.num_train=50000 data.num_val=100 data.num_test=100
```

### DDP (multi-GPU) training

```bash
# 2 GPUs
python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hsent \
    data.num_train=500000 \
    data.use_cache=true \
    trainer.devices=2 \
    trainer.strategy=ddp \
    trainer.max_steps=150000 \
    data.batch_size=64 \
    model.learning_rate=8.5e-4 \
    model.warmup_steps=2000 \
    +trainer.limit_val_batches=0 \
    +trainer.num_sanity_val_steps=0 \
    sampling.num_samples=0 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hsent-500k-ddp-2gpu \
    wandb.log_model=true \
    wandb.log_graphs=true \
    wandb.eval_every_n_val=0

# 4 GPUs, 20 epochs
python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hsent \
    data.num_train=50000 \
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
    wandb.name=hsent-500k-ddp-4gpu-20epochs \
    wandb.log_model=true \
    wandb.log_graphs=true \
    wandb.eval_every_n_val=0

# 4 GPUs, 30 epochs, val every 261 steps
python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hsent \
    data.num_train=50000 \
    data.num_val=1000 \
    data.use_cache=true \
    trainer.devices=4 \
    trainer.strategy=ddp \
    trainer.max_epochs=30 \
    trainer.max_steps=-1 \
    trainer.val_check_interval=261 \
    data.batch_size=96 \
    model.learning_rate=1.5e-3 \
    model.warmup_steps=260 \
    sampling.num_samples=0 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hsent-50k-2ddp-4gpu-30e-kmeans1 \
    wandb.log_model=true \
    wandb.log_graphs=true
```

### Shorter sequence length (e.g. 512)

```bash
python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hsent \
    tokenizer.max_length=512 \
    tokenizer.truncation_length=512 \
    sampling.max_length=512 \
    data.num_train=50000 \
    data.num_val=1000 \
    data.use_cache=true \
    trainer.devices=4 \
    trainer.strategy=ddp \
    trainer.max_epochs=30 \
    trainer.max_steps=-1 \
    trainer.val_check_interval=131 \
    data.batch_size=96 \
    model.learning_rate=1.5e-3 \
    model.warmup_steps=260 \
    sampling.num_samples=0 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hsent-50k-ddp-4gpu-30e-kmeans1 \
    wandb.log_model=true \
    wandb.log_graphs=true
```

### Very large dataset (e.g. 1.5M train)

```bash
python scripts/train.py \
    model.model_name=gpt2-xs \
    tokenizer=hsent \
    tokenizer.max_length=512 \
    tokenizer.truncation_length=512 \
    sampling.max_length=512 \
    data.num_train=1500000 \
    data.num_val=1000 \
    data.use_cache=true \
    trainer.devices=4 \
    trainer.strategy=ddp \
    trainer.max_epochs=5 \
    trainer.max_steps=-1 \
    trainer.val_check_interval=416 \
    data.batch_size=96 \
    model.learning_rate=6.0e-4 \
    model.warmup_steps=416 \
    sampling.num_samples=0 \
    wandb.enabled=true \
    wandb.project=molecular-graph-gen \
    wandb.name=hsent-1600k-ddp-4gpu-5e-median \
    wandb.log_model=true \
    wandb.log_graphs=true
```

---

## Precomputation (tokenized cache)

Precomputing builds tokenized cache files so training can skip on-the-fly tokenization. Only **SC (spectral)** and **HAC** coarsening need this; MC and MAS are fast enough on-the-fly.

### Precompute with bash script (recommended)

Uses parallel **screen** sessions for large MOSES runs. Optionally use precomputed SMILES so chunks read from one file instead of CSV (run `python scripts/export_moses_smiles.py` first).

```bash
# H-SENT + spectral, 8 chunks (default)
bash bash_scripts/precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc --chunks=8

# H-SENT + spectral, 50K train, 1K val, 4 chunks
./bash_scripts/precompute_benchmarks.sh \
  --tokenizer=hsent \
  --coarsening=sc \
  --train-samples=50000 \
  --val-samples=1000 \
  --chunks=4

# HDT + spectral, 500K train, 1K val, 4 chunks
./bash_scripts/precompute_benchmarks.sh \
  --tokenizer=hdt \
  --coarsening=sc \
  --train-samples=500000 \
  --val-samples=1000 \
  --chunks=4

# Use precomputed SMILES file (faster; run export_moses_smiles.py first)
./bash_scripts/precompute_benchmarks.sh --use-precomputed-smiles --tokenizer=hsent --coarsening=sc --chunks=4

# 1.5M train samples, 4 chunks
./bash_scripts/precompute_benchmarks.sh \
  --train-samples=1500000 \
  --chunks=4 \
  --tokenizer=hsent \
  --coarsening=sc
```

### Stop precompute screen sessions

If you need to cancel running precompute jobs:

```bash
./bash_scripts/stop_precompute_benchmarks.sh --tokenizer=hdt --coarsening=sc --chunks=4

# Stop all MOSES + COCONUT precompute screens
./bash_scripts/stop_precompute_benchmarks.sh --all
```

### Cache usage in training

After precomputing, point training at the cache:

```bash
python scripts/train.py ... data.use_cache=true
```

Cache files live under `data/cache/` with names like `moses_train_hsent_50000_<hash>.pt`. The hash includes tokenizer and spectral settings (`n_init`, `k_min_factor`, `k_max_factor`) so different configs get different caches.

---

## Evaluation and comparison

### Single checkpoint test

```bash
python scripts/test.py \
    model.checkpoint_path=outputs/train/.../best.ckpt \
    sampling.num_samples=1000

# Use precomputed SMILES and/or precomputed PGD reference graphs (faster repeated evals)
python scripts/test.py \
    model.checkpoint_path=outputs/train/.../best.ckpt \
    data.use_precomputed_smiles=true \
    metrics.reference_graphs_path=outputs/eval_run/reference_graphs/reference_graphs_moses_test_100.pt
```

### Precompute PGD reference graphs (once per eval run)

Speeds up evaluation when running many checkpoints: convert reference SMILES to graphs once, then pass the `.pt` path to each `test.py` run.

```bash
python scripts/precompute_reference_graphs.py experiment=moses reference_graphs.output_dir=outputs/eval_run
# Prints path, e.g. outputs/eval_run/reference_graphs/reference_graphs_moses_test_100.pt
```

### eval_benchmarks_auto.sh (mapping file + optional caching)

Evaluates a **list of checkpoints** from a mapping file, optionally uses precomputed SMILES and reference graphs, and can skip runs that already have results.

**Mapping file format** (one line per checkpoint; order = column order in comparison table):

```
directory_name
directory_name    Display Label
# comments and empty lines are skipped
```

Checkpoints are looked for under `BENCHMARK_DIR` (default `outputs/benchmark`) as `BENCHMARK_DIR/<directory_name>/best.ckpt` or `last.ckpt`.

```bash
./bash_scripts/eval_benchmarks_auto.sh \
    outputs/benchmark/compare_ckpts.txt \
    outputs/eval \
    --last \
    --core-only \
    --use-precomputed-smiles
```

- `--last`: use `last.ckpt` instead of `best.ckpt`
- `--core-only`: only validity, uniqueness, novelty (no FCD, PGD, motif, realistic_gen)
- `--use-precomputed-smiles`: pass `data.use_precomputed_smiles=true` to each test run and use precomputed SMILES when precomputing reference graphs
- `--recompute DIR1,DIR2`: force re-run test/gen for those directory names

The script precomputes reference graphs once and passes `metrics.reference_graphs_path` to each `test.py` so all checkpoints share the same PGD reference set.

### compare_results.py

Build a comparison table image from test (and optional realistic_gen) outputs:

```bash
python scripts/compare_results.py
python scripts/compare_results.py --filter "moses"
python scripts/compare_results.py --output comparison.png
python scripts/compare_results.py --all    # show all runs, not just best per tokenizer+coarsening
python scripts/compare_results.py --test-only   # exclude realistic gen metrics
```

Table sections include Training Info (e.g. `coarsening_strategy`, `reference_split`, `generation_time`), Core Quality, Distribution Matching, Structural, Motif MMD, and Realistic Generation.

---

## Config overrides (quick reference)

| Override | Effect |
|----------|--------|
| `data.use_precomputed_smiles=true` | Load train/test from `data/moses_smiles/moses_smiles.txt` (run `export_moses_smiles.py` first) |
| `data.use_cache=true` | Use precomputed tokenized cache from `data/cache/` (run `precompute_benchmarks.sh` first) |
| `metrics.reference_graphs_path=<.pt path>` | Load PGD reference graphs from file instead of converting SMILES each run |
| `metrics.core_only=true` | Only validity, uniqueness, novelty |

See [setup_training.md](setup_training.md) and [bash_scripts/README.md](../bash_scripts/README.md) for full option lists.
