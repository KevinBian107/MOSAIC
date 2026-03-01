# Configs

[Hydra](https://hydra.cc/)-based configuration. Base configs define shared defaults; experiment configs override dataset-specific values.

## Override Order

```
tokenizer (sent/hsent/hdt/hdtc)  →  base (train.yaml / test.yaml)  →  experiment (moses / coconut)
```

Experiment configs are applied **last**, so any value defined in both base and experiment uses the experiment value.

## Base Configs

| File | Purpose |
|------|---------|
| `train.yaml` | Training (model, optimizer, trainer, sampling, logging) |
| `test.yaml` | Evaluation metrics (validity, FCD, PGD, motif) |
| `realistic_gen.yaml` | Generation quality analysis |

## Experiment Configs

| File | Dataset |
|------|---------|
| `experiment/moses.yaml` | MOSES (drug-like molecules from ZINC) |
| `experiment/coconut.yaml` | COCONUT (complex natural products) |

## Parameter Comparison

### Training

| Parameter | Base (`train.yaml`) | MOSES | COCONUT |
|-----------|---------------------|-------|---------|
| `data.dataset_name` | — | moses | coconut |
| `data.num_train` | — | 500,000 | 5,000 |
| `data.num_val` | — | 1,000 | 500 |
| `data.num_test` | — | 1,000 | 500 |
| `data.batch_size` | 32 | — | — |
| `model.learning_rate` | 8.49e-4 | — | 6e-4 |
| `model.warmup_steps` | 1,414 | — | 1,000 |
| `trainer.target_samples_seen` | 16,000,000 | — | 1,600,000 |
| `trainer.max_steps` | 250,000 | 250,000 | 50,000 |
| `trainer.val_checks_per_epoch` | 5 | 5 | 1 |
| `tokenizer.truncation_length` | — | 512 | 2,048 |
| `sampling.num_samples` | 1,000 | 1,000 | 100 |
| `sampling.max_length` | 512 | — | 2,048 |

"—" means the experiment does not override the base value (or the base does not set it, deferring to the experiment).

### Evaluation (`test.yaml`)

| Parameter | Base (`test.yaml`) | MOSES | COCONUT |
|-----------|---------------------|-------|---------|
| `sampling.num_samples` | 1,000 | 1,000 | 100 |
| `sampling.max_length` | 512 | — | 2,048 |
| `metrics.core_only` | false | — | — |
| `metrics.compute_fcd` | true | — | — |
| `metrics.compute_pgd` | true | — | — |

## Tokenizer Configs

Located in `tokenizer/`. Each defines tokenizer-specific defaults (type, coarsening strategy, labeled_graph, etc.). Selected via `tokenizer=<name>` on the CLI or in defaults.

## Usage

```bash
# Train with defaults (HDTC on MOSES)
python scripts/train.py

# Train COCONUT with HDT tokenizer
python scripts/train.py tokenizer=hdt experiment=coconut

# Override any parameter from CLI
python scripts/train.py model.learning_rate=1e-3 trainer.max_steps=100000
```
