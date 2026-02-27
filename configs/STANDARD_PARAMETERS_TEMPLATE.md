# Standard parameters template

**Instructions:** Fill in the `YOUR_VALUE` placeholders (or replace the suggested value) for each parameter. Leave a value as-is if you want to keep the current default. When done, tell the assistant to "transfer the template into the configs" and they will update `train.yaml`, `test.yaml`, `realistic_gen.yaml`, and the experiment/tokenizer configs accordingly.

Use this either **section by section** (edit one block, then we sync that block to configs) or **fill everything**, then one full transfer.

---

## 1. Global (train, test, realistic_gen)

| Parameter | Current | Used in | YOUR_VALUE |
|-----------|---------|---------|------------|
| seed | 42 | train, test, realistic_gen | |
| resume | false | train only | |

```yaml
# → train.yaml (and test/realistic_gen for seed)
seed: 42        # current: 42
resume: false      # current: false (train only)
```

---

## 2. Model

| Parameter | Current | Used in | YOUR_VALUE |
|-----------|---------|---------|------------|
| model_name | gpt2-xs | train | |
| learning_rate | 8.49e-4 | train | |
| weight_decay | 0.1 | train | |
| warmup_steps | 1414 | train | |
| checkpoint_path | (required at runtime) | test, realistic_gen | (leave placeholder in yaml) |
| is_autograph | false | test | |
| pretrained_path | null | train (coconut finetune) | (experiment override only) |

```yaml
# → train.yaml
model:
  model_name: gpt2-xs
  learning_rate: 8.49e-4
  weight_decay: 0.1
  warmup_steps: 1414
  max_steps: ${trainer.max_steps}

# → test.yaml (checkpoint_path required at runtime; do not set default here)
model:
  checkpoint_path: ___   # placeholder; must be passed
  is_autograph: false
```

---

## 3. Data (shared where applicable)

| Parameter | Current (train / test / real) | Used in | YOUR_VALUE |
|-----------|------------------------------|---------|------------|
| batch_size | 32 / 32 / 32 | all | |
| num_workers | 8 / 0 / 0 | train vs test/real | |
| test_num_workers | 0 | train, test | |
| include_hydrogens | false | all | |
| data_root | data | all | |
| use_cache | true | train | |
| cache_dir | data/cache | train | |
| use_precomputed_smiles | true | test, realistic_gen | |
| precomputed_smiles_dir | data/moses_smiles | test, realistic_gen | |
| min_atoms | 20 (code default) / 20 | train, test, coconut | |
| max_atoms | 100 (code default) / 100 | train, test, coconut | |
| min_rings | 3 (code default) / 3 | train, test, coconut | |

```yaml
# → train.yaml
data:
  batch_size: 32
  num_workers: 8
  test_num_workers: 0
  include_hydrogens: false
  data_root: data
  use_cache: true
  cache_dir: data/cache

# → test.yaml, realistic_gen.yaml
data:
  batch_size: 32
  num_workers: 0
  include_hydrogens: false
  data_root: data
  use_precomputed_smiles: true
  precomputed_smiles_dir: data/moses_smiles
```

---

## 4. Trainer (train only)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| target_samples_seen | 16000000 (train default) | |
| max_epochs | null | |
| max_steps | 250000 (train); experiments override | |
| val_checks_per_epoch | 5 (preferred) | |
| validate_every_n_epochs | 1 (preferred) | |
| val_check_interval | 1000 (legacy fallback, steps-based) | |
| check_val_every_n_epoch | 1 (legacy fallback) | |
| limit_val_batches | (not in yaml; code default 1.0) | |
| precision | 32 | |
| gradient_clip_val | 1.0 | |
| accelerator | auto | |
| devices | 1 | |
| num_nodes | 1 | |
| strategy | auto | |
| accumulate_grad_batches | 1 | |
| num_sanity_val_steps | (not in yaml; code default 2) | |

```yaml
# → train.yaml (base)
trainer:
  target_samples_seen: 16000000   # primary budget; derives max_steps from effective_batch_size
  max_epochs: null
  max_steps: 250000             # fallback only when target_samples_seen is null
  val_checks_per_epoch: 5
  validate_every_n_epochs: 1
  val_check_interval: ___      # legacy fallback (used only when val_checks_per_epoch is null)
  check_val_every_n_epoch: 1 # legacy fallback
  precision: 32
  gradient_clip_val: 1.0
  accelerator: auto
  devices: 1
  num_nodes: 1
  strategy: auto
  accumulate_grad_batches: ___
```

---

## 5. Sampling (unified standard for train + test + eval)

| Parameter | Current train | Current test | Current real | YOUR_VALUE |
|-----------|---------------|--------------|--------------|------------|
| num_samples | 1000 | 1000 | 1000 (generation) | |
| batch_size | 32 | 32 | 32 | |
| top_k | 10 | 10 | 10 | |
| temperature | 1.0 | 1.0 | 1.0 | |
| max_length | 512 | 512 | 512 | |

```yaml
# → train.yaml, test.yaml, realistic_gen.yaml (same values for standard)
sampling:
  num_samples: 1000   # e.g. 500 for eval; 100 for train post-run
  batch_size: 32
  top_k: 10
  temperature: 1.0
  max_length: 512   # Base default; coconut experiment overrides to 2048
```

---

## 6. Tokenizer (defaults per tokenizer type)

| Parameter | sent | hsent | hdt | hdtc | YOUR_VALUE (if change) |
|-----------|------|-------|-----|------|------------------------|
| type | sent | hsent | hdt | hdtc | (identifier) |
| max_length | -1 | -1 | -1 | -1 | |
| truncation_length | 2048 | 2048 | 2048 | 2048 | |
| undirected | true | - | - | - | |
| labeled_graph | true | true | true | true | |
| node_order | - | BFS | BFS | BFS | |
| min_community_size | - | 4 | 4 | - | |
| coarsening_strategy | - | spectral | spectral | - | |
| motif_alpha | - | 1.0 | 1.0 | - | |
| normalize_by_motif_size | - | false | false | - | |
| include_rings | - | - | - | true | |

```yaml
# → configs/tokenizer/sent.yaml (base)
type: sent
max_length: ___
truncation_length: 2048   # Base; override via experiment (moses=512, coconut=2048, qm9=512)
undirected: true
labeled_graph: true

# → configs/tokenizer/hsent.yaml, hdt.yaml (base)
type: hsent  # or hdt
max_length: ___
truncation_length: 2048   # Base; override via experiment (moses=512, coconut=2048, qm9=512)
node_order: ___
min_community_size: ___
coarsening_strategy: ___
motif_alpha: ___
normalize_by_motif_size: ___
labeled_graph: ___

# → configs/tokenizer/hdtc.yaml (base)
type: hdtc
max_length: ___
truncation_length: 2048   # Base; override via experiment (moses=512, coconut=2048, qm9=512)
node_order: ___
include_rings: ___
labeled_graph: ___
```

---

## 7. Logs

| Parameter | train | test | realistic_gen | YOUR_VALUE |
|-----------|-------|------|---------------|------------|
| base_dir | outputs/train | outputs/test | outputs/realistic_gen | |
| run_name | ${data.dataset_name}_${tokenizer.type}_n${data.num_train} | ${data.dataset_name}_${tokenizer.type}_gen${sampling.num_samples} | ${data.dataset_name}_${tokenizer.type} | (or keep) |
| path | ${logs.base_dir}/${logs.run_name}_${now:...} | (same pattern) | (same pattern) | (keep formula) |

```yaml
# → train.yaml
logs:
  base_dir: ___
  run_name: ___
  path: ${logs.base_dir}/${logs.run_name}_${now:%Y%m%d-%H%M%S}

# → test.yaml, realistic_gen.yaml (run_name differs)
logs:
  base_dir: ___
  run_name: ___   # test: dataset_tokenizer_gen${sampling.num_samples}; real: dataset_tokenizer
  path: ${logs.base_dir}/${logs.run_name}_${now:%Y%m%d-%H%M%S}
```

---

## 8. Wandb (train only)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| enabled | true | |
| project | molecular-graph-gen | |
| entity | null | |
| name | null | |
| tags | [] | |
| notes | null | |
| log_model | true | |
| log_graphs | true | |
| max_logged_molecules | 12 | |
| eval_every_n_val | 250 | |
| eval_num_samples | 5 | |

```yaml
# → train.yaml
wandb:
  enabled: true
  project: molecular-graph-gen
  entity: ___
  name: ___
  tags: []
  notes: ___
  log_model: true
  log_graphs: true
  max_logged_molecules: ___
  eval_every_n_val: ___
  eval_num_samples: ___
```

---

## 9. Metrics (test only)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| core_only | false | |
| compute_motif | true | |
| compute_fcd | true | |
| compute_pgd | true | |
| pgd_reference_size | 500 | |
| reference_size | null (auto target = 10% train; capped to available pool) | |
| reference_split | "test" | |
| reference_graphs_path | null | |

```yaml
# → test.yaml
metrics:
  core_only: false
  compute_motif: true
  compute_fcd: true
  compute_pgd: true
  pgd_reference_size: 1000   # should be >= sampling.num_samples for full PGD
  reference_size: ___   # null/<=0 => auto target 10% of train size (capped by available pool)
  reference_split: test
  reference_graphs_path: null
```

---

## 10. Visualization (test only)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| enabled | true | |
| max_molecules | 12 | |
| dpi | 150 | |

```yaml
# → test.yaml
visualization:
  enabled: true
  max_molecules: 12
  dpi: 150
```

---

## 11. Generation (realistic_gen only)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| num_samples | 500 | |
| batch_size | 32 | |

```yaml
# → realistic_gen.yaml
generation:
  num_samples: 1000
  batch_size: 32
```

---

## 12. Experiment overrides (only if you want to change dataset-specific defaults)

These override the base train (and test when they use the same experiment). Fill only if you want a different standard per dataset.

### MOSES (experiment/moses.yaml)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| data.dataset_name | moses | moses |
| data.num_train | 1000000 | 500000 |
| data.num_val | 1000 | 1000 |
| data.num_test | 1000 | 1000 |
| trainer.max_steps | 500000 | 250000 |
| trainer.val_checks_per_epoch | 5 | |
| trainer.validate_every_n_epochs | 1 | |
| sampling.num_samples | 100 | (train post-run) |
| sampling.top_k | 10 | |

### COCONUT (experiment/coconut.yaml)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| data.dataset_name | coconut | |
| data_file | data/coconut_complex.smi | |
| data.num_train | 5000 | |
| data.num_val | 500 | |
| data.num_test | 500 | |
| data.min_atoms | 20 | |
| data.max_atoms | 100 | |
| data.min_rings | 3 | |
| model.learning_rate | 1e-5 | |
| trainer.max_steps | 50000 | |
| trainer.val_checks_per_epoch | 1 | |
| trainer.validate_every_n_epochs | 1 | |
| tokenizer.truncation_length | 2048 | |
| sampling.num_samples | 1000 | |
| sampling.top_k | 10 | |
| sampling.max_length | 2048 | |

### QM9 (experiment/qm9.yaml)

| Parameter | Current | YOUR_VALUE |
|-----------|---------|------------|
| data.dataset_name | qm9 | |
| data.num_train | null | |
| data.num_val | null | |
| data.num_test | null | |
| trainer.max_steps | 50000 | |
| trainer.val_checks_per_epoch | 1 | |
| trainer.validate_every_n_epochs | 1 | |
| tokenizer.truncation_length | 512 | |
| sampling.num_samples | 1000 | |
| sampling.top_k | 50 | |

---