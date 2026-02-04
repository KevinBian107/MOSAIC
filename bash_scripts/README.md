# Bash Scripts

Utility scripts for running batch operations.

## finetune_benchmarks.sh

Fine-tunes all pretrained models in `outputs/benchmark/` on the COCONUT complex natural products dataset for transfer learning evaluation.

### What it does

1. Finds all `best.ckpt` files in `outputs/benchmark/`
2. Extracts tokenizer type and coarsening strategy from directory names
3. Runs `scripts/finetune.py` on each checkpoint with appropriate settings
4. Saves fine-tuned models to `outputs/finetune/`

### Usage

```bash
# Run from project root
./bash_scripts/finetune_benchmarks.sh

# Dry run (show commands without executing)
./bash_scripts/finetune_benchmarks.sh --dry-run

# Disable WandB logging
./bash_scripts/finetune_benchmarks.sh --no-wandb

# Custom number of training steps (default: 50000)
./bash_scripts/finetune_benchmarks.sh --steps=100000

# Show help
./bash_scripts/finetune_benchmarks.sh --help
```

### Output

Fine-tuned models are saved to:
- `outputs/finetune/coconut_{tokenizer}_{coarsening}/best.ckpt` - Best checkpoint
- `outputs/finetune/coconut_{tokenizer}_{coarsening}/last.ckpt` - Last checkpoint
- `outputs/finetune/coconut_{tokenizer}_{coarsening}/config.yaml` - Training config

### Prerequisites

Before running, ensure you have prepared the COCONUT dataset:

```bash
# Download and prepare COCONUT data (~191MB download)
python scripts/prepare_coconut_data.py
```

This creates `data/coconut_complex.smi` with ~10,000 complex natural products.

---

## eval_benchmarks.sh

Evaluates all checkpoints in `outputs/benchmark/` and generates a comparison table.

### What it does

1. Finds all `best.ckpt` files in `outputs/benchmark/`
2. Runs `scripts/test.py` on each checkpoint (computes validity, uniqueness, novelty, FCD, PGD, etc.)
3. Runs `scripts/realistic_gen.py` on each checkpoint (computes motif rate, substitution patterns, etc.)
4. Runs `scripts/compare_results.py` to generate a comparison table PNG

### Usage

```bash
# Run from project root
./bash_scripts/eval_benchmarks.sh

# Only run test.py (skip realistic generation analysis)
./bash_scripts/eval_benchmarks.sh --test-only

# Only run realistic_gen.py (skip standard test metrics)
./bash_scripts/eval_benchmarks.sh --gen-only

# Show help
./bash_scripts/eval_benchmarks.sh --help
```

### Output

Results are saved to:
- `outputs/test/{run_name}/results.json` - Test metrics
- `outputs/realistic_gen/{run_name}/results.json` - Realistic generation metrics
- `outputs/test/comparison.png` - Comparison table image

### Adding checkpoints

Place your trained checkpoints in `outputs/benchmark/`. The script looks for `best.ckpt` files:

```
outputs/benchmark/
├── my_model_1/
│   └── best.ckpt
├── my_model_2/
│   └── best.ckpt
└── ...
```

### Naming convention

The script automatically extracts tokenizer type and coarsening strategy from directory names.

**Directory name pattern:** `moses_{tokenizer}_{coarsening}_{other}_...`

**Tokenizer types:**
- `hsent` - Hierarchical SENT (supports coarsening)
- `hdt` - Hierarchical DFS-based Tokenizer (supports coarsening)
- `hdtc` - HDT Compositional (no coarsening, uses FunctionalHierarchyBuilder)
- `sent` - Flat SENT (no coarsening)

**Coarsening strategies** (only for `hsent` and `hdt`):
- `mc` → `motif_community` - Direct motif-based community assignment
- `sc` → `spectral` - Standard spectral clustering (default)
- `mas` → `motif_aware_spectral` - Spectral clustering with motif preservation

**Examples:**
```
moses_hsent_mc_n100000_20260126/   → tokenizer=hsent, coarsening=motif_community
moses_hdt_sc_n100000_20260126/    → tokenizer=hdt, coarsening=spectral
moses_hsent_mas_n100000_20260126/ → tokenizer=hsent, coarsening=motif_aware_spectral
moses_hdtc_n100000_20260126/      → tokenizer=hdtc (no coarsening)
moses_sent_n100000_20260126/      → tokenizer=sent (no coarsening)
```
