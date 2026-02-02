# Bash Scripts

Utility scripts for running batch operations.

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
