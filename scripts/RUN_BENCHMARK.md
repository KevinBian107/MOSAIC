# How to Run Spectral Clustering Benchmark

## Prerequisites

The script needs to be run in an environment with:
- torch_geometric
- rdkit
- sklearn
- numpy
- torch

## Running the Benchmark

```bash
# Activate your conda environment (e.g., mosaic or the one with torch_geometric)
conda activate <your_env_name>

# Run with 10 samples (quick test)
python scripts/benchmark_spectral.py --num-samples 10 --output spectral_benchmark_10samples.json

# Run with more samples for better statistics
python scripts/benchmark_spectral.py --num-samples 500 --output spectral_benchmark.json
```

## Samples Used

The benchmark uses:

1. **MOSES Dataset**: First N molecules from MOSES training split
   - Diverse drug-like molecules
   - Various sizes (small to medium-large)
   - Includes molecules with different structural features

2. **Additional Test Molecules** (for diversity):
   - `c1ccccc1` - benzene (small, symmetric, 6 atoms)
   - `c1ccc2ccccc2c1` - naphthalene (medium, fused rings, 10 atoms)
   - `CC(=O)OC1=CC=CC=C1C(=O)O` - aspirin (medium, functional groups, ~21 atoms)
   - `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` - caffeine (medium-large, heterocycles, ~24 atoms)
   - `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O` - ibuprofen (medium-large, branched, ~26 atoms)

Total: N MOSES molecules + 5 test molecules

## What Gets Tested

The script tests 16 different configurations:

1. **Baseline**: k=[0.7,1.3], n_init=10, kmeans (current defaults)
2. **n_init variations**: 1, 5, 10, 20, 50, 100
3. **k-range variations**: [0.8,1.2], [0.9,1.1], [0.95,1.05]
4. **assign_labels**: kmeans vs discretize
5. **Combined optimizations**: tighter k-range + lower n_init

## Output

Results are saved as JSON with:
- Time metrics (mean, median, std, throughput)
- Quality metrics (modularity mean, median, std, min, max)
- Number of communities found
- Speedup vs baseline
- Quality ratio vs baseline

## Expected Runtime

- 10 samples: ~2-5 minutes
- 500 samples: ~30-60 minutes
- 1000 samples: ~1-2 hours
