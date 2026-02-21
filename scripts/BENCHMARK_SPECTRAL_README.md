# Spectral Clustering Benchmark

This script benchmarks different parameter combinations for `SpectralCoarsening` to find optimal trade-offs between compute efficiency and modularity quality.

## Usage

```bash
# Run with default settings (500 graphs)
python scripts/benchmark_spectral.py

# Run with more samples for better statistics
python scripts/benchmark_spectral.py --num-samples 1000

# Specify output file
python scripts/benchmark_spectral.py --output results/spectral_bench.json

# Use different random seed
python scripts/benchmark_spectral.py --seed 123
```

## What It Tests

The benchmark tests various combinations of:

1. **n_init**: Number of spectral clustering initializations (1, 5, 10, 20, 50, 100)
2. **k_min_factor / k_max_factor**: Range of cluster counts to search
   - Current: [0.7, 1.3] * sqrt(n)
   - Tighter: [0.9, 1.1] * sqrt(n) (faster, less exploration)
   - Very tight: [0.95, 1.05] * sqrt(n) (fastest, minimal exploration)
3. **assign_labels**: Label assignment method
   - "kmeans": Current default (slower but more appropriate for molecular graphs)
   - "discretize": Faster but forces equal-sized partitions (may be inappropriate)

## Metrics Collected

For each configuration:

- **Time metrics**:
  - Mean/median/std time per graph
  - Total time
  - Throughput (graphs/second)

- **Quality metrics**:
  - Mean/median/std modularity scores
  - Min/max modularity
  - Mean number of communities found

## Output

Results are saved as JSON with:
- Baseline configuration (current defaults)
- All tested configurations with metrics
- Summary statistics comparing speedups and quality ratios

## Interpreting Results

The script automatically identifies:

1. **Baseline**: Current defaults (k=[0.7,1.3], n_init=10, kmeans)
2. **Speedup vs baseline**: How much faster each config is
3. **Quality ratio**: Modularity relative to baseline
4. **Best trade-offs**:
   - Fastest configuration with >95% quality
   - Highest quality configuration

## Example Output

```
Baseline (current defaults):
  Time: 0.0234s/graph
  Modularity: 0.4523

Speedup vs baseline:
  0.90-1.10, n_init=5, kmeans: 1.85x speedup, 0.987x modularity
  0.90-1.10, n_init=1, kmeans: 2.34x speedup, 0.956x modularity

Best configurations:
  Fastest with >95% quality: k=[0.90,1.10], n_init=5, kmeans
    1.85x speedup, 0.987x modularity
  Highest quality: k=[0.70,1.30], n_init=50, kmeans
    Modularity: 0.4634
```

## Recommendations

Based on results, you can:

1. **Maximize speed**: Use tighter k range + lower n_init (if quality acceptable)
2. **Maximize quality**: Use wider k range + higher n_init (if time acceptable)
3. **Balance**: Find config with >95% quality but significant speedup

## Notes

- Tests on diverse molecular graphs from MOSES dataset
- Includes small (benzene) to medium-large (ibuprofen) molecules
- Results may vary with graph size distribution
- For production use, test on your specific dataset distribution
