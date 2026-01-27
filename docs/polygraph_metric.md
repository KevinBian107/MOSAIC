# PolyGraph Discrepancy (PGD) Metric

## Overview

PolyGraph Discrepancy (PGD) is a classifier-based metric for evaluating graph generation quality, introduced in the PolyGraph benchmark framework (arXiv:2510.06122). It provides a standardized, bounded score in [0, 1] that measures how distinguishable generated graphs are from reference graphs.

## What is PGD?

PGD trains a binary classifier to distinguish between generated and reference graphs using learned graph descriptors. The classifier's accuracy serves as the metric: a score of 0 means the generated graphs are indistinguishable from the reference distribution (perfect), while 1 means they are easily distinguishable (poor).

### Key Advantages

1. **Bounded Range**: Score ∈ [0, 1] provides interpretable quality assessment
2. **No Hyperparameters**: Unlike MMD which requires kernel bandwidth tuning
3. **Multi-Descriptor**: Aggregates information across multiple graph properties
4. **Learned Features**: GIN-based descriptors capture complex distributional differences

### Interpretation Guidelines

- **< 0.1**: Excellent - generated distribution matches reference very closely
- **< 0.3**: Good - generated distribution is similar to reference
- **< 0.5**: Moderate - noticeable differences but reasonable quality
- **≥ 0.5**: Poor - generated graphs easily distinguishable from reference

## How PGD Works

### 1. Graph Descriptor Extraction

PGD uses Graph Isomorphism Networks (GIN) to extract feature vectors from both generated and reference graphs. These descriptors capture:
- Node degree distributions
- Spectral properties
- Clustering coefficients
- Higher-order structural patterns

### 2. Binary Classification

A TabPFN (Tabular Prior-Fitted Network) classifier is trained to distinguish:
- Class 0: Reference graphs (from training dataset)
- Class 1: Generated graphs (from model output)

### 3. Metric Computation

The PGD score is derived from the classifier's ability to separate the two distributions. Lower scores indicate the classifier struggles to distinguish them, meaning the generator produces realistic graphs.

### 4. Balanced Dataset Requirement

**Critical Implementation Detail**: PGD requires equal numbers of reference and generated graphs for proper binary classification. Our implementation handles this by:

```python
if len(reference_graphs) > len(generated_graphs):
    sampled_ref = random.sample(reference_graphs, len(generated_graphs))
```

This ensures balanced training without biasing the classifier toward either class.

## Sample Size Recommendations

Based on the PolyGraph paper findings:

### Minimum for Stability
- **~256 samples**: PGD's mean and variance stabilize at this threshold
- Below this, scores may have high variance and be unreliable

### Recommended for Benchmarks
- **1000-2048 samples**: Ideal range for stable, reproducible measurements
- The paper introduces datasets with 2048 samples to address traditional benchmarks that used only 20-40 graphs

### Our Implementation
- **Generated**: 1000 samples (default configuration)
- **Reference**: 5000 samples (configurable via `metrics.pgd_reference_size`)
- **Effective**: 875-1000 balanced pairs (after validity filtering)

This configuration is well above the stability threshold and follows best practices from the literature.

### Why Sample Size Matters

Traditional graph generation benchmarks use tiny datasets (20-40 graphs), leading to:
- High bias and variance in distribution metrics (especially MMD)
- Unreliable model rankings that change with different random seeds
- Poor reproducibility across evaluations

PGD addresses this with larger sample requirements but provides much more stable assessments.

## Implementation Details

### Basic Usage

```python
from src.evaluation.polygraph_metric import PolygraphMetric

# Initialize with reference graphs
metric = PolygraphMetric(
    reference_graphs=reference_graphs,  # List of PyG Data objects
    max_reference_size=5000,            # Memory constraint
)

# Compute PGD score
results = metric(generated_graphs)
pgd_score = results["pgd"]  # Score in [0, 1], lower is better
```

### Configuration Parameters

In `configs/test.yaml`:

```yaml
metrics:
  compute_pgd: true              # Enable/disable PGD computation
  pgd_reference_size: 5000       # Max reference graphs to use
```

### Memory Considerations

PGD computation involves:
1. Converting SMILES to graphs (~5000 reference molecules)
2. GIN descriptor extraction (neural network forward passes)
3. TabPFN classifier training (limited to 10k samples total)

The `pgd_reference_size` parameter controls memory usage. Recommended values:
- **5000**: Good balance (default)
- **10000**: Maximum for single evaluation
- **2000**: Minimum for stable estimates

### Integration with MOSAIC

PGD complements MOSAIC's existing evaluation metrics:

| Metric Type | What It Measures | MOSAIC Implementation |
|-------------|------------------|----------------------|
| **Chemical Validity** | Valid molecules | validity, uniqueness, novelty |
| **Chemical Similarity** | Nearest-neighbor distance | SNN, fragment/scaffold similarity |
| **Distribution Quality** | Generated vs reference | **PGD (new)**, FCD |
| **Motif Preservation** | Functional group patterns | MSC, MFD, MPR (MOSAIC-specific) |

### Comparison with Other Metrics

#### vs. Maximum Mean Discrepancy (MMD)
- **PGD**: Bounded [0,1], no hyperparameters, stable with 256+ samples
- **MMD**: Unbounded, requires kernel bandwidth tuning, high variance at small sample sizes

#### vs. Fréchet ChemNet Distance (FCD)
- **PGD**: General graph structure, works on any graph type
- **FCD**: Molecular-specific, uses ChemNet features trained on drug-like molecules

#### vs. Motif Metrics (MSC/MFD/MPR)
- **PGD**: Overall distribution matching, general structural patterns
- **Motif Metrics**: Specific functional group preservation (MOSAIC's unique contribution)

## Results Interpretation

### Example Evaluation Results

```json
{
  "pgd": 0.176,
  "fcd": 4.789,
  "validity": 0.738
}
```

**Interpretation**:
- PGD of 0.176 is **good** (< 0.3 threshold)
- Generated distribution closely matches reference
- Model captures structural patterns effectively

### Hierarchical vs Flat Tokenization

From our MOSES evaluations:

| Model | PGD ↓ | FCD ↓ | Interpretation |
|-------|-------|-------|----------------|
| SENT (flat) | 0.309 | 7.577 | Good, but room for improvement |
| H-SENT | **0.176** | 4.789 | **Excellent distribution matching** |
| HDT | 0.208 | **4.549** | Good distribution, best FCD |

The hierarchical tokenizers (H-SENT, HDT) significantly outperform flat SENT on distribution quality (PGD, FCD), demonstrating that explicit motif encoding improves generation fidelity.

## Best Practices

### When to Use PGD

✅ **Use PGD when:**
- Comparing different generation methods on the same dataset
- Evaluating how well a model captures the training distribution
- Benchmarking against published baselines (PolyGraph paper includes many)
- You have 256+ generated and reference graphs

❌ **Don't rely solely on PGD when:**
- Sample size < 256 (use with caution)
- Molecular properties matter more than structure (use FCD, QED, SA)
- Specific motifs are critical (combine with MOSAIC's motif metrics)

### Recommended Evaluation Protocol

For comprehensive molecular generation evaluation, compute:

1. **Basic Quality**: validity, uniqueness, novelty
2. **Chemical Similarity**: SNN, fragment/scaffold similarity
3. **Distribution Matching**: PGD (structure), FCD (chemistry)
4. **Motif Preservation**: MSC, MFD (MOSAIC-specific)

This provides coverage across validity, diversity, distribution quality, and functional group preservation.

## References

- **PolyGraph Paper**: arXiv:2510.06122 - "PolyGraph Discrepancy: a classifier-based metric for graph generation"
- **Implementation**: `polygraph-benchmark` package (https://github.com/BorgwardtLab/polygraph-benchmark)
- **MOSAIC Integration**: `src/evaluation/polygraph_metric.py`

## Citation

If you use PGD in your research, please cite:

```bibtex
@article{polygraph2024,
  title={PolyGraph Discrepancy: A Classifier-Based Metric for Graph Generation},
  url={https://arxiv.org/abs/2510.06122},
  year={2024}
}
```
