# Evaluation Metrics

This document describes the evaluation metrics used in MOSAIC for assessing molecular graph generation quality.

---

## Overview

Metrics are organized into three categories:

| Category | Purpose | Key Metrics |
|----------|---------|-------------|
| **Graph Metrics** | Structural similarity | Degree, Spectral, Clustering MMD, PGD |
| **Molecular Metrics** | Chemical validity & diversity | Validity, Uniqueness, Novelty, FCD, SNN |
| **Motif Metrics** | Substructure preservation | Motif distribution, Ring systems, BRICS |

---

## 1. Graph Metrics

Structural metrics comparing generated graphs to reference distributions using Maximum Mean Discrepancy (MMD).

### Maximum Mean Discrepancy (MMD)

Measures distributional distance between two sample sets:

$$\text{MMD}^2 = \mathbb{E}[k(x,x')] + \mathbb{E}[k(y,y')] - 2\mathbb{E}[k(x,y)]$$

Where k is a kernel function (Gaussian, TV, or EMD-based).

### Degree Distribution MMD

Compares degree histograms between generated and reference graphs.

$$\text{deg\_hist}(G) = [|\{v : \deg(v) = 0\}|, |\{v : \deg(v) = 1\}|, \ldots]$$

### Spectral Distribution MMD

Compares eigenvalue distributions of normalized Laplacian matrices.

$$L_{\text{norm}} = I - D^{-1/2} A D^{-1/2}$$

Eigenvalues are binned into histograms and compared via MMD.

### Clustering Coefficient MMD

Compares distributions of local clustering coefficients.

$$C_v = \frac{2 \cdot |\{(u,w) : u,w \in N(v), (u,w) \in E\}|}{|N(v)| \cdot (|N(v)| - 1)}$$

### PolyGraph Discrepancy (PGD)

**Type:** Classifier-based graph distribution discrepancy

**Range:** [0, 1] (lower is better)

A classifier-based metric that measures the quality of generated graphs by training a binary classifier to distinguish between reference and generated distributions. The classifier's ability to discriminate serves as a measure of distribution mismatch.

**Advantages over MMD:**
- Bounded range [0, 1] for interpretability
- No hyperparameter tuning required (MMD needs kernel bandwidth)
- Captures complex distributional differences through learned representations
- Single unified score across multiple graph properties

**Formulation:**
The metric trains a classifier $f: \mathcal{G} \to \{0, 1\}$ to distinguish between:
- Real graphs $G_r \sim P_{ref}$ (labeled 0)
- Generated graphs $G_g \sim P_{gen}$ (labeled 1)

$$\text{PGD} = \text{accuracy}(f) - 0.5$$

Normalized to [0, 1] where:
- PGD ≈ 0: Generator produces distribution indistinguishable from reference (optimal)
- PGD ≈ 1: Distributions are easily distinguishable (poor quality)

**Interpretation:**
- PGD < 0.1: Excellent generation quality (indistinguishable from reference)
- PGD < 0.3: Good generation quality
- PGD < 0.5: Moderate quality (noticeable differences)
- PGD ≥ 0.5: Poor quality (easily distinguishable)

**Reference:**
- Paper: https://arxiv.org/abs/2510.06122
- GitHub: https://github.com/BorgwardtLab/polygraph-benchmark

---

## 2. Molecular Metrics

Chemical and structural metrics for molecular generation quality.

### Validity

Fraction of generated SMILES that parse to valid, sanitizable molecules.

$$\text{Validity} = \frac{|\{\text{valid molecules}\}|}{|\{\text{generated}\}|}$$

### Uniqueness

Fraction of unique molecules among valid generations (using canonical SMILES).

$$\text{Uniqueness} = \frac{|\{\text{unique canonical SMILES}\}|}{|\{\text{valid molecules}\}|}$$

### Novelty

Fraction of generated molecules not present in the training set.

$$\text{Novelty} = \frac{|\{\text{valid} \cap \text{not in training}\}|}{|\{\text{valid molecules}\}|}$$

### SNN (Similarity to Nearest Neighbor)

Average Tanimoto similarity to the nearest neighbor in the reference set using Morgan fingerprints.

$$\text{SNN} = \frac{1}{|G|} \sum_{g \in G} \max_{r \in R} \text{Tanimoto}(\text{fp}(g), \text{fp}(r))$$

### FCD (Fréchet ChemNet Distance)

Compares distributions of ChemNet activations between generated and reference molecules. Lower is better.

$$\text{FCD} = \|\mu_g - \mu_r\|^2 + \text{Tr}(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2})$$

### Fragment Similarity

Cosine similarity of BRICS fragment frequency distributions.

$$\text{FragSim} = \frac{\vec{f}_g \cdot \vec{f}_r}{\|\vec{f}_g\| \|\vec{f}_r\|}$$

Where $\vec{f}$ is the normalized frequency vector of BRICS fragments.

### Scaffold Similarity

Cosine similarity of Bemis-Murcko scaffold frequency distributions.

$$\text{ScaffSim} = \frac{\vec{s}_g \cdot \vec{s}_r}{\|\vec{s}_g\| \|\vec{s}_r\|}$$

### Internal Diversity

Average pairwise Tanimoto distance within generated molecules.

$$\text{IntDiv} = \frac{2}{n(n-1)} \sum_{i<j} (1 - \text{Tanimoto}(\text{fp}_i, \text{fp}_j))$$

---

## 3. Motif Distribution Metrics

Metrics for comparing motif distributions between generated and reference molecules.

### Functional Group MMD

MMD between RDKit functional group count vectors (82 patterns from `rdkit.Chem.Fragments`).

### SMARTS Motif MMD

MMD between vectors of SMARTS pattern match counts:

| Category | Patterns |
|----------|----------|
| Aromatic rings | benzene, pyridine, pyrrole, furan, thiophene, imidazole, pyrimidine, naphthalene |
| Functional groups | hydroxyl, carboxyl, carbonyl, aldehyde, ester, amide, amine (1°/2°/3°), nitro, nitrile |
| Halogens | F, Cl, Br, I |
| Others | ether, thioether, sulfone, sulfonamide, phosphate |

### Ring System MMD

MMD between ring system feature vectors:

- Number of rings (total, aromatic, aliphatic, saturated)
- Number of heterocycles (total, aromatic)
- Spiro atoms, bridgehead atoms
- Ring size distribution (3, 4, 5, 6, 7, 8-membered)

### BRICS Fragment MMD

L2 distance between normalized BRICS fragment frequency distributions.

---

## 4. Additional Motif Metrics

### Motif Histogram Distribution

Distribution comparison of individual motif frequencies across molecules.

**Purpose**: Measure whether the model generates motifs with the same frequency distribution as the training data.

**Approach**:
- For each motif type m, compute histogram of counts: P(count(m) = k)
- Compare histograms using KL divergence or Wasserstein distance

$$D_{\text{motif}}(m) = D_{KL}(P_{\text{gen}}(\text{count}(m)) \| P_{\text{ref}}(\text{count}(m)))$$

**Output Metrics**:
- `motif_hist_mean`: Average KL divergence across all motif types
- `motif_hist_max`: Maximum KL divergence (worst motif)
- `motif_hist_{name}`: Per-motif KL divergence

### Motif Co-occurrence (Combinations)

Measures which motifs tend to appear together in the same molecule.

**Purpose**: Capture higher-order structural patterns - certain motifs naturally co-occur (e.g., benzene + hydroxyl in phenols).

**Approach**:
- Build motif co-occurrence matrix C where Cᵢⱼ = P(motif j present | motif i present)
- Compare matrices using Frobenius norm

$$C_{ij} = \frac{|\{M : m_i \in M \land m_j \in M\}|}{|\{M : m_i \in M\}|}$$

$$D_{\text{co-occur}} = \|C_{\text{gen}} - C_{\text{ref}}\|_F$$

**Output Metrics**:
- `motif_cooccur_frobenius`: Frobenius norm of matrix difference
- `motif_cooccur_mean_abs`: Mean absolute element-wise difference

---

## 5. Planned Metrics

The following metrics are planned for future implementation:

### PolyGraph

Graph-level evaluation using polynomial features of graph structure.

**Purpose**: Capture global graph properties beyond local statistics.

**Approach**:
- Compute polynomial features of adjacency matrix eigenspectrum
- Compare feature distributions via MMD

$$\phi(G) = [\text{Tr}(A), \text{Tr}(A^2), \text{Tr}(A^3), \ldots, \text{Tr}(A^k)]$$

Trace of Aᵏ counts closed walks of length k, capturing:
- Tr(A²): Number of edges
- Tr(A³): 6 × number of triangles
- Higher powers: Larger cycle structures

**Reference**: PolyGraph metric from graph generation literature.

---

## Usage

### GraphMetrics

```python
from src.evaluation import GraphMetrics

evaluator = GraphMetrics(reference_graphs, compute_emd=False)
results = evaluator.compute(generated_graphs)
# {'degree': 0.023, 'spectral': 0.015, 'clustering': 0.031}
```

### MolecularMetrics

```python
from src.evaluation import MolecularMetrics

evaluator = MolecularMetrics(reference_smiles)
results = evaluator.compute(generated_smiles)
# {'validity': 0.92, 'uniqueness': 0.87, 'novelty': 0.95, ...}
```

### MotifDistributionMetric

```python
from src.evaluation import MotifDistributionMetric

evaluator = MotifDistributionMetric(reference_smiles)
results = evaluator.compute(generated_smiles)
# {'motif_fg_mmd': 0.012, 'motif_smarts_mmd': 0.008, ...}

# Get motif summary
summary = evaluator.get_motif_summary(generated_smiles)
```

### MotifHistogramMetric

```python
from src.evaluation import MotifHistogramMetric

evaluator = MotifHistogramMetric(reference_smiles, distance_fn="kl")
results = evaluator.compute(generated_smiles)
# {'motif_hist_mean': 0.05, 'motif_hist_max': 0.12, 'motif_hist_benzene': 0.03, ...}

# Use Wasserstein distance instead of KL divergence
evaluator_w = MotifHistogramMetric(reference_smiles, distance_fn="wasserstein")
```

### MotifCooccurrenceMetric

```python
from src.evaluation import MotifCooccurrenceMetric

evaluator = MotifCooccurrenceMetric(reference_smiles)
results = evaluator.compute(generated_smiles)
# {'motif_cooccur_frobenius': 0.15, 'motif_cooccur_mean_abs': 0.02}

# Get top co-occurring motif pairs
summary = evaluator.get_cooccurrence_summary(generated_smiles, top_k=10)
# {'top_pairs': [('benzene', 'hydroxyl', 0.85), ...]}
```

### PolygraphMetric

```python
from src.evaluation import PolygraphMetric
from src.data.molecular import smiles_to_graph

# Convert SMILES to graph objects
reference_graphs = [smiles_to_graph(s) for s in reference_smiles[:10000]]

evaluator = PolygraphMetric(reference_graphs, max_reference_size=10000)
results = evaluator.compute(generated_graphs)
# {'pgd': 0.15}

# Interpretation:
# pgd < 0.1: Excellent
# pgd < 0.3: Good (our result)
# pgd < 0.5: Moderate
# pgd >= 0.5: Poor
```

---

## Metric Summary Table

| Metric | Range | Ideal | Measures |
|--------|-------|-------|----------|
| Validity | [0, 1] | 1.0 | Chemical correctness |
| Uniqueness | [0, 1] | 1.0 | Diversity |
| Novelty | [0, 1] | 1.0 | Generalization |
| SNN | [0, 1] | ~0.5-0.7 | Similarity to training |
| FCD | [0, ∞) | 0.0 | Distribution match |
| Fragment Sim | [0, 1] | 1.0 | Fragment distribution |
| Scaffold Sim | [0, 1] | 1.0 | Scaffold distribution |
| Internal Div | [0, 1] | ~0.8-0.9 | Output diversity |
| Degree MMD | [0, ∞) | 0.0 | Degree distribution |
| Spectral MMD | [0, ∞) | 0.0 | Spectral properties |
| Clustering MMD | [0, ∞) | 0.0 | Local clustering |
| **PGD** | **[0, 1]** | **0.0** | **Classifier-based graph quality** |
| Motif MMD | [0, ∞) | 0.0 | Motif preservation |
| Motif Hist Mean | [0, ∞) | 0.0 | Per-motif count distributions |
| Motif Hist Max | [0, ∞) | 0.0 | Worst motif distribution |
| Motif Co-occur | [0, ∞) | 0.0 | Motif combination patterns |

---

## References

1. **MOSES**: Molecular Sets benchmark - Polykovskiy et al. (2020)
2. **FCD**: Fréchet ChemNet Distance - Preuer et al. (2018)
3. **MMD**: Maximum Mean Discrepancy - Gretton et al. (2012)
4. **BRICS**: Breaking of Retrosynthetically Interesting Chemical Substructures - Degen et al. (2008)
5. **PolyGraph**: A Classifier-Based Metric for Graph Generation - https://arxiv.org/abs/2510.06122 (2024)
