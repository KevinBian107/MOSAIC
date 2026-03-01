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

#### Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Bounded Range** | Score ∈ [0, 1] provides interpretable quality assessment |
| **No Hyperparameters** | Unlike MMD which requires kernel bandwidth tuning |
| **Multi-Descriptor** | Aggregates information across multiple graph properties |
| **Learned Features** | GIN-based descriptors capture complex distributional differences |

#### How PGD Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        PGD Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Graph Descriptor Extraction                             │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │  Reference   │     │  Generated   │                          │
│  │   Graphs     │     │   Graphs     │                          │
│  └──────┬───────┘     └──────┬───────┘                          │
│         │                    │                                   │
│         ▼                    ▼                                   │
│  ┌──────────────────────────────────────┐                       │
│  │     GIN (Graph Isomorphism Network)  │                       │
│  │     Extract feature vectors          │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                            │
│  Step 2: Binary Classification                                   │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │         TabPFN Classifier            │                       │
│  │   Class 0: Reference graphs          │                       │
│  │   Class 1: Generated graphs          │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                            │
│  Step 3: Compute Score                                           │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │  PGD = classifier_accuracy - 0.5     │                       │
│  │                                      │                       │
│  │  Low accuracy → PGD ≈ 0 → GOOD      │                       │
│  │  High accuracy → PGD ≈ 1 → POOR     │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**GIN Descriptors Capture:**
- Node degree distributions
- Spectral properties
- Clustering coefficients
- Higher-order structural patterns

#### Comparison with Other Metrics

| Metric | Range | Hyperparameters | Stability | Best For |
|--------|-------|-----------------|-----------|----------|
| **PGD** | [0, 1] | None | Stable at 256+ samples | Overall graph structure |
| **MMD** | [0, ∞) | Kernel bandwidth | High variance at small samples | Local statistics |
| **FCD** | [0, ∞) | None | Stable | Molecular chemistry (drug-like) |

**PGD vs MMD:**
- PGD: Bounded, no tuning, stable with 256+ samples
- MMD: Unbounded, requires kernel bandwidth tuning, high variance at small sample sizes

**PGD vs FCD:**
- PGD: General graph structure, works on any graph type
- FCD: Molecular-specific, uses ChemNet features trained on drug-like molecules

#### When to Use PGD

✅ **Use PGD when:**
- Comparing different generation methods on the same dataset
- Evaluating how well a model captures the training distribution
- Benchmarking against published baselines (PolyGraph paper includes many)
- You have 256+ generated and reference graphs

❌ **Don't rely solely on PGD when:**
- Sample size < 256 (high variance, unreliable)
- Molecular properties matter more than structure (use FCD, QED, SA)
- Specific motifs are critical (combine with motif metrics)

#### Interpretation Guidelines

| PGD Score | Quality | Meaning |
|-----------|---------|---------|
| < 0.1 | Excellent | Generated distribution matches reference very closely |
| < 0.3 | Good | Generated distribution is similar to reference |
| < 0.5 | Moderate | Noticeable differences but reasonable quality |
| ≥ 0.5 | Poor | Generated graphs easily distinguishable from reference |

#### Sample Size Recommendations

Based on the PolyGraph paper findings:

| Sample Size | Stability | Recommendation |
|-------------|-----------|----------------|
| < 256 | High variance, unreliable | Not recommended |
| ~256 | Mean and variance stabilize | Minimum for stability |
| 1000-2048 | Ideal range | Recommended for benchmarks |

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

### BRICS Fragment L2

L2 distance between normalized BRICS fragment frequency distributions (not MMD).

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

## 5. Realistic Generation Metrics

Metrics for evaluating whether generated molecules exhibit realistic structural patterns matching training data.

### What Makes a Molecule Realistic?

A realistic molecule exhibits the same **structural patterns** as molecules found in real-world chemical databases. Beyond basic validity, a realistic molecule should:

1. **Follow chemical preferences** - Real molecules exhibit biases in how functional groups attach, where substituents are placed, and which structures co-occur
2. **Match training distribution** - Generated molecules should statistically resemble the training data, not just be valid

#### Examples of Chemical Biases

**1. Functional Group Attachment Biases**

Not all attachment points are equally likely. Real chemistry favors certain patterns:

```
COMMON (seen in training):          RARE (chemically unusual):

     OH                                  OH   OH
      |                                   |    |
  ●───●───●───COOH                    ●───●───●───●
      |                                   |
     NH₂                                 OH

Phenol + carboxylic acid              Multiple adjacent -OH on
at chain terminus                     aliphatic chain (unstable)
```

**2. Substituent Placement Biases**

For di-substituted benzenes, position matters:

```
COMMON (~60% in drugs):             LESS COMMON (~15%):

    COOH                                COOH
      |                                   |
      ●                                   ●
     / \                                 / \
    ●   ●                               ●   ●
    |   |                                \ /
    ●   ●                                 ●
     \ /                                  |
      ●                                   ●
      |                                   |
     NH₂                                 NH₂

Para-aminobenzoic acid              Meta-aminobenzoic acid
(PABA - symmetric, stable)          (less symmetric)
```

**3. Functional Group Co-occurrence Biases**

Certain groups naturally appear together in drug-like molecules:

```
COMMON CO-OCCURRENCES:              RARE CO-OCCURRENCES:

    O    H                              O    O
    ‖    |                              ‖    ‖
────C────N────[aromatic]           ────C────C────
                                        |    |
Amide + aromatic ring                  OH   OH
(found in ~40% of drugs)
                                   Adjacent carboxylic acids
                                   (would react/dehydrate)


    F                                   I    I
    |                                   |    |
   [aromatic]───CF₃                [aromatic]
                                        |
Fluorine + trifluoromethyl             I
(common in modern drugs,
metabolically stable)              Multiple iodines
                                   (rare, too heavy/reactive)
```

A model that generates only valid molecules but ignores these biases has not truly learned chemistry. For example, generating many meta-substituted benzenes when training data is predominantly para-substituted indicates poor distribution learning.

### Benzene Substitution Patterns

Benzene rings are ubiquitous in drug-like molecules. How substituents attach follows well-known chemical principles:

**Substitution Count**:
- **Mono-substituted**: One group attached (e.g., toluene, phenol)
- **Di-substituted**: Two groups attached
- **Tri-substituted**: Three groups attached
- **Poly-substituted**: Four or more groups attached

**Di-substitution Position (ortho/meta/para)**:

```
    ortho (1,2)         meta (1,3)          para (1,4)
       X                   X                    X
      /                   /                    /
     ●                   ●                    ●
    / \                 / \                  / \
   ●   ●               ●   ●                ●   ●
    \ /                 \ /                  \ /
     ●                   ●                    ●
      \                   \                    \
       X                   ●                    X
                            \
                             X
```

Real molecules show preferences:
- **Para** is often favored (less steric hindrance, symmetric)
- **Ortho** can be favored for intramolecular interactions
- **Meta** is typically less common

### Functional Group Distribution

Different functional groups appear with different frequencies in drug-like molecules:

| Functional Group | Description | Typical Role |
|-----------------|-------------|--------------|
| Hydroxyl (-OH) | Alcohol/phenol | H-bonding, solubility |
| Amino (-NH2) | Amine | H-bonding, basicity |
| Carbonyl (C=O) | Ketone/aldehyde | H-bonding, reactivity |
| Carboxyl (-COOH) | Carboxylic acid | Ionizable, H-bonding |
| Halogen (-F, -Cl, -Br) | Halide | Lipophilicity, metabolic stability |
| Ether (-OR) | Ether/methoxy | Solubility, metabolic site |
| Methyl (-CH3) | Methyl | Lipophilicity, steric |
| Cyano (-CN) | Nitrile | H-bond acceptor |

### Total Variation Distance (TV)

Measures the maximum difference between two probability distributions:

$$\text{TV}(P, Q) = 0.5 \times \sum |P(x) - Q(x)|$$

- **TV = 0**: Identical distributions
- **TV = 1**: Completely different distributions
- **TV < 0.1**: Good match

**Output Metrics**:
- `substitution_tv`: TV distance for substitution count distribution
- `functional_group_tv`: TV distance for functional group distribution

### KL Divergence

Measures information lost when using generated distribution to approximate training:

$$\text{KL}(P \| Q) = \sum P(x) \times \log(P(x) / Q(x))$$

- **KL = 0**: Identical distributions
- **KL > 0**: Divergence (lower is better)

**Output Metrics**:
- `substitution_kl`: KL divergence for substitution patterns
- `functional_group_kl`: KL divergence for functional groups

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

### Graph Metrics

| Metric | Range | Ideal | Measures |
|--------|-------|-------|----------|
| Degree MMD | [0, ∞) | 0.0 | Degree distribution |
| Spectral MMD | [0, ∞) | 0.0 | Spectral properties |
| Clustering MMD | [0, ∞) | 0.0 | Local clustering |
| **PGD** | **[0, 1]** | **0.0** | **Classifier-based graph quality** |

### Molecular Metrics

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

### Motif Metrics

| Metric | Range | Ideal | Measures |
|--------|-------|-------|----------|
| FG MMD | [0, ∞) | 0.0 | Functional group preservation |
| SMARTS MMD | [0, ∞) | 0.0 | SMARTS pattern preservation |
| Ring MMD | [0, ∞) | 0.0 | Ring system preservation |
| BRICS L2 | [0, ∞) | 0.0 | BRICS fragment distribution |
| Motif Hist Mean | [0, ∞) | 0.0 | Per-motif count distributions |
| Motif Hist Max | [0, ∞) | 0.0 | Worst motif distribution |
| Motif Co-occur | [0, ∞) | 0.0 | Motif combination patterns |

### Realistic Generation Metrics

| Metric | Range | Ideal | Measures |
|--------|-------|-------|----------|
| Motif Rate | [0, 1] | ~training | Fraction with target motif |
| Subst TV | [0, 1] | 0.0 | Substitution pattern match |
| Subst KL | [0, ∞) | 0.0 | Substitution pattern divergence |
| FG TV | [0, 1] | 0.0 | Functional group match |
| FG KL | [0, ∞) | 0.0 | Functional group divergence |

---

## References

1. **MOSES**: Molecular Sets benchmark - Polykovskiy et al. (2020)
2. **FCD**: Fréchet ChemNet Distance - Preuer et al. (2018)
3. **MMD**: Maximum Mean Discrepancy - Gretton et al. (2012)
4. **BRICS**: Breaking of Retrosynthetically Interesting Chemical Substructures - Degen et al. (2008)
5. **PolyGraph**: A Classifier-Based Metric for Graph Generation - https://arxiv.org/abs/2510.06122 (2024)
