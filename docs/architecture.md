# Architecture Guide

This document describes the architecture and design decisions of the molecular graph generation codebase.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SMILES String        ┌──────────────────┐                          │
│  (MOSES/QM9)          │   Molecular      │     PyG Data             │
│       │               │   Conversion     │                          │
│       ▼               │                  │                          │
│  "CCO"               │  - Atom features │     edge_index,          │
│  "c1ccccc1O"  ──────▶│  - Bond features │────▶ x, edge_attr,       │
│  ...                 │  - Node/edge     │     smiles               │
│                      └──────────────────┘                          │
│                              │                                       │
│                              ▼                                       │
│                      ┌──────────────────┐     Token Sequence        │
│                      │  SENT            │                           │
│                      │  Tokenizer       │     [SOS, n0, n1, [,      │
│                      │                  │──▶  n0, ], n2, ..., EOS]  │
│                      │  - Random walk   │                           │
│                      │  - Back-edges    │                           │
│                      └──────────────────┘                           │
│                              │                                       │
│                              ▼                                       │
│                      ┌──────────────────┐                           │
│                      │  HF Transformer  │                           │
│                      │  (GPT-2/LLaMA)   │                           │
│                      │                  │                           │
│                      │  Next token pred │                           │
│                      └──────────────────┘                           │
│                              │                                       │
│                              ▼                                       │
│                      ┌──────────────────┐     ┌──────────────────┐  │
│                      │  Generation      │     │  Evaluation      │  │
│                      │  (Top-k + Temp)  │────▶│  (Molecular +    │  │
│                      │  → SMILES conv.  │     │   Motif Metrics) │  │
│                      └──────────────────┘     └──────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### `src/data/` - Data Module

**molecular.py**
- `smiles_to_graph()`: Convert SMILES to PyG Data with atom/bond features
- `graph_to_smiles()`: Convert PyG Data back to SMILES
- `load_moses_dataset()`: Load MOSES benchmark data
- `load_qm9_smiles()`: Load QM9 dataset
- `MolecularDataset`: Dataset class for molecular graphs

**datamodule.py**
- `MolecularDataModule`: PyTorch Lightning data module
  - Supports MOSES and QM9 datasets
  - Handles tokenization and batching
  - Provides train/val/test SMILES lists for metrics

### `src/tokenizers/` - Tokenization Module

**base.py**
- `Tokenizer`: Abstract base class defining the interface
  - `tokenize()`: Graph → token sequence
  - `decode()`: Token sequence → graph
  - `batch_converter()`: Collation function
- `BatchConverter`: Pads and batches token sequences

**sent.py**
- `SENTTokenizer`: SENT tokenization from AutoGraph
  - Random walk traversal
  - Back-edge encoding with bracket tokens
  - Special tokens: SOS, EOS, RESET, LADJ, RADJ, PAD

### `src/models/` - Model Module

**transformer.py**
- `TransformerLM`: HuggingFace transformer wrapper
  - Supports GPT-2, LLaMA, GPT-NeoX
  - Configurable sizes: xs, s, m
  - `generate()`: Autoregressive graph generation
- `GraphGeneratorModule`: PyTorch Lightning module
  - Training with next-token prediction
  - Cosine learning rate schedule with warmup
  - Generation and evaluation methods

### `src/evaluation/` - Evaluation Module

**dist_helper.py**
- `compute_mmd()`: Maximum Mean Discrepancy computation
- `gaussian()`, `gaussian_tv()`, `gaussian_emd()`: Kernel functions

**metrics.py**
- `GraphMetrics`: Standard graph generation metrics
  - Degree distribution MMD
  - Spectral properties MMD
  - Clustering coefficient MMD
- `compute_validity_metrics()`: Uniqueness and novelty

**molecular_metrics.py**
- `MolecularMetrics`: AutoGraph-style molecular metrics
  - `compute_validity()`: RDKit valency check
  - `compute_uniqueness()`: Unique SMILES count
  - `compute_novelty()`: Not in training set
  - `compute_snn()`: Nearest neighbor similarity
  - `compute_fragment_similarity()`: BRICS fragment comparison
  - `compute_scaffold_similarity()`: Bemis-Murcko scaffold comparison
  - `compute_fcd()`: Frechet ChemNet Distance

**motif_distribution.py**
- `MotifDistributionMetric`: Novel motif distribution comparison
  - `get_functional_group_counts()`: RDKit functional groups
  - `get_motif_counts()`: SMARTS pattern matching
  - `get_ring_system_info()`: Ring analysis
  - `get_brics_fragments()`: BRICS decomposition
  - Computes MMD between reference and generated distributions

## Configuration System

The codebase uses Hydra for configuration management:

```
configs/
├── train.yaml          # Training configuration
├── test.yaml           # Testing configuration
└── experiment/
    ├── moses.yaml      # MOSES dataset defaults
    └── qm9.yaml        # QM9 dataset defaults
```

Configuration hierarchy:
1. `train.yaml` provides base defaults
2. `experiment/*.yaml` overrides for specific datasets
3. Command-line arguments override everything

## Node and Edge Features

### Atom (Node) Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Atom type | 10 | One-hot (C, N, O, F, P, S, Cl, Br, I, other) |
| Atomic number | 1 | Integer |
| Formal charge | 1 | Integer |
| Total Hs | 1 | Number of hydrogens |
| Is aromatic | 1 | Boolean |
| Is in ring | 1 | Boolean |
| Degree | 1 | Number of bonds |

### Bond (Edge) Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Bond type | 5 | One-hot (single, double, triple, aromatic, other) |
| Is aromatic | 1 | Boolean |
| Is in ring | 1 | Boolean |
| Is conjugated | 1 | Boolean |

## Design Decisions

### Why MOSES/QM9?

These are standard benchmarks for molecular generation:
- **MOSES**: Drug-like molecules, standard metrics suite
- **QM9**: Small molecules with property optimization targets
- Both have established baselines and evaluation protocols

### Why SENT Tokenization?

SENT provides a lossless graph-to-sequence encoding that:
- Preserves all structural information
- Has linear sequence length in number of edges
- Is compatible with standard transformer architectures

### Why Motif Distribution Analysis?

Comparing motif distributions enables:
- Understanding what structural patterns the model learns
- Identifying generation biases (e.g., too many/few benzene rings)
- More interpretable evaluation than single-number metrics

### Why PyTorch Lightning?

Lightning provides:
- Clean separation of model and training logic
- Built-in support for distributed training
- Easy checkpointing and logging

### Why Hydra?

Hydra enables:
- Hierarchical configuration management
- Easy experiment sweeps
- Reproducible experiments
