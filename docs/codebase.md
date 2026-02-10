# Architecture Guide

This document describes the architecture and design decisions of MOSAIC, a motif-preserving graph generation codebase.

## Project Overview

MOSAIC (MOtif-aware Structural Abstraction for graph tokenIzation and Composition) is a framework for state-of-the-art methods in motif-preserving graph generation. The codebase supports:

- **Multiple datasets**: Synthetic and real molecular graphs (MOSES, QM9)
- **Multiple tokenization schemes**: Flat (SENT) and hierarchical (H-SENT, HDT)
- **Multiple evaluation metrics**: Standard graph metrics and molecular-specific measures

## Directory Structure

```
MOSAIC/
├── src/                          # Core source code
│   ├── data/                     # Data loading and processing
│   ├── tokenizers/               # Graph tokenization schemes
│   │   ├── base.py               # Abstract tokenizer interface
│   │   ├── sent.py               # Flat SENT tokenizer
│   │   └── hierarchical/         # Hierarchical tokenizers
│   │       ├── hsent.py          # H-SENT tokenizer class
│   │       ├── hdt.py            # HDT tokenizer class (DFS-based)
│   │       ├── structures.py     # Partition, Bipartite, HierarchicalGraph
│   │       ├── coarsening.py     # Spectral clustering
│   │       ├── ordering.py       # Node ordering strategies
│   │       └── visualization.py  # Visualization utilities
│   ├── models/                   # Neural network models
│   └── evaluation/               # Evaluation metrics
├── tests/                        # Test suite
│   └── fixtures/                 # Reusable test fixtures
├── scripts/                      # Entry point scripts
├── configs/                      # Hydra configuration files
├── docs/                         # Documentation
└── scratch/                      # Development workspace
```

## Pipeline Overview

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  SMILES String        ┌──────────────────┐                                        │
│  (MOSES/QM9)          │   Molecular      │     PyG Data                           │
│       │               │   Conversion     │                                        │
│       ▼               │                  │                                        │
│  "CCO"               │  - Atom features │     edge_index,                        │
│  "c1ccccc1O"  ──────▶│  - Bond features │────▶ x, edge_attr,                     │
│  ...                 │  - Node/edge     │     smiles                             │
│                      └──────────────────┘                                        │
│                              │                                                     │
│          ┌───────────────────┼───────────────────┐                                │
│          │                   │                   │                                │
│          ▼                   ▼                   ▼                                │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                       │
│  │ SENT          │   │ H-SENT        │   │ HDT           │                       │
│  │ (Flat)        │   │ (Hierarchical)│   │ (Hierarchical)│                       │
│  │               │   │               │   │               │                       │
│  │ - Random walk │   │ - Spectral    │   │ - Spectral    │                       │
│  │ - Back-edges  │   │   clustering  │   │   clustering  │                       │
│  │               │   │ - Partition + │   │ - DFS nesting │                       │
│  │               │   │   bipartite   │   │ - Back-edges  │                       │
│  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                       │
│          │                   │                   │                                │
│          └───────────────────┼───────────────────┘                                │
│                              │                                                     │
│                              ▼                                                     │
│                      ┌──────────────────┐                                         │
│                      │  HF Transformer  │                                         │
│                      │  (GPT-2/LLaMA)   │                                         │
│                      │                  │                                         │
│                      │  Next token pred │                                         │
│                      └──────────────────┘                                         │
│                              │                                                     │
│                              ▼                                                     │
│                      ┌──────────────────┐     ┌──────────────────┐                │
│                      │  Generation      │     │  Evaluation      │                │
│                      │  (Top-k + Temp)  │────▶│  (Molecular +    │                │
│                      │  → SMILES conv.  │     │   Motif Metrics) │                │
│                      └──────────────────┘     └──────────────────┘                │
│                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
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
- `SENTTokenizer`: Flat SENT tokenization from AutoGraph
  - Random walk traversal with deterministic seeding
  - Back-edge encoding with bracket tokens
  - Special tokens: SOS, EOS, RESET, LADJ, RADJ, PAD
  - Linear sequence length in number of edges

**hierarchical/** - Hierarchical Tokenization

*H-SENT (hsent.py)*
- `HSENTTokenizer`: H-SENT tokenizer class
  - Hierarchical decomposition via spectral clustering
  - Recursive partition encoding with SENT-style walks
  - Explicit bipartite encoding for inter-community edges
  - Special tokens: SOS, EOS, PAD, RESET, LADJ, RADJ, LCOM, RCOM, LBIP, RBIP, SEP

*HDT (hdt.py)*
- `HDTTokenizer`: Hierarchical DFS-based tokenizer
  - ~45% fewer tokens than H-SENT via implicit hierarchy encoding
  - DFS traversal through hierarchy with ENTER/EXIT tokens
  - Cross-community edges encoded as back-edges (no bipartite blocks)
  - Special tokens: SOS, EOS, PAD, ENTER, EXIT, LEDGE, REDGE (IDX_OFFSET=7)
  - Smaller vocabulary than H-SENT

*Shared Components*
- `Partition`, `Bipartite`, `HierarchicalGraph`: Data structures (structures.py)
  - `Partition`: Induced subgraph within a community
  - `Bipartite`: Edges between two communities
  - `HierarchicalGraph`: Container with `reconstruct()` method
- `SpectralCoarsening`, `MotifAwareCoarsening`: Graph partitioning (coarsening.py)
  - Modularity-optimized spectral clustering
  - Optional motif-aware clustering with affinity augmentation
  - Configurable `min_community_size` for recursion depth
- `HACCoarsening`: Agglomerative clustering with connectivity constraint (hac.py)
  - Bottom-up merging with adjacency features and connectivity constraint
  - Supports ward, complete, average, single linkage criteria
  - Same recursive hierarchy building as SpectralCoarsening
- Node ordering strategies (ordering.py)
  - BFS, DFS: Standard traversals from highest-degree node
  - BFSAC, BFSDC: BFS with ascending/descending cutset weight
- Visualization utilities (visualization.py)
  - `visualize_hierarchy()`: HiGen-style block matrix visualization
  - `visualize_graph_communities()`: Graph with community coloring
  - `quick_visualize()`: Combined visualization

See [Hierarchical Graph Guide](hgraph.md) and [Tokenization Guide](tokenization.md) for mathematical details.

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
├── tokenizer/          # Tokenizer configurations
│   ├── hdt.yaml        # HDT with standard spectral clustering
│   └── hdt_motif.yaml  # HDT with motif-aware clustering
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