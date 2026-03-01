# Architecture Guide

This document describes the architecture and design decisions of MOSAIC, a motif-preserving graph generation codebase.

## Project Overview

MOSAIC (MOtif-aware Structural Abstraction for graph tokenIzation and Composition) is a framework for state-of-the-art methods in motif-preserving graph generation. The codebase supports:

- **Multiple datasets**: MOSES (drug-like), COCONUT (natural products), QM9
- **Multiple tokenization schemes**: Flat (SENT), hierarchical (H-SENT, HDT), compositional (HDTC)
- **Multiple coarsening strategies**: Spectral, HAC, Motif Community
- **Multiple evaluation metrics**: Standard graph metrics and molecular-specific measures

## Directory Structure

```
MOSAIC/
├── src/                          # Core source code
│   ├── data/                     # Data loading and processing
│   ├── tokenizers/               # Graph tokenization schemes
│   │   ├── base.py               # Abstract tokenizer interface
│   │   ├── structures.py         # Partition, Bipartite, HierarchicalGraph
│   │   ├── ordering.py           # Node ordering strategies
│   │   ├── visualization.py      # Visualization utilities
│   │   ├── sent/                 # Flat SENT tokenizer
│   │   ├── hsent/                # H-SENT tokenizer (hierarchical SENT)
│   │   ├── hdt/                  # HDT tokenizer (hierarchical DFS)
│   │   ├── hdtc/                 # HDTC tokenizer (compositional)
│   │   ├── coarsening/           # Coarsening strategies
│   │   │   ├── spectral.py       # Spectral clustering
│   │   │   ├── hac.py            # Hierarchical agglomerative clustering
│   │   │   ├── motif_community.py # Motif-aware community detection
│   │   │   └── functional_hierarchy.py # HDTC functional hierarchy
│   │   └── motif/                # Motif detection and patterns
│   ├── models/                   # Neural network models
│   ├── evaluation/               # Evaluation metrics
│   └── realistic_gen/            # Generation quality analysis
├── scripts/                      # Entry point scripts
│   ├── preprocess/               # Data preprocessing and caching
│   ├── comparison/               # Result comparison and benchmarking
│   └── visualization/            # Visualization and demo scripts
├── bash_scripts/                 # Batch benchmark automation
│   ├── train/                    # Training scripts
│   └── eval/                     # Evaluation scripts
├── configs/                      # Hydra configuration files
├── tests/                        # Test suite
├── property_experiment/          # Post-hoc analysis experiments
└── docs/                         # Documentation
```

## Pipeline Overview

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  SMILES String        ┌──────────────────┐                                        │
│  (MOSES/COCONUT)      │   Molecular      │     PyG Data                           │
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
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │ SENT     │  │ H-SENT   │  │ HDT      │  │ HDTC     │                          │
│  │ (Flat)   │  │ (Hier.)  │  │ (Hier.)  │  │ (Comp.)  │                          │
│  │          │  │          │  │          │  │          │                          │
│  │ Random   │  │ Spectral/│  │ Spectral/│  │ Func.    │                          │
│  │ walk +   │  │ HAC/MC + │  │ HAC/MC + │  │ hierarchy│                          │
│  │ back-edge│  │ partition│  │ DFS nest │  │ + DFS    │                          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                          │
│       │              │              │              │                               │
│       └──────────────┴──────────────┴──────────────┘                               │
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
- `MolecularDataset`: Dataset class for molecular graphs

**datamodule.py**
- `MolecularDataModule`: PyTorch Lightning data module
  - Supports MOSES, COCONUT, and QM9 datasets
  - Handles tokenization and batching
  - Provides train/val/test SMILES lists for metrics

### `src/tokenizers/` - Tokenization Module

**base.py**
- `Tokenizer`: Abstract base class defining the interface
  - `tokenize()`: Graph → token sequence
  - `decode()`: Token sequence → graph
  - `batch_converter()`: Collation function
- `BatchConverter`: Pads and batches token sequences

**sent/** - Flat SENT tokenizer (from AutoGraph)
- Random walk traversal with deterministic seeding
- Back-edge encoding with bracket tokens
- Special tokens: SOS, EOS, RESET, LADJ, RADJ, PAD (IDX_OFFSET=6)

**hsent/** - H-SENT tokenizer (hierarchical SENT)
- Recursive partition encoding with SENT-style walks
- Explicit bipartite encoding for inter-community edges
- Special tokens: SOS, EOS, PAD, RESET, LADJ, RADJ, LCOM, RCOM, LBIP, RBIP, SEP (IDX_OFFSET=11)

**hdt/** - HDT tokenizer (hierarchical DFS)
- ~45% fewer tokens than H-SENT via implicit hierarchy encoding
- DFS traversal through hierarchy with ENTER/EXIT tokens
- Cross-community edges encoded as back-edges (no bipartite blocks)
- Special tokens: SOS, EOS, PAD, ENTER, EXIT, LEDGE, REDGE (IDX_OFFSET=7)

**hdtc/** - HDTC tokenizer (compositional)
- Functional hierarchy: Ring systems → Functional groups → Scaffolds
- DFS-based encoding like HDT, with typed abstract nodes
- Special tokens: SOS, EOS, PAD, ENTER, EXIT, LEDGE, REDGE + R/F/S type tokens (IDX_OFFSET=12)

**coarsening/** - Coarsening strategies
- `SpectralCoarsening`: Modularity-optimized spectral clustering
- `HACCoarsening`: Agglomerative clustering with connectivity constraint
- `MotifCommunityCoarsening`: Motif-aware community detection
- `FunctionalHierarchy`: HDTC functional group hierarchy (no coarsening needed)

**Shared components**
- `Partition`, `Bipartite`, `HierarchicalGraph`: Data structures (structures.py)
- Node ordering strategies: BFS, DFS, BFSAC, BFSDC (ordering.py)
- Visualization utilities (visualization.py)

See [Hierarchical Graph Guide](hgraph.md) and [Tokenization Guide](tokenization.md) for details.

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

The codebase uses [Hydra](https://hydra.cc/) for configuration. See [configs/README.md](../configs/README.md) for the full parameter comparison table.

```
configs/
├── train.yaml          # Base training configuration
├── test.yaml           # Base evaluation configuration
├── realistic_gen.yaml  # Base generation analysis configuration
├── tokenizer/          # Tokenizer configurations
│   ├── sent.yaml       # SENT
│   ├── hsent.yaml      # H-SENT
│   ├── hdt.yaml        # HDT
│   └── hdtc.yaml       # HDTC (default)
└── experiment/
    ├── moses.yaml      # MOSES dataset overrides
    └── coconut.yaml    # COCONUT dataset overrides
```

Override order: `tokenizer → base (train/test.yaml) → experiment → CLI`

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