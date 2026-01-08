# Architecture Guide

This document describes the architecture and design decisions of the motif-preserving graph generation codebase.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Graph          ┌──────────────────┐                          │
│  (PyG Data)           │   SENT           │     Token Sequence       │
│       │               │   Tokenizer      │                          │
│       ▼               │                  │                          │
│  ┌─────────┐          │  - Random walk   │     [SOS, n0, n1, [,     │
│  │ edge_   │─────────▶│  - Back-edges    │────▶ n0, ], n2, ..., EOS]│
│  │ index   │          │  - SENT format   │                          │
│  └─────────┘          └──────────────────┘                          │
│                                                                      │
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
│                      │  (Top-k + Temp)  │────▶│  (Graph + Motif) │  │
│                      └──────────────────┘     └──────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### `src/data/` - Data Module

**motif.py**
- `MotifType`: Enum of supported motif types
- `MotifDetector`: Detects and labels motifs in graphs
  - `detect()`: Returns motif labels, types, and counts
  - `get_motif_vector()`: Returns normalized motif count vector

**synthetic.py**
- `SyntheticGraphGenerator`: Wraps NetworkX generators with motif detection
  - `generate()`: Generate graphs with motif labels
  - `generate_dataset()`: Generate train/val/test splits
- `create_mixed_dataset()`: Create dataset from multiple generators

**datamodule.py**
- `GraphDataModule`: PyTorch Lightning data module
  - Handles data loading, tokenization, and batching
  - Supports multiple generators in a single dataset

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

**motif_metrics.py**
- `MotifMetrics`: Novel motif preservation metrics
  - `compute_msc()`: Motif Signature Comparison
  - `compute_mfd()`: Motif Frequency Distribution
  - `compute_mpr()`: Motif Preservation Rate

## Configuration System

The codebase uses Hydra for configuration management:

```
configs/
├── train.yaml          # Training configuration
├── test.yaml           # Testing configuration
└── experiment/
    └── synthetic.yaml  # Synthetic experiment defaults
```

Configuration hierarchy:
1. `train.yaml` provides base defaults
2. `experiment/*.yaml` overrides for specific experiments
3. Command-line arguments override everything

## Design Decisions

### Why SENT Tokenization?

SENT provides a lossless graph-to-sequence encoding that:
- Preserves all structural information
- Has linear sequence length in number of edges
- Is compatible with standard transformer architectures

### Why Motif Labels?

Motif labels enable:
- Ground truth for evaluating motif preservation
- Potential for motif-aware tokenization schemes
- Quantitative comparison of generation methods

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
