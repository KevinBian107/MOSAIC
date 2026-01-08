# Motif-Preserving Graph Generation

A benchmarking framework for evaluating graph generation methods on **motif preservation**.

## Overview

This codebase provides tools for:

1. **Synthetic graph generation** with known motif labels (ground truth)
2. **Graph tokenization** using the SENT scheme from AutoGraph
3. **Transformer-based graph generation** models
4. **Evaluation metrics** for both standard graph properties and motif preservation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/user/motif-preserving-tokenization.git
cd motif-preserving-tokenization

# Create conda environment
conda env create -f environment.yaml
conda activate motif-graph

# Install in development mode
pip install -e ".[dev]"
```

### Training a Model

```bash
# Train with default configuration
python scripts/train.py

# Train with custom settings
python scripts/train.py \
    model.model_name=gpt2-s \
    trainer.max_steps=100000 \
    data.generators=[{name:erdos_renyi,n:50,p:0.2},{name:barabasi_albert,n:50,m:3}]
```

### Evaluating a Model

```bash
python scripts/test.py model.checkpoint_path=outputs/model.ckpt
```

## Project Structure

```
motif-preserving-tokenization/
├── src/
│   ├── data/           # Data loading and generation
│   ├── tokenizers/     # Graph tokenization schemes
│   ├── models/         # Neural network models
│   └── evaluation/     # Evaluation metrics
├── configs/            # Hydra configuration files
├── scripts/            # Training and evaluation scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Key Concepts

### Motifs

Motifs are recurring structural patterns in graphs. We focus on:

| Motif | Description |
|-------|-------------|
| Triangle | 3-clique (3 nodes, all connected) |
| 4-Cycle | Square (4 nodes in a cycle) |
| 5-Cycle | Pentagon (5 nodes in a cycle) |
| 4-Clique | Complete graph on 4 nodes |
| Star | Hub node connected to multiple leaves |

### SENT Tokenization

SENT (Sequence of Edge-indicating Neighborhoods) converts graphs to token sequences via random walk with back-edge encoding:

```
Graph: 0--1--2
       |  |
       +--+

Tokens: [SOS, 0, 1, [, 0, ], 2, [, 0, 1, ], EOS]
```

### Evaluation Metrics

**Standard Metrics:**
- Degree distribution (MMD)
- Spectral properties (MMD)
- Clustering coefficients (MMD)

**Motif Metrics:**
- MSC (Motif Signature Comparison): MMD between motif count vectors
- MFD (Motif Frequency Distribution): Per-motif-type comparison
- MPR (Motif Preservation Rate): For conditional generation

## Next Steps

- [Architecture Guide](architecture.md)
- [Contributing Guide](contributing.md)
- [API Reference](api/index.md)
