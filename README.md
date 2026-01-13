# MOSAIC: MOtif-aware Structural Abstraction for graph tokenIzation and Composition
This project explores a simple idea: instead of asking generative models to recover motif-level structure implicitly, we encode motifs directly into the representation used for graph generation or treat it as a loss regularization upon training, both independent of the inner processing of sequence transformer.

## Quick Start

### Installation

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate mosaic
```

### Training

```bash
# Train with default configuration (uses flat SENT tokenization)
python scripts/train.py

# Train with hierarchical H-SENT tokenization
python scripts/train.py tokenizer.type=hsent

# Train on QM9 dataset
python scripts/train.py data.dataset_name=qm9

# Train with custom model and settings
python scripts/train.py \
    model.model_name=gpt2-s \
    trainer.max_steps=100000 \
    tokenizer.type=hsent
```

### Evaluation

```bash
python scripts/test.py model.checkpoint_path=outputs/model.ckpt
```

### Visualization

```bash
# Visualize hierarchical tokenization of molecules
python scripts/visualize_htoken.py --name cholesterol

# Visualize by SMILES string
python scripts/visualize_htoken.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"

# Run demo with multiple complex molecules and save
python scripts/visualize_htoken.py --demo --output-dir ./figures --no-show

# List available molecules
python scripts/visualize_htoken.py --list

# Visualize molecular motifs
python scripts/visualize_motifs.py --smiles "c1ccccc1O"
```

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
MOSAIC/
├── src/
│   ├── data/           # Data loading, generation, and motif detection
│   ├── tokenizers/     # Graph tokenization (SENT, H-SENT)
│   │   └── hierarchical/   # Hierarchical tokenization module
│   ├── models/         # Transformer models
│   └── evaluation/     # Standard and motif metrics
├── configs/            # Hydra configuration
├── scripts/            # Training, evaluation, and visualization scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Key Metrics

### Standard Graph Metrics
- Degree distribution (MMD)
- Spectral properties (MMD)
- Clustering coefficients (MMD)

### Motif Metrics
- **MSC** (Motif Signature Comparison): MMD between motif count vectors
- **MFD** (Motif Frequency Distribution): Per-motif-type comparison
- **MPR** (Motif Preservation Rate): For conditional generation

## Documentation

See the [docs/](docs/) directory for:
- [Getting Started](docs/index.md)
- [Hierarchical Tokenization (H-SENT)](docs/htokenization.md)
- [Visualize Hierarchical Tokenization (H-SENT)](docs/vis_htokenization.md)
- [Architecture Guide](docs/architecture.md)
- [Setup Guide](docs/setup.md)
- [Contributing Guide](docs/contributing.md)

## Acknowledgement

This codebase was developed based on insights from：
- The official [AutoGraph](https://github.com/BorgwardtLab/AutoGraph) repository.
- The official [HiGen](https://github.com/Karami-m/HiGen_main) repository.