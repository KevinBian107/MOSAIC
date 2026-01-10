# MOSAIC: MOtif-aware Structural Abstraction for graph tokenIzation and Composition
This project explores a simple idea: instead of asking generative models to recover motif-level structure implicitly, we encode motifs directly into the representation used for graph generation or treat it as a loss regularization upon training, both independent of the inner processing of sequence transformer.

The codebase is designed for flexibility, currently supporting synthetic datasets and the original AutoGraph tokenization scheme, with easy extensibility for additional datasets (e.g., molecules, proteins) and methods.

## Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom model and data
python scripts/train.py \
    model.model_name=gpt2-s \
    trainer.max_steps=100000 \
    data.generators=[{name:erdos_renyi},{name:barabasi_albert}]
```

### Evaluation

```bash
python scripts/test.py model.checkpoint_path=outputs/model.ckpt
```

### Running Tests

```bash
pytest
```

## Project Structure

```
MOSAIC/
├── src/
│   ├── data/           # Data loading, generation, and motif detection
│   ├── tokenizers/     # Graph tokenization (SENT)
│   ├── models/         # Transformer models
│   └── evaluation/     # Standard and motif metrics
├── configs/            # Hydra configuration
├── scripts/            # Training and evaluation scripts
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
- [Setting up](docs/setup.md)
- [Getting Started](docs/index.md)
- [Architecture Guide](docs/architecture.md)
- [Contributing Guide](docs/contributing.md)

## Acknowledgement

This codebase was developed based on insights from the official [AutoGraph](https://github.com/BorgwardtLab/AutoGraph) repository.