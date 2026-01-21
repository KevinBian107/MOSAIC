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

## Documentation

See the [docs/](docs/) directory for:
- [Codebase Guide](docs/codebase.md)
- [Server Setup Guide](docs/server_setup.md)
- [Training Setup Guide](docs/setup_training.md)
- [Contributing Guide](docs/contributing.md)
- [H-graph Construction](docs/hgraph.md)
- [Tokenization](docs/tokenization.md)
- [Visualize Tokenization](docs/visualization.md)

## Acknowledgement

This codebase was developed based on insights from：
- The official [AutoGraph](https://github.com/BorgwardtLab/AutoGraph) repository.
- The official [HiGen](https://github.com/Karami-m/HiGen_main) repository.