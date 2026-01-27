# MOSAIC: MOtif-aware Structural Abstraction for graph tokenIzation and Composition

This project explores a simple idea: instead of asking generative models to recover motif-level structure implicitly, we encode motifs directly into the representation used for graph generation, independent of the sequence transformer's processing. We are interesting in constructing a representation in the form of flat tokens.

## Core Approach

For generating graphs using tokens with hierarchical insights, we need 3 things:

1. **Create the input H-graph**: Build a hierarchical representation of the graph using coarsening strategies (HAC, Spectral Clustering, Motif Community).

2. **Tokenize the input H-graph**: Convert the hierarchy to a token sequence using H-SENT (Vanilla HiGen) or HDT (DFS-based). Note that we need to preserve enough information (leaf edge connections) for the inverse problem to flatten the H-graph.

3. **Flatten the generated H-graph**: Reconstruct the flat graph from tokens via bipartite edge union for H-SENT, or union of back edges for HDT.

![HDT](/docs/figure/hdt_sample.gif)
> HDT generation of novel molecules

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

### Realistic Generation

Generate molecules and analyze how well they match the structural patterns of training data. This evaluates whether the model learns realistic chemical preferences.

```bash
# Generate and analyze with HDT
python scripts/realistic_gen.py

# Generate and analyze with SENT
python scripts/realistic_gen.py \
    model.checkpoint_path=outputs/train/moses_sent_n1000000_20260123-140906/best.ckpt \
    tokenizer=sent

# Custom number of samples
python scripts/realistic_gen.py \
    generation.num_samples=500
```

This produces:
- **Generated SMILES**: Saved to `generated_smiles.txt`
- **Statistical analysis**: Substitution patterns (mono/di/tri), ortho/meta/para ratios, functional group frequencies
- **Distribution metrics**: Total Variation distance and KL divergence vs training data
- **Molecule visualizations**: Side-by-side comparison of training vs generated molecular structures

### Demo

```bash
# Generation Demo
python scripts/generation_demo.py
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
│   ├── tokenizers/     # Graph tokenization (SENT, H-SENT, HDT)
│   │   └── hierarchical/   # Hierarchical tokenization module
│   ├── models/         # Transformer models
│   ├── evaluation/     # Standard and motif metrics
│   └── realistic_gen/  # Generation quality analysis
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
- [Evaluation Metrics](docs/metric.md)
- [Realistic Generation](docs/realistic.md)
- [Visualize Tokenization](docs/visualization.md)

## Acknowledgement

This codebase was developed based on insights from：
- The official [AutoGraph](https://github.com/BorgwardtLab/AutoGraph) repository.
- The official [HiGen](https://github.com/Karami-m/HiGen_main) repository.