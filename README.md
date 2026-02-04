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
python scripts/train.py tokenizer.type=hdtc

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
python scripts/test.py
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

### Transfer Learning / Fine-tuning

Fine-tune a pretrained model on COCONUT complex natural products to evaluate transfer learning capabilities.

```bash
# First, prepare the complex molecule dataset (one-time setup)
# Downloads and filters COCONUT natural products (~191MB download)
python scripts/prepare_coconut_data.py

# Custom filtering thresholds
python scripts/prepare_coconut_data.py \
    --n-molecules 10000 \
    --min-atoms 25 \
    --min-rings 3

# Fine-tune from MOSES checkpoint
python scripts/finetune.py \
    model.pretrained_path=outputs/benchmark/moses_hdtc_n500000_20260129-171812/best.ckpt \
    experiment=coconut \
    trainer.max_steps=50000

# Or use train.py directly with COCONUT experiment
python scripts/train.py \
    experiment=coconut \
    model.pretrained_path=outputs/moses_hdtc/best.ckpt

# Evaluate fine-tuned model
python scripts/eval_finetune.py \
    model.checkpoint_path=outputs/coconut_finetune/best.ckpt \
    generation.num_samples=1000

# Compare the results of fine-tuned model
python scripts/compare_finetune_results.py
```

### Table Comparison

```bash
python scripts/compare_results.py
```

### Trained Checkpoint
gdown our trained checkpoint for different models from [this google drive](https://drive.google.com/drive/folders/1aMo5cQvexJ11GyIXACQys_UlA1CxKv7e?usp=drive_link).

### Demo

```bash
# Generation Demo
python scripts/visualization/generation_demo.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
MOSAIC/
├── src/
│   ├── data/              # Data loading, generation, and motif detection
│   ├── tokenizers/        # Graph tokenization (SENT, H-SENT, HDT, HDTC)
│   │   ├── coarsening/    # Coarsening strategies (spectral, motif-aware)
│   │   └── motif/         # Motif detection and patterns
│   ├── models/            # Transformer models
│   ├── evaluation/        # Standard and motif metrics
│   └── realistic_gen/     # Generation quality analysis
├── configs/               # Hydra configuration
├── scripts/               # Training, evaluation, and visualization scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
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