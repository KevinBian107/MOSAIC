# [Beyond Flat Walks: Compositional Abstraction for Autoregressive Molecular Generation](https://kbian.org/MOSAIC-website/#)

This project explores a simple idea: instead of asking generative models to recover motif-level structure implicitly, we encode motifs directly into the representation used for graph generation, independent of the sequence transformer's processing. We are interesting in constructing a representation in the form of flat tokens.

![HDT](/docs/figure/encoding.png)
![HDT](/docs/figure/decoding.png)

## Core Approach

For generating graphs using tokens with hierarchical insights, we need 3 things:

1. **Create the input H-graph**: Build a hierarchical representation of the graph using coarsening strategies (HAC, Spectral Clustering, Motif Community).

2. **Tokenize the input H-graph**: Convert the hierarchy to a token sequence using H-SENT (Vanilla HiGen) or HDT (DFS-based). Note that we need to preserve enough information (leaf edge connections) for the inverse problem to flatten the H-graph.

3. **Flatten the generated H-graph**: Reconstruct the flat graph from tokens via bipartite edge union for H-SENT, or union of back edges for HDT.

## Quick Start

### Installation

```bash
# Create training environment
conda env create -f environment.yaml
conda activate mosaic

# Create evaluation/test environment (required for full metrics)
conda env create -f environment_eval.yaml
conda activate mosaic-eval
```

### Training

The configs in `configs/` are the default hyperparameters used for our experiments. Training uses [Hydra](https://hydra.cc/) for configuration — experiment-specific overrides (dataset size, LR, steps) are in `configs/experiment/`, and tokenizer defaults in `configs/tokenizer/`.

```bash
# Train HDTC on MOSES (default)
python scripts/train.py

# Train with different tokenizers
python scripts/train.py tokenizer=sent     # Flat SENT (baseline)
python scripts/train.py tokenizer=hsent    # Hierarchical SENT
python scripts/train.py tokenizer=hdt      # Hierarchical DFS
python scripts/train.py tokenizer=hdtc     # Compositional (default)

# Train on COCONUT dataset
python scripts/train.py experiment=coconut

# Evaluate trained model
# (Use the eval environment: conda activate mosaic-eval)
python scripts/test.py model.checkpoint_path=outputs/train/.../best.ckpt
python scripts/realistic_gen.py model.checkpoint_path=outputs/train/.../best.ckpt

# Create comparison table
python scripts/comparison/compare_results.py
```

### Trained Checkpoints

Download our trained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1aMo5cQvexJ11GyIXACQys_UlA1CxKv7e?usp=drive_link).

### Batch Benchmark Scripts

The `bash_scripts/` directory automates the full benchmark pipeline. See [bash_scripts/README.md](bash_scripts/README.md) for details.

```bash
# Train all tokenizer variants
./bash_scripts/train/train_benchmarks.sh
./bash_scripts/train/train_benchmarks.sh --coconut

# Evaluate all trained models
./bash_scripts/eval/eval_benchmarks.sh
./bash_scripts/eval/eval_benchmarks.sh --coconut

# Faster eval flow: GPU sequential + CPU-parallel motif phase
./bash_scripts/eval/eval_benchmarks_2phase.sh
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
│   ├── preprocess/        # Data preprocessing and caching
│   ├── comparison/        # Result comparison and benchmarking
│   └── visualization/     # Visualization and demo scripts
├── bash_scripts/          # Batch benchmark automation scripts
├── property_experiment/   # Post-hoc analysis experiments
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Documentation

See the [docs/](docs/) directory for:
- [Codebase Guide](docs/codebase.md)
- [Reproducibility Across GPUs](docs/designs/reproducibility_across_gpus.md)

Setup guides (`docs/setups/`):
- [Server Setup Guide](docs/setups/server_setup.md)
- [GCP Setup](docs/setups/setup_gcp.md)
- [Training Setup Guide](docs/setups/setup_training.md)

Design docs (`docs/designs/`):
- [H-graph Construction](docs/designs/hgraph.md)
- [Tokenization](docs/designs/tokenization.md)
- [Evaluation Metrics](docs/designs/metric.md)
- [Visualize Tokenization](docs/designs/visualization.md)
- [Property Experiments](docs/designs/property_experiment.md)

## Acknowledgement

This codebase was developed based on insights from：
- The official [AutoGraph](https://github.com/BorgwardtLab/AutoGraph) repository.
- The official [HiGen](https://github.com/Karami-m/HiGen_main) repository.