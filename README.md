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

### Scaffold-Primed Generation

Generate molecules starting from a scaffold structure. This enables zero-shot generation of complex molecules by priming the model with known structural motifs.

```bash
# Generate from a named scaffold (e.g., naphthalene)
python scripts/primed_gen.py \
    model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
    scaffold.name=naphthalene \
    scaffold.num_samples=10

# Generate from custom SMILES
python scripts/primed_gen.py \
    model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
    scaffold.smiles="c1ccc2ccccc2c1"

# Generate from all Tier 2 scaffolds (fused bicyclic)
python scripts/primed_gen.py \
    model.checkpoint_path=outputs/train/moses_hdt_*/best.ckpt \
    scaffold.tier=2

# Use different tokenizer (HSENT)
python scripts/primed_gen.py \
    model.checkpoint_path=outputs/train/moses_hsent_*/best.ckpt \
    tokenizer=hsent \
    scaffold.name=carbazole
```

**Scaffold Tiers:**
- **Tier 1**: Simple monocyclic (benzene, pyridine, furan, etc.)
- **Tier 2**: Fused bicyclic (naphthalene, indole, quinoline, etc.)
- **Tier 3**: Complex polycyclic (carbazole, pyrene, phenanthrene, etc.)

<details>
<summary>Python API</summary>

```python
from src.models import GraphGeneratorModule
from src.transfer_learning import PrimedGenerator

# Load trained model
model = GraphGeneratorModule.load_from_checkpoint("path/to/checkpoint.ckpt")

# Create primed generator
generator = PrimedGenerator(model)

# Generate from a named scaffold
graphs, time = generator.generate_from_scaffold("naphthalene", num_samples=10)

# Generate from custom SMILES
graphs, time = generator.generate_from_smiles("c1ccc2ccccc2c1", num_samples=5)

# Generate from all Tier 2 scaffolds
results, time = generator.generate_by_tier(tier=2, samples_per_scaffold=5)

# List available scaffolds
print(generator.list_available_scaffolds())
```
</details>

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
│   ├── realistic_gen/     # Generation quality analysis
│   └── transfer_learning/ # Scaffold priming for complex generation
│       ├── scaffolds/     # Scaffold library and tier patterns
│       ├── primers/       # Tokenizer-specific primers
│       └── generation/    # Primed generation utilities
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