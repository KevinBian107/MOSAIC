# Molecular Graph Generation with Motif Analysis

A framework for training and evaluating molecular graph generation models with **motif distribution analysis**.

## Overview

This codebase provides tools for:

1. **Molecular dataset loading** from MOSES and QM9 benchmarks
2. **Graph tokenization** using the SENT scheme from AutoGraph
3. **Transformer-based molecular generation** models
4. **AutoGraph-style evaluation metrics** (validity, uniqueness, novelty, FCD, SNN, fragment/scaffold similarity)
5. **Motif distribution analysis** comparing functional groups and fragments between training and generated molecules

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/user/mosaic.git
cd mosaic

# Create conda environment
conda env create -f environment.yaml
conda activate mosaic

# Install in development mode (optional)
pip install -e ".[dev]"
```

### Training a Model

```bash
# Train with default configuration (MOSES dataset)
python scripts/train.py

# Train on QM9 dataset
python scripts/train.py experiment=qm9

# Train with custom settings
python scripts/train.py \
    model.model_name=gpt2-s \
    trainer.max_steps=100000 \
    data.num_train=50000
```

### Evaluating a Model

```bash
python scripts/test.py model.checkpoint_path=outputs/model.ckpt
```

### Visualizing Molecular Motifs

```bash
# Visualize a single molecule with detailed motif analysis
python scripts/visualize_motifs.py --smiles "c1ccccc1O"

# Visualize molecules from MOSES dataset
python scripts/visualize_motifs.py --dataset moses --num_molecules 6

# Visualize molecules from QM9 dataset
python scripts/visualize_motifs.py --dataset qm9 --num_molecules 6

# Save visualization to file
python scripts/visualize_motifs.py --smiles "CC(=O)Nc1ccc(O)cc1" --output paracetamol.png
```

## Project Structure

```
mosaic/
├── src/
│   ├── data/           # Molecular data loading (MOSES, QM9)
│   ├── tokenizers/     # Graph tokenization schemes
│   ├── models/         # Neural network models
│   └── evaluation/     # Evaluation metrics
├── configs/            # Hydra configuration files
├── scripts/            # Training, evaluation, and visualization scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Datasets

### MOSES

The MOSES benchmark contains ~1.9M drug-like molecules filtered from ZINC. Default configuration uses a subset for efficient training.

- **Source**: [molecularsets/moses](https://github.com/molecularsets/moses)
- **Molecules**: Drug-like, MW 250-350 Da
- **Max atoms**: ~27 heavy atoms

### QM9

QM9 contains ~130K small organic molecules with up to 9 heavy atoms.

- **Source**: MoleculeNet benchmark
- **Atom types**: C, N, O, F
- **Properties**: Quantum mechanical properties available

## Evaluation Metrics

### AutoGraph-Style Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Validity | Fraction passing RDKit valency checks | 0-1 (higher=better) |
| Uniqueness | Fraction of unique canonical SMILES | 0-1 (higher=better) |
| Novelty | Fraction not in training set | 0-1 (higher=better) |
| SNN | Similarity to nearest neighbor (Tanimoto) | 0-1 (higher=more similar) |
| Frag | Fragment frequency similarity (BRICS) | 0-1 (higher=more similar) |
| Scaff | Scaffold frequency similarity | 0-1 (higher=more similar) |
| FCD | Frechet ChemNet Distance | 0+ (lower=better) |

### Motif Distribution Metrics

| Metric | Description |
|--------|-------------|
| motif_fg_mmd | MMD of functional group distributions |
| motif_smarts_mmd | MMD of SMARTS pattern matches |
| motif_ring_mmd | MMD of ring system information |
| motif_brics_mmd | MMD of BRICS fragment distributions |

## Molecular Motifs

We detect and analyze various molecular substructures:

### SMARTS-Based Motifs

- **Aromatic rings**: benzene, pyridine, furan, thiophene, etc.
- **Functional groups**: hydroxyl, carboxyl, carbonyl, amine, etc.
- **Halogens**: fluorine, chlorine, bromine, iodine

### RDKit Functional Groups

The 82 built-in functional group patterns from `rdkit.Chem.Fragments`.

### BRICS Fragments

Retrosynthetically interesting fragments from BRICS decomposition.

### Ring Systems

- Number of rings (total, aromatic, aliphatic)
- Ring sizes (3-8 membered)
- Heterocycles and bridgehead atoms

## SENT Tokenization

SENT (Sequence of Edge-indicating Neighborhoods) converts molecular graphs to token sequences via random walk with back-edge encoding:

```
Molecule: ethanol (CCO)
Graph: C-C-O

Tokens: [SOS, 0, 1, [, 0, ], 2, EOS]
        (start, C0, C1, [back-edge to C0], O, end)
```

## Next Steps

- [Architecture Guide](architecture.md)
- [Setup Guide](setup.md)
- [Contributing Guide](contributing.md)
