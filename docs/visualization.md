# Visualize Graph Tokenization Schemes

This document describes the graph tokenization systems and how to visualize them.

## Overview

MOSAIC provides four tokenization schemes for converting graphs to sequences:

| Scheme | Description | Key Feature |
|--------|-------------|-------------|
| **SENT** | Flat random walk with back-edges | Simple, baseline |
| **H-SENT** | Hierarchical with explicit partition blocks | Interpretable structure |
| **HDT** | Hierarchical DFS with implicit nesting | ~45% fewer tokens |
| **HDTC** | Compositional with functional groups | Guarantees chemical motif preservation |

## Visualization

### Quick Start: Compare Tokenization Schemes

```bash
conda activate mosaic

# Compare SENT, H-SENT, HDT, and HDTC on a molecule
python scripts/visualization/visualize_tokenization.py --name cholesterol --output-dir ./figures

# Run demo with complex molecules (cholesterol, morphine, caffeine, penicillin)
python scripts/visualization/visualize_tokenization.py --demo --output-dir ./figures

# List available molecules
python scripts/visualization/visualize_tokenization.py --list
```

![tokenization](figure/tokenization.png)

### Visualization Panels

The comparison shows:

| Panel | Description |
|-------|-------------|
| **Molecule with Motifs** | Original graph with detected ring structures highlighted |
| **SENT** | Random walk traversal with visit order on nodes |
| **H-SENT** | Community structure with cross-community edges |
| **HDT** | Hierarchical tree with bidirectional parent↔child arrows |
| **HDTC** | Two-level functional hierarchy with ring/functional group communities |

Compare standard vs motif-aware coarsening:

```bash
# Compare standard vs motif-aware coarsening
python scripts/visualization/visualize_motif_aware_sc.py --name cholesterol --alpha 10.0
python scripts/visualization/visualize_motif_aware_sc.py --demo --output-dir ./figures
```

![motif](figure/motif.png)

Visualize the $M$ matrix:

```bash
# Custom SMILES
python scripts/visualization/visualize_motif_m_matrix.py --smiles "c1ccc2ccccc2c1" --name naphthalene

# Demo with 9 complex molecules
python scripts/visualization/visualize_motif_m_matrix.py --demo --output-dir ./figures

# List all molecules with motif counts
python scripts/visualization/visualize_motif_m_matrix.py --list

# Normalize M by motif size (prevents large motifs from dominating)
python scripts/visualization/visualize_motif_m_matrix.py --name cholesterol --normalize
```

![m_matrix](figure/m_matrix.png)

### Generation Demo

Visualize step-by-step molecule generation with animated GIFs. Uses Hydra
configuration (`configs/generation_demo.yaml`) with per-model entries for
checkpoint path, tokenizer type, and labeled graph settings.

Each tokenizer type has a specialized visualizer:

| Tokenizer | Side Panel | Features |
|-----------|-----------|----------|
| **SENT** | None | Random walk phase tracking |
| **H-SENT** | Block diagram | Partition fill bars, bipartite arrows |
| **HDT** | Abstract tree | Community hierarchy, current-community highlight |
| **HDTC** | Typed abstract tree | R/F/S type labels, type-based coloring |

All tokenizers support motif detection (ring and functional group highlighting)
via `FunctionalGroupDetector`, with progressive reveal as atoms/edges appear.

```bash
# Default: generate with models listed in configs/generation_demo.yaml
python scripts/visualization/generation_demo.py

# Override generation settings
python scripts/visualization/generation_demo.py generation.num_samples=5 animation.fps=3

# Change output directory
python scripts/visualization/generation_demo.py output.dir=outputs/my_demo

# Disable motif highlighting
python scripts/visualization/generation_demo.py motif.enabled=false

# Single model override (HDTC example)
python scripts/visualization/generation_demo.py \
    'models=[{name: my_hdtc, checkpoint_path: outputs/train/hdtc/best.ckpt, tokenizer_type: hdtc, labeled_graph: true}]'
```

## References
- [HiGen: Hierarchical Graph Generative Networks](https://arxiv.org/abs/2305.19337) - Hierarchical decomposition approach
- [AutoGraph](https://arxiv.org/abs/2306.10310) - SENT tokenization scheme
