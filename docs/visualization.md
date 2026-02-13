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

### Community Structure Comparison (HAC vs Spectral)

Compare how HAC (Affinity Coarsening) and Spectral Coarsening partition molecules
across MOSES (simple drug-like) and COCONUT (complex natural products) datasets.

Produces two types of figures:
- **Example progression figures** (4 molecules spanning small→large): 2x2 grid per molecule showing molecule+community overlay and hierarchy tree for both HAC and Spectral.
- **Aggregate statistics figure**: 2x3 grid comparing hierarchy depth, community count, largest community size, singleton fraction, non-singleton sizes, and depth vs atom count.

```bash
conda activate mosaic

# Default run (200 molecules/dataset, 4 examples, outputs to tmp/)
python scripts/visualization/compare_community_structure.py --no-show

# Custom output directory and sample size
python scripts/visualization/compare_community_structure.py \
    --output-dir ./figures/community_comparison \
    --num-stats-samples 500 \
    --no-show

# Use different cache files
python scripts/visualization/compare_community_structure.py \
    --moses-cache data/cache/moses_train_hdt_1000_d111408d.pt \
    --coconut-cache data/cache/coconut_train_hdt_5000_d111408d.pt \
    --no-show

# Adjust coarsening granularity
python scripts/visualization/compare_community_structure.py \
    --min-community-size 6 \
    --no-show
```

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `tmp/feature/hac-improvement/community_comparison` | Output directory |
| `--num-examples` | `4` | Number of example molecules |
| `--num-stats-samples` | `200` | Molecules per dataset for statistics |
| `--moses-cache` | `data/cache/moses_train_hdt_1000_d111408d.pt` | MOSES cache path |
| `--coconut-cache` | `data/cache/coconut_train_hdt_5000_d111408d.pt` | COCONUT cache path |
| `--min-community-size` | `4` | Minimum community size for coarsening |
| `--seed` | `42` | Random seed |
| `--dpi` | `150` | Output image DPI |

## References
- [HiGen: Hierarchical Graph Generative Networks](https://arxiv.org/abs/2305.19337) - Hierarchical decomposition approach
- [AutoGraph](https://arxiv.org/abs/2306.10310) - SENT tokenization scheme
