# Hierarchical Graph Tokenization (H-SENT)

This document describes the hierarchical graph tokenization system and how to visualize it.

## Overview

H-SENT (Hierarchical SENT) extends the SENT tokenization scheme with hierarchical graph decomposition inspired by [HiGen](https://arxiv.org/abs/2305.19843). Instead of treating the graph as a flat structure, H-SENT:

1. **Decomposes** the graph into communities using spectral clustering
2. **Encodes** each community (partition) separately using SENT-style walks
3. **Connects** communities via bipartite edge encodings
4. **Reconstructs** the original graph from tokens (roundtrip verified)

## Visualization

### Quick Start

```bash
conda activate mosaic

# Visualize a molecule by name
python scripts/visualize_htoken.py --name cholesterol

# Visualize by SMILES string
python scripts/visualize_htoken.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"

# Save to file without displaying
python scripts/visualize_htoken.py --name caffeine --output caffeine.png --no-show

# Full 4-panel view (includes token sequence)
python scripts/visualize_htoken.py --name morphine --full

# Run demo with multiple complex molecules
python scripts/visualize_htoken.py --demo --output-dir ./figures
```

### Available Molecules

List all pre-defined molecules:

```bash
python scripts/visualize_htoken.py --list
```

Available molecules include: benzene, naphthalene, caffeine, aspirin, ibuprofen, cholesterol, testosterone, morphine, penicillin_g, dopamine, serotonin, nicotine, glucose, and more.

### Visualization Panels

The default 3-panel visualization shows:

| Panel | Description |
|-------|-------------|
| **Graph with Communities** | Original graph with nodes colored by community assignment |
| **Block Matrix** | Reordered adjacency matrix showing partition structure |
| **Hierarchy Structure** | Text summary of partitions and bipartite connections |

The `--full` option adds a 4th panel showing the token sequence.

## Python API

### Basic Usage

```python
from torch_geometric.data import Data
from src.tokenizers.hierarchical import HSENTTokenizer, quick_visualize

# Create tokenizer
tokenizer = HSENTTokenizer(seed=42)
tokenizer.set_num_nodes(100)

# Visualize a graph
fig = quick_visualize(data, tokenizer, save_path="output.png")
```

### Individual Visualizations

```python
from src.tokenizers.hierarchical import (
    HSENTTokenizer,
    visualize_hierarchy,
    visualize_graph_communities,
    visualize_block_matrix,
    visualize_tokens,
)

# Build hierarchy
tokenizer = HSENTTokenizer(seed=42)
hg = tokenizer.coarsener.build_hierarchy(data)
tokens = tokenizer.tokenize_hierarchy(hg)

# Individual plots
fig1 = visualize_hierarchy(hg, data)           # 3-panel HiGen style
fig2 = visualize_graph_communities(hg, data)   # Graph with colors
fig3 = visualize_block_matrix(hg)              # Block adjacency
fig4 = visualize_tokens(tokens, tokenizer)     # Token sequence
```

### Converting SMILES to Graph

```python
import torch
from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())

data = smiles_to_graph("c1ccccc1")  # benzene
```

## Hierarchical Decomposition

### How It Works

1. **Spectral Clustering**: Graph is partitioned into communities using modularity-optimized spectral clustering

2. **Partitions (Diagonal Blocks)**: Each community becomes a partition containing:
   - Induced subgraph edges (intra-community)
   - Global node indices for reconstruction

3. **Bipartites (Off-Diagonal Blocks)**: Edges between communities are stored as bipartite graphs

4. **Token Encoding**: Each partition is encoded using SENT-style random walk with back-edges

### Token Format

```
[SOS]
[LCOM] [part_id] [num_nodes] [global_indices...] [SEP] [SENT walk...] [RCOM]
[LCOM] ... [RCOM]
[LBIP] [left_id] [right_id] [edge pairs...] [RBIP]
[LBIP] ... [RBIP]
[EOS]
```

### Example: Cholesterol

```
Molecule: Cholesterol (C27H46O)
Atoms: 28
Bonds: 31

Hierarchical Decomposition:
- 5 communities (partitions)
- Partition sizes: [4, 4, 8, 7, 5]
- 4 bipartite connections
- Total tokens: 187
```

## Roundtrip Verification

The tokenization is verified to be lossless:

```python
# Forward: graph -> hierarchy -> tokens
hg = tokenizer.coarsener.build_hierarchy(data)
tokens = tokenizer.tokenize_hierarchy(hg)

# Backward: tokens -> hierarchy -> graph
hg_reconstructed = tokenizer.parse_tokens(tokens)
data_reconstructed = hg_reconstructed.reconstruct()

# Verify edges match exactly
assert set(original_edges) == set(reconstructed_edges)
```

All 32 tests pass including molecular roundtrips for aspirin, caffeine, ibuprofen, cholesterol, and more.

## Dependencies

The hierarchical tokenization requires:

```yaml
- scikit-learn    # SpectralClustering
- python-louvain  # Modularity computation (optional)
```

Install if missing:

```bash
pip install scikit-learn
```

## References

- [HiGen: Hierarchical Graph Generative Networks](https://arxiv.org/abs/2305.19843) - Hierarchical decomposition approach
- [AutoGraph](https://arxiv.org/abs/2306.10310) - SENT tokenization scheme
