# Visualize Hierarchical Graph Tokenization (H-SENT)

This document describes the hierarchical graph tokenization system and how to visualize it.

## Overview

H-SENT (Hierarchical SENT) extends the SENT tokenization scheme with hierarchical graph decomposition inspired by [HiGen](https://arxiv.org/abs/2305.19337). Instead of treating the graph as a flat structure, H-SENT:

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

Comparing SENT and H-SENT:

```python
# Run demo comparing H-SENT and SENT
python scripts/visualize_tokenization.py --demo --output-dir ./figures
python scripts/visualize_tokenization.py --motif-aware --alpha 10.0 --name cholesterol
```

Compare standard vs motif-aware coarsening:

```python
# Compare standard vs motif-aware coarsening
python scripts/visualize_motif_coarsening.py --name cholesterol --alpha 10.0
python scripts/visualize_motif_coarsening.py --demo --output-dir ./figures
```

Visualize the $M$ matrix:

```python
# Custom SMILES                                                                                            
python scripts/visualize_m.py --smiles "c1ccc2ccccc2c1" --name naphthalene                                 
                                                                                                            
# Demo with 9 complex molecules                                                                            
python scripts/visualize_m.py --demo --output-dir ./figures                                                
                                                                                                            
# List all molecules with motif counts                                                                     
python scripts/visualize_m.py --list                                                                       
                                                                                                            
# Normalize M by motif size (prevents large motifs from dominating)                                        
python scripts/visualize_m.py --name cholesterol --normalize
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

## Motif-Aware Coarsening

### Overview

Motif-aware coarsening extends the standard spectral clustering to preserve molecular ring structures (motifs) during graph partitioning. It uses a modified affinity matrix:

```
A' = A + α · M
```

Where:
- `A` = original adjacency matrix
- `M` = motif co-membership matrix (`M[i,j]` = number of motifs containing both atoms i and j)
- `α` = hyperparameter controlling motif influence (default: 1.0, higher = stronger preservation)

### Comparing Coarsening Methods

Use the comparison script to visualize the difference:

```bash
# Compare on a single molecule
python scripts/visualize_motif_coarsening.py --name cholesterol --alpha 10.0

# Compare with custom SMILES
python scripts/visualize_motif_coarsening.py --smiles "c1ccc(-c2ccccc2)cc1" --name biphenyl

# Run demo on complex molecules
python scripts/visualize_motif_coarsening.py --demo --output-dir ./figures --no-show

# List available molecules with motif counts
python scripts/visualize_motif_coarsening.py --list
```

### Comparison Output

The script generates a 4-panel comparison showing:

| Panel | Description |
|-------|-------------|
| **Top-Left** | Standard spectral coarsening with community colors |
| **Top-Right** | Standard block matrix |
| **Bottom-Left** | Motif-aware coarsening with community colors |
| **Bottom-Right** | Motif-aware block matrix |

Each panel shows the cohesion metric (% of motifs kept intact in single communities).

### Example Results (α=10.0)

| Molecule | Atoms | Motifs | Std Cohesion | MA Cohesion | Improvement |
|----------|-------|--------|--------------|-------------|-------------|
| cholesterol | 28 | 4 | 50% | 100% | +50% |
| morphine | 21 | 2 | 50% | 100% | +50% |
| resveratrol | 17 | 2 | 0% | 100% | +100% |
| quercetin | 22 | 2 | 50% | 100% | +50% |

### Using Motif-Aware Tokenization

```python
from src.tokenizers.hierarchical import HSENTTokenizer

# Enable motif-aware coarsening
tokenizer = HSENTTokenizer(
    seed=42,
    motif_aware=True,      # Enable motif awareness
    motif_alpha=2.0,       # Strength of motif preference (default: 1.0)
)

# Tokenize as usual
tokenizer.set_num_nodes(100)
tokens = tokenizer.tokenize(data)
```

### Tuning α

| α Value | Effect |
|---------|--------|
| 0 | Standard spectral clustering (no motif awareness) |
| 1.0 | Motif co-membership treated equally to actual edges |
| 2.0-5.0 | Moderate preference for keeping motifs together |
| 10.0+ | Strong preference; may reduce modularity |

**Recommendation:** Start with α=1.0 and increase if ring structures are being split.

---

## Hierarchical Decomposition

### How It Works

1. **Spectral Clustering**: Graph is partitioned into communities using modularity-optimized spectral clustering (or motif-aware variant)

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

- [HiGen: Hierarchical Graph Generative Networks](https://arxiv.org/abs/2305.19337) - Hierarchical decomposition approach
- [AutoGraph](https://arxiv.org/abs/2306.10310) - SENT tokenization scheme
