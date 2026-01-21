# Visualize Graph Tokenization Schemes

This document describes the graph tokenization systems and how to visualize them.

## Overview

MOSAIC provides three tokenization schemes for converting graphs to sequences:

| Scheme | Description | Key Feature |
|--------|-------------|-------------|
| **SENT** | Flat random walk with back-edges | Simple, baseline |
| **H-SENT** | Hierarchical with explicit partition blocks | Interpretable structure |
| **HDT** | Hierarchical DFS with implicit nesting | ~45% fewer tokens |

## Visualization

### Quick Start: Compare All Three Schemes

```bash
conda activate mosaic

# Compare SENT, H-SENT, and HDT on a molecule
python scripts/visualize_tokenization.py --name cholesterol

# Run demo with complex molecules (cholesterol, morphine, caffeine, penicillin)
python scripts/visualize_tokenization.py --demo --output-dir ./figures

# Save without displaying
python scripts/visualize_tokenization.py --name morphine --output morphine.png --no-show

# List available molecules
python scripts/visualize_tokenization.py --list
```

![tokenization](figure/tokenization.png)

### Visualization Panels

The 4-column comparison shows:

| Panel | Description |
|-------|-------------|
| **Molecule with Motifs** | Original graph with detected ring structures highlighted |
| **SENT** | Random walk traversal with visit order on nodes |
| **H-SENT** | Community structure with cross-community edges |
| **HDT** | Hierarchical tree with bidirectional parent↔child arrows |

Compare standard vs motif-aware coarsening:

```python
# Compare standard vs motif-aware coarsening
python scripts/visualize_motif_coarsening.py --name cholesterol --alpha 10.0
python scripts/visualize_motif_coarsening.py --demo --output-dir ./figures
```

![motif](figure/motif.png)

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

![m_matrix](figure/m_matrix.png)

## References

- [HiGen: Hierarchical Graph Generative Networks](https://arxiv.org/abs/2305.19337) - Hierarchical decomposition approach
- [AutoGraph](https://arxiv.org/abs/2306.10310) - SENT tokenization scheme
