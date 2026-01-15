# Hierarchical Graph Representation (HGraph)

This document describes the hierarchical graph data structures and coarsening algorithms used to decompose graphs into community-based hierarchies.

## Overview

Hierarchical graph decomposition enables explicit encoding of multi-level graph structure by partitioning nodes into communities. This representation is the foundation for hierarchical tokenization schemes.

### Motivation

Standard flat representations treat graphs as monolithic structures, missing the hierarchical organization present in many real-world graphs. Molecular graphs, for example, contain:

- **Functional groups** (hydroxyl, carboxyl, amino)
- **Ring systems** (benzene, pyridine, cyclohexane)
- **Scaffolds** (multi-ring fused structures)

These substructures naturally form **communities** where internal connectivity is denser than external connectivity. Hierarchical decomposition exploits this property to:

1. **Preserve motifs by design**: Dense subgraphs cluster into the same community
2. **Create structured representations**: Hierarchical organization enables shorter, more interpretable sequences
3. **Enable constraint-based generation**: Super-edge weights constrain the number of inter-community connections

## Data Structures

### Partition

A `Partition` represents the induced subgraph of a single community (diagonal block in the adjacency block matrix).

```python
@dataclass
class Partition:
    part_id: int                              # Unique partition identifier
    global_node_indices: list[int]            # Local -> global index mapping
    edge_index: Tensor                        # [2, num_edges] in LOCAL indices
    child_hierarchy: Optional[HierarchicalGraph]  # For nested hierarchies

    def local_to_global(self, local_idx: int) -> int
    def global_to_local(self, global_idx: int) -> int
    def get_all_edges_global(self) -> set[tuple[int, int]]
```

**Key Properties:**
- `num_nodes`: Number of nodes in this partition
- `num_edges`: Number of internal edges
- `is_leaf`: True if no child hierarchy exists

**Index Coordination:**
```
Local indices: (0, 1, 2) within partition
       ↓ local_to_global
Global indices: (5, 7, 9) in original graph
```

### Bipartite

A `Bipartite` represents edges between two communities (off-diagonal block in the adjacency block matrix).

```python
@dataclass
class Bipartite:
    left_part_id: int                         # Source partition ID
    right_part_id: int                        # Target partition ID
    edge_index: Tensor                        # [2, num_edges] in LOCAL indices
```

Edges are stored as pairs of local indices: the first row contains indices local to `left_part_id`, the second row contains indices local to `right_part_id`.

### HierarchicalGraph

`HierarchicalGraph` is the top-level container for the complete hierarchical decomposition.

```python
@dataclass
class HierarchicalGraph:
    partitions: list[Partition]               # All communities at this level
    bipartites: list[Bipartite]               # All inter-community edges
    community_assignment: list[int]           # Maps global node → partition ID

    def get_partition(self, part_id: int) -> Partition
    def get_all_edges_global(self) -> set[tuple[int, int]]
    def reconstruct(self) -> Data
    def get_level_info(self) -> dict
```

**Key Properties:**
- `num_nodes`: Total number of nodes
- `num_communities`: Number of partitions
- `depth`: Maximum depth of nested hierarchies

## Coarsening Algorithms

Coarsening algorithms partition nodes into communities based on graph structure.

### Spectral Coarsening

Spectral coarsening uses spectral clustering with modularity optimization to discover community structure.

#### Algorithm

1. **Build adjacency matrix** $A$ from edge_index
2. **Symmetrize** for undirected graphs: $A \leftarrow (A + A^T) / 2$
3. **Search for optimal $k$**:
   $$k^* = \arg\max_{k \in [k_{\min}, k_{\max}]} Q_k$$
   where:
   - $k_{\min} = \max(2, \lfloor\sqrt{n} \times 0.7\rfloor)$
   - $k_{\max} = \min(n-1, \lfloor\sqrt{n} \times 1.3\rfloor)$
4. **Run SpectralClustering** for each $k$ with `affinity="precomputed"`
5. **Select partition** with maximum modularity

#### Modularity

Partition quality is measured by **modularity**:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where:
- $m = \frac{1}{2}\sum_{ij} A_{ij}$ is the total edge weight
- $k_i = \sum_j A_{ij}$ is the degree of node $i$
- $c_i$ is the community assignment of node $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$, else $0$

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_community_size` | int | `4` | Minimum nodes before stopping recursion |
| `k_min_factor` | float | `0.7` | Lower bound factor for $k$ search |
| `k_max_factor` | float | `1.3` | Upper bound factor for $k$ search |
| `n_init` | int | `100` | SpectralClustering initializations |
| `seed` | int | `None` | Random seed for reproducibility |

### Motif-Aware Coarsening

Motif-aware coarsening extends spectral clustering to preserve chemically meaningful structures by augmenting the affinity matrix.

#### Motivation

Standard spectral clustering optimizes modularity but is agnostic to chemically meaningful structures. This can lead to **ring systems being split** across communities, which destroys motif semantics during generation.

#### Modified Affinity Matrix

The adjacency matrix is augmented with a **motif co-membership matrix**:

$$A' = A + \alpha \cdot M$$

where:
- $A$ is the original adjacency matrix
- $M$ is the motif co-membership matrix
- $\alpha \geq 0$ controls motif influence

#### Motif Co-Membership Matrix

The matrix $M$ encodes which atom pairs share membership in detected motifs:

$$M_{ij} = \sum_{m \in \mathcal{M}} \mathbf{1}[i \in m \land j \in m]$$

where $\mathcal{M}$ is the set of all detected motif instances.

**Properties of $M$:**
- Symmetric: $M_{ij} = M_{ji}$
- Non-negative: $M_{ij} \geq 0$
- Diagonal: $M_{ii} = $ number of motifs containing atom $i$
- For overlapping motifs (e.g., fused rings), $M_{ij}$ can be $> 1$

**Example**: For naphthalene (two fused benzene rings sharing atoms 3 and 8):
- Atoms in benzene 1 only: $M_{ij} = 1$
- Atoms in benzene 2 only: $M_{ij} = 1$
- Atoms 3 and 8 (shared): $M_{3,8} = 2$ (both benzenes)

#### Motif Detection

Motifs are detected using **SMARTS patterns** via RDKit when a SMILES string is available:

```python
CLUSTERING_MOTIFS = {
    # Aromatic 6-membered rings
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    # Aromatic 5-membered rings
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1cnc[nH]1",
    # Fused ring systems
    "naphthalene": "c1ccc2ccccc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    # Saturated rings
    "cyclopentane": "C1CCCC1",
    "cyclohexane": "C1CCCCC1",
    "cyclohexene": "C1=CCCCC1",
}
```

The focus is on **ring systems** because:
1. Rings form the structural backbone of molecules
2. Splitting a ring destroys its chemical identity
3. Functional groups (OH, COOH) are typically small and naturally cluster

#### The α Hyperparameter

| $\alpha$ Value | Interpretation |
|----------------|----------------|
| $\alpha = 0$ | Standard spectral clustering (no motif awareness) |
| $\alpha = 1$ | Motif co-membership weighted equally to actual edges |
| $\alpha = 2$-$5$ | Moderate preference for motif preservation |
| $\alpha = 10$+ | Strong motif preservation; may reduce modularity |

**Tuning Guidelines:**
- Start with $\alpha = 1.0$ (default)
- Increase $\alpha$ if ring systems are being split
- Decrease $\alpha$ if communities become too large (poor modularity)
- For molecules with many overlapping motifs, use lower $\alpha$

#### Motif Cohesion Metric

Motif preservation is measured using the **cohesion rate**:

$$\text{Cohesion} = \frac{|\{m \in \mathcal{M} : m \subseteq C_i \text{ for some } i\}|}{|\mathcal{M}|}$$

A motif is "intact" if all its atoms belong to a single community. Cohesion of 1.0 means all motifs are preserved.

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | `1.0` | Weight for motif affinity |
| `motif_patterns` | dict | `None` | Custom SMARTS patterns (uses defaults if None) |
| `normalize_by_motif_size` | bool | `False` | Normalize M by motif size |

### Future Coarsening Methods

The coarsening framework is designed to be extensible. Planned additions include:

- **Hierarchical Agglomerative Clustering (HAC)**: Bottom-up clustering that iteratively merges similar nodes/clusters
- **Metis-based Partitioning**: Graph partitioning using the METIS library
- **Learned Coarsening**: Neural network-based community detection

## Hierarchical Decomposition

### Block Structure

Given a partition $\mathcal{P} = \{C_1, C_2, \ldots, C_k\}$, the graph decomposes into:

**Diagonal Blocks (Partitions)**: For each community $C_i$, the induced subgraph $G[C_i] = (C_i, E_i)$ where:
$$E_i = \{(u, v) \in E : u \in C_i \land v \in C_i\}$$

**Off-Diagonal Blocks (Bipartites)**: For each pair of communities $(C_i, C_j)$ where $i < j$, the bipartite subgraph contains:
$$E_{ij} = \{(u, v) \in E : u \in C_i \land v \in C_j\}$$

### Recursive Structure

The decomposition can be applied recursively for multi-level hierarchies:

```
build_hierarchy(G, recursive=True):
    if |V(G)| < min_community_size:
        return SinglePartition(G)

    communities = spectral_partition(G)

    if len(communities) <= 1:
        return SinglePartition(G)

    partitions = []
    for each community C in communities:
        partition = extract_induced_subgraph(C)
        if recursive and |C| >= min_community_size:
            partition.child_hierarchy = build_hierarchy(partition)
        partitions.append(partition)

    bipartites = extract_all_bipartites(communities)

    return HierarchicalGraph(partitions, bipartites)
```

The `min_community_size` parameter controls recursion depth:
- Small values produce deep hierarchies
- Large values produce shallow hierarchies with larger leaf partitions

## Graph Reconstruction

### Flattening Algorithm

The original graph is reconstructed by combining all edges from all levels:

$$E_{\text{reconstructed}} = \bigcup_{i} E_i \cup \bigcup_{i<j} E_{ij}$$

For nested hierarchies, edges are collected recursively.

### Index Conversion

**Partition edges** $(u_{\text{local}}, v_{\text{local}}) \in E_i$:
$$u_{\text{global}} = \text{global\_indices}_i[u_{\text{local}}]$$
$$v_{\text{global}} = \text{global\_indices}_i[v_{\text{local}}]$$

**Bipartite edges** $(u_{\text{local}}, v_{\text{local}}) \in E_{ij}$:
$$u_{\text{global}} = \text{global\_indices}_i[u_{\text{local}}]$$
$$v_{\text{global}} = \text{global\_indices}_j[v_{\text{local}}]$$

### Implementation

```python
def reconstruct(hg: HierarchicalGraph) -> Data:
    """Reconstruct PyG Data from hierarchical graph."""
    all_edges = set()

    # Collect edges from all partitions (recursively)
    for partition in hg.partitions:
        all_edges.update(partition.get_all_edges_global())

    # Collect edges from all bipartites
    for bipartite in hg.bipartites:
        left_part = hg.get_partition(bipartite.left_part_id)
        right_part = hg.get_partition(bipartite.right_part_id)
        for i in range(bipartite.edge_index.size(1)):
            left_local = bipartite.edge_index[0, i].item()
            right_local = bipartite.edge_index[1, i].item()
            u = left_part.local_to_global(left_local)
            v = right_part.local_to_global(right_local)
            all_edges.add((u, v))
            all_edges.add((v, u))  # Undirected

    # Convert to PyG Data
    edge_list = list(all_edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=hg.num_nodes)
```

### Lossless Guarantee

The encoding is provably lossless:

$$\forall G: \quad \text{reconstruct}(\text{build\_hierarchy}(G)) \equiv G$$

This is verified by roundtrip tests on all graph types including:
- Synthetic graphs (triangles, paths, stars, complete graphs)
- Molecular graphs (aspirin, caffeine, cholesterol, morphine)

## References

1. **HiGen**: Hierarchical Graph Generative Networks - [arXiv:2305.19843](https://arxiv.org/abs/2305.19337)
2. **Spectral Clustering**: Ng, Jordan, Weiss (2001) - On Spectral Clustering
3. **Modularity**: Newman (2006) - Modularity and community structure in networks
