# Hierarchical SENT Tokenization (H-SENT)

This document provides a comprehensive technical description of the Hierarchical SENT (H-SENT) tokenization scheme for motif-preserving graph generation.

## Introduction

H-SENT extends the flat SENT tokenization with hierarchical graph decomposition, enabling explicit encoding of multi-level graph structure. The approach is inspired by [HiGen](https://arxiv.org/abs/2305.19843)'s hierarchical generation paradigm, adapted for transformer-based autoregressive models.

### Motivation

Standard flat tokenization schemes treat graphs as monolithic structures, missing the hierarchical organization present in many real-world graphs. Molecular graphs, for example, contain:

- **Functional groups** (hydroxyl, carboxyl, amino)
- **Ring systems** (benzene, pyridine, cyclohexane)
- **Scaffolds** (multi-ring fused structures)

These substructures naturally form **communities** where internal connectivity is denser than external connectivity. H-SENT exploits this property to:

1. **Preserve motifs by design**: Dense subgraphs cluster into the same community
2. **Create structured token sequences**: Hierarchical organization enables shorter, more interpretable sequences
3. **Enable constraint-based generation**: Super-edge weights constrain the number of inter-community connections

## Mathematical Formulation

### Graph Partitioning

Given a graph $G = (V, E)$ with $n = |V|$ nodes, we seek a partition $\mathcal{P} = \{C_1, C_2, \ldots, C_k\}$ such that:

$$V = \bigcup_{i=1}^{k} C_i \quad \text{and} \quad C_i \cap C_j = \emptyset \quad \forall i \neq j$$

The partition quality is measured by **modularity**:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where:
- $m = \frac{1}{2}\sum_{ij} A_{ij}$ is the total edge weight
- $k_i = \sum_j A_{ij}$ is the degree of node $i$
- $c_i$ is the community assignment of node $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$, else $0$

### Spectral Coarsening

The partitioning uses **spectral clustering** with modularity optimization:

1. **Compute affinity matrix**: $A$ (adjacency matrix with self-loops added)

2. **Search for optimal $k$**:
   $$k^* = \arg\max_{k \in [k_{\min}, k_{\max}]} Q_k$$

   where:
   - $k_{\min} = \max(2, \lfloor\sqrt{n} \times 0.7\rfloor)$
   - $k_{\max} = \min(n-1, \lfloor\sqrt{n} \times 1.3\rfloor)$

3. **Apply SpectralClustering** with `affinity="precomputed"` for each $k$

4. **Select partition** with maximum modularity

### Hierarchical Decomposition

The partition defines two types of substructures:

**Diagonal Blocks (Partitions)**: For each community $C_i$, the induced subgraph $G[C_i] = (C_i, E_i)$ where:
$$E_i = \{(u, v) \in E : u \in C_i \land v \in C_i\}$$

**Off-Diagonal Blocks (Bipartites)**: For each pair of communities $(C_i, C_j)$ where $i < j$, the bipartite subgraph contains:
$$E_{ij} = \{(u, v) \in E : u \in C_i \land v \in C_j\}$$

### Recursive Structure

The decomposition can be applied recursively. For a partition $P$ with induced subgraph $G[P]$:

```
build_hierarchy(G):
    if |V(G)| < min_community_size:
        return SinglePartition(G)

    P = spectral_partition(G)

    if |P| <= 1:
        return SinglePartition(G)

    for each community C in P:
        partition = extract_induced_subgraph(C)
        if |C| >= min_community_size:
            partition.child = build_hierarchy(partition)

    return HierarchicalGraph(partitions, bipartites)
```

The `min_community_size` parameter controls the recursion depth:
- Small values produce deep hierarchies
- Large values produce shallow hierarchies with larger leaf partitions

## Token Vocabulary

H-SENT uses 11 special tokens followed by node indices:

| Token ID | Symbol | Description |
|----------|--------|-------------|
| 0 | `SOS` | Start of sequence |
| 1 | `EOS` | End of sequence |
| 2 | `PAD` | Padding |
| 3 | `RESET` | New component (from SENT) |
| 4 | `LADJ` | Left back-edge bracket |
| 5 | `RADJ` | Right back-edge bracket |
| 6 | `LCOM` | Start community block |
| 7 | `RCOM` | End community block |
| 8 | `LBIP` | Start bipartite block |
| 9 | `RBIP` | End bipartite block |
| 10 | `SEP` | Separator |

Node indices are offset by `IDX_OFFSET = 11`:
$$\text{token\_id}(v) = v + 11$$

Total vocabulary size: $11 + n_{\max}$ where $n_{\max}$ is the maximum number of nodes.

## Token Encoding

### Sequence Structure

The complete token sequence follows this grammar:

```
sequence    := [SOS] num_comm partition* bipartite* [EOS]
num_comm    := <community_count + IDX_OFFSET>
partition   := [LCOM] part_id num_nodes global_idx* [SEP] content [RCOM]
bipartite   := [LBIP] left_id right_id num_edges edge_pair* [RBIP]
edge_pair   := left_local right_local
content     := nested_hierarchy | sent_walk
```

### Partition Encoding

Each partition is encoded as:

1. **Header**: `[LCOM] part_id num_nodes global_idx_0 ... global_idx_{n-1} [SEP]`
2. **Content**: Either nested hierarchy (recursive) or SENT-style walk (leaf)
3. **Footer**: `[RCOM]`

For leaf partitions, the content uses SENT-style back-edge encoding:

$$\text{walk} = v_0, v_1, [\text{LADJ}, b_1, \ldots, \text{RADJ}], v_2, \ldots$$

where $b_i$ are indices of previously visited nodes connected to the current node.

### Bipartite Encoding

Inter-community edges are encoded as:

```
[LBIP] left_part_id right_part_id num_edges
    left_local_0 right_local_0
    left_local_1 right_local_1
    ...
[RBIP]
```

where `left_local` and `right_local` are **local indices** within their respective partitions.

### Example: Two Connected Triangles

Consider a graph with nodes $\{0,1,2,3,4,5\}$ forming two triangles $(0,1,2)$ and $(3,4,5)$ connected by edge $(2,3)$:

```
Graph Structure:
    0---1          3---4
     \ /    2-3     \ /
      2-----------3  5
```

**Hierarchical Decomposition**:
- Partition 0: nodes $\{0,1,2\}$ with edges $\{(0,1),(1,2),(2,0)\}$
- Partition 1: nodes $\{3,4,5\}$ with edges $\{(3,4),(4,5),(5,3)\}$
- Bipartite: edge $(2,3)$ mapping to $(2_{\text{local}}, 0_{\text{local}})$

**Token Sequence**:
```
[SOS]
13                                    # 2 communities (2 + 11 = 13)
[LCOM] 11 13 11 12 13 [SEP]           # Part 0: 2 nodes [0,1,2]
    11 12 14 [LADJ] 12 [RADJ]         # SENT walk with back-edge
[RCOM]
[LCOM] 12 13 14 15 16 [SEP]           # Part 1: 2 nodes [3,4,5]
    14 15 17 [LADJ] 15 [RADJ]         # SENT walk with back-edge
[RCOM]
[LBIP] 11 12 11 15 16 17 [RBIP]       # Bipartite: 1 edge
[EOS]
```

## Decoding Algorithm

### Token Parsing

The decoding process reconstructs the hierarchical structure from tokens:

```
parse_tokens(tokens):
    idx = 0
    skip SOS

    num_communities = tokens[idx++] - IDX_OFFSET

    partitions = []
    while tokens[idx] == LCOM:
        part, idx = parse_partition(tokens, idx)
        partitions.append(part)

    bipartites = []
    while tokens[idx] == LBIP:
        bipart, idx = parse_bipartite(tokens, idx)
        bipartites.append(bipart)

    return HierarchicalGraph(partitions, bipartites)
```

### Partition Parsing

```
parse_partition(tokens, idx):
    assert tokens[idx++] == LCOM
    part_id = tokens[idx++] - IDX_OFFSET
    num_nodes = tokens[idx++] - IDX_OFFSET

    global_indices = []
    for _ in range(num_nodes):
        global_indices.append(tokens[idx++] - IDX_OFFSET)

    assert tokens[idx++] == SEP

    # Parse SENT-style walk with back-edges
    edges = parse_sent_walk(tokens, idx until RCOM)

    return Partition(part_id, global_indices, edges)
```

### Graph Reconstruction

The original graph is reconstructed by combining all edges:

$$E_{\text{reconstructed}} = \bigcup_{i} E_i \cup \bigcup_{i<j} E_{ij}$$

For each partition edge $(u_{\text{local}}, v_{\text{local}}) \in E_i$:
$$u_{\text{global}} = \text{global\_indices}_i[u_{\text{local}}]$$
$$v_{\text{global}} = \text{global\_indices}_i[v_{\text{local}}]$$

For each bipartite edge $(u_{\text{local}}, v_{\text{local}}) \in E_{ij}$:
$$u_{\text{global}} = \text{global\_indices}_i[u_{\text{local}}]$$
$$v_{\text{global}} = \text{global\_indices}_j[v_{\text{local}}]$$

## Motif Preservation

### Why H-SENT Preserves Motifs

Motifs are preserved through three mechanisms:

1. **Spectral Clustering Maximizes Modularity**: Modularity measures the density of intra-community edges versus inter-community edges. Since motifs are dense subgraphs, they naturally cluster into the same community.

2. **Complete Edge Encoding**: Within each partition, SENT-style back-edge encoding captures ALL internal edges, not just a spanning tree. Every edge in the original motif is preserved.

3. **Explicit Bipartite Encoding**: Inter-community edges are explicitly enumerated, ensuring no connection is lost.

### Motif Types and Their Encoding

| Motif Type | H-SENT Encoding |
|------------|-----------------|
| Triangle (3-clique) | Single partition with 3 back-edges |
| Benzene ring | Single partition with 6 nodes, cyclic back-edges |
| Functional group | Single partition (dense internal connectivity) |
| Fused rings | May span partitions if large; bipartite captures bridges |
| Bridge edges | Bipartite encoding between adjacent communities |

### Lossless Guarantee

The encoding is provably lossless:

$$\forall G: \quad \text{decode}(\text{encode}(G)) \equiv G$$

This is verified by roundtrip tests on all graph types including:
- Synthetic graphs (triangles, paths, stars, complete graphs)
- Molecular graphs (aspirin, caffeine, cholesterol, morphine)

## Node Ordering Strategies

The order of nodes within partitions affects the token sequence. Four strategies are supported:

### BFS (Breadth-First Search)

Starting from the highest-degree node, traverse level by level:

```
BFS(G):
    start = argmax_v degree(v)
    queue = [start]
    visited = {start}
    order = [start]

    while queue:
        v = queue.pop(0)
        for u in neighbors(v):
            if u not in visited:
                visited.add(u)
                queue.append(u)
                order.append(u)

    return order
```

### DFS (Depth-First Search)

Starting from the highest-degree node, explore deeply before backtracking:

```
DFS(G):
    start = argmax_v degree(v)
    stack = [start]
    visited = {start}
    order = [start]

    while stack:
        v = stack.pop()
        for u in neighbors(v):
            if u not in visited:
                visited.add(u)
                stack.append(u)
                order.append(u)

    return order
```

### BFSAC (BFS with Ascending Cutset)

Modified BFS that prefers nodes with **fewer** edges to already-visited nodes:

$$\text{cutset}(v) = |\{u \in \text{visited} : (v, u) \in E\}|$$

At each step, select the unvisited neighbor with minimum cutset weight.

### BFSDC (BFS with Descending Cutset)

Modified BFS that prefers nodes with **more** edges to already-visited nodes.

## Data Structures

### Partition

Represents an induced subgraph within a community:

```python
@dataclass
class Partition:
    part_id: int                              # Unique ID
    global_node_indices: list[int]            # Local -> global mapping
    edge_index: Tensor                        # [2, num_edges] in LOCAL indices
    child_hierarchy: Optional[HierarchicalGraph]  # For recursion

    def local_to_global(self, idx: int) -> int
    def global_to_local(self, idx: int) -> int
```

### Bipartite

Represents edges between two communities:

```python
@dataclass
class Bipartite:
    left_part_id: int
    right_part_id: int
    edge_index: Tensor  # [2, num_edges] in LOCAL indices
```

### HierarchicalGraph

Container for the complete hierarchical decomposition:

```python
@dataclass
class HierarchicalGraph:
    partitions: list[Partition]
    bipartites: list[Bipartite]
    community_assignment: list[int]  # node -> community

    def reconstruct(self) -> Data:
        """Reconstruct PyG Data from hierarchy."""
```

## Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| Spectral clustering | $O(n^3)$ via SVD |
| Partition extraction | $O(m)$ |
| Tokenization | $O(n + m)$ |
| Parsing | $O(T)$ where $T$ = sequence length |
| Reconstruction | $O(n + m)$ |

**Sequence Length**: For a graph with $n$ nodes and $m$ edges:
- Best case: $O(n)$ (star graph, single partition)
- Typical case: $O(n + m)$ (balanced communities)
- Worst case: $O(n + m + kc)$ where $k$ = communities, $c$ = cross-edges

## Configuration

### HSENTTokenizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_order` | str | `"BFS"` | Ordering strategy: BFS, DFS, BFSAC, BFSDC |
| `min_community_size` | int | `2` | Minimum nodes before stopping recursion |
| `seed` | int | `None` | Random seed for reproducibility |

### Usage Example

```python
from src.tokenizers.hierarchical import HSENTTokenizer
from torch_geometric.data import Data
import torch

# Create a graph
edges = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (2,3)]
edge_list = [(s,d) for s,d in edges] + [(d,s) for s,d in edges]
edge_index = torch.tensor(edge_list, dtype=torch.long).t()
graph = Data(edge_index=edge_index, num_nodes=6)

# Initialize tokenizer
tokenizer = HSENTTokenizer(
    node_order="BFS",
    min_community_size=2,
    seed=42
)
tokenizer.set_num_nodes(100)

# Tokenize
tokens = tokenizer.tokenize(graph)

# Decode (roundtrip)
reconstructed = tokenizer.decode(tokens)

# Verify
assert reconstructed.num_nodes == graph.num_nodes
```

## References

1. **HiGen**: Hierarchical Graph Generative Networks - [arXiv:2305.19843](https://arxiv.org/abs/2305.19843)
2. **AutoGraph**: SENT tokenization scheme - [arXiv:2306.10310](https://arxiv.org/abs/2306.10310)
3. **Spectral Clustering**: Ng, Jordan, Weiss (2001) - On Spectral Clustering
4. **Modularity**: Newman (2006) - Modularity and community structure in networks

## See Also

- [Visualization Guide](vis_htokenization.md) - How to visualize H-SENT decompositions
- [Architecture Overview](architecture.md) - Overall codebase structure
