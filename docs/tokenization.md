# Graph Tokenization

This document describes the tokenization schemes available in MOSAIC for converting hierarchical graphs into sequential token representations suitable for autoregressive transformer models.

## Overview

MOSAIC provides two hierarchical tokenization schemes:

| Tokenizer | Description | Use Case |
|-----------|-------------|----------|
| **H-SENT** | Explicit hierarchy with partition/bipartite blocks | Debugging, interpretability |
| **HDT** | Implicit hierarchy via DFS nesting (~45% fewer tokens) | Production, efficiency |

Both schemes share:
- **SENT-style encoding**: Back-edge markers for complete edge capture
- **Spectral coarsening**: Modularity-optimized community detection
- **Roundtrip guarantee**: Lossless encode/decode cycle

---

# H-SENT: Hierarchical SENT

H-SENT combines HiGen's hierarchical decomposition with SENT-style sequential encoding. It uses explicit partition and bipartite blocks to encode hierarchy structure.

## H-SENT Token Vocabulary

The vocabulary consists of 11 special tokens followed by node indices:

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

**Total vocabulary size**: $11 + n_{\max}$ where $n_{\max}$ is the maximum number of nodes.

## H-SENT Sequence Grammar

```
sequence    := [SOS] num_comm partition* bipartite* [EOS]
num_comm    := <community_count + IDX_OFFSET>
partition   := [LCOM] part_id num_nodes global_idx* [SEP] content [RCOM]
bipartite   := [LBIP] left_id right_id num_edges edge_pair* [RBIP]
edge_pair   := left_local right_local
content     := nested_hierarchy | sent_walk
```

### Partition Block

Each partition is encoded as:

```
[LCOM] <part_id> <num_nodes> <global_idx_0> ... <global_idx_{n-1}> [SEP] <sent_walk> [RCOM]
        ↓          ↓              ↓
      which      how many     mapping from local → global indices
    partition    nodes        (local 0 = global_idx_0, etc.)
```

**Components:**
1. **Header**: `[LCOM] part_id num_nodes global_idx_0 ... global_idx_{n-1} [SEP]`
2. **Content**: Either nested hierarchy (recursive) or SENT-style walk (leaf)
3. **Footer**: `[RCOM]`

### SENT Walk Encoding

For leaf partitions, the content uses SENT-style back-edge encoding:

```
<node_0> <node_1> [LADJ] <back_edge_targets...> [RADJ] <node_2> ...
   ↓        ↓               ↓
 start   walk to      edges to previously visited nodes
         next node    (not covered by the walk)
```

The walk visits nodes in a canonical order (see [Node Ordering](#node-ordering-strategies)). After each node, back-edges to previously visited neighbors are enclosed in `[LADJ]...[RADJ]` brackets.

**Important**: Only explicit back-edges capture graph edges. Sequential adjacency in the walk does NOT imply an edge, since ordering methods (BFS/DFS) may visit non-adjacent nodes sequentially.

### Bipartite Block

Inter-community edges are encoded as:

```
[LBIP] <left_part_id> <right_part_id> <num_edges> <left_0> <right_0> ... [RBIP]
            ↓               ↓             ↓          ↓        ↓
       partition i     partition j    edge count   local    local
                                                   idx in   idx in
                                                   part i   part j
```

All integer values are encoded as `value + IDX_OFFSET`.

## H-SENT Token Decoding

### Parsing Algorithm

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

    # Parse SENT-style walk with back-edges (NOT sequential adjacency)
    edges = parse_sent_walk(tokens, idx until RCOM)

    return Partition(part_id, global_indices, edges)
```

### Back-Edge Parsing

When parsing SENT walks, edges are only created for explicit back-edge markers:

```
parse_sent_walk(tokens, idx):
    edges = []
    visited_order = []  # Maps position → local node index

    while tokens[idx] != RCOM:
        if tokens[idx] == LADJ:
            idx++
            current_node = visited_order[-1]
            while tokens[idx] != RADJ:
                target_pos = tokens[idx++] - IDX_OFFSET
                target_node = visited_order[target_pos]
                edges.append((current_node, target_node))
            idx++  # skip RADJ
        else:
            node = tokens[idx++] - IDX_OFFSET
            visited_order.append(node)

    return edges
```

## H-SENT Configuration

### HSENTTokenizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_order` | str | `"BFS"` | Ordering strategy: BFS, DFS, BFSAC, BFSDC |
| `min_community_size` | int | `4` | Minimum nodes before stopping recursion |
| `seed` | int | `None` | Random seed for reproducibility |
| `motif_aware` | bool | `False` | Enable motif-aware coarsening |
| `motif_alpha` | float | `1.0` | Motif affinity weight |
| `motif_patterns` | dict | `None` | Custom SMARTS patterns |
| `normalize_by_motif_size` | bool | `False` | Normalize motif matrix by size |

### H-SENT Usage Example

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

---

# HDT: Hierarchical Depth Trial

HDT is a simplified DFS-based tokenization scheme that achieves **~45% token reduction** over H-SENT by encoding hierarchy structure implicitly via DFS nesting rather than explicit partition and bipartite blocks.

## Key Differences from H-SENT

| Aspect | H-SENT | HDT |
|--------|--------|-----|
| Hierarchy encoding | `[LCOM]...[RCOM]` blocks | `[ENTER]...[EXIT]` nesting |
| Cross-community edges | Explicit `[LBIP]...[RBIP]` blocks | Back-edges to visited atoms |
| Vocabulary size | IDX_OFFSET = 11 | IDX_OFFSET = 7 |
| Token count | Baseline | ~45% fewer |

## HDT Token Vocabulary

| Token ID | Symbol | Description |
|----------|--------|-------------|
| 0 | `SOS` | Start of sequence |
| 1 | `EOS` | End of sequence |
| 2 | `PAD` | Padding |
| 3 | `ENTER` | Enter super node (followed by level, id) |
| 4 | `EXIT` | Exit current super node |
| 5 | `LEDGE` | Left edge bracket |
| 6 | `REDGE` | Right edge bracket |
| 7+ | Node indices | `value + IDX_OFFSET` |

**Total vocabulary size**: $7 + n_{\max}$ (smaller than H-SENT's $11 + n_{\max}$)

## HDT Sequence Grammar

```
sequence    := [SOS] hierarchy [EOS]
hierarchy   := [ENTER] <level> <local_id> (hierarchy | atoms)* [EXIT]
atoms       := atom+
atom        := <global_idx> back_edges?
back_edges  := [LEDGE] <target_idx>+ [REDGE]
```

## Key Insight: Bipartites Become Back-Edges

In HDT, cross-community edges are **not encoded in separate bipartite blocks**. Instead, they are automatically captured as back-edges when atoms in different communities reference previously visited atoms:

```
Community A visited: atoms 0, 1, 2
Community B visited: atom 3 with edge to atom 2
                     ↓
Token sequence: ... 3 [LEDGE] 2 [REDGE] ...
                      └── back-edge captures cross-community edge!
```

## HDT Configuration

### HDTTokenizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_order` | str | `"BFS"` | Ordering strategy: BFS, DFS, BFSAC, BFSDC |
| `min_community_size` | int | `4` | Minimum nodes before stopping recursion |
| `seed` | int | `None` | Random seed for reproducibility |
| `motif_aware` | bool | `False` | Enable motif-aware coarsening |
| `motif_alpha` | float | `1.0` | Motif affinity weight |

### HDT Usage Example

```python
from src.tokenizers.hierarchical import HDTTokenizer
from torch_geometric.data import Data
import torch

# Create a graph
edges = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (2,3)]
edge_list = [(s,d) for s,d in edges] + [(d,s) for s,d in edges]
edge_index = torch.tensor(edge_list, dtype=torch.long).t()
graph = Data(edge_index=edge_index, num_nodes=6)

# Initialize tokenizer
tokenizer = HDTTokenizer(
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

### Hydra Configuration

```yaml
# configs/tokenizer/hdt.yaml
type: hdt
max_length: -1
truncation_length: 2048
node_order: BFS
min_community_size: 4
motif_aware: false
```

---

# Comparison Examples

## Two Triangles Example

Consider a graph with nodes $\{0,1,2,3,4,5\}$ forming two triangles $(0,1,2)$ and $(3,4,5)$ connected by edge $(2,3)$:

```
Graph Structure:

    0---1       4---5
     \ /         \ /
      2-----------3

  Triangle 1    Triangle 2
  (0,1,2)       (3,4,5)
         edge (2,3)
```

### Hierarchical Decomposition

- Partition 0: nodes $\{0,1,2\}$ with edges $\{(0,1),(1,2),(2,0)\}$
- Partition 1: nodes $\{3,4,5\}$ with edges $\{(3,4),(4,5),(5,3)\}$
- Bipartite: edge $(2,3)$ mapping to $(2_{\text{local}}, 0_{\text{local}})$

### H-SENT Token Sequence (~30+ tokens)

```
[SOS]
13                                    # 2 communities (2 + 11 = 13)
[LCOM] 11 14 11 12 13 [SEP]           # Part 0: id=0, 3 nodes, globals=[0,1,2]
    11 12 13 [LADJ] 11 [RADJ]         # Walk 0→1→2, back-edge 2→0
[RCOM]
[LCOM] 12 14 14 15 16 [SEP]           # Part 1: id=1, 3 nodes, globals=[3,4,5]
    14 15 16 [LADJ] 14 [RADJ]         # Walk 3→4→5, back-edge 5→3
[RCOM]
[LBIP] 11 12 12 13 11 [RBIP]          # Bipartite: parts 0&1, 1 edge, local_2→local_0
[EOS]
```

### HDT Token Sequence (~20 tokens)

```
[SOS]
  [ENTER] L0 :0              // Enter root hierarchy
    8 9 10 [LEDGE] 8 [REDGE] // Atoms 0,1,2 with back-edge 2→0
    11 [LEDGE] 10 [REDGE]    // Atom 3 with back-edge 3→2 (CROSS!)
    12 13 [LEDGE] 11 [REDGE] // Atoms 4,5 with back-edge 5→3
  [EXIT]
[EOS]
```

**Token savings**: ~33% fewer tokens for this simple example.

## Extended Example: Indole-Benzene Compound (18 atoms)

This example demonstrates HDT on a molecule with nested motifs: an indole system (benzene + pyrrole fused) linked to another benzene ring.

```
Structure:

         ┌──────── INDOLE SYSTEM ────────┐
         │                               │
         │    1───2       7              │
         │   /     \     / \             │
         │  0       3───6   8            │      LINKER        BENZENE B
         │   \     /     \ /             │        │              │
         │    5───4       N(9)           │        │      14────15
         │                 \             │        │     /        \
         │   [benzene A]  [pyrrole]      │       10────13        16
         │   (0-5)        (6-9)          │        │     \        /
         └───────────────────────────────┘        11    12────17
                                                   │
                                              [benzene B]
                                               (12-17)
```

### 4-Level Hierarchical Decomposition

```
Level 0:  ROOT (L0:0)
            │
            ├───────────────────────────────┬──────────────────┐
            │                               │                  │
Level 1:  INDOLE (L1:0)                 LINKER (L1:1)    BENZENE_B (L1:2)
          (atoms 0-9)                    (atoms 10-11)     (atoms 12-17)
            │
            ├────────────┬────────────┐
            │            │            │
Level 2:  BENZ_A (L2:0) SHARED (L2:1) PYRROLE (L2:2)
          (0,1,2,5)      (3,4)        (6,7,8,9)
```

### DFS Traversal and Token Encoding

| Step | Action | Token(s) | Edges |
|------|--------|----------|-------|
| 1 | Enter ROOT | `[ENTER] 7 7` | |
| 2 | Enter INDOLE | `[ENTER] 8 7` | |
| 3 | Enter BENZ_A | `[ENTER] 9 7` | |
| 4-7 | Visit atoms 0,1,2,5 | `7 8[E7] 9[E8] 12[E7]` | 1-0, 2-1, 5-0 |
| 8 | Exit BENZ_A | `[EXIT]` | |
| 9 | Enter SHARED | `[ENTER] 9 8` | |
| 10-11 | Visit atoms 3,4 | `10[E9,12] 11[E10,12]` | 3-2, 3-5, 4-3, 4-5 |
| 12 | Exit SHARED | `[EXIT]` | |
| 13 | Enter PYRROLE | `[ENTER] 9 9` | |
| 14-17 | Visit atoms 6,7,8,9 | `13[E10] 14[E13] 15[E14] 16[E11,13,15]` | 6-3, 7-6, 8-7, 9-4, 9-6, 9-8 |
| 18 | Exit PYRROLE | `[EXIT]` | |
| 19 | Exit INDOLE | `[EXIT]` | |
| 20 | Enter LINKER | `[ENTER] 8 8` | |
| 21-22 | Visit atoms 10,11 | `17[E16] 18[E17]` | **10-9 (CROSS!)**, 11-10 |
| 23 | Exit LINKER | `[EXIT]` | |
| 24 | Enter BENZENE_B | `[ENTER] 8 9` | |
| 25-30 | Visit atoms 12-17 | `19[E18] 20[E19] ...` | **12-11 (CROSS!)**, ... |
| 31 | Exit BENZENE_B | `[EXIT]` | |
| 32 | Exit ROOT | `[EXIT]` | |

**Cross-community edges (10-9, 12-11)** are automatically captured as back-edges!

### Token Count Comparison

| Component | HDT | H-SENT |
|-----------|-----|--------|
| SOS, EOS | 2 | 2 |
| Hierarchy tokens | 14 (`[ENTER]` + level + id) | ~20+ (partition headers) |
| Exit tokens | 7 | included in `[RCOM]` |
| Atom tokens | 18 | 18 |
| Edge encoding | ~22 | ~22 + bipartite blocks |
| **Total** | **~56 tokens** | **~100+ tokens** |

**~45% reduction** by eliminating super edges and explicit hierarchy encoding!

---

# Node Ordering Strategies

The order in which nodes are visited within a partition affects the token sequence. Four strategies are supported:

## BFS (Breadth-First Search)

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

**Properties:**
- Produces balanced orderings
- Good for general-purpose tokenization

## DFS (Depth-First Search)

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

**Properties:**
- Captures connectivity structure
- May produce longer back-edge lists

## BFSAC (BFS with Ascending Cutset)

Modified BFS that prefers nodes with **fewer** edges to already-visited nodes:

$$\text{cutset}(v) = |\{u \in \text{visited} : (v, u) \in E\}|$$

At each step, select the unvisited neighbor with minimum cutset weight.

**Properties:**
- Tends to minimize back-edges
- Produces more "linear" orderings

## BFSDC (BFS with Descending Cutset)

Modified BFS that prefers nodes with **more** edges to already-visited nodes.

**Properties:**
- Tends to maximize locality
- Groups highly connected nodes together

---

# Sequence Length Analysis

For a graph with $n$ nodes and $m$ edges:

| Case | H-SENT Complexity | HDT Complexity |
|------|-------------------|----------------|
| Best case | $O(n)$ | $O(n)$ |
| Typical case | $O(n + m + kc)$ | $O(n + m)$ |
| Worst case | $O(n + m + kc)$ | $O(n + m)$ |

Where $k$ = number of communities, $c$ = cross-community edges.

**H-SENT breakdown:**
- Header tokens: $O(k)$ for $k$ communities
- Partition tokens: $O(n)$ for global indices + $O(m_{\text{intra}})$ for back-edges
- Bipartite tokens: $O(m_{\text{inter}})$ for inter-community edges

**HDT breakdown:**
- Hierarchy tokens: $O(k)$ for ENTER/EXIT pairs
- Atom tokens: $O(n)$
- Back-edge tokens: $O(m)$ for all edges (intra + inter)

---

# Roundtrip Guarantee

Both encoding-decoding cycles are lossless:

$$\forall G: \quad \text{decode}(\text{tokenize}(G)) \equiv G$$

This is verified by extensive roundtrip tests on:
- Synthetic graphs (triangles, paths, stars, complete graphs)
- Molecular graphs (aspirin, caffeine, ibuprofen, naphthalene, etc.)

---

# When to Use Which Tokenizer

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Token efficiency is critical | **HDT** | ~45% fewer tokens |
| Need explicit hierarchy structure | H-SENT | Clear partition/bipartite blocks |
| Large graphs with many communities | **HDT** | Scales better |
| Debugging/interpretability | H-SENT | Easier to parse manually |
| Production deployment | **HDT** | Lower memory, faster training |

---

# References

1. **HiGen**: Hierarchical Graph Generative Networks - [arXiv:2305.19843](https://arxiv.org/abs/2305.19337)
2. **AutoGraph**: SENT tokenization scheme - [arXiv:2306.10310](https://arxiv.org/abs/2306.10310)
