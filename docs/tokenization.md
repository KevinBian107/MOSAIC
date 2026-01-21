# Graph Tokenization

This document describes the three tokenization schemes available in MOSAIC for converting graphs into sequential token representations.

---

## Overview

| Tokenizer | Structure | Cross-Community Edges | Use Case |
|-----------|-----------|----------------------|----------|
| **SENT** | Flat (no hierarchy) | N/A | Baseline, simple graphs |
| **H-SENT** | Explicit hierarchy blocks | Explicit bipartite blocks | Interpretability, debugging |
| **HDT** | Implicit hierarchy (DFS nesting) | Back-edges to visited atoms | Production, efficiency (~45% fewer tokens) |

All schemes share:
- **Back-edge encoding**: Edges to previously visited nodes via bracket tokens
- **Roundtrip guarantee**: Lossless encode/decode cycle

---

## Running Example

Throughout this document, we use the **Two Triangles** graph:

```
    0---1       4---5
     \ /         \ /
      2-----------3

Triangle A: nodes {0, 1, 2}, edges {(0,1), (1,2), (2,0)}
Triangle B: nodes {3, 4, 5}, edges {(3,4), (4,5), (5,3)}
Bridge: edge (2, 3)
```

**Hierarchical decomposition** (when used):
- Partition 0: nodes {0, 1, 2} with global indices [0, 1, 2]
- Partition 1: nodes {3, 4, 5} with global indices [3, 4, 5]
- Bipartite: edge (2, 3) connecting the partitions

---

## 1. SENT (Flat Tokenization)

SENT (Sequence of Edge-indicating Neighborhoods) from AutoGraph performs a random walk traversal, encoding edges via back-edge brackets.

### Token Vocabulary

| ID | Token | Description |
|----|-------|-------------|
| 0 | SOS | Start of sequence |
| 1 | RESET | New component start |
| 2 | LADJ | Left back-edge bracket |
| 3 | RADJ | Right back-edge bracket |
| 4 | EOS | End of sequence |
| 5 | PAD | Padding |
| 6+ | Node indices | Walk position + 6 |

### Encoding Rules

1. Traverse graph via random walk, emitting each new node's **walk position** (0-indexed)
2. After visiting a node, emit back-edges to previously visited neighbors in `[LADJ ... RADJ]` brackets
3. Use `RESET` when starting a new connected component or backtracking

### Example: Two Triangles

Assuming walk order: 0 → 1 → 2 → (backtrack) → 3 → 4 → 5

```
[SOS] 6 7 [LADJ] 6 [RADJ] 8 [LADJ] 6 7 [RADJ] [RESET] 9 [LADJ] 8 [RADJ] 10 [LADJ] 9 [RADJ] 11 [LADJ] 9 10 [RADJ] [EOS]
```

**Token breakdown**:
- `6`: Node at position 0 (original node 0)
- `7`: Node at position 1 (original node 1)
- `[LADJ] 6 [RADJ]`: Edge from position 1 to position 0 (edge 1→0)
- `8`: Node at position 2 (original node 2)
- `[LADJ] 6 7 [RADJ]`: Edges from position 2 to positions 0 and 1 (edges 2→0, 2→1)
- `[RESET]`: Start new trail
- `9`: Node at position 3 (original node 3)
- `[LADJ] 8 [RADJ]`: Edge from position 3 to position 2 (bridge edge 3→2)
- And so on...

### Complexity

- **Token count**: O(n + m) where n = nodes, m = edges
- **Vocabulary size**: 6 + n_max

### Flattening

Reconstruct by:
1. Creating nodes for each walk position
2. Adding edges from sequential adjacency (previous → current)
3. Adding edges from back-edge brackets

---

## 2. H-SENT (Hierarchical SENT)

H-SENT combines HiGen's hierarchical decomposition with SENT-style encoding. Hierarchy is explicit via partition and bipartite blocks.

### Token Vocabulary

| ID | Token | Description |
|----|-------|-------------|
| 0 | SOS | Start of sequence |
| 1 | EOS | End of sequence |
| 2 | PAD | Padding |
| 3 | RESET | Component restart |
| 4 | LADJ | Left back-edge bracket |
| 5 | RADJ | Right back-edge bracket |
| 6 | LCOM | Start community block |
| 7 | RCOM | End community block |
| 8 | LBIP | Start bipartite block |
| 9 | RBIP | End bipartite block |
| 10 | SEP | Separator |
| 11+ | Indices | Value + 11 |

### Sequence Grammar

```
sequence := [SOS] <num_communities> partition* bipartite* [EOS]
partition := [LCOM] <part_id> <num_nodes> <global_indices>* [SEP] <sent_content> [RCOM]
bipartite := [LBIP] <left_id> <right_id> <num_edges> <edge_pairs>* [RBIP]
```

### Example: Two Triangles

With BFS node order within partitions:

```
[SOS] 13 [LCOM] 11 14 11 12 13 [SEP] 11 12 [LADJ] 11 [RADJ] 13 [LADJ] 11 12 [RADJ] [RCOM] [LCOM] 12 14 14 15 16 [SEP] 14 15 [LADJ] 14 [RADJ] 16 [LADJ] 14 15 [RADJ] [RCOM] [LBIP] 11 12 12 13 11 [RBIP] [EOS]
```

**Token breakdown**:

**Header**: `13` = 2 communities (2 + 11)

**Partition 0**: `[LCOM] 11 14 11 12 13 [SEP] ... [RCOM]`
- `11` = part_id 0, `14` = 3 nodes, `11 12 13` = global indices [0, 1, 2]
- Content: SENT walk using LOCAL indices within partition
  - `11` = local node 0
  - `12 [LADJ] 11 [RADJ]` = local node 1 with back-edge to walk position 0
  - `13 [LADJ] 11 12 [RADJ]` = local node 2 with back-edges to positions 0 and 1

**Partition 1**: `[LCOM] 12 14 14 15 16 [SEP] ... [RCOM]`
- `12` = part_id 1, `14` = 3 nodes, `14 15 16` = global indices [3, 4, 5]
- Content: Same SENT structure for the second triangle

**Bipartite**: `[LBIP] 11 12 12 13 11 [RBIP]`
- `11 12` = between partitions 0 and 1
- `12` = 1 edge
- `13 11` = local indices (2 in part 0, 0 in part 1) → edge (global 2, global 3)

### Complexity

- **Token count**: O(n + m + k·c) where k = communities, c = cross-community edges
- **Vocabulary size**: 11 + n_max

### Flattening

Reconstruct by:
1. Parsing partition blocks to get intra-community edges
2. Parsing bipartite blocks to get inter-community edges
3. Converting local indices to global using partition headers
4. Union of all edges

---

## 3. HDT (Hierarchical DFS Tokenization)

HDT achieves ~45% token reduction over H-SENT by encoding hierarchy implicitly via DFS traversal with ENTER/EXIT markers. Cross-community edges become back-edges to previously visited atoms.

### Token Vocabulary

| ID | Token | Description |
|----|-------|-------------|
| 0 | SOS | Start of sequence |
| 1 | EOS | End of sequence |
| 2 | PAD | Padding |
| 3 | ENTER | Enter super node (followed by level, id) |
| 4 | EXIT | Exit current super node |
| 5 | LEDGE | Left edge bracket |
| 6 | REDGE | Right edge bracket |
| 7+ | Indices | Value + 7 |

### Sequence Grammar

```
sequence := [SOS] hierarchy [EOS]
hierarchy := [ENTER] <level> <id> (hierarchy | atoms)* [EXIT]
atoms := atom+
atom := <global_idx> back_edges?
back_edges := [LEDGE] <global_targets>* [REDGE]
```

### Key Insight: Bipartites Become Back-Edges

In HDT, cross-community edges are NOT in separate blocks. They are captured as back-edges when atoms in later partitions reference previously visited atoms from earlier partitions.

### Example: Two Triangles

With BFS node order within partitions:

```
[SOS] [ENTER] 7 7 [ENTER] 8 7 7 8 [LEDGE] 7 [REDGE] 9 [LEDGE] 7 8 [REDGE] [EXIT] [ENTER] 8 8 10 [LEDGE] 9 [REDGE] 11 [LEDGE] 10 [REDGE] 12 [LEDGE] 10 11 [REDGE] [EXIT] [EXIT] [EOS]
```

**Token breakdown**:

**Root entry**: `[ENTER] 7 7` = Enter level 0, id 0

**Partition 0**: `[ENTER] 8 7 ... [EXIT]` = Enter level 1, id 0
- `7` = global node 0
- `8 [LEDGE] 7 [REDGE]` = global node 1 with back-edge to global 0
- `9 [LEDGE] 7 8 [REDGE]` = global node 2 with back-edges to globals 0 and 1

**Partition 1**: `[ENTER] 8 8 ... [EXIT]` = Enter level 1, id 1
- `10 [LEDGE] 9 [REDGE]` = global node 3 with back-edge to global 2 (**cross-partition edge!**)
- `11 [LEDGE] 10 [REDGE]` = global node 4 with back-edge to global 3
- `12 [LEDGE] 10 11 [REDGE]` = global node 5 with back-edges to globals 3 and 4

**Root exit**: `[EXIT]`

Note how the bridge edge (2, 3) is captured when visiting global 3: it has a back-edge to global 2 (which was visited in partition 0).

### Complexity

- **Token count**: O(n + m) - eliminates bipartite overhead
- **Vocabulary size**: 7 + n_max (smaller than H-SENT)

### Flattening

Reconstruct by:
1. Tracking all visited atoms in order
2. Parsing ENTER/EXIT to identify partition boundaries
3. Adding edges from back-edge brackets (includes both intra and inter-partition edges)
4. Union of all back-edges

---

## Comparison Summary

### Token Count (Two Triangles Example)

| Tokenizer | Token Count | Breakdown |
|-----------|-------------|-----------|
| SENT | ~18 | 2 (SOS/EOS) + 6 (nodes) + 10 (back-edges + reset) |
| H-SENT | ~35 | 2 (SOS/EOS) + 2 (num_comm) + 22 (partitions) + 9 (bipartite) |
| HDT | ~25 | 2 (SOS/EOS) + 8 (hierarchy markers) + 6 (nodes) + 9 (back-edges) |

### When to Use Which

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Simple graphs, baseline | **SENT** | No hierarchy overhead |
| Debugging, interpretability | **H-SENT** | Explicit structure |
| Production, large graphs | **HDT** | ~45% fewer tokens than H-SENT |
| Motif preservation | **H-SENT** or **HDT** | Hierarchical structure |

---

## Node Ordering Strategies

All tokenizers support configurable node ordering within partitions:

| Strategy | Description |
|----------|-------------|
| **BFS** | Breadth-first from highest-degree node |
| **DFS** | Depth-first from highest-degree node |
| **BFSAC** | BFS preferring nodes with fewer visited neighbors |
| **BFSDC** | BFS preferring nodes with more visited neighbors |

---

## References

1. **AutoGraph**: SENT tokenization - arXiv:2306.10310
2. **HiGen**: Hierarchical Graph Generative Networks - arXiv:2305.19337
