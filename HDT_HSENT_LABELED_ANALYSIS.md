# HDT & H-SENT Labeled Graph Support Analysis

## Executive Summary

**Critical Finding**: HDT and H-SENT currently **DO NOT support atom/bond type encoding**. The hierarchical structures discard `data.x` (node features) and `data.edge_attr` (edge features) during tokenization.

**Impact**: Without atom/bond types, HDT and H-SENT cannot effectively generate molecules. The generated graphs would only have connectivity information, not chemical semantics.

**Confirmation of User's Assumption**: ✅ **CORRECT** - The same/similar changes made to SENT for labeled graphs should theoretically work for HDT/H-SENT, but they have **NOT been implemented yet**.

**Special Focus**: This analysis pays particular attention to the encode/decode and tokenize/detokenize pipeline to ensure correctness.

---

## 1. Current State Analysis

### SENT Labeled Graph Support (✅ Implemented)

**How it works**:

**INPUT**: `Data` object with:
- `edge_index`: Graph connectivity [2, num_edges]
- `x`: Node labels (integer atom types) [num_nodes]
- `edge_attr`: Edge labels (integer bond types) [num_edges]

**ENCODING/TOKENIZATION** (`src/tokenizers/sent.py` lines 132-148):
```python
if self.labeled_graph:
    walk, _ = sample_labeled_sent_from_graph(
        edge_index=edge_index,
        node_labels=data.x,        # ← Atom types
        edge_labels=data.edge_attr,  # ← Bond types
        node_idx_offset=self.node_idx_offset,
        edge_idx_offset=self.edge_idx_offset,
        ...
    )
```

**TOKEN SEQUENCE STRUCTURE** (labeled SENT):
```
[SOS] node0 atom0 node1 atom1 [LADJ] backref0 bond0 ... [RADJ] node2 atom2 ... [EOS]
      ^^^^^ ^^^^^  ^^^^^ ^^^^^         ^^^^^^^^ ^^^^^
      Node  Atom   Node  Atom          Target   Bond
      ID    Type   ID    Type          Ref      Type
```

**DECODING/DETOKENIZATION** (`src/tokenizers/sent.py` lines 270-280):
```python
if self.labeled_graph:
    edge_index, node_labels, edge_labels = get_graph_from_labeled_sent(
        walk_index=tokens,
        ...
    )
    return Data(
        edge_index=edge_index,
        x=node_labels,       # ← Restored atom types
        edge_attr=edge_labels,  # ← Restored bond types
        num_nodes=num_nodes
    )
```

**VOCABULARY STRUCTURE**:
```
Unlabeled SENT:
[0-5]                   : Special tokens (SOS, EOS, PAD, RESET, LADJ, RADJ)
[6, 6+max_num_nodes)   : Node indices

Labeled SENT:
[0-5]                          : Special tokens
[6, 6+max_num_nodes)          : Node indices
[6+max_num_nodes, 6+max_num_nodes+num_atom_types)    : Atom type tokens
[6+max_num_nodes+num_atom_types, vocab_size)         : Bond type tokens
```

### HDT/H-SENT Current State (❌ NOT Implemented)

**How they work now** (src/tokenizers/hierarchical/hdt.py lines 186-196):
```python
def tokenize(self, data: Data) -> Tensor:
    hg = self.coarsener.build_hierarchy(data, recursive=False)  # ← LOSES data.x, data.edge_attr
    return self.tokenize_hierarchy(hg)
```

**Current Token Sequence** (unlabeled HDT):
```
[SOS] [ENTER] level0 id0 node0 node1 [LEDGE] backref0 [REDGE] [EXIT] [EOS]
              ^^^^^^ ^^^  ^^^^^ ^^^^^         ^^^^^^^^
              Level  ID   Node  Node         Target
              Info   Info ID    ID           Ref

❌ NO ATOM TYPES OR BOND TYPES!
```

**The Problem Chain**:

1. **`build_hierarchy()` discards features**:
   ```python
   # Input
   Data(edge_index, x, edge_attr, num_nodes)

   # Output
   HierarchicalGraph(partitions, bipartites, community_assignment)
   # ❌ data.x and data.edge_attr are LOST
   ```

2. **`HierarchicalGraph` structure doesn't store features** (src/tokenizers/hierarchical/structures.py):
   ```python
   @dataclass
   class Partition:
       part_id: int
       global_node_indices: list[int]  # ← Just node IDs
       edge_index: Tensor              # ← Just connectivity [2, num_edges]
       child_hierarchy: Optional[HierarchicalGraph] = None
       # ❌ NO node_features or edge_features!
   ```

3. **`reconstruct()` returns unlabeled graph** (src/tokenizers/hierarchical/structures.py lines 231-249):
   ```python
   def reconstruct(self) -> Data:
       all_edges = self.get_all_edges_global()
       edge_index = torch.tensor(all_edges, dtype=torch.long).t()
       return Data(edge_index=edge_index, num_nodes=self.num_nodes)
       # ❌ NO x or edge_attr in returned Data!
   ```

---

## 2. Detailed Token Encoding/Decoding Analysis

### Critical Insight: Token Sequence Interleaving

The key to labeled graph support is **interleaving structure tokens with feature tokens**:

**SENT Approach** (proven working):
```
Structure:  node0    node1    [LADJ] backref0      [RADJ] node2
Features:         atom0    atom1            bond0              atom2

Combined:   node0 atom0 node1 atom1 [LADJ] backref0 bond0 [RADJ] node2 atom2
```

**HDT/H-SENT Must Use Same Pattern**:
```
Structure:  [ENTER] level id node0    node1    [LEDGE] backref0      [REDGE]
Features:                         atom0    atom1             bond0

Combined:   [ENTER] level id node0 atom0 node1 atom1 [LEDGE] backref0 bond0 [REDGE]
```

### Token Encoding Rules (CRITICAL)

**Rule 1: Node Encoding**
```
EVERY node reference MUST be followed by its atom type (if labeled_graph=True)

Unlabeled:  node_token
Labeled:    node_token atom_type_token
```

**Rule 2: Edge Encoding**
```
EVERY back-edge reference MUST be followed by its bond type (if labeled_graph=True)

Unlabeled:  [LEDGE] target0 target1 [REDGE]
Labeled:    [LEDGE] target0 bond0 target1 bond1 [REDGE]
```

**Rule 3: Special Tokens Are Unaffected**
```
[SOS], [EOS], [PAD], [ENTER], [EXIT], [LEDGE], [REDGE] remain unchanged
These are purely structural markers
```

### Token Decoding Rules (CRITICAL)

**Rule 1: Expect Atom Type After Node**
```python
if token >= IDX_OFFSET and token < node_idx_offset:
    # This is a node ID
    node_id = token - IDX_OFFSET
    idx += 1

    if self.labeled_graph:
        # MUST read next token as atom type
        atom_token = tokens[idx]
        atom_type = atom_token - node_idx_offset
        node_features[node_id] = atom_type
        idx += 1
```

**Rule 2: Expect Bond Type After Back-Edge Target**
```python
if tokens[idx] == LEDGE:
    idx += 1
    while tokens[idx] != REDGE:
        target_token = tokens[idx]
        target_id = visited_atoms[target_token - IDX_OFFSET]
        idx += 1

        if self.labeled_graph:
            # MUST read next token as bond type
            bond_token = tokens[idx]
            bond_type = bond_token - edge_idx_offset
            edge_features[(current_node, target_id)] = bond_type
            idx += 1

        edges.append((current_node, target_id))
    idx += 1  # Skip REDGE
```

---

## 3. Required Changes (DETAILED)

### Change 1: Update Data Structures to Store Features

**File**: `src/tokenizers/hierarchical/structures.py`

**Critical**: Features must be preserved through the entire hierarchy pipeline.

```python
@dataclass
class Partition:
    part_id: int
    global_node_indices: list[int]
    edge_index: Tensor
    child_hierarchy: Optional[HierarchicalGraph] = None
    # NEW: Store node features for nodes in this partition (LOCAL indices)
    node_features: Optional[Tensor] = None  # Shape: [num_nodes_in_partition]
```

**Why local indices?** Each partition has its own local indexing (0, 1, 2, ...) that maps to global indices via `global_node_indices`.

```python
@dataclass
class Bipartite:
    left_part_id: int
    right_part_id: int
    edges: list[tuple[int, int]]  # (left_local, right_local)
    # NEW: Store edge features for each edge in edges list
    edge_features: Optional[list[int]] = None  # Same length as edges
```

**Why list?** Bipartite edges are already stored as a list of tuples, so edge features naturally align as a parallel list.

```python
@dataclass
class HierarchicalGraph:
    partitions: list[Partition]
    bipartites: list[Bipartite]
    community_assignment: list[int]
    # NEW: Global node features (indexed by global node ID 0...num_nodes-1)
    node_features: Optional[Tensor] = None  # Shape: [num_nodes]
    # NEW: Global edge features as dictionary for efficient lookup
    edge_features: Optional[dict[tuple[int, int], int]] = None  # {(src, dst): bond_type}
```

**Why both global and partition-level features?**
- **Global**: Used during tokenization to look up features for any node
- **Partition-level**: Useful for recursive hierarchies, encapsulation

### Change 2: Update `build_hierarchy()` to Preserve Features

**File**: `src/tokenizers/hierarchical/coarsening.py`

**Critical**: This is where features are currently lost. Must extract and propagate them.

```python
def build_hierarchy(self, data: Data, recursive: bool = False) -> HierarchicalGraph:
    # ... existing partitioning logic (spectral clustering, etc.) ...

    # ═══════════════════════════════════════════════════════════
    # CRITICAL SECTION: Extract Features from Input Graph
    # ═══════════════════════════════════════════════════════════

    # Extract node features (atom types)
    node_features_global = None
    if hasattr(data, 'x') and data.x is not None:
        node_features_global = data.x  # Shape: [num_nodes]

    # Extract edge features (bond types) into lookup dictionary
    edge_features_global = None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_features_global = {}
        for i in range(data.edge_index.shape[1]):
            src = int(data.edge_index[0, i])
            dst = int(data.edge_index[1, i])
            bond_type = int(data.edge_attr[i])
            edge_features_global[(src, dst)] = bond_type

    # ═══════════════════════════════════════════════════════════
    # Create Partitions with Node Features
    # ═══════════════════════════════════════════════════════════

    partitions = []
    for part_id in range(num_communities):
        # Get nodes in this partition (GLOBAL indices)
        node_indices_global = [i for i, c in enumerate(community_assignment) if c == part_id]

        # Extract PARTITION node features (convert global → local indexing)
        part_node_features = None
        if node_features_global is not None:
            # node_indices_global = [5, 12, 7]  (global IDs)
            # We want features for these nodes in LOCAL order [0, 1, 2]
            part_node_features = node_features_global[node_indices_global]

        # Extract internal edges (existing code)
        # ...

        partition = Partition(
            part_id=part_id,
            global_node_indices=node_indices_global,
            edge_index=part_edge_index,
            node_features=part_node_features,  # NEW
        )
        partitions.append(partition)

    # ═══════════════════════════════════════════════════════════
    # Create Bipartites with Edge Features
    # ═══════════════════════════════════════════════════════════

    bipartites = []
    for (left_id, right_id), edge_list in bipartite_edge_map.items():
        # edge_list contains (left_local, right_local, global_src, global_dst)

        # Extract edge features for bipartite edges
        bip_edge_features = None
        if edge_features_global is not None:
            bip_edge_features = []
            for (left_local, right_local, global_src, global_dst) in edge_list:
                bond_type = edge_features_global.get((global_src, global_dst), 0)
                bip_edge_features.append(bond_type)

        bipartite = Bipartite(
            left_part_id=left_id,
            right_part_id=right_id,
            edges=[(left, right) for (left, right, _, _) in edge_list],
            edge_features=bip_edge_features,  # NEW
        )
        bipartites.append(bipartite)

    # ═══════════════════════════════════════════════════════════
    # Return Hierarchy with Features
    # ═══════════════════════════════════════════════════════════

    return HierarchicalGraph(
        partitions=partitions,
        bipartites=bipartites,
        community_assignment=community_assignment,
        node_features=node_features_global,      # NEW
        edge_features=edge_features_global,      # NEW
    )
```

### Change 3: Add `labeled_graph` Parameter to HDT/H-SENT

**File**: `src/tokenizers/hierarchical/hdt.py` and `hsent.py`

```python
class HDTTokenizer(Tokenizer):
    # Existing special tokens
    SOS: int = 0
    EOS: int = 1
    PAD: int = 2
    ENTER: int = 3
    EXIT: int = 4
    LEDGE: int = 5
    REDGE: int = 6
    IDX_OFFSET: int = 7

    def __init__(
        self,
        node_order: OrderingMethod = "BFS",
        max_length: int = -1,
        truncation_length: Optional[int] = None,
        undirected: bool = True,
        seed: Optional[int] = None,
        min_community_size: int = 4,
        # ... existing parameters ...
        labeled_graph: bool = False,  # NEW
    ) -> None:
        # ... existing initialization ...

        # NEW: Labeled graph support (matches SENT)
        self.labeled_graph = labeled_graph
        self.num_node_types = 0
        self.num_edge_types = 0
        self.node_idx_offset: Optional[int] = None
        self.edge_idx_offset: Optional[int] = None

    def set_num_node_and_edge_types(
        self, num_node_types: int, num_edge_types: int
    ) -> None:
        """Set number of node and edge types for labeled graphs.

        CRITICAL: Must be called before tokenization if labeled_graph=True.

        Token layout:
        [0-6]                                   : Special tokens
        [7, 7+max_num_nodes)                    : Node IDs
        [7+max_num_nodes, 7+max_num_nodes+num_node_types)     : Atom types
        [7+max_num_nodes+num_node_types, vocab_size)          : Bond types
        """
        if self.labeled_graph:
            if self.max_num_nodes is None:
                raise ValueError("Call set_num_nodes() first")

            self.num_node_types = num_node_types
            self.num_edge_types = num_edge_types

            # Node type tokens come after node ID tokens
            self.node_idx_offset = self.IDX_OFFSET + self.max_num_nodes

            # Edge type tokens come after node type tokens
            self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    @property
    def vocab_size(self) -> int:
        """Compute vocabulary size.

        Unlabeled: IDX_OFFSET + max_num_nodes
        Labeled:   IDX_OFFSET + max_num_nodes + num_node_types + num_edge_types
        """
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes() first")

        if self.labeled_graph:
            if self.edge_idx_offset is None:
                raise ValueError("Call set_num_node_and_edge_types() first for labeled graphs")
            return self.edge_idx_offset + self.num_edge_types
        else:
            return self.IDX_OFFSET + self.max_num_nodes
```

### Change 4: Update Tokenization to Encode Features

**File**: `src/tokenizers/hierarchical/hdt.py`

**CRITICAL SECTION**: This is where atom/bond types are interleaved with structure tokens.

```python
def _tokenize_partition_dfs(
    self,
    partition: Partition,
    level: int,
    hg: HierarchicalGraph,  # Need full hierarchy for edge features
    full_adj: dict[int, set[int]],
    visited_atoms: list[int],
    tokens: list[int],
) -> None:
    """Tokenize a single partition via DFS (LABELED VERSION).

    Token sequence pattern:
    [ENTER] level id node0 atom0 node1 atom1 [LEDGE] target0 bond0 [REDGE] ...
    """
    # ═══════════════════════════════════════════════════════════
    # Enter Partition (hierarchy markers)
    # ═══════════════════════════════════════════════════════════
    tokens.append(self.ENTER)
    tokens.append(level + self.IDX_OFFSET)
    tokens.append(partition.part_id + self.IDX_OFFSET)

    # ... Handle recursive child hierarchy if exists ...

    # ═══════════════════════════════════════════════════════════
    # Traverse Atoms in Partition (DFS/BFS order)
    # ═══════════════════════════════════════════════════════════

    # Get ordering (BFS, DFS, etc.)
    order = self._get_partition_ordering(partition, full_adj)

    visited_set = set(visited_atoms)  # For O(1) membership check

    for local_idx in order:
        # Convert local index → global index
        global_idx = partition.local_to_global(local_idx)

        # ───────────────────────────────────────────────────────
        # ENCODE NODE ID
        # ───────────────────────────────────────────────────────
        node_token = global_idx + self.IDX_OFFSET
        tokens.append(node_token)

        # ───────────────────────────────────────────────────────
        # ENCODE ATOM TYPE (if labeled)
        # ───────────────────────────────────────────────────────
        if self.labeled_graph:
            if hg.node_features is None:
                raise ValueError("labeled_graph=True but node_features is None")

            # Get atom type for this global node
            atom_type = int(hg.node_features[global_idx])
            atom_token = self.node_idx_offset + atom_type
            tokens.append(atom_token)

        # Mark as visited
        visited_atoms.append(global_idx)
        visited_set.add(global_idx)

        # ═══════════════════════════════════════════════════════════
        # Encode Back-Edges (to previously visited atoms)
        # ═══════════════════════════════════════════════════════════

        # Find neighbors that have been visited (back-edges)
        neighbors = full_adj.get(global_idx, set())
        back_edge_targets = [v for v in neighbors if v in visited_set]

        if back_edge_targets:
            tokens.append(self.LEDGE)

            for target_global in back_edge_targets:
                # ───────────────────────────────────────────────────────
                # ENCODE TARGET REFERENCE (position in visited list)
                # ───────────────────────────────────────────────────────
                target_position = visited_atoms.index(target_global)
                target_token = target_position + self.IDX_OFFSET
                tokens.append(target_token)

                # ───────────────────────────────────────────────────────
                # ENCODE BOND TYPE (if labeled)
                # ───────────────────────────────────────────────────────
                if self.labeled_graph:
                    if hg.edge_features is None:
                        raise ValueError("labeled_graph=True but edge_features is None")

                    # Lookup bond type for edge (current_node, target)
                    bond_type = hg.edge_features.get((global_idx, target_global), 0)
                    bond_token = self.edge_idx_offset + bond_type
                    tokens.append(bond_token)

            tokens.append(self.REDGE)

    # Exit partition
    tokens.append(self.EXIT)
```

**Example Token Sequence** (2-node partition with labeled graph):
```
Input Partition:
  Nodes: [5, 12] (global indices)
  Node features: [6, 7] (C, N atom types)
  Edge: (5, 12) with bond type 1 (single bond)

Output Tokens (assuming level=1, part_id=0):
  [ENTER] 8 7     # Enter level 1, partition 0 (8 = 1+7, 7 = 0+7)
  12 6            # Node 5 (12 = 5+7), Atom type C (6 = 0+node_idx_offset, assuming offset=6)
  19 7            # Node 12 (19 = 12+7), Atom type N (7 = 1+node_idx_offset)
  [LEDGE] 7 1 [REDGE]   # Back-edge to position 0 (7 = 0+7), bond type 1
  [EXIT]
```

### Change 5: Update Parsing to Decode Features

**File**: `src/tokenizers/hierarchical/hdt.py`

**CRITICAL SECTION**: This is where we reconstruct atom/bond types from tokens.

```python
def parse_tokens(self, tokens: Tensor) -> HierarchicalGraph:
    """Parse HDT tokens to HierarchicalGraph (LABELED VERSION).

    CRITICAL: Must correctly parse interleaved node/atom and target/bond tokens.
    """
    tokens_list = tokens.tolist()
    idx = 0

    # Skip SOS
    if tokens_list[idx] == self.SOS:
        idx += 1

    # ═══════════════════════════════════════════════════════════
    # Initialize Feature Storage (if labeled)
    # ═══════════════════════════════════════════════════════════
    node_features_dict = {} if self.labeled_graph else None
    edge_features_dict = {} if self.labeled_graph else None

    # Track visited atoms for back-edge resolution
    visited_atoms: list[int] = []

    # ... Parse hierarchy structure (ENTER/EXIT blocks) ...

    # ═══════════════════════════════════════════════════════════
    # Parse Atoms (within a partition)
    # ═══════════════════════════════════════════════════════════

    while idx < len(tokens_list) and tokens_list[idx] not in [self.EXIT, self.EOS]:
        token = tokens_list[idx]

        if token == self.ENTER:
            # Recursively parse child partition
            # ... (existing logic) ...

        elif token == self.EXIT:
            break

        elif token >= self.IDX_OFFSET:
            # ───────────────────────────────────────────────────────
            # DECODE NODE ID
            # ───────────────────────────────────────────────────────
            node_global = token - self.IDX_OFFSET
            idx += 1

            # ───────────────────────────────────────────────────────
            # DECODE ATOM TYPE (if labeled)
            # ───────────────────────────────────────────────────────
            if self.labeled_graph:
                if idx >= len(tokens_list):
                    raise ValueError(f"Expected atom type token after node {node_global}, but reached end of sequence")

                atom_token = tokens_list[idx]
                atom_type = atom_token - self.node_idx_offset

                # Store atom type
                node_features_dict[node_global] = atom_type
                idx += 1

            # Add to visited list
            visited_atoms.append(node_global)

            # ═══════════════════════════════════════════════════════════
            # Decode Back-Edges
            # ═══════════════════════════════════════════════════════════

            if idx < len(tokens_list) and tokens_list[idx] == self.LEDGE:
                idx += 1  # Skip LEDGE

                while idx < len(tokens_list) and tokens_list[idx] != self.REDGE:
                    # ───────────────────────────────────────────────────────
                    # DECODE TARGET REFERENCE
                    # ───────────────────────────────────────────────────────
                    target_token = tokens_list[idx]
                    target_position = target_token - self.IDX_OFFSET
                    target_global = visited_atoms[target_position]
                    idx += 1

                    # ───────────────────────────────────────────────────────
                    # DECODE BOND TYPE (if labeled)
                    # ───────────────────────────────────────────────────────
                    if self.labeled_graph:
                        if idx >= len(tokens_list):
                            raise ValueError(f"Expected bond type token after target {target_global}, but reached end of sequence")

                        bond_token = tokens_list[idx]
                        bond_type = bond_token - self.edge_idx_offset

                        # Store bond type
                        edge_features_dict[(node_global, target_global)] = bond_type
                        idx += 1

                    # Record edge
                    edges.append((node_global, target_global))

                if idx < len(tokens_list) and tokens_list[idx] == self.REDGE:
                    idx += 1  # Skip REDGE
                else:
                    raise ValueError(f"Expected REDGE at position {idx}, got {tokens_list[idx]}")

        else:
            idx += 1  # Skip other special tokens

    # ═══════════════════════════════════════════════════════════
    # Convert Feature Dictionaries to Tensors
    # ═══════════════════════════════════════════════════════════

    node_features_tensor = None
    if self.labeled_graph and node_features_dict is not None:
        # Create tensor [num_nodes] with atom types
        num_nodes = max(node_features_dict.keys()) + 1 if node_features_dict else 0
        node_features_tensor = torch.zeros(num_nodes, dtype=torch.long)
        for node_id, atom_type in node_features_dict.items():
            node_features_tensor[node_id] = atom_type

    # ═══════════════════════════════════════════════════════════
    # Create HierarchicalGraph with Features
    # ═══════════════════════════════════════════════════════════

    return HierarchicalGraph(
        partitions=partitions,
        bipartites=bipartites,
        community_assignment=community_assignment,
        node_features=node_features_tensor,  # NEW
        edge_features=edge_features_dict,    # NEW
    )
```

### Change 6: Update `reconstruct()` to Return Features

**File**: `src/tokenizers/hierarchical/structures.py`

**CRITICAL**: Must reconstruct Data object with all features.

```python
def reconstruct(self) -> Data:
    """Reconstruct the original graph with features from hierarchical representation.

    CRITICAL: Must return Data with x and edge_attr if features exist.
    """
    # ═══════════════════════════════════════════════════════════
    # Reconstruct Edge Index (existing logic)
    # ═══════════════════════════════════════════════════════════
    all_edges = self.get_all_edges_global()

    if all_edges:
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ═══════════════════════════════════════════════════════════
    # Reconstruct Edge Attributes (bond types)
    # ═══════════════════════════════════════════════════════════
    edge_attr = None
    if self.edge_features is not None and edge_index.shape[1] > 0:
        edge_attr_list = []

        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])

            # Lookup bond type from edge_features dictionary
            bond_type = self.edge_features.get((src, dst), 0)
            edge_attr_list.append(bond_type)

        edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

    # ═══════════════════════════════════════════════════════════
    # Return Data with Features
    # ═══════════════════════════════════════════════════════════
    return Data(
        edge_index=edge_index,
        num_nodes=self.num_nodes,
        x=self.node_features,       # NEW: Atom types
        edge_attr=edge_attr,         # NEW: Bond types
    )
```

---

## 4. Testing Strategy (CRITICAL)

### Phase 1: Vocabulary Size Verification

```python
# Test unlabeled
tokenizer_unlabeled = HDTTokenizer(labeled_graph=False)
tokenizer_unlabeled.set_num_nodes(50)
assert tokenizer_unlabeled.vocab_size == 7 + 50  # 57

# Test labeled
tokenizer_labeled = HDTTokenizer(labeled_graph=True)
tokenizer_labeled.set_num_nodes(50)
tokenizer_labeled.set_num_node_and_edge_types(num_node_types=9, num_edge_types=4)
expected = 7 + 50 + 9 + 4  # 70
assert tokenizer_labeled.vocab_size == expected
assert tokenizer_labeled.node_idx_offset == 57
assert tokenizer_labeled.edge_idx_offset == 66
```

### Phase 2: Token Sequence Validation

```python
# Create simple labeled graph: C-N single bond
data = Data(
    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    x=torch.tensor([6, 7], dtype=torch.long),  # C (index 0), N (index 1)
    edge_attr=torch.tensor([1, 1], dtype=torch.long),  # Single bond
    num_nodes=2,
)

tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=1)
tokenizer.set_num_nodes(10)
tokenizer.set_num_node_and_edge_types(9, 4)

tokens = tokenizer.tokenize(data)

# Verify token pattern (example expected sequence):
# [SOS] [ENTER] level id node0 atom0 node1 atom1 [LEDGE] ref0 bond0 [REDGE] [EXIT] [EOS]
# Exact values depend on hierarchy, but structure must be:
# - Every node followed by atom
# - Every back-edge followed by bond
```

### Phase 3: Roundtrip Testing (MOST CRITICAL)

```python
def test_roundtrip_labeled():
    """Test that encode→decode preserves all features."""

    # Create test graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],  # Edges
        [1, 0, 2, 1, 0, 2],
    ], dtype=torch.long)

    x = torch.tensor([6, 7, 8], dtype=torch.long)  # C, N, O
    edge_attr = torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.long)  # Single, double bonds

    original = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=3)

    # Tokenize
    tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=2)
    tokenizer.set_num_nodes(50)
    tokenizer.set_num_node_and_edge_types(9, 4)

    tokens = tokenizer.tokenize(original)

    # Decode
    reconstructed = tokenizer.decode(tokens)

    # ═══════════════════════════════════════════════════════════
    # VERIFY ROUNDTRIP CORRECTNESS
    # ═══════════════════════════════════════════════════════════

    # 1. Node features preserved
    assert torch.equal(original.x, reconstructed.x), \
        f"Node features mismatch: {original.x} != {reconstructed.x}"

    # 2. Edge connectivity preserved (may be reordered)
    assert reconstructed.num_edges == original.num_edges

    # 3. Edge attributes preserved (check each edge)
    for i in range(original.num_edges):
        src_orig = int(original.edge_index[0, i])
        dst_orig = int(original.edge_index[1, i])
        bond_orig = int(original.edge_attr[i])

        # Find matching edge in reconstructed
        found = False
        for j in range(reconstructed.num_edges):
            src_rec = int(reconstructed.edge_index[0, j])
            dst_rec = int(reconstructed.edge_index[1, j])
            if src_rec == src_orig and dst_rec == dst_orig:
                bond_rec = int(reconstructed.edge_attr[j])
                assert bond_rec == bond_orig, \
                    f"Bond type mismatch for edge ({src_orig}, {dst_orig}): {bond_orig} != {bond_rec}"
                found = True
                break

        assert found, f"Edge ({src_orig}, {dst_orig}) not found in reconstructed graph"
```

### Phase 4: Molecular Roundtrip Testing

```python
def test_molecular_roundtrip():
    """Test on real molecules to ensure chemical validity."""
    from rdkit import Chem
    from src.data.molecular import smiles_to_graph, graph_to_smiles

    test_molecules = [
        "CC",           # Ethane (simple)
        "C1CC1",        # Cyclopropane (ring)
        "c1ccccc1",     # Benzene (aromatic)
        "CC(=O)O",      # Acetic acid (double bond, heteroatom)
    ]

    tokenizer = HDTTokenizer(labeled_graph=True, min_community_size=3)
    tokenizer.set_num_nodes(100)
    tokenizer.set_num_node_and_edge_types(9, 4)

    for smiles in test_molecules:
        print(f"Testing: {smiles}")

        # Convert to graph
        data = smiles_to_graph(smiles, labeled=True)

        # Tokenize → Decode
        tokens = tokenizer.tokenize(data)
        reconstructed = tokenizer.decode(tokens)

        # Convert back to SMILES
        reconstructed_smiles = graph_to_smiles(reconstructed)

        # Verify chemical equivalence
        original_canonical = Chem.CanonSmiles(smiles)
        reconstructed_canonical = Chem.CanonSmiles(reconstructed_smiles) if reconstructed_smiles else None

        assert reconstructed_canonical == original_canonical, \
            f"SMILES mismatch: {original_canonical} != {reconstructed_canonical}"

        print(f"✓ Passed: {smiles} → {reconstructed_smiles}")
```

---

## 5. Implementation Priority & Questions

### Recommendation: Implement Labeled Support for HDT First

**Rationale**:
1. HDT is more efficient (~45% fewer tokens) - better for production
2. If HDT labeled works, H-SENT labeled is mechanically identical
3. Can validate the approach on one tokenizer before duplicating to another

### Implementation Order:

1. **Data Structures** (2-3 hours)
   - Update `Partition`, `Bipartite`, `HierarchicalGraph` structures
   - Add feature fields

2. **Feature Preservation** (2-3 hours)
   - Update `build_hierarchy()` to extract and store features
   - Test that features flow through correctly

3. **Tokenization with Features** (3-4 hours)
   - Update `_tokenize_partition_dfs()` to interleave atom/bond tokens
   - Validate token sequences manually

4. **Parsing with Features** (3-4 hours)
   - Update `parse_tokens()` to decode atom/bond tokens
   - This is the most error-prone - needs careful index tracking

5. **Reconstruction** (1-2 hours)
   - Update `reconstruct()` to return `Data(x=..., edge_attr=...)`

6. **Testing** (3-4 hours)
   - Write comprehensive roundtrip tests
   - Test on molecular graphs

**Total Estimate**: ~14-20 hours of focused work

### Critical Questions:

1. **Should I implement for both HDT and H-SENT simultaneously, or HDT first?**
   - Recommendation: HDT first to validate approach

2. **Should unlabeled mode still work after changes?**
   - Recommendation: Yes, keep backward compatibility with `labeled_graph=False`

3. **Should I create a separate branch for this work?**
   - Recommendation: Yes, `feature/labeled-hierarchical-tokenizers`

4. **Want me to start implementation now, or wait for your approval?**