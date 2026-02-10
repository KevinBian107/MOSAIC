# Hierarchical Graph (H-Graph)

This document describes the hierarchical graph representation and the two core operations: **construction** (building the hierarchy) and **flattening** (reconstructing the original graph).

---

## Overview

A hierarchical graph decomposes a flat graph into a multi-level structure where nodes are grouped into **communities** (partitions). This enables:

1. **Motif preservation**: Dense subgraphs (rings, functional groups) naturally cluster together
2. **Structured generation**: Hierarchical organization enables shorter token sequences
3. **Constraint-based generation**: Super-edge weights constrain inter-community connections

---

## 1. Construction

### Data Structures

**Partition**: Induced subgraph of a single community (diagonal block in adjacency matrix).
- `part_id`: Unique identifier
- `global_node_indices`: Mapping from local to global node indices
- `edge_index`: Internal edges in local indices
- `child_hierarchy`: Optional nested hierarchy

**Bipartite**: Edges between two communities (off-diagonal block).
- `left_part_id`, `right_part_id`: Connected partition IDs
- `edge_index`: Cross-community edges in local indices

**HierarchicalGraph**: Top-level container.
- `partitions`: List of all communities
- `bipartites`: List of inter-community edge sets
- `community_assignment`: Maps each node to its partition ID

### Block Structure

Given a graph G = (V, E) and partition P = {C₁, C₂, ..., Cₖ}:

**Diagonal blocks (Partitions)**: For each community Cᵢ, the induced subgraph:

$$G[C_i] = (C_i, E_i) \quad \text{where} \quad E_i = \{(u, v) \in E : u \in C_i \land v \in C_i\}$$

**Off-diagonal blocks (Bipartites)**: For each pair (Cᵢ, Cⱼ) where i < j:

$$E_{ij} = \{(u, v) \in E : u \in C_i \land v \in C_j\}$$

---

### Coarsening Strategies

Four strategies are available for partitioning nodes into communities:

#### 1. Spectral Clustering (SC)

Uses spectral clustering with modularity optimization:

1. Build adjacency matrix **A** from edges
2. Symmetrize for undirected graphs: **A** ← (**A** + **A**ᵀ) / 2
3. Search for optimal k in range [k_min, k_max]:
   - k_min = max(2, ⌊√n × 0.7⌋)
   - k_max = min(n-1, ⌊√n × 1.3⌋)
4. Run SpectralClustering for each k, select partition with maximum modularity

**Modularity** measures partition quality:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- m = total edge weight (½ΣᵢⱼAᵢⱼ)
- kᵢ = degree of node i (Σⱼ Aᵢⱼ)
- cᵢ = community assignment of node i
- δ(cᵢ, cⱼ) = 1 if cᵢ = cⱼ, else 0

#### 2. Hierarchical Agglomerative Clustering (HAC)

Bottom-up clustering with adjacency connectivity constraint. Uses sklearn's `AgglomerativeClustering` to ensure only adjacent nodes are merged.

**Algorithm**:

1. Build adjacency matrix **A** and use rows as node features (neighborhood pattern)
2. Create connectivity constraint from **A** (only adjacent nodes can merge)
3. Search for optimal k in range [k_min, k_max] (same formula as SC)
4. For each k, run `AgglomerativeClustering(n_clusters=k, connectivity=A)` with the chosen linkage
5. Select partition with maximum modularity Q

**Linkage criteria**:

| Linkage | Criterion | Best for |
|---------|-----------|----------|
| Ward (default) | Minimize within-cluster variance | Balanced, compact clusters |
| Complete | Maximum distance between cluster members | Tight, spherical clusters |
| Average | Mean distance between cluster members | General purpose |
| Single | Minimum distance between cluster members | Elongated clusters |

**Configuration**:

```yaml
# configs/tokenizer/hsent.yaml or hdt.yaml
coarsening_strategy: hac
hac_linkage: ward           # ward, complete, average, single
hac_feature_type: adjacency # adjacency matrix rows as features
```

**Comparison with SC**:

| Aspect | Spectral Clustering | HAC |
|--------|-------------------|-----|
| Approach | Top-down (eigendecomposition) | Bottom-up (merge) |
| Connectivity | Implicit via affinity | Explicit constraint |
| Determinism | Depends on init | Deterministic |
| Speed | Slower (eigensolve) | Faster for small graphs |

#### 3. Motif-Based Clustering

Uses detected motifs directly as communities:

1. Detect motifs using SMARTS patterns (rings, functional groups)
2. Each motif instance becomes a community
3. Overlapping motifs (sharing atoms) are merged via union-find into a single community
4. Non-motif atoms are handled by the singleton rule (see below)
5. If two motif share atoms together, they will be grouped as one bigger motif, not layered motif

**Motif detection** uses SMARTS patterns:

| Motif | SMARTS | Description |
|-------|--------|-------------|
| Benzene | `c1ccccc1` | Aromatic 6-ring |
| Pyridine | `c1ccncc1` | N-containing aromatic |
| Naphthalene | `c1ccc2ccccc2c1` | Fused bicyclic |
| Cyclohexane | `C1CCCCC1` | Saturated 6-ring |

**Singleton rule** for non-motif atoms:

Atoms not belonging to any detected motif initially form singleton communities (one atom per community). These are then merged based on adjacency:

1. For each singleton atom, find its neighbors in the molecular graph
2. If a neighbor belongs to a non-singleton community (i.e., a motif), merge the singleton into that community
3. If multiple neighbors belong to different communities, merge into the first found
4. If no neighbor is in a motif, the singleton remains as its own community

This ensures non-motif atoms (e.g., chain carbons, substituents) are absorbed into adjacent motif communities, reducing fragmentation while preserving motif boundaries.

**Example**: In phenol (benzene + OH):
- Benzene ring → 1 community (6 atoms)
- Oxygen atom → initially singleton, merged into benzene community (adjacent)
- Result: 1 community containing all 7 atoms

**Example**: In biphenyl (two connected benzene rings):
- First benzene → community 1 (6 atoms)
- Second benzene → community 2 (6 atoms)
- Result: 2 separate communities (rings don't share atoms)

This approach guarantees motif preservation by design.

---

### Motif-Aware Coarsening (Hybrid)

Extends SC or HAC to preserve chemically meaningful structures by augmenting the affinity matrix:

$$A' = A + \alpha \cdot M$$

Where **M** is the **motif co-membership matrix**:

$$M_{ij} = \sum_{m \in \mathcal{M}} \mathbf{1}[i \in m \land j \in m]$$

- M is symmetric: Mᵢⱼ = Mⱼᵢ
- Mᵢⱼ counts how many motifs contain both atoms i and j
- For overlapping motifs (e.g., fused rings), Mᵢⱼ can be > 1

**α hyperparameter**:

| α Value | Effect |
|---------|--------|
| 0 | Standard clustering (no motif awareness) |
| 1 | Motif co-membership weighted equally to edges |
| 2-5 | Moderate motif preference |
| 10+ | Strong motif preservation |

**Motif cohesion metric**:

$$\text{Cohesion} = \frac{|\{m \in \mathcal{M} : m \subseteq C_i \text{ for some } i\}|}{|\mathcal{M}|}$$

A motif is "intact" if all its atoms belong to a single community. Cohesion of 1.0 means all motifs are preserved.

#### 4. Functional Group Coarsening (HDTC)

The most direct approach to motif preservation: use functional group detection as the partitioning criterion itself.

**Process**:
1. **Detect functional groups** via SMARTS patterns with priority-based overlap resolution
2. **Assign atoms to communities**:
   - Each detected group → one community
   - Remaining atoms → singleton communities
3. **Build super-graph** representing inter-community connectivity

**Key advantage**: Guarantees 100% motif cohesion by construction, since functional groups define the communities directly.

**Two-Level Hierarchy**:

Unlike SC/HAC which produce multi-level recursive hierarchies, functional coarsening produces a flat two-level structure:

| Level | Content |
|-------|---------|
| **Level 1** | Communities (rings, functional groups, singletons) |
| **Level 2** | Super-graph (community-to-community edges) |

**Functional Group Priority**:

When SMARTS patterns overlap, higher-priority patterns take precedence:

| Priority | Type | Rationale |
|----------|------|-----------|
| 30 (highest) | Ring systems | Preserve cyclic structures intact |
| 20 | Multi-atom groups | Carboxyl, ester, amide, etc. |
| 10 (lowest) | Single-atom groups | Hydroxyl, halides, amines |

**Detected Pattern Types**:

*Ring patterns* (priority 30):
- Aromatic 6-membered: benzene, pyridine, pyrimidine, pyrazine
- Aromatic 5-membered: pyrrole, furan, thiophene, imidazole
- Fused rings: naphthalene, indole, quinoline
- Saturated: cyclopropane, cyclobutane, cyclopentane, cyclohexane

*Functional group patterns* (priority 20):
- Carboxylic acid derivatives: carboxyl, ester, amide, anhydride
- Carbonyl derivatives: aldehyde, ketone
- Nitrogen-containing: nitro, nitrile, imine, azo
- Sulfur-containing: sulfonyl, sulfoxide, thioether
- Phosphorus-containing: phosphate, phosphonate

*Single-atom patterns* (priority 10):
- Halogens: fluoride, chloride, bromide, iodide
- Oxygen: hydroxyl, ether oxygen, epoxide
- Nitrogen: primary/secondary/tertiary amine
- Carbon: methyl, tert-butyl, isopropyl

**Example: Aspirin (acetylsalicylic acid)**

Detected communities:
1. Benzene ring (6 atoms) - TYPE_RING
2. Ester group (C=O-O-) - TYPE_FUNC
3. Carboxyl group (C=O-OH) - TYPE_FUNC

Super-edges connect these communities.

---

## 2. Flattening

Flattening reconstructs the original graph from the token sequence. The approach differs based on the tokenization method.

### H-SENT Flattening (Bipartite Edge Union)

H-SENT uses explicit partition and bipartite blocks, so flattening combines both:

$$E_{\text{reconstructed}} = \bigcup_{i} E_i \cup \bigcup_{i<j} E_{ij}$$

Where:
- Eᵢ = edges within partition i (from SENT back-edges in partition blocks)
- Eᵢⱼ = edges between partitions i and j (from bipartite blocks)

**Index conversion**:

For partition edges (u_local, v_local) ∈ Eᵢ:
$$u_{\text{global}} = \text{global\_indices}_i[u_{\text{local}}]$$
$$v_{\text{global}} = \text{global\_indices}_i[v_{\text{local}}]$$

For bipartite edges (u_local, v_local) ∈ Eᵢⱼ:
$$u_{\text{global}} = \text{global\_indices}_i[u_{\text{local}}]$$
$$v_{\text{global}} = \text{global\_indices}_j[v_{\text{local}}]$$

### HDT Flattening (Back-Edge Union)

HDT encodes ALL edges (both intra and inter-partition) as back-edges to previously visited atoms:

$$E_{\text{reconstructed}} = \bigcup \text{back\_edges}$$

Since cross-partition edges are captured when visiting atoms that have neighbors in earlier partitions, no separate bipartite reconstruction is needed.

**Process**:
1. Track visited atoms in DFS order
2. Parse ENTER/EXIT to identify partition boundaries
3. Collect all back-edges (these include both intra and inter-partition edges)
4. Convert global indices directly (HDT uses global indices throughout)

### Comparison

| Aspect | H-SENT | HDT |
|--------|--------|-----|
| Intra-partition edges | From SENT back-edges | From back-edges |
| Inter-partition edges | From bipartite blocks | From back-edges (automatic) |
| Index conversion | Local → Global | Already global |

### Lossless Guarantee

Both methods are provably lossless:

$$\forall G: \quad \text{flatten}(\text{tokenize}(G)) \equiv G$$

This is verified by roundtrip tests on synthetic graphs (triangles, paths, stars, complete graphs) and molecular graphs (aspirin, caffeine, cholesterol, morphine).

---

## References

1. **HiGen**: Hierarchical Graph Generative Networks - arXiv:2305.19337
2. **Spectral Clustering**: Ng, Jordan, Weiss (2001) - On Spectral Clustering
3. **Modularity**: Newman (2006) - Modularity and community structure in networks
4. **HAC**: Müllner (2011) - Modern hierarchical, agglomerative clustering algorithms
5. **HDTC**: Compositional tokenization with functional group preservation (this work)
