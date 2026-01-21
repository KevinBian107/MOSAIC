# Labeled Graph Support Implementation Progress

## Completed Changes ✅

### 1. Data Structures (structures.py)
- ✅ Added `node_features: Optional[Tensor]` to `Partition`
- ✅ Added `edge_features: Optional[Tensor]` to `Bipartite`
- ✅ Added `node_features: Optional[Tensor]` and `edge_features: Optional[dict]` to `HierarchicalGraph`
- ✅ Updated `reconstruct()` to return `Data(x=..., edge_attr=...)`

### 2. Coarsening Logic (coarsening.py)
- ✅ Updated `build_hierarchy()` to extract node/edge features from input `Data`
- ✅ Updated `build_hierarchy()` to pass features to `Partition` objects
- ✅ Updated `build_hierarchy()` to pass features to recursive calls
- ✅ Updated `_build_single_partition()` to handle features
- ✅ Updated `_extract_bipartites()` to extract and pass edge features

## Remaining Changes ⏳

### 3. HDT Tokenizer
- ⏳ Add `labeled_graph` parameter to `__init__`
- ⏳ Add `num_node_types`, `num_edge_types`, `node_idx_offset`, `edge_idx_offset` attributes
- ⏳ Add `set_num_node_and_edge_types()` method
- ⏳ Update `vocab_size` property to account for atom/bond types
- ⏳ Update tokenization to interleave atom/bond type tokens
- ⏳ Update parsing to decode atom/bond type tokens

### 4. H-SENT Tokenizer
- ⏳ Same changes as HDT (parallel implementation)

### 5. Testing
- ⏳ Create roundtrip tests with labeled graphs
- ⏳ Test on molecular data

## Status
Currently at: ~40% complete
Next: Add labeled_graph parameter to HDT/H-SENT tokenizers

## Testing Notes
Once tokenizers are updated, must test:
1. Vocabulary size calculations
2. Token sequence format (node followed by atom, target followed by bond)
3. Roundtrip reconstruction preserves all features
4. Molecular graph roundtrip (SMILES → graph → tokens → graph → SMILES)
