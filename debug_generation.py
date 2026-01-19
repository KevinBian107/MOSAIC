#!/usr/bin/env python
"""Debug script to inspect model generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import SENTTokenizer
from src.data.molecular import graph_to_smiles, NUM_ATOM_TYPES, NUM_BOND_TYPES

# Load checkpoint
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/2026-01-16/00-52-54/last.ckpt"

print(f"Loading checkpoint: {checkpoint_path}")
print()

# Create tokenizer
tokenizer = SENTTokenizer(
    max_length=512,
    truncation_length=256,
    undirected=True,  # Adjust if needed
    seed=42
)

# Extract vocab size from checkpoint and configure tokenizer
print("Extracting vocab size from checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if "state_dict" in checkpoint:
    wte_key = "model.model.transformer.wte.weight"
    if wte_key in checkpoint["state_dict"]:
        checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
        print(f"Checkpoint vocab size: {checkpoint_vocab_size}")

        # Detect if labeled or unlabeled
        checkpoint_max_num_nodes_labeled = checkpoint_vocab_size - tokenizer.idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES

        if checkpoint_max_num_nodes_labeled > 0 and checkpoint_max_num_nodes_labeled <= 100:
            # Labeled graph model
            print("Detected labeled SENT checkpoint")
            tokenizer.labeled_graph = True
            tokenizer.set_num_nodes(checkpoint_max_num_nodes_labeled)
            tokenizer.set_num_node_and_edge_types(
                num_node_types=NUM_ATOM_TYPES,
                num_edge_types=NUM_BOND_TYPES,
            )
            print(f"Set tokenizer: max_num_nodes={checkpoint_max_num_nodes_labeled}, labeled_graph=True")
        else:
            # Unlabeled model
            checkpoint_max_num_nodes = checkpoint_vocab_size - tokenizer.idx_offset
            print(f"Detected unlabeled SENT checkpoint")
            print(f"Setting tokenizer max_num_nodes to {checkpoint_max_num_nodes}")
            tokenizer.set_num_nodes(checkpoint_max_num_nodes)

print()

# Load model
model = GraphGeneratorModule.load_from_checkpoint(
    checkpoint_path,
    tokenizer=tokenizer
)
model.eval()

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer special tokens: SOS={tokenizer.sos}, EOS={tokenizer.eos}, PAD={tokenizer.pad}")
print(f"Tokenizer labeled_graph: {tokenizer.labeled_graph}")
print(f"Tokenizer max_num_nodes: {tokenizer.max_num_nodes}")
if tokenizer.labeled_graph:
    print(f"Tokenizer num_node_types: {tokenizer.num_node_types}")
    print(f"Tokenizer num_edge_types: {tokenizer.num_edge_types}")
print()

# Generate one sample
print("Generating 1 sample...")
print("=" * 60)

# First generate raw tokens to inspect
print("\n[DEBUG] Generating raw tokens...")
init_ids = torch.tensor([[tokenizer.sos]], dtype=torch.long, device=model.device)
generated_tokens = model.model.model.generate(
    init_ids,
    do_sample=True,
    top_k=10,
    temperature=1.0,
    max_length=256,
)
print(f"Generated token shape: {generated_tokens.shape}")
print(f"Generated tokens (first 50): {generated_tokens[0][:50].tolist()}")
print(f"Generated tokens (full): {generated_tokens[0].tolist()}")

# Now generate graphs using the model's generate function
print("\n[DEBUG] Generating graphs via model.generate()...")
graphs, gen_time = model.generate(num_samples=1)

print(f"Number of graphs generated: {len(graphs)}")
print()

if len(graphs) > 0:
    g = graphs[0]
    print(f"[GRAPH INFO]")
    print(f"  Type: {type(g)}")
    print(f"  Attributes: {g.keys if hasattr(g, 'keys') else 'N/A'}")
    print(f"  Number of nodes: {g.num_nodes if hasattr(g, 'num_nodes') else 'N/A'}")
    print(f"  Number of edges: {g.edge_index.size(1) if hasattr(g, 'edge_index') else 'N/A'}")

    if hasattr(g, 'x'):
        print(f"  Node features shape: {g.x.shape}")
        print(f"  Node features dtype: {g.x.dtype}")
        print(f"  Node features (all): {g.x}")

    if hasattr(g, 'edge_index'):
        print(f"  Edge index shape: {g.edge_index.shape}")
        print(f"  Edge index (first 10 edges): {g.edge_index[:, :10] if g.edge_index.size(1) > 0 else 'No edges'}")

    if hasattr(g, 'edge_attr') and g.edge_attr is not None:
        print(f"  Edge attr shape: {g.edge_attr.shape}")
        print(f"  Edge attr (first 10): {g.edge_attr[:10]}")

    print()

    # Try to convert to SMILES
    print("[SMILES CONVERSION]")
    try:
        smiles = graph_to_smiles(g)
        print(f"  Success: {smiles is not None}")
        print(f"  SMILES type: {type(smiles)}")
        print(f"  SMILES value: '{smiles}'")
        print(f"  SMILES length: {len(smiles) if smiles else 0}")
    except Exception as e:
        print(f"  ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
else:
    print("ERROR: No graphs generated!")
