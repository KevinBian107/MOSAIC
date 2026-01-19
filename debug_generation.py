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
print(f"Tokenizer special tokens: BOS={tokenizer.bos_idx}, EOS={tokenizer.eos_idx}, PAD={tokenizer.pad_idx}")
print()

# Generate one sample
print("Generating 1 sample...")
graphs, gen_time = model.generate(num_samples=1)

print(f"Number of graphs generated: {len(graphs)}")
print()

if len(graphs) > 0:
    g = graphs[0]
    print(f"Graph type: {type(g)}")
    print(f"Graph attributes: {g.keys if hasattr(g, 'keys') else 'N/A'}")
    print(f"Number of nodes: {g.num_nodes if hasattr(g, 'num_nodes') else 'N/A'}")
    print(f"Number of edges: {g.edge_index.size(1) if hasattr(g, 'edge_index') else 'N/A'}")
    print(f"Node features shape: {g.x.shape if hasattr(g, 'x') else 'N/A'}")
    print(f"Node features dtype: {g.x.dtype if hasattr(g, 'x') else 'N/A'}")
    print(f"First 5 node features: {g.x[:5] if hasattr(g, 'x') else 'N/A'}")
    print()

    # Try to convert to SMILES
    print("Converting to SMILES...")
    smiles = graph_to_smiles(g)
    print(f"SMILES type: {type(smiles)}")
    print(f"SMILES value: {smiles}")
    print(f"SMILES length: {len(smiles) if smiles else 0}")
else:
    print("No graphs generated!")
