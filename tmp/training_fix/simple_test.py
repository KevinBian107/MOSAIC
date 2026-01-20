#!/usr/bin/env python
"""Simple test script - just generate and print SMILES."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.transformer import GraphGeneratorModule
from src.tokenizers import SENTTokenizer
from src.data.molecular import graph_to_smiles, NUM_ATOM_TYPES, NUM_BOND_TYPES

# Config
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/2026-01-16/00-52-54/last.ckpt"
num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10

print(f"Checkpoint: {checkpoint_path}")
print(f"Generating {num_samples} samples...")
print()

# Create tokenizer
tokenizer = SENTTokenizer(
    max_length=512,
    truncation_length=256,
    undirected=True,
    seed=42
)

# Extract vocab size from checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if "state_dict" in checkpoint:
    wte_key = "model.model.transformer.wte.weight"
    if wte_key in checkpoint["state_dict"]:
        checkpoint_vocab_size = checkpoint["state_dict"][wte_key].shape[0]
        checkpoint_max_num_nodes_labeled = checkpoint_vocab_size - tokenizer.idx_offset - NUM_ATOM_TYPES - NUM_BOND_TYPES

        if checkpoint_max_num_nodes_labeled > 0 and checkpoint_max_num_nodes_labeled <= 100:
            tokenizer.labeled_graph = True
            tokenizer.set_num_nodes(checkpoint_max_num_nodes_labeled)
            tokenizer.set_num_node_and_edge_types(
                num_node_types=NUM_ATOM_TYPES,
                num_edge_types=NUM_BOND_TYPES,
            )
        else:
            checkpoint_max_num_nodes = checkpoint_vocab_size - tokenizer.idx_offset
            tokenizer.set_num_nodes(checkpoint_max_num_nodes)

# Load model
model = GraphGeneratorModule.load_from_checkpoint(
    checkpoint_path,
    tokenizer=tokenizer
)
model.eval()

# Generate
print("Generating...")
graphs, gen_time = model.generate(num_samples=num_samples)
print(f"Generated {len(graphs)} graphs in {gen_time:.2f}s per sample\n")

# Convert to SMILES
generated_smiles = []
for i, g in enumerate(graphs):
    smiles = graph_to_smiles(g)
    if smiles:
        generated_smiles.append(smiles)
        print(f"{i+1:3d}. {smiles}")
    else:
        print(f"{i+1:3d}. FAILED")

print(f"\nSuccess rate: {len(generated_smiles)}/{len(graphs)} = {100*len(generated_smiles)/len(graphs):.1f}%")
