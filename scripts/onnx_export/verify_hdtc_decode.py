"""Verify HDTC decoding pipeline by generating molecules and rendering 2D images.

This script:
1. Loads the HDTC checkpoint (via HuggingFace GPT-2)
2. Generates molecules with short max_length for fast verification
3. Decodes tokens to graphs
4. Converts to SMILES via graph_to_smiles
5. Renders 2D molecule images using RDKit

Usage:
    python scripts/verify_hdtc_decode.py
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from collections import OrderedDict
from transformers import GPT2Config, GPT2LMHeadModel

from src.tokenizers.hdtc.tokenizer import HDTCTokenizer
from src.data.molecular import graph_to_smiles, ATOM_TYPES, NUM_ATOM_TYPES, NUM_BOND_TYPES


def load_hf_model(checkpoint_path: str):
    """Load HF GPT-2 model from Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    wte_key = "model.model.transformer.wte.weight"
    wpe_key = "model.model.transformer.wpe.weight"
    vocab_size = state_dict[wte_key].shape[0]
    n_embd = state_dict[wte_key].shape[1]
    n_positions = state_dict[wpe_key].shape[0]

    n_layer = 0
    while f"model.model.transformer.h.{n_layer}.ln_1.weight" in state_dict:
        n_layer += 1

    config = GPT2Config(
        vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
        n_head=12, n_positions=n_positions,
        bos_token_id=0, eos_token_id=1, pad_token_id=2,
    )

    prefix = "model.model."
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix):]] = value

    model = GPT2LMHeadModel(config)
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    print(f"Model: vocab={vocab_size}, embd={n_embd}, layers={n_layer}, positions={n_positions}")
    return model, vocab_size


def create_tokenizer(vocab_size: int):
    """Create HDTC tokenizer matching the checkpoint config."""
    tokenizer = HDTCTokenizer(
        node_order="BFS", max_length=-1, truncation_length=2048,
        include_rings=True, labeled_graph=True,
    )
    max_num_nodes = vocab_size - 12 - NUM_ATOM_TYPES - NUM_BOND_TYPES
    tokenizer.max_num_nodes = max_num_nodes
    tokenizer.set_num_node_and_edge_types(NUM_ATOM_TYPES, NUM_BOND_TYPES)

    print(f"Tokenizer: max_nodes={max_num_nodes}, vocab={tokenizer.vocab_size}")
    print(f"  node_idx_offset={tokenizer.node_idx_offset}, edge_idx_offset={tokenizer.edge_idx_offset}")
    return tokenizer


def generate_batch(model, num_samples=8, max_length=512, top_k=10, temperature=1.0):
    """Generate a batch of molecule token sequences on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = torch.zeros(num_samples, 1, dtype=torch.long, device=device)  # SOS=0

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            max_length=max_length,
            pad_token_id=2,
            eos_token_id=1,
        )
    elapsed = time.time() - t0
    print(f"  Generated batch of {num_samples} in {elapsed:.1f}s on {device}")

    all_tokens = []
    for i in range(num_samples):
        tokens = generated[i].cpu().tolist()
        # Trim after EOS
        if 1 in tokens:
            eos_idx = tokens.index(1)
            tokens = tokens[:eos_idx + 1]
        all_tokens.append(tokens)
    return all_tokens


def tokens_to_string(tokens, tokenizer):
    """Convert token list to readable string."""
    names = {
        0: "SOS", 1: "EOS", 2: "PAD",
        3: "CS", 4: "CE", 5: "LE", 6: "RE",
        7: "SS", 8: "SE", 9: "R", 10: "F", 11: "S",
    }
    parts = []
    for t in tokens:
        if t in names:
            parts.append(names[t])
        elif t >= tokenizer.edge_idx_offset:
            bond_types = ["SGL", "DBL", "TRP", "ARO", "UNK"]
            bt = t - tokenizer.edge_idx_offset
            parts.append(f"B{bond_types[bt] if bt < len(bond_types) else bt}")
        elif t >= tokenizer.node_idx_offset:
            at = t - tokenizer.node_idx_offset
            parts.append(f"A{ATOM_TYPES[at] if at < len(ATOM_TYPES) else at}")
        elif t >= 12:
            parts.append(f"N{t - 12}")
        else:
            parts.append(f"?{t}")
    return " ".join(parts)


def main():
    checkpoint_path = "checkpoints/coconut_hdtc_20260209-170216/best.ckpt"

    print("=" * 60)
    print("HDTC Decode Verification")
    print("=" * 60)

    # Load model
    print("\n--- Loading model ---")
    model, vocab_size = load_hf_model(checkpoint_path)
    tokenizer = create_tokenizer(vocab_size)

    # Generate molecules as a batch (fast on GPU)
    num_samples = 8

    print(f"\n--- Generating {num_samples} molecules (batch) ---")
    all_tokens = generate_batch(model, num_samples=num_samples, max_length=512, top_k=10, temperature=1.0)
    all_smiles = []

    for i, tokens in enumerate(all_tokens):
        print(f"\nMolecule {i}: {len(tokens)} tokens")

        # Print readable tokens
        readable = tokens_to_string(tokens[:60], tokenizer)
        print(f"  First 60: {readable}")

        # Count token types
        counts = {"communities": 0, "super": False, "atoms": 0, "bonds": 0, "backedges": 0}
        for t in tokens:
            if t == 3: counts["communities"] += 1
            elif t == 7: counts["super"] = True
            elif t == 5: counts["backedges"] += 1
            elif tokenizer.node_idx_offset <= t < tokenizer.edge_idx_offset: counts["atoms"] += 1
            elif t >= tokenizer.edge_idx_offset: counts["bonds"] += 1
        print(f"  Stats: communities={counts['communities']}, atoms={counts['atoms']}, bonds={counts['bonds']}, backedges={counts['backedges']}, super={counts['super']}")

        # Decode
        try:
            tensor_tokens = torch.tensor(tokens, dtype=torch.long)
            graph = tokenizer.decode(tensor_tokens)
            print(f"  Decoded: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

            if graph.x is not None:
                atom_symbols = [ATOM_TYPES[int(graph.x[j])] if int(graph.x[j]) < len(ATOM_TYPES) else "?" for j in range(min(graph.num_nodes, 15))]
                print(f"  Atoms: {atom_symbols}")

            smiles = graph_to_smiles(graph)
            print(f"  SMILES: {smiles}")
            all_smiles.append(smiles)
        except Exception as e:
            print(f"  Decode error: {e}")
            import traceback; traceback.print_exc()
            all_smiles.append(None)

    # Summary
    valid = sum(1 for s in all_smiles if s is not None)
    print(f"\n--- Summary ---")
    print(f"Generated: {len(all_smiles)}, Valid SMILES: {valid} ({100*valid/max(len(all_smiles),1):.0f}%)")

    # Render valid molecules
    mols = []
    labels = []
    for i, smi in enumerate(all_smiles):
        if smi is not None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                labels.append(f"Mol {i}: {smi[:50]}")

    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(400, 300), legends=labels)
        img.save("verify_hdtc_decode.png")
        print(f"\nSaved {len(mols)} molecules to verify_hdtc_decode.png")
    else:
        print("\nNo valid molecules to render!")

    # Save token dump
    token_dump = [{"id": i, "tokens": t, "smiles": s} for i, (t, s) in enumerate(zip(all_tokens, all_smiles))]
    with open("verify_hdtc_tokens.json", "w") as f:
        json.dump(token_dump, f, indent=2)
    print(f"Saved token dump to verify_hdtc_tokens.json")


if __name__ == "__main__":
    main()
