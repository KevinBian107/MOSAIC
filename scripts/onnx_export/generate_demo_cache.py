"""Generate pre-cached molecules for the MOSAIC live demo fallback.

Generates molecules using a trained model and saves them as JSON with:
- SMILES, validity, atom count
- Community structure (type, atom indices, internal edges) [HDTC only]
- Super-edges [HDTC only]
- 2D coordinates (from RDKit)
- Token sequence (for replay animation)

Supports both HDTC and SENT tokenizer types.

Usage:
    python scripts/generate_demo_cache.py \
        --checkpoint path/to/checkpoint.ckpt \
        --output demo_cache.json \
        --num_molecules 100 \
        --tokenizer_type hdtc

    python scripts/generate_demo_cache.py \
        --checkpoint path/to/sent_checkpoint.ckpt \
        --output sent_demo_cache.json \
        --num_molecules 100 \
        --tokenizer_type sent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.molecular import ATOM_TYPES, BOND_TYPES, graph_to_smiles
from scripts.visualization.pipeline_overview import load_model


def molecule_to_demo_data(
    tokens: torch.Tensor,
    tokenizer,
    mol_idx: int,
) -> dict | None:
    """Convert generated tokens into demo-ready JSON data.

    Args:
        tokens: 1D token tensor from generation.
        tokenizer: HDTC tokenizer instance.
        mol_idx: Index for this molecule.

    Returns:
        Dict with molecule data, or None if invalid.
    """
    try:
        # Parse tokens to hierarchy
        hierarchy = tokenizer.parse_tokens(tokens)

        # Reconstruct graph
        data = hierarchy.reconstruct()
        if data.num_nodes == 0:
            return None

        # Get SMILES
        smiles = graph_to_smiles(data)
        if smiles is None:
            return None

        # Validate with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer()
        coords_2d = []
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coords_2d.append([round(pos.x, 3), round(pos.y, 3)])

        # Extract community structure
        communities = []
        for comm in hierarchy.communities:
            comm_data = {
                "id": comm.community_id,
                "type": comm.community_type,
                "atomIndices": list(comm.atom_indices),
                "internalEdges": [
                    [int(s), int(d)] for s, d in comm.internal_edges
                    if s < d  # Only one direction
                ],
            }
            if comm.node_features is not None:
                comm_data["nodeFeatures"] = comm.node_features.tolist()
            communities.append(comm_data)

        # Extract super-edges
        super_edges = []
        for se in hierarchy.super_edges:
            super_edges.append({
                "sourceCommunity": se.source_community,
                "targetCommunity": se.target_community,
                "sourceAtom": se.source_atom,
                "targetAtom": se.target_atom,
            })

        # Extract atom info
        atoms = []
        for i in range(data.num_nodes):
            atom_type_idx = int(data.x[i]) if data.x is not None else 0
            element = ATOM_TYPES[atom_type_idx] if atom_type_idx < len(ATOM_TYPES) else "C"
            atoms.append({
                "index": i,
                "element": element,
                "typeIdx": atom_type_idx,
                "x": coords_2d[i][0] if i < len(coords_2d) else 0,
                "y": coords_2d[i][1] if i < len(coords_2d) else 0,
            })

        # Extract bond info
        bonds = []
        added = set()
        if data.edge_index is not None and data.edge_index.size(1) > 0:
            for k in range(data.edge_index.size(1)):
                src = int(data.edge_index[0, k])
                dst = int(data.edge_index[1, k])
                if src < dst and (src, dst) not in added:
                    added.add((src, dst))
                    bond_type_idx = 0
                    if data.edge_attr is not None and k < data.edge_attr.size(0):
                        bond_type_idx = int(data.edge_attr[k])
                    bonds.append({
                        "src": src,
                        "dst": dst,
                        "typeIdx": bond_type_idx,
                    })

        # Token sequence (strip PAD)
        token_seq = tokens.tolist()
        # Trim after EOS
        eos_id = 1
        if eos_id in token_seq:
            eos_pos = token_seq.index(eos_id)
            token_seq = token_seq[:eos_pos + 1]

        return {
            "id": mol_idx,
            "smiles": smiles,
            "isValid": True,
            "numAtoms": data.num_nodes,
            "communities": communities,
            "superEdges": super_edges,
            "atoms": atoms,
            "bonds": bonds,
            "tokens": token_seq,
            "coords2D": coords_2d,
        }

    except Exception as e:
        print(f"  Warning: Failed to process molecule {mol_idx}: {e}")
        return None


def sent_molecule_to_demo_data(
    tokens: torch.Tensor,
    tokenizer,
    mol_idx: int,
) -> dict | None:
    """Convert SENT-generated tokens into demo-ready JSON data.

    SENT has no hierarchy/communities - just a flat graph.

    Args:
        tokens: 1D token tensor from generation.
        tokenizer: SENT tokenizer instance.
        mol_idx: Index for this molecule.

    Returns:
        Dict with molecule data, or None if invalid.
    """
    try:
        # Decode tokens to graph
        data = tokenizer.decode(tokens)
        if data.num_nodes == 0:
            return None

        # Get SMILES
        smiles = graph_to_smiles(data)
        if smiles is None:
            return None

        # Validate with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer()
        coords_2d = []
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coords_2d.append([round(pos.x, 3), round(pos.y, 3)])

        # Extract atom info
        atoms = []
        for i in range(data.num_nodes):
            atom_type_idx = int(data.x[i]) if data.x is not None else 0
            element = ATOM_TYPES[atom_type_idx] if atom_type_idx < len(ATOM_TYPES) else "C"
            atoms.append({
                "index": i,
                "element": element,
                "typeIdx": atom_type_idx,
                "x": coords_2d[i][0] if i < len(coords_2d) else 0,
                "y": coords_2d[i][1] if i < len(coords_2d) else 0,
            })

        # Extract bond info
        bonds = []
        added = set()
        if data.edge_index is not None and data.edge_index.size(1) > 0:
            for k in range(data.edge_index.size(1)):
                src = int(data.edge_index[0, k])
                dst = int(data.edge_index[1, k])
                if src < dst and (src, dst) not in added:
                    added.add((src, dst))
                    bond_type_idx = 0
                    if data.edge_attr is not None and k < data.edge_attr.size(0):
                        bond_type_idx = int(data.edge_attr[k])
                    bonds.append({
                        "src": src,
                        "dst": dst,
                        "typeIdx": bond_type_idx,
                    })

        # Token sequence (strip PAD)
        if isinstance(tokens, torch.Tensor):
            token_seq = tokens.tolist()
        else:
            token_seq = list(tokens)
        # Ensure all values are plain ints
        token_seq = [int(t) for t in token_seq]
        # Trim after EOS (SENT EOS = 4)
        eos_id = int(tokenizer.eos)
        if eos_id in token_seq:
            eos_pos = token_seq.index(eos_id)
            token_seq = token_seq[:eos_pos + 1]

        return {
            "id": mol_idx,
            "smiles": smiles,
            "isValid": True,
            "numAtoms": int(data.num_nodes),
            "communities": [],  # SENT has no communities
            "superEdges": [],   # SENT has no super-graph
            "atoms": atoms,
            "bonds": bonds,
            "tokens": token_seq,
            "coords2D": coords_2d,
        }

    except Exception as e:
        print(f"  Warning: Failed to process SENT molecule {mol_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate demo cache")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to HDTC Lightning checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_cache.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num_molecules",
        type=int,
        default=100,
        help="Target number of valid molecules to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Generation batch size",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="hdtc",
        choices=["hdtc", "sent"],
        help="Tokenizer type (hdtc or sent)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model
    print(f"Loading model (tokenizer_type={args.tokenizer_type})...")
    model, tokenizer = load_model(
        args.checkpoint,
        tokenizer_type=args.tokenizer_type,
        labeled_graph=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate molecules in batches
    print(f"Generating molecules (target: {args.num_molecules} valid)...")
    all_molecules = []
    total_generated = 0
    mol_idx = 0

    while len(all_molecules) < args.num_molecules:
        batch_size = min(args.batch_size, args.num_molecules - len(all_molecules) + 20)
        sos = torch.full((batch_size, 1), tokenizer.sos, dtype=torch.long, device=device)

        results, token_lengths = model.model.generate(
            sos,
            top_k=args.top_k,
            temperature=args.temperature,
            return_tokens=True,
        )

        total_generated += batch_size

        process_fn = (
            sent_molecule_to_demo_data
            if args.tokenizer_type == "sent"
            else molecule_to_demo_data
        )

        for tokens in results:
            demo_data = process_fn(tokens.cpu(), tokenizer, mol_idx)
            if demo_data is not None:
                all_molecules.append(demo_data)
                mol_idx += 1
                if len(all_molecules) >= args.num_molecules:
                    break

        print(f"  Generated {total_generated}, valid so far: {len(all_molecules)}")

    # Save to JSON
    output = {
        "version": 1,
        "model": f"{args.tokenizer_type}_coconut",
        "numMolecules": len(all_molecules),
        "generationParams": {
            "topK": args.top_k,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "molecules": all_molecules,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    # Print stats
    file_size = output_path.stat().st_size / 1024
    avg_atoms = sum(m["numAtoms"] for m in all_molecules) / len(all_molecules)
    avg_tokens = sum(len(m["tokens"]) for m in all_molecules) / len(all_molecules)

    print(f"\nDone! Saved {len(all_molecules)} molecules to {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Avg atoms: {avg_atoms:.1f}")
    print(f"  Avg tokens: {avg_tokens:.1f}")
    print(f"  Validity: {len(all_molecules)/total_generated*100:.1f}%")


if __name__ == "__main__":
    main()
