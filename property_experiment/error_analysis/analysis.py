"""Error analysis for valence violations in generated molecules.

Builds molecules without sanitization and classifies where valence
violations occur (ring interior, ring boundary, chain boundary, chain interior).

Validity is determined by the standard RDKit sanitization check (graph_to_smiles).
Violation analysis is only performed on invalid molecules.
"""

from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.data.molecular import (
    ATOM_TYPES,
    BOND_TYPES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    graph_to_smiles,
)


def build_mol_no_sanitize(data: Data) -> Optional[Chem.RWMol]:
    """Build an RDKit RWMol from a PyG Data object WITHOUT sanitization.

    Reuses atom/bond construction logic from src/data/molecular.py:graph_to_smiles
    but skips Chem.SanitizeMol() so we can inspect raw valence states.

    Args:
        data: PyG Data object with node and edge features.

    Returns:
        RWMol object, or None if construction fails.
    """
    try:
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return None

        edge_index = data.edge_index
        mol = Chem.RWMol()

        # Detect if using integer labels or one-hot features
        labeled = data.x.dtype == torch.long or data.x.dtype == torch.int64

        # Add atoms
        for i in range(num_nodes):
            if labeled:
                atom_type_idx = int(data.x[i])
                if atom_type_idx < len(ATOM_TYPES):
                    atom_symbol = ATOM_TYPES[atom_type_idx]
                else:
                    atom_symbol = "C"
            else:
                node_feat = data.x[i].numpy()
                atom_type_idx = int(np.argmax(node_feat[:NUM_ATOM_TYPES]))
                if atom_type_idx < len(ATOM_TYPES):
                    atom_symbol = ATOM_TYPES[atom_type_idx]
                else:
                    atom_symbol = "C"

            atom = Chem.Atom(atom_symbol)

            if not labeled and len(data.x[i]) > NUM_ATOM_TYPES + 1:
                formal_charge = int(data.x[i][NUM_ATOM_TYPES + 1])
                atom.SetFormalCharge(formal_charge)

            mol.AddAtom(atom)

        # Add bonds (only process each edge once)
        added_bonds = set()
        if edge_index is not None and edge_index.numel() > 0:
            for k in range(edge_index.size(1)):
                i = int(edge_index[0, k])
                j = int(edge_index[1, k])
                if i >= num_nodes or j >= num_nodes:
                    continue
                if i < j and (i, j) not in added_bonds:
                    added_bonds.add((i, j))

                    if data.edge_attr is not None and data.edge_attr.size(0) > k:
                        if labeled:
                            bond_type_idx = int(data.edge_attr[k])
                        else:
                            edge_feat = data.edge_attr[k].numpy()
                            bond_type_idx = int(np.argmax(edge_feat[:NUM_BOND_TYPES]))

                        if bond_type_idx < len(BOND_TYPES):
                            bond_type = BOND_TYPES[bond_type_idx]
                        else:
                            bond_type = Chem.rdchem.BondType.SINGLE
                    else:
                        bond_type = Chem.rdchem.BondType.SINGLE

                    mol.AddBond(i, j, bond_type)

        return mol

    except Exception:
        return None


def find_valence_violations(mol: Chem.RWMol) -> list[dict]:
    """Find atoms with valence violations in an unsanitized molecule.

    Uses RDKit's partial sanitization (SANITIZE_PROPERTIES) to compute
    valence correctly (handles aromatic kekulization, implicit Hs, etc.),
    then checks each atom for violations.

    Args:
        mol: Unsanitized RWMol object.

    Returns:
        List of dicts with keys: atom_idx, atom_type, actual_valence, allowed_valence.
    """
    pt = Chem.GetPeriodicTable()

    # Run partial sanitization to compute valence properties without
    # raising on violations. This handles aromatic kekulization properly.
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )
    except Exception:
        pass

    # Now compute explicit valence (won't raise after partial sanitize)
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    violations = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()

        try:
            actual_valence = atom.GetTotalValence()
        except Exception:
            # Fallback: count bond orders manually
            actual_valence = _compute_explicit_valence(mol, idx)

        # GetDefaultValence returns a tuple of allowed valences
        default_valence = pt.GetDefaultValence(atomic_num)
        if isinstance(default_valence, tuple):
            max_allowed = max(default_valence)
        else:
            max_allowed = default_valence

        if actual_valence > max_allowed:
            violations.append(
                {
                    "atom_idx": idx,
                    "atom_type": atom.GetSymbol(),
                    "actual_valence": actual_valence,
                    "allowed_valence": max_allowed,
                }
            )

    return violations


def _compute_explicit_valence(mol: Chem.RWMol, atom_idx: int) -> int:
    """Fallback: compute explicit valence from bonds manually.

    Only used when UpdatePropertyCache + GetTotalValence fails.
    """
    bond_order_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 1,  # conservative: treat as single
    }
    atom = mol.GetAtomWithIdx(atom_idx)
    valence = 0
    for bond in atom.GetBonds():
        valence += bond_order_map.get(bond.GetBondType(), 1)
    valence += atom.GetNumExplicitHs()
    return valence


def classify_atom_role(mol: Chem.RWMol, atom_idx: int) -> str:
    """Classify an atom's structural role based on ring membership.

    Categories:
        ring_interior:   atom in ring, ALL neighbors also in some ring
        ring_boundary:   atom in ring, at least one neighbor NOT in any ring
        chain_boundary:  atom NOT in ring, at least one neighbor IN a ring
        chain_interior:  atom NOT in ring, no neighbors in rings

    Args:
        mol: RWMol object (must have ring info computed via FastFindRings).
        atom_idx: Index of the atom to classify.

    Returns:
        One of the four role strings.
    """
    ring_info = mol.GetRingInfo()
    atom_in_ring = ring_info.NumAtomRings(atom_idx) > 0

    atom = mol.GetAtomWithIdx(atom_idx)
    neighbor_indices = [n.GetIdx() for n in atom.GetNeighbors()]

    if atom_in_ring:
        # Check if all neighbors are also in rings
        all_neighbors_in_ring = all(
            ring_info.NumAtomRings(n) > 0 for n in neighbor_indices
        )
        if all_neighbors_in_ring and len(neighbor_indices) > 0:
            return "ring_interior"
        else:
            return "ring_boundary"
    else:
        # Check if any neighbor is in a ring
        any_neighbor_in_ring = any(
            ring_info.NumAtomRings(n) > 0 for n in neighbor_indices
        )
        if any_neighbor_in_ring:
            return "chain_boundary"
        else:
            return "chain_interior"


def analyze_molecule(data: Data) -> dict:
    """Analyze a single generated molecule for valence violations.

    Uses graph_to_smiles() (standard RDKit sanitization) for validity.
    Only performs violation analysis on invalid molecules.

    Args:
        data: PyG Data object representing a generated molecule.

    Returns:
        Dict with keys: valid, num_atoms, decode_failure, violations.
    """
    # Standard validity check via RDKit sanitization
    smiles = graph_to_smiles(data)
    is_valid = smiles is not None

    mol = build_mol_no_sanitize(data)

    if mol is None:
        return {
            "valid": False,
            "num_atoms": 0,
            "decode_failure": True,
            "violations": [],
        }

    num_atoms = mol.GetNumAtoms()

    # If valid, no need to analyze violations
    if is_valid:
        return {
            "valid": True,
            "num_atoms": num_atoms,
            "decode_failure": False,
            "violations": [],
        }

    # Invalid molecule — find and classify violations
    Chem.FastFindRings(mol)

    violations_raw = find_valence_violations(mol)

    violations = []
    for v in violations_raw:
        role = classify_atom_role(mol, v["atom_idx"])
        violations.append(
            {
                "atom_idx": v["atom_idx"],
                "role": role,
                "atom_type": v["atom_type"],
                "actual_valence": v["actual_valence"],
                "allowed_valence": v["allowed_valence"],
            }
        )

    return {
        "valid": False,
        "num_atoms": num_atoms,
        "decode_failure": False,
        "violations": violations,
    }


def analyze_batch(graphs: list[Data]) -> dict:
    """Analyze a batch of generated molecules.

    Args:
        graphs: List of PyG Data objects.

    Returns:
        Summary dict with aggregated statistics ready for plotting.
    """
    results = []
    for g in graphs:
        results.append(analyze_molecule(g))

    total = len(results)
    num_valid = sum(1 for r in results if r["valid"])
    num_invalid = sum(1 for r in results if not r["valid"] and not r["decode_failure"])
    num_decode_failures = sum(1 for r in results if r["decode_failure"])

    # Aggregate violation counts per role (only from invalid molecules)
    role_counts = {
        "ring_interior": 0,
        "ring_boundary": 0,
        "chain_boundary": 0,
        "chain_interior": 0,
    }

    # Per atom-type breakdown
    atom_type_counts = {}

    # Count invalid molecules with vs without valence violations
    num_invalid_with_valence = 0
    num_invalid_other = 0

    total_violations = 0
    for r in results:
        if not r["valid"] and not r["decode_failure"]:
            if len(r["violations"]) > 0:
                num_invalid_with_valence += 1
            else:
                num_invalid_other += 1

        for v in r["violations"]:
            role_counts[v["role"]] += 1
            total_violations += 1

            at = v["atom_type"]
            if at not in atom_type_counts:
                atom_type_counts[at] = {
                    "ring_interior": 0,
                    "ring_boundary": 0,
                    "chain_boundary": 0,
                    "chain_interior": 0,
                }
            atom_type_counts[at][v["role"]] += 1

    # Boundary ratio
    boundary_violations = role_counts["ring_boundary"] + role_counts["chain_boundary"]
    boundary_ratio = (
        boundary_violations / total_violations if total_violations > 0 else 0.0
    )

    return {
        "total": total,
        "num_valid": num_valid,
        "num_invalid": num_invalid,
        "num_invalid_with_valence": num_invalid_with_valence,
        "num_invalid_other": num_invalid_other,
        "num_decode_failures": num_decode_failures,
        "validity_rate": num_valid / total if total > 0 else 0.0,
        "total_violations": total_violations,
        "role_counts": role_counts,
        "boundary_ratio": boundary_ratio,
        "atom_type_counts": atom_type_counts,
        "per_molecule": results,
    }
