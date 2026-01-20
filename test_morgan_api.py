#!/usr/bin/env python
"""Test which Morgan fingerprint API is available."""

print("Testing Morgan fingerprint API...")
print()

# Test 1: Check if new API exists
try:
    from rdkit.Chem import rdMolDescriptors
    has_new_api = hasattr(rdMolDescriptors, 'GetMorganGenerator')
    print(f"✓ New API (GetMorganGenerator) available: {has_new_api}")

    if has_new_api:
        # Test using the new API
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CCO')
        gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
        fp = gen.GetFingerprint(mol)
        print(f"✓ New API works! Fingerprint type: {type(fp)}")
except Exception as e:
    print(f"✗ New API error: {e}")
    has_new_api = False

print()

# Test 2: Check old API
try:
    from rdkit.Chem import AllChem
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CCO')
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    print(f"✓ Old API works! Fingerprint type: {type(fp)}")
except Exception as e:
    print(f"✗ Old API error: {e}")

print()

# Test 3: Test our helper function
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.evaluation.molecular_metrics import get_morgan_fingerprint, USE_NEW_API
    from rdkit import Chem

    print(f"Our code is using: {'NEW API' if USE_NEW_API else 'OLD API'}")

    mol = Chem.MolFromSmiles('CCO')
    fp = get_morgan_fingerprint(mol, radius=2, n_bits=2048)
    print(f"✓ Helper function works! Fingerprint type: {type(fp)}")
except Exception as e:
    print(f"✗ Helper function error: {e}")
    import traceback
    traceback.print_exc()
