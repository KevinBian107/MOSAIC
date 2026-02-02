"""Tier-based scaffold pattern definitions.

This module defines scaffolds organized by complexity:
- Tier 1: Simple monocyclic rings (6-8 atoms)
- Tier 2: Fused bicyclic systems (9-12 atoms)
- Tier 3: Complex polycyclic systems (13+ atoms)

Each scaffold includes:
- smiles: Canonical SMILES representation
- category: Structural classification
"""

# Tier 1: Simple monocyclic scaffolds (~6-8 atoms)
TIER1_SCAFFOLDS: dict[str, dict[str, str]] = {
    # Aromatic 6-membered rings
    "benzene": {"smiles": "c1ccccc1", "category": "aromatic_6"},
    "pyridine": {"smiles": "c1ccncc1", "category": "heterocyclic_6"},
    "pyrimidine": {"smiles": "c1cncnc1", "category": "heterocyclic_6"},
    "pyrazine": {"smiles": "c1cnccn1", "category": "heterocyclic_6"},
    "pyridazine": {"smiles": "c1ccnnc1", "category": "heterocyclic_6"},
    "triazine": {"smiles": "c1ncncn1", "category": "heterocyclic_6"},
    # Aromatic 5-membered rings
    "pyrrole": {"smiles": "c1cc[nH]c1", "category": "heterocyclic_5"},
    "furan": {"smiles": "c1ccoc1", "category": "heterocyclic_5"},
    "thiophene": {"smiles": "c1ccsc1", "category": "heterocyclic_5"},
    "imidazole": {"smiles": "c1cnc[nH]1", "category": "heterocyclic_5"},
    "oxazole": {"smiles": "c1cocn1", "category": "heterocyclic_5"},
    "thiazole": {"smiles": "c1cscn1", "category": "heterocyclic_5"},
}

# Tier 2: Fused bicyclic scaffolds (~9-12 atoms)
TIER2_SCAFFOLDS: dict[str, dict[str, str]] = {
    # Fused aromatic (5+6)
    "indole": {"smiles": "c1ccc2[nH]ccc2c1", "category": "fused_hetero_56"},
    "benzofuran": {"smiles": "c1ccc2occc2c1", "category": "fused_hetero_56"},
    "benzothiophene": {"smiles": "c1ccc2sccc2c1", "category": "fused_hetero_56"},
    "benzimidazole": {"smiles": "c1ccc2[nH]cnc2c1", "category": "fused_hetero_56"},
    "benzoxazole": {"smiles": "c1ccc2ocnc2c1", "category": "fused_hetero_56"},
    "benzothiazole": {"smiles": "c1ccc2scnc2c1", "category": "fused_hetero_56"},
    # Fused aromatic (6+6)
    "naphthalene": {"smiles": "c1ccc2ccccc2c1", "category": "fused_aromatic"},
    "quinoline": {"smiles": "c1ccc2ncccc2c1", "category": "fused_hetero_66"},
    "isoquinoline": {"smiles": "c1ccc2cnccc2c1", "category": "fused_hetero_66"},
    "quinazoline": {"smiles": "c1ccc2ncncc2c1", "category": "fused_hetero_66"},
    "quinoxaline": {"smiles": "c1ccc2nccnc2c1", "category": "fused_hetero_66"},
    "cinnoline": {"smiles": "c1ccc2nnccc2c1", "category": "fused_hetero_66"},
}

# Tier 3: Complex polycyclic scaffolds (13+ atoms)
TIER3_SCAFFOLDS: dict[str, dict[str, str]] = {
    # Hetero-polycyclic
    "carbazole": {
        "smiles": "c1ccc2c(c1)[nH]c1ccccc12",
        "category": "hetero_polycyclic",
    },
    "acridine": {"smiles": "c1ccc2nc3ccccc3cc2c1", "category": "hetero_polycyclic"},
    "phenanthridine": {
        "smiles": "c1ccc2c(c1)ccc1cnccc21",
        "category": "hetero_polycyclic",
    },
    "dibenzofuran": {
        "smiles": "c1ccc2c(c1)oc1ccccc12",
        "category": "hetero_polycyclic",
    },
    "dibenzothiophene": {
        "smiles": "c1ccc2c(c1)sc1ccccc12",
        "category": "hetero_polycyclic",
    },
    "xanthene": {"smiles": "c1ccc2c(c1)Cc1ccccc1O2", "category": "hetero_polycyclic"},
    # Polycyclic aromatic hydrocarbons (PAHs)
    "phenanthrene": {
        "smiles": "c1ccc2c(c1)ccc1ccccc21",
        "category": "polycyclic_aromatic",
    },
    "anthracene": {"smiles": "c1ccc2cc3ccccc3cc2c1", "category": "polycyclic_aromatic"},
    "fluorene": {"smiles": "c1ccc2c(c1)Cc1ccccc1-2", "category": "polycyclic_aromatic"},
    "pyrene": {"smiles": "c1cc2ccc3cccc4ccc(c1)c2c34", "category": "polycyclic_large"},
    "fluoranthene": {
        "smiles": "c1ccc2c(c1)-c1cccc3cccc-2c13",
        "category": "polycyclic_large",
    },
}

# Combined dictionary for easy access
ALL_SCAFFOLDS: dict[str, dict[str, str]] = {
    **TIER1_SCAFFOLDS,
    **TIER2_SCAFFOLDS,
    **TIER3_SCAFFOLDS,
}

# Tier mapping for lookup
SCAFFOLD_TIERS: dict[str, int] = {}
for name in TIER1_SCAFFOLDS:
    SCAFFOLD_TIERS[name] = 1
for name in TIER2_SCAFFOLDS:
    SCAFFOLD_TIERS[name] = 2
for name in TIER3_SCAFFOLDS:
    SCAFFOLD_TIERS[name] = 3
