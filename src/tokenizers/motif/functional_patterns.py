"""SMARTS patterns for functional group detection in molecular graphs.

This module defines SMARTS patterns for detecting functional groups during
HDTC tokenization. Patterns are organized by priority to resolve overlaps.

Priority levels:
- ring: 30 (highest) - Ring systems should be preserved intact
- multi_atom: 20 - Multi-atom functional groups (e.g., carboxyl, ester)
- single_atom: 10 (lowest) - Single-atom functional groups (e.g., hydroxyl)
"""

# Pattern priority levels for overlap resolution
PATTERN_PRIORITY: dict[str, int] = {
    "ring": 30,
    "multi_atom": 20,
    "single_atom": 10,
}

# Functional group patterns: {name: (SMARTS, pattern_type)}
# Pattern type is used to look up priority in PATTERN_PRIORITY
FUNCTIONAL_GROUP_PATTERNS: dict[str, tuple[str, str]] = {
    # Carboxylic acid derivatives (multi_atom)
    "carboxyl": ("[CX3](=O)[OX2H1]", "multi_atom"),
    "ester": ("[CX3](=O)[OX2][#6]", "multi_atom"),
    "amide": ("[CX3](=O)[NX3]", "multi_atom"),
    "anhydride": ("[CX3](=O)[OX2][CX3](=O)", "multi_atom"),
    # Carbonyl derivatives (multi_atom)
    "aldehyde": ("[CX3H1](=O)", "multi_atom"),
    "ketone": ("[#6][CX3](=O)[#6]", "multi_atom"),
    # Nitrogen-containing (multi_atom)
    "nitro": ("[NX3+](=O)[O-]", "multi_atom"),
    "nitrile": ("[CX2]#[NX1]", "multi_atom"),
    "imine": ("[CX3]=[NX2]", "multi_atom"),
    "azo": ("[NX2]=[NX2]", "multi_atom"),
    # Sulfur-containing (multi_atom)
    "sulfonyl": ("[SX4](=O)(=O)", "multi_atom"),
    "sulfoxide": ("[SX3](=O)", "multi_atom"),
    "thioether": ("[#6][SX2][#6]", "multi_atom"),
    # Phosphorus-containing (multi_atom)
    "phosphate": ("[PX4](=O)([OX2])([OX2])[OX2]", "multi_atom"),
    "phosphonate": ("[PX4](=O)([OX2])([OX2])[#6]", "multi_atom"),
    # Halogen-containing (single_atom)
    "fluoride": ("[FX1]", "single_atom"),
    "chloride": ("[ClX1]", "single_atom"),
    "bromide": ("[BrX1]", "single_atom"),
    "iodide": ("[IX1]", "single_atom"),
    # Oxygen-containing (single_atom)
    "hydroxyl": ("[OX2H]", "single_atom"),
    "ether_o": ("[OX2]([#6])[#6]", "single_atom"),
    "epoxide": ("[OX2r3]", "single_atom"),
    # Nitrogen-containing (single_atom)
    "primary_amine": ("[NX3H2]", "single_atom"),
    "secondary_amine": ("[NX3H1]([#6])[#6]", "single_atom"),
    "tertiary_amine": ("[NX3]([#6])([#6])[#6]", "single_atom"),
    # Carbon-containing (single_atom)
    "methyl": ("[CH3]", "single_atom"),
    "tert_butyl": ("[CX4]([CH3])([CH3])[CH3]", "multi_atom"),
    "isopropyl": ("[CX4H]([CH3])[CH3]", "multi_atom"),
}

# Ring patterns from existing CLUSTERING_MOTIFS
# These are treated as "ring" pattern type for priority purposes
RING_PATTERNS: dict[str, str] = {
    # Aromatic 6-membered rings
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "pyrazine": "c1cnccn1",
    # Aromatic 5-membered rings
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1cnc[nH]1",
    "oxazole": "c1cocn1",
    "thiazole": "c1cscn1",
    # Fused ring systems
    "naphthalene": "c1ccc2ccccc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",
    # Saturated rings
    "cyclopropane": "C1CC1",
    "cyclobutane": "C1CCC1",
    "cyclopentane": "C1CCCC1",
    "cyclohexane": "C1CCCCC1",
    # Partially unsaturated
    "cyclohexene": "C1=CCCCC1",
    "cyclopentene": "C1=CCCC1",
}


def get_all_patterns() -> dict[str, tuple[str, str]]:
    """Get all patterns including both functional groups and rings.

    Returns:
        Dictionary mapping pattern name to (SMARTS, pattern_type) tuple.
    """
    all_patterns = dict(FUNCTIONAL_GROUP_PATTERNS)
    for name, smarts in RING_PATTERNS.items():
        all_patterns[name] = (smarts, "ring")
    return all_patterns
