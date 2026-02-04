"""SMARTS patterns for motif detection in molecular graphs.

This module defines the SMARTS patterns used for detecting structural motifs
during graph coarsening. These patterns focus on ring systems that should
be kept together during hierarchical decomposition.
"""

# Ring-focused motif patterns for clustering
# These patterns focus on structural motifs that should stay together
CLUSTERING_MOTIFS: dict[str, str] = {
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
    # Fused ring systems (Tier 2)
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
    # Complex polycyclic systems (Tier 3)
    "carbazole": "c1ccc2c(c1)[nH]c1ccccc12",
    "phenanthrene": "c1ccc2c(c1)ccc1ccccc21",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "fluorene": "c1ccc2c(c1)Cc1ccccc1-2",
    "acridine": "c1ccc2nc3ccccc3cc2c1",
    "dibenzofuran": "c1ccc2c(c1)oc1ccccc12",
    "dibenzothiophene": "c1ccc2c(c1)sc1ccccc12",
}
