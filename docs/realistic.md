# Realistic Molecule Generation

This document defines what it means for a generative model to produce "realistic" molecules and describes the metrics we use to evaluate this.

## What Makes a Molecule Realistic?

A realistic molecule is one that exhibits the same **structural patterns** as molecules found in real-world chemical databases. Beyond basic validity (correct atoms, bonds, valence), a realistic molecule should:

1. **Follow chemical preferences** - Real molecules exhibit biases in how functional groups attach, where substituents are placed, and which structures co-occur
2. **Match training distribution** - Generated molecules should statistically resemble the training data, not just be valid

A model that generates only valid molecules but with unusual structural patterns has not truly learned chemistry.

## Evaluation Metrics

### 1. Benzene Substitution Patterns

Benzene rings are ubiquitous in drug-like molecules. How substituents attach to benzene follows well-known chemical principles:

#### Substitution Count
- **Mono-substituted**: One group attached (e.g., toluene, phenol)
- **Di-substituted**: Two groups attached
- **Tri-substituted**: Three groups attached
- **Poly-substituted**: Four or more groups attached

Training data has characteristic ratios of these patterns. A realistic generator should match these ratios.

#### Di-substitution Position (ortho/meta/para)

For di-substituted benzenes, the relative positions matter:

```
    ortho (1,2)         meta (1,3)          para (1,4)
       X                   X                    X
      /                   /                    /
     ●                   ●                    ●
    / \                 / \                  / \
   ●   ●               ●   ●                ●   ●
    \ /                 \ /                  \ /
     ●                   ●                    ●
      \                   \                    \
       X                   ●                    X
                            \
                             X
```

Real molecules show preferences:
- **Para** is often favored (less steric hindrance, symmetric)
- **Ortho** can be favored for intramolecular interactions
- **Meta** is typically less common

A realistic generator should capture these preferences from training data.

### 2. Functional Group Distribution

Different functional groups appear with different frequencies in drug-like molecules:

| Functional Group | Description | Typical Role |
|-----------------|-------------|--------------|
| Hydroxyl (-OH) | Alcohol/phenol | H-bonding, solubility |
| Amino (-NH2) | Amine | H-bonding, basicity |
| Carbonyl (C=O) | Ketone/aldehyde | H-bonding, reactivity |
| Carboxyl (-COOH) | Carboxylic acid | Ionizable, H-bonding |
| Halogen (-F, -Cl, -Br) | Halide | Lipophilicity, metabolic stability |
| Ether (-OR) | Ether/methoxy | Solubility, metabolic site |
| Methyl (-CH3) | Methyl | Lipophilicity, steric |
| Cyano (-CN) | Nitrile | H-bond acceptor |

A realistic generator should produce functional groups in proportions similar to training data.

## Quantitative Metrics

We use two distance metrics to compare generated vs training distributions:

### Total Variation Distance (TV)

Measures the maximum difference between two probability distributions:

```
TV(P, Q) = 0.5 × Σ |P(x) - Q(x)|
```

- **TV = 0**: Identical distributions
- **TV = 1**: Completely different distributions
- **TV < 0.1**: Good match

### KL Divergence

Measures information lost when using generated distribution to approximate training:

```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

- **KL = 0**: Identical distributions
- **KL > 0**: Divergence (lower is better)

## Usage

Run realistic generation evaluation:

```bash
python scripts/realistic_gen.py \
    model.checkpoint_path=outputs/train/moses_hdt_n50000_20260122-185129/best.ckpt \
    generation.num_samples=500
```

Output includes:
- `generated_smiles.txt` - Generated molecules
- `results.json` - All metrics
- `analysis_*.png` - Distribution comparison bar charts
- `substitution_molecules_*.png` - Example molecules by substitution pattern
- `functional_group_molecules_*.png` - Example molecules by functional group

## Future Extensions

Additional realism metrics to consider:

- **Ring system distribution** - Types and sizes of ring systems
- **Molecular weight distribution** - Should match training range
- **LogP distribution** - Lipophilicity patterns
- **Scaffold diversity** - Core structure variety
- **Stereochemistry patterns** - Chiral center preferences
- **Heteroatom placement** - Where N, O, S appear in structures
