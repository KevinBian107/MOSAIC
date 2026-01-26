# Motif-Conditional Molecule Generation

This document outlines use cases for motif-conditional generation and explains why HDT's hierarchical structure provides advantages over flat tokenization approaches like SENT.

## What is Motif-Conditional Generation?

Motif-conditional generation allows users to specify molecular substructures (motifs) that must appear in generated molecules. Instead of hoping a model produces molecules with desired features, you directly provide the motif and let the model complete the molecule around it.

Common motifs include:
- **Ring systems**: benzene, pyridine, piperidine, furan, thiophene
- **Functional groups**: carboxylic acid, amine, amide, sulfonamide, hydroxyl
- **Privileged scaffolds**: quinoline, indole, benzodiazepine
- **BRICS fragments**: retrosynthetically meaningful building blocks

## Use Cases

### 1. Structure-Activity Relationship (SAR) Exploration

Medicinal chemists iteratively modify hit compounds to improve potency, selectivity, and drug-like properties. Motif-conditional generation supports this workflow:

- Fix the core pharmacophore, generate diverse R-groups
- Keep the binding motif, explore different linkers
- Maintain key hydrogen bond donors/acceptors while varying the scaffold

This is fundamentally how drug optimization works: preserve what's essential, vary what's flexible.

### 2. Scaffold Hopping

Given a known active compound, generate structurally distinct molecules that maintain biological activity. This is critical for:

- **Patent navigation**: Find novel chemical matter outside existing IP
- **Backup compounds**: Develop alternatives in case lead compound fails
- **Selectivity optimization**: Same target engagement, different off-target profile

Motif conditioning enables this by fixing the pharmacophore motifs while allowing the model to generate novel connecting scaffolds.

### 3. Fragment-Based Drug Design (FBDD)

In FBDD, small fragments (150-300 Da) are identified as weak binders, then grown or linked into drug-like molecules. Motif-conditional generation directly supports:

- **Fragment growing**: Provide fragment as motif, generate extensions
- **Fragment linking**: Provide two fragments, generate linkers connecting them
- **Fragment merging**: Combine features of multiple fragments into one molecule

### 4. Synthesizability-Constrained Generation

Limit generation to molecules constructible from available building blocks:

- Specify motifs corresponding to purchasable reagents
- Ensure generated molecules can be made via known reactions
- Reduce wet-lab attrition from unsynthesizable designs

### 5. ADMET Optimization

Certain motifs are known to improve drug-like properties:

- **Solubility**: Add piperazine, morpholine, or other solubilizing groups
- **Metabolic stability**: Introduce fluorine or block known metabolic soft spots
- **Permeability**: Include motifs that improve membrane crossing

By conditioning on these motifs, you bias generation toward molecules with favorable ADMET profiles.

### 6. Multi-Parameter Optimization

Combine motif conditioning with property conditioning:

- "Generate molecules with a benzene ring AND LogP < 3"
- "Include a carboxylic acid motif AND molecular weight < 500"

This provides fine-grained control over both structure and properties.

## Why HDT is Better Than SENT for Motif Conditioning

### The Core Problem with Flat Tokenization

SENT and similar flat tokenizers represent molecules as linear sequences without explicit structural boundaries. When you provide a motif as a prefix:

- The model has no signal that the motif is "complete"
- It may attempt to modify the motif rather than extend from it
- Continuation points are ambiguous
- The model must implicitly learn motif boundaries from data

### HDT's Structural Advantage

HDT uses ENTER/EXIT tokens to explicitly delimit partitions, which correspond to motifs in the molecular hierarchy. This provides three key advantages:

#### 1. Explicit Motif Boundaries

When you provide a motif followed by an EXIT token, the model receives an unambiguous signal: this structural unit is complete. The motif becomes a sealed black box that cannot be modified, only connected to.

With SENT, the model must infer where the motif "ends" from context. This inference can fail, especially for unusual motifs or edge cases like fused ring systems.

#### 2. Grammar-Constrained Continuation

After an EXIT token, HDT's grammar restricts what can follow:
- ENTER: begin a new partition (connecting to existing atoms via back-edges)
- EXIT: close the current hierarchy level
- EOS: end generation

This constraint is enforced at the tokenization level, not learned. The model cannot produce invalid continuations that would corrupt the provided motif.

SENT has no such constraints. Any token can follow any other token, and validity is purely learned.

#### 3. Structured Connectivity

HDT encodes connections between partitions through back-edges: when a new partition's atoms need to connect to existing atoms, they explicitly reference those atoms in the token sequence.

This means:
- The model explicitly "decides" which atoms in the provided motif to connect to
- Connections are interpretable in the token sequence
- The model learns patterns like "new aromatic rings typically connect to existing rings at specific positions"

With SENT, connectivity emerges implicitly from the sequence order and adjacency brackets. There's no explicit representation of "this new atom connects to that existing motif."

### Alignment of Structure and Task

The fundamental advantage is alignment: HDT's tokenization structure matches the task structure.

Motif-conditional generation asks: "Here's a complete structural unit. Generate more structure around it."

HDT represents molecules as: "Here's a complete structural unit (ENTER...EXIT). Here's another unit. Here's how they connect."

SENT represents molecules as: "Here's an atom. Here's another atom. They're connected."

When tokenization aligns with the task, the model doesn't need to learn implicit boundaries—they're explicit in the representation. This should lead to:
- Better preservation of provided motifs
- More chemically sensible attachment points
- Fewer invalid or awkward extensions

### Hierarchical Conditioning Opportunities

HDT's hierarchy enables conditioning at multiple levels:

- **Root level**: Global molecular properties (size, lipophilicity)
- **Partition level**: Motif types and local properties
- **Atom level**: Specific atom types and connectivity

This hierarchical conditioning is natural in HDT but would require architectural additions for flat tokenizers.

## Comparison Summary

| Aspect | SENT | HDT |
|--------|------|-----|
| Motif boundary | Implicit (learned) | Explicit (EXIT token) |
| Continuation constraint | None (any token valid) | Grammar-enforced |
| Connection representation | Implicit (sequence order) | Explicit (back-edges) |
| Motif modification risk | Higher | Lower (sealed by EXIT) |
| Hierarchical conditioning | Requires architecture changes | Natural fit |
| Interpretability | Low | Higher (visible partition structure) |

## Limitations and Considerations

HDT's advantages come with trade-offs:

1. **Sequence length**: HDT sequences are longer than SENT due to structural tokens, though shorter than H-SENT
2. **Hierarchy quality**: Benefits depend on meaningful motif-based coarsening; poor hierarchy construction reduces advantages
3. **Motif vocabulary**: For motif-type conditioning, need to define and detect motif types during tokenization
4. **Training data**: Model must see diverse motif combinations during training to generalize well

## Conclusion

Motif-conditional generation addresses real needs in drug discovery: SAR exploration, scaffold hopping, fragment-based design, and synthesizability constraints. HDT's explicit hierarchical structure aligns naturally with these tasks, providing grammar-enforced motif boundaries, explicit connectivity representation, and opportunities for hierarchical conditioning that flat tokenizers lack.
