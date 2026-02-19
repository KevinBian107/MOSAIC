# Property Experiments

Post-hoc analysis experiments that probe specific properties of generated molecules across trained models. Each experiment lives in `property_experiment/<name>/` with a matching Hydra config at `configs/property_experiment/<name>.yaml`.

---

## Structure

```
property_experiment/
├── __init__.py
├── error_analysis/          # Experiment 1: Valence violation location analysis
│   ├── __init__.py
│   ├── analysis.py          # Core analysis functions
│   ├── visualize.py         # Figure generation
│   └── run_experiment.py    # Hydra entry point
└── <future_experiment>/     # Add new experiments here

configs/property_experiment/
├── error_analysis.yaml
└── <future_experiment>.yaml
```

Output is saved to `outputs/property_experiment/<name>/`.

---

## Experiments

### 1. Error Analysis — Valence Violations at Motif Boundaries

**Hypothesis:** Flexible coarsening methods (SC, HAC) merge singleton/linker atoms into motif communities where they don't belong, corrupting the structural prior and causing valence violations at ring-chain boundaries.

**Method:**
1. Generate 500 molecules from each of 5 COCONUT models (H-SENT SC, H-SENT HAC, HDT SC, HDT HAC, HDTC)
2. Build each molecule without RDKit sanitization to preserve raw valence states
3. Identify atoms with valence violations (actual valence > max allowed)
4. Classify each violating atom's structural role:
   - **ring_interior** — in a ring, all neighbors also in rings
   - **ring_boundary** — in a ring, has neighbor(s) outside rings
   - **chain_boundary** — not in a ring, has neighbor(s) in rings (linker atoms SC/HAC misassign)
   - **chain_interior** — not in a ring, no ring neighbors

**Output:** 3-panel figure (`valence_violation_analysis.png`)
- Panel A: Validity rate per model
- Panel B: Stacked distribution of violation locations per model
- Panel C: Boundary error ratio `(ring_boundary + chain_boundary) / total`

If SC/HAC models show higher boundary error ratios than HDTC, the hypothesis is supported.

**Run:**
```bash
python property_experiment/error_analysis/run_experiment.py

# Override sample count
python property_experiment/error_analysis/run_experiment.py generation.num_samples=100
```

---

## Adding a New Experiment

1. Create `property_experiment/<name>/` with `__init__.py`, analysis module, and `run_experiment.py`
2. Add `configs/property_experiment/<name>.yaml`
3. Point `@hydra.main(config_path=..., config_name=...)` to the new config
4. Document the experiment in this file
