# Reproducibility Across GPUs

A guide for ensuring fair model benchmarking across different hardware setups, and for interpreting training metrics on WandB.

## 1. Definitions

### Core Concepts

| Term | What it is | Where it lives |
|------|-----------|----------------|
| **Batch** | A group of samples processed together in one forward pass on a single device. | Pure PyTorch (`DataLoader(batch_size=...)`) |
| **Batch Size** ($B$) | The number of samples in one batch, on one device. | `data.batch_size` |
| **Epoch** | One full pass through the entire training dataset. After one epoch, every sample has been seen exactly once. | Universal concept; managed by Lightning `Trainer` |
| **Optimizer Step** | One call to `optimizer.step()` — i.e. one weight update. This is what Lightning's `global_step` counts. | Pure PyTorch (`optimizer.step()`) |
| **Gradient Accumulation** ($A$) | Number of forward/backward passes whose gradients are summed before a single optimizer step. Simulates a larger batch without more GPU memory. | Lightning `Trainer(accumulate_grad_batches=...)` |
| **Effective Batch Size** ($B_{\text{eff}}$) | The true number of samples that contribute to one optimizer step. This determines gradient noise and training dynamics. | Conceptual — computed from other params |
| **Training Data Size** ($N$) | Total number of unique samples in the training set. | `data.num_train` (or dataset length) |
| **Max Steps** ($S$) | Hard limit on optimizer steps. Training stops when `global_step` reaches this, regardless of epochs. | Lightning `Trainer(max_steps=...)` |
| **Total Samples Seen** | Cumulative count of samples the model has processed over the entire run. | Conceptual — computed from other params |
| **Learning Rate** (LR) | Step size for weight updates. | Pure PyTorch (`torch.optim.Optimizer(lr=...)`) |
| **Warmup Steps** | Number of optimizer steps over which LR ramps from ~0 to the target LR. | Pure PyTorch (LR scheduler), configured in `model.warmup_steps` |
| **Target Samples Seen** ($T$) | A budget: "stop training after seeing this many samples total." Our default training policy. | `trainer.target_samples_seen` |

### Hardware Parameters

| Symbol | Meaning | Config location |
|--------|---------|-----------------|
| $G$ | GPUs per node | `trainer.devices` |
| $M$ | Number of nodes | `trainer.num_nodes` |

### How DDP Changes the Math

When you apply DDP (Distributed Data Parallel), the model and DataLoader are **replicated** to every GPU. Each GPU gets its own **distinct** batch of $B$ samples — PyTorch does **not** split $B$ across GPUs.

So in one forward pass, the system processes $B \times G \times M$ samples total. Gradients are averaged across all devices before the optimizer step.

## 2. Core Formulas

**Effective batch size:**

$$B_{\text{eff}} = B \times G \times M \times A$$

**Optimizer steps per epoch:**

$$S_{\text{epoch}} = \left\lfloor \frac{N}{B_{\text{eff}}} \right\rfloor$$

**Total samples seen at step $s$:**

$$\text{samples\_seen}(s) = s \times B_{\text{eff}}$$

**Samples-seen budget → steps (our default policy):**

$$S = \left\lceil \frac{T}{B_{\text{eff}}} \right\rceil$$

**Total epochs completed:**

$$\text{epochs} = \frac{S}{S_{\text{epoch}}}$$

## 3. The Domino Effect: How Parameters Interact

Changing one parameter creates a chain reaction:

| You change... | Direct effect | Downstream consequence |
|---------------|--------------|----------------------|
| Add GPUs ($G$) | $B_{\text{eff}}$ increases | Steps per epoch drops; if using `target_samples_seen`, `max_steps` drops too |
| Increase $B$ | $B_{\text{eff}}$ increases | Same as above; also needs more GPU memory |
| Increase $A$ | $B_{\text{eff}}$ increases | Same training dynamics as more GPUs, but slower (sequential not parallel) |
| Increase $B_{\text{eff}}$ (any way) | Gradient estimates become more stable | LR may need to scale up to take advantage (linear or sqrt scaling rules) |
| Hardcode `max_steps` + add GPUs | Steps stay fixed but $B_{\text{eff}}$ grows | Model sees **more** total data than intended — more epochs |
| Use `target_samples_seen` + add GPUs | Total data is fixed but steps drop | **Fewer optimizer steps** — different training dynamics even though data budget is the same |

## 4. Fair Benchmarking: What Must Stay Identical

When comparing tokenization strategies (SENT vs HSENT vs HDT vs HDTC), we need to isolate the tokenizer as the only variable. Different tokenizers produce **different sequence lengths** for the same molecule — this is the point of the comparison. But the optimizer trajectory must be identical.

### Must-lock parameters

| Parameter | Why | Notes |
|-----------|-----|-------|
| **Effective batch size** ($B_{\text{eff}}$) | Gradient variance must be identical across runs | Use `accumulate_grad_batches` to compensate for GPU count differences |
| **Max steps** ($S$) | Same number of weight updates = same optimization trajectory | Do **not** use `max_epochs` — an "epoch" means different compute for different tokenizers |
| **Peak learning rate** | Directly controls convergence speed | Keep identical across all runs |
| **LR scheduler** | Warmup + decay shape affects final performance | Keep identical |
| **Warmup steps** | Must be an absolute integer, not a fraction of epochs | Already the case in our config (`warmup_steps: 1414`) |
| **Seed** | Reproducibility of initialization and data order | `seed_everything(42, workers=True)` |

### DDP + fair benchmarking

You **can** run different models on different GPU counts (e.g. Model A on 1 GPU, Model B on 4 GPUs). But you must equalize $B_{\text{eff}}$:

| Setup | $B$ | $G$ | $A$ | $B_{\text{eff}}$ | Fair? |
|-------|-----|-----|-----|-------------------|-------|
| 1 GPU, batch 32, accum 4 | 32 | 1 | 4 | **128** | Yes |
| 4 GPUs, batch 32, accum 1 | 32 | 4 | 1 | **128** | Yes |
| 1 GPU, batch 32, accum 1 | 32 | 1 | 1 | **32** | No — different $B_{\text{eff}}$ |

> **Note on DDP non-determinism:** DDP introduces microscopic floating-point non-determinism from asynchronous gradient reductions. This is negligible for macro-benchmarking — final metric differences will be far larger than this noise.

### What NOT to use for comparison

**Do not compare validation loss across tokenizers.** Cross-entropy loss depends on vocabulary size and token distribution. A tokenizer with vocab 50 has fundamentally different loss dynamics than one with vocab 150. Always evaluate using downstream generation metrics: Validity, Uniqueness, Novelty, FCD, etc.

## 5. Our Default Training Policy: `target_samples_seen`

We use `target_samples_seen` as the primary training budget. This means:

$$\text{max\_steps} = \left\lceil \frac{\text{target\_samples\_seen}}{B_{\text{eff}}} \right\rceil$$

**Implication:** if you change GPUs (and thus $B_{\text{eff}}$) without adjusting `accumulate_grad_batches`, the model trains for fewer optimizer steps. The total data volume stays constant, but the optimization trajectory changes.

**For rigorous benchmarking:** either (a) keep $B_{\text{eff}}$ fixed so `max_steps` is derived identically, or (b) switch to explicit `max_steps` with fixed $B_{\text{eff}}$.

## 6. Real Examples from Our Checkpoints

The goal is to make sure that we have the same sampels seen, if we fix the same batch size across GPU, then we just need less max steps to achieve so.

### MOSES HDTC — 4-GPU DDP vs equivalent 1-GPU

Run: `moses_hdtc_20260221-224537`, finished at step 250,000.

| Symbol | 4-GPU DDP (actual run) | 1-GPU equivalent |
|--------|----------------------|------------------|
| $N$ | 1,000,000 | 1,000,000 |
| $B$ | 16 | 16 |
| $G$ | 4 | 1 |
| $A$ | 1 | 1 |
| $B_{\text{eff}} = B \times G \times A$ | $16 \times 4 \times 1 =$ **64** | $16 \times 1 \times 1 =$ **16** |
| $S_{\text{epoch}} = \lfloor N / B_{\text{eff}} \rfloor$ | $\lfloor 1\text{M} / 64 \rfloor =$ **15,625** | $\lfloor 1\text{M} / 16 \rfloor =$ **62,500** |
| $S$ (max_steps) | 250,000 | **1,000,000** |
| Epochs $= S / S_{\text{epoch}}$ | **16** | **16** |
| Samples seen $= S \times B_{\text{eff}}$ | **16M** | **16M** |

Same data seen, same number of epochs — but the 1-GPU run needs **4x the global_steps** on WandB.

## 7. Unified X-Axis: `train/samples_seen`

**Do not compare WandB loss curves by `global_step` when $B_{\text{eff}}$ differs.** At the same step, a larger effective batch has seen more data.

We log `train/samples_seen` every training step:

$$\texttt{train/samples\_seen} = \texttt{global\_step} \times B \times G \times A$$

### How to use in WandB

1. Open your WandB panel with `train/loss`
2. Click the **x-axis** dropdown (bottom-left of chart)
3. Select **`train/samples_seen`** instead of `Step`
4. All runs now share a comparable x-axis regardless of batch/GPU/accum config

`effective_batch_size` is also logged to each run's WandB config for quick reference.

### Conversion table

| Run config | $B_{\text{eff}}$ | step 10K = | step 100K = | step 250K = |
|-----------|-------------------|------------|-------------|-------------|
| $B$=16, $G$=1, $A$=1 | 16 | 160K samples | 1.6M | 4M |
| $B$=16, $G$=4, $A$=1 | 64 | 640K samples | 6.4M | 16M |
| $B$=32, $G$=1, $A$=2 | 64 | 640K samples | 6.4M | 16M |
| $B$=8, $G$=1, $A$=8 | 64 | 640K samples | 6.4M | 16M |
| $B$=32, $G$=4, $A$=1 | 128 | 1.28M samples | 12.8M | 32M |

The last three rows all have $B_{\text{eff}} = 64$ — their WandB curves align perfectly when plotted against `train/samples_seen`.

## 8. Eval Pipeline: `samples_seen` in `results.json`

Both `test.py` and `realistic_gen.py` automatically compute `samples_seen` for any checkpoint and include it in `results.json`:

```json
{
  "global_step": 250000,
  "effective_batch_size": 64,
  "samples_seen": 16000000
}
```

**How it works:** the eval scripts read `global_step` from the checkpoint, then look for `config.yaml` in the same directory (saved during training) to recover $B$, $G$, $A$.

**Old checkpoints** (trained before this change): `global_step` is always available in the `.ckpt` file. If the `config.yaml` is co-located, `samples_seen` is computed automatically. If the checkpoint was moved and `config.yaml` is missing, the script logs a warning and sets `effective_batch_size` and `samples_seen` to `null` — you can compute it manually:

```
samples_seen = global_step × batch_size × num_GPUs × num_nodes × accumulate_grad_batches
```

## 9. Gotcha: Logged `Steps per epoch` is Wrong When $A > 1$

The training log (`train.py:1240`) computes:

$$\text{logged steps/epoch} = \left\lfloor \frac{N}{B \times G} \right\rfloor$$

This does **not** divide by $A$. So the logged value is **batches per epoch**, not optimizer steps. True optimizer steps per epoch = logged value / $A$.

## 10. Config Reference

- `batch_size` ($B$): `configs/train.yaml` → `data.batch_size`
- `num_GPUs` ($G$): `configs/train.yaml` → `trainer.devices` (or `WORLD_SIZE` env)
- `accum` ($A$): `configs/train.yaml` → `trainer.accumulate_grad_batches` (default 1)
- Step logging: `src/models/transformer.py` → `_shared_step()` logs `train/loss` with `on_step=True` (keyed to `global_step`)
- LR scheduler: `src/models/transformer.py` → `configure_optimizers()` uses `interval="step"`
- `val_checks_per_epoch`: preferred validation cadence (default 5)
- `validate_every_n_epochs`: preferred epoch interval (default 1)
- Legacy fallback: `val_check_interval` / `check_val_every_n_epoch`
