# Training Steps & Samples Seen

How to interpret the step count on WandB and calculate how much data the model has actually seen.

## What is `global_step`?

The x-axis on WandB is PyTorch Lightning's `global_step` — the **number of optimizer updates**, not forward passes. It increments once per `optimizer.step()` call.

## Definitions

| Symbol | Meaning | Config location |
|--------|---------|-----------------|
| $N$ | Training dataset size (number of molecules) | `data.num_train` |
| $B$ | Per-GPU batch size | `data.batch_size` |
| $G$ | GPUs per node | `trainer.devices` |
| $M$ | Number of nodes | `trainer.num_nodes` |
| $A$ | Gradient accumulation steps | `trainer.accumulate_grad_batches` |
| $T$ | Target samples seen | `trainer.target_samples_seen` |
| $S$ | `max_steps` (WandB x-axis limit) | `trainer.max_steps` |

## Core Formulas

**Effective batch size:**

$$B_{\text{eff}} = B \times G \times M \times A$$

**Optimizer steps per epoch:**

$$S_{\text{epoch}} = \left\lfloor \frac{N}{B_{\text{eff}}} \right\rfloor$$

**Total samples seen at step $s$:**

$$\text{samples\_seen}(s) = s \times B_{\text{eff}}$$

**Samples-seen budget to steps (current default policy):**

$$S = \left\lceil \frac{T}{B_{\text{eff}}} \right\rceil$$

**Total epochs completed:**

$$\text{epochs} = \frac{S}{S_{\text{epoch}}}$$

## Real Examples from Our Checkpoints

### MOSES HDTC — 4-GPU DDP vs equivalent 1-GPU

Run: `moses_hdtc_20260221-224537`, finished at step 250,000. To see the same 16M samples on 1 GPU, you'd need 4x the steps.

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

Same data seen, same number of epochs — but the 1-GPU run needs **4x the global_steps** on WandB to get there. If you compare the two WandB curves by step, the 4-GPU run looks 4x faster to converge, but it's just the x-axis scaling.

## Unified X-Axis: `train/samples_seen`

**Do not compare WandB loss curves by `global_step` when $B_{\text{eff}}$ differs.** At the same step, a larger effective batch has seen more data.

We log `train/samples_seen` every training step so you can normalize across any combination of $B$, $G$, $A$:

$$\texttt{train/samples\_seen} = \texttt{global\_step} \times B \times G \times A$$

### How to use in WandB

1. Open your WandB panel with `train/loss`
2. Click the **x-axis** dropdown (bottom-left of chart)
3. Select **`train/samples_seen`** instead of `Step`
4. All runs now share a comparable x-axis regardless of batch/GPU/accum config

`effective_batch_size` is also logged to each run's WandB config for quick reference.

### Conversion table

To convert a `global_step` $s$ to `samples_seen`, look up $B_{\text{eff}}$ from the run's config:

| Run config | $B_{\text{eff}}$ | step 10K = | step 100K = | step 250K = |
|-----------|-------------------|------------|-------------|-------------|
| $B$=16, $G$=1, $A$=1 | 16 | 160K samples | 1.6M | 4M |
| $B$=16, $G$=4, $A$=1 | 64 | 640K samples | 6.4M | 16M |
| $B$=32, $G$=1, $A$=2 | 64 | 640K samples | 6.4M | 16M |
| $B$=8, $G$=1, $A$=8 | 64 | 640K samples | 6.4M | 16M |
| $B$=32, $G$=4, $A$=1 | 128 | 1.28M samples | 12.8M | 32M |

The last three rows all have $B_{\text{eff}} = 64$ — their WandB curves align perfectly when plotted against `train/samples_seen`, even though their `global_step` counts differ by up to 4x.

## Eval Pipeline: `samples_seen` in `results.json`

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

## Gotcha: Logged `Steps per epoch` is Wrong When $A > 1$

The training log (`train.py:1240`) computes:

$$\text{logged steps/epoch} = \left\lfloor \frac{N}{B \times G} \right\rfloor$$

This does **not** divide by $A$. So the logged value is **batches per epoch**, not optimizer steps. True optimizer steps per epoch = logged value / $A$.

## Config Reference

- `batch_size` ($B$): `configs/train.yaml` → `data.batch_size`
- `num_GPUs` ($G$): `configs/train.yaml` → `trainer.devices` (or `WORLD_SIZE` env)
- `accum` ($A$): `configs/train.yaml` → `trainer.accumulate_grad_batches` (default 1)
- Step logging: `src/models/transformer.py` → `_shared_step()` logs `train/loss` with `on_step=True` (keyed to `global_step`)
- LR scheduler: `src/models/transformer.py` → `configure_optimizers()` uses `interval="step"`
- `val_checks_per_epoch`: preferred validation cadence (default 5)
- `validate_every_n_epochs`: preferred epoch interval (default 1)
- Legacy fallback: `val_check_interval` / `check_val_every_n_epoch`
