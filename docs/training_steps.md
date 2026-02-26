# Training Steps & Samples Seen

How to interpret the step count on WandB and calculate how much data the model has actually seen.

## What is `global_step`?

The x-axis on WandB is PyTorch Lightning's `global_step` — the **number of optimizer updates**, not forward passes. It increments once per `optimizer.step()` call.

## Definitions

| Symbol | Meaning | Config location |
|--------|---------|-----------------|
| $N$ | Training dataset size (number of molecules) | `data.num_train` |
| $B$ | Per-GPU batch size | `data.batch_size` |
| $G$ | Number of GPUs (DDP world size) | `trainer.devices` or `WORLD_SIZE` env |
| $A$ | Gradient accumulation steps | `trainer.accumulate_grad_batches` |
| $S$ | `max_steps` (WandB x-axis limit) | `trainer.max_steps` |

## Core Formulas

**Effective batch size:**

$$B_{\text{eff}} = B \times G \times A$$

**Optimizer steps per epoch:**

$$S_{\text{epoch}} = \left\lfloor \frac{N}{B_{\text{eff}}} \right\rfloor$$

**Total samples seen at step $s$:**

$$\text{samples\_seen}(s) = s \times B_{\text{eff}}$$

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

## Comparing Runs Fairly

**Do not compare WandB loss curves by `global_step` when $B_{\text{eff}}$ differs.** At the same step, a larger effective batch has seen more data.

Compare by:
- **Samples seen**: $s \times B_{\text{eff}}$
- **Epochs**: WandB logs `train/loss_epoch`

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
- `val_check_interval`: validation every this many **global_steps** (default 1000)
