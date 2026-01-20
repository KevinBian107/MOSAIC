# Merge Plan: training_fix → main

## Summary
26 commits with ~2061 insertions across 16 files. Changes include major features, critical bug fixes, and infrastructure improvements.

---

## Category 1: Major Features (Keep All)

### 1.1 Labeled SENT Tokenization Support
**Impact**: Core functionality for molecular generation with atom/bond types
- `src/tokenizers/sent.py` - Added labeled_graph support
- `src/tokenizers/sent_utils/*.py` - Cython implementation for labeled SENT
- `src/data/molecular.py` - Integer labels vs one-hot features
- `scripts/train.py` - Pass labeled_graph parameter
- `scripts/test.py` - Auto-detect labeled checkpoints
**Commits**: 82eaf81, 933ec56, e155f92

### 1.2 Auto-Resume Training from Checkpoints
**Impact**: Critical for training on systems that restart (Kubernetes)
- `scripts/train.py`:
  - Early checkpoint detection (before data loading)
  - Auto-resume with `resume: true` config
  - Simple checkpoint naming: `best.ckpt`, `last.ckpt`
- `configs/train.yaml` - Added `resume` parameter
**Commits**: 3af63ee, 3647fc0, 5ab867b, 6d9e7a3

### 1.3 New Model Size (gpt2-xxs)
**Impact**: Smaller model option for faster iteration
- `src/models/transformer.py` - Added xxs configuration (4 layers, 4 heads, 256 dim)
**Commits**: 6019c55

---

## Category 2: Critical Bug Fixes (Keep All)

### 2.1 Training Fixes
- **Training loss logging**: Changed to log per-step instead of per-epoch
  - `src/models/transformer.py:260` - `on_step=(phase=="train")`
  - **Impact**: Wandb shows continuous training curves
  - **Commit**: 09ee7ff

- **Validation interval auto-adjustment**: Prevent crash when `val_check_interval > steps_per_epoch`
  - `scripts/train.py:218-233` - Auto-cap validation interval
  - **Impact**: Training works on datasets of any size
  - **Commit**: dea3511

### 2.2 Testing Fixes
- **Metric computation bugs** (CRITICAL):
  - Invalid SMILES filtering → use sentinel value "INVALID"
  - Reference dataset not loading in test mode → load train_smiles
  - **Files**: `scripts/test.py`, `src/data/datamodule.py`
  - **Impact**: validity, novelty, snn, frag_similarity, scaff_similarity now correct
  - **Commits**: 13a6d27, 06ffc7c

- **FCD multiprocessing fix**:
  - Force single-threaded FCD computation with `n_jobs=1`
  - Add error handling to prevent hanging
  - **File**: `src/evaluation/molecular_metrics.py`
  - **Impact**: FCD computation works without character-level parsing errors
  - **Commits**: 41c46d9, a6f00af

- **Morgan fingerprint deprecation warnings**:
  - Suppress RDKit deprecation warnings
  - **File**: `src/evaluation/molecular_metrics.py`
  - **Impact**: Clean test output
  - **Commits**: 743ddf7, e867292

- **CUDA tensor conversion**:
  - Fix numpy conversion for CUDA tensors in SENT decoder
  - **File**: `src/tokenizers/sent.py`
  - **Impact**: Generation works on GPU
  - **Commit**: f44ee7b

### 2.3 Server Compatibility
- **NumPy version fix**: Pin to <2.0 for PyTorch 2.2.1 compatibility
  - **File**: `environment_server.yaml`
  - **Commit**: 90f2de9

---

## Category 3: Development/Debug Tools (Optional - Keep)

### 3.1 Debug Scripts (Useful for development)
- `debug_generation.py` - Inspect model generation pipeline
- `simple_test.py` - Quick generation test without metrics
- `test_morgan_api.py` - Verify RDKit API usage
**Commits**: 61a6581, fe106bb, a780a22, 9af2fda, 84e2c32
**Recommendation**: Keep in `tmp/` or root (they're in .gitignore already)

---

## Category 4: Documentation (Keep)

### 4.1 Setup Guide
- `docs/SETUP_AND_TRAINING.md` - Comprehensive setup and training guide
**Commits**: b6fc9bf, 81d1e55
**Recommendation**: Keep and update if needed

---

## Category 5: Infrastructure (Keep)

### 5.1 Server Environment
- `environment_server.yaml` - CUDA 12.2 compatible environment file
**Commit**: 7a2ec06
**Recommendation**: Keep for remote server setup

---

## Merge Strategy

### Option A: Squash Merge (RECOMMENDED)
**Pros**:
- Clean single commit in main
- All fixes bundled together
- Easy to revert if needed
- Clear PR description

**Cons**:
- Loses individual commit history
- Can't cherry-pick specific commits later

**Command**:
```bash
# Create PR
gh pr create --title "feat: training improvements and bug fixes" \
  --body "$(cat MERGE_PLAN.md)" \
  --base main --head training_fix

# After approval, squash merge
gh pr merge --squash --delete-branch
```

### Option B: Regular Merge
**Pros**:
- Preserves all commit history
- Can see evolution of changes

**Cons**:
- 26 commits clutters main history
- Many "fix: fix previous fix" type commits

**Command**:
```bash
gh pr create --title "feat: training improvements and bug fixes" \
  --base main --head training_fix

gh pr merge --merge --delete-branch
```

---

## Recommended Merge Message

```
feat: training improvements and bug fixes

Major Features:
- Add labeled SENT tokenization support for molecular generation
- Add auto-resume training from checkpoints (critical for K8s)
- Add gpt2-xxs model size (4 layers, 4 heads, 256 dim)

Critical Bug Fixes:
- Fix training loss logging (per-step instead of per-epoch)
- Fix validation interval auto-adjustment
- Fix metric computation (validity, novelty, snn, similarity)
- Fix FCD multiprocessing issues
- Fix CUDA tensor conversion in SENT decoder
- Suppress Morgan fingerprint deprecation warnings

Infrastructure:
- Add server environment file (CUDA 12.2)
- Add comprehensive setup documentation

Development Tools:
- Add debug scripts for generation testing

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## Pre-Merge Checklist

- [ ] All tests pass
- [ ] Linting passes (`ruff format`, `ruff check`)
- [ ] Documentation updated
- [ ] No sensitive data in commits
- [ ] Branch is up to date with main

---

## Recommendation

**Use Option A (Squash Merge)** because:
1. Many commits are incremental fixes ("fix: fix the fix")
2. Cleaner main branch history
3. Easier to understand what changed in one place
4. All changes are cohesive (training improvements)
