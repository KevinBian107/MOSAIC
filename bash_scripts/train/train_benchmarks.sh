#!/bin/bash
# Train benchmark models from scratch on MOSES or COCONUT dataset.
#
# This script trains all tokenizer variants (SENT, H-SENT, HDT, HDTC) from scratch.
# If a checkpoint exists but training is incomplete (global_step < max_steps),
# the script automatically resumes from the checkpoint in the same output directory.
# By default, trains all coarsening variants (MC, SC, HAC) for hierarchical tokenizers.
#
# Usage:
#   ./bash_scripts/train/train_benchmarks.sh              # Train on MOSES (default)
#   ./bash_scripts/train/train_benchmarks.sh --coconut    # Train on COCONUT
#   ./bash_scripts/train/train_benchmarks.sh --ddp        # Train with DDP on 4 GPUs
#   ./bash_scripts/train/train_benchmarks.sh --devices=2  # Train with DDP on 2 GPUs
#   ./bash_scripts/train/train_benchmarks.sh --skip-sc-hac  # Only train MC variants
#   ./bash_scripts/train/train_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/train/train_benchmarks.sh --help       # Show help
#
# Output directories:
#   MOSES:   outputs/benchmark/moses_{tokenizer}_{coarsening}_...
#   COCONUT: outputs/benchmark_coconut/coconut_{tokenizer}_{coarsening}_...

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
DATASET="moses"
OUTPUT_DIR="$PROJECT_ROOT/outputs/benchmark"
MAX_STEPS=500000
WANDB_ENABLED=true
DRY_RUN=false
FORCE=false
SKIP_SC_HAC=false
NUM_DEVICES=1
DEVICES_SET=false  # Track whether --devices was explicitly set

# Tokenizers to train
# Format: "tokenizer:coarsening" where coarsening is "none", "mc", "sc", or "hac"
TOKENIZERS=(
    "sent:none"
    "hsent:mc"
    "hsent:sc"
    "hsent:hac"
    "hdt:mc"
    "hdt:sc"
    "hdt:hac"
    "hdtc:none"
)

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --no-wandb)
            WANDB_ENABLED=false
            ;;
        --force)
            FORCE=true
            ;;
        --coconut)
            DATASET="coconut"
            OUTPUT_DIR="$PROJECT_ROOT/outputs/benchmark_coconut"
            MAX_STEPS=50000  # Smaller dataset, fewer steps
            ;;
        --ddp)
            # Only set default of 4 GPUs if --devices wasn't explicitly specified
            if [ "$DEVICES_SET" = false ]; then
                NUM_DEVICES=4
            fi
            ;;
        --devices=*)
            NUM_DEVICES="${arg#*=}"
            DEVICES_SET=true
            ;;
        --skip-sc-hac)
            SKIP_SC_HAC=true
            ;;
        --steps=*)
            MAX_STEPS="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train all tokenizer variants from scratch."
            echo ""
            echo "Options:"
            echo "  --dry-run         Show what would be run without executing"
            echo "  --force           Re-run from scratch even if checkpoint exists"
            echo "  --coconut         Train on COCONUT instead of MOSES"
            echo "  --ddp             Enable DDP multi-GPU training (default: 4 GPUs)"
            echo "  --devices=N       Set number of GPUs (implies DDP when N > 1)"
            echo "  --skip-sc-hac     Only train MC variants (skip SC and HAC coarsening)"
            echo "  --no-wandb        Disable WandB logging"
            echo "  --steps=N         Set max training steps (default: 500000 MOSES, 50000 COCONUT)"
            echo "                    Same step count is used regardless of GPU count (fair benchmarking)"
            echo ""
            echo "Tokenizers trained (default: all 8 variants):"
            echo "  - SENT (flat, no coarsening)"
            echo "  - H-SENT + MC (hierarchical SENT with motif community)"
            echo "  - H-SENT + SC (spectral clustering)"
            echo "  - H-SENT + HAC (hierarchical agglomerative clustering)"
            echo "  - HDT + MC (hierarchical DFS with motif community)"
            echo "  - HDT + SC (spectral clustering)"
            echo "  - HDT + HAC (hierarchical agglomerative clustering)"
            echo "  - HDTC (compositional, uses functional hierarchy)"
            echo ""
            echo "With --skip-sc-hac, only trains MC variants:"
            echo "  - SENT, H-SENT + MC, HDT + MC, HDTC"
            echo ""
            echo "Output directories:"
            echo "  MOSES:   outputs/benchmark/{dataset}_{tokenizer}_{coarsening}_n{steps}_{date}/"
            echo "  COCONUT: outputs/benchmark_coconut/{dataset}_{tokenizer}_{coarsening}_n{steps}_{date}/"
            exit 0
            ;;
    esac
done

# Filter to MC-only if requested
if [ "$SKIP_SC_HAC" = true ]; then
    TOKENIZERS=(
        "sent:none"
        "hsent:mc"
        "hdt:mc"
        "hdtc:none"
    )
fi

cd "$PROJECT_ROOT"

# Fair benchmarking: hold effective batch size constant regardless of GPU count.
# DDP replicates the batch to each GPU, so we REDUCE per-GPU batch size to
# keep B_eff = batch_per_gpu × num_gpus × accum_grad_batches unchanged.
# LR, warmup, and max_steps do NOT change — the optimization trajectory
# is mathematically identical regardless of hardware.
#
# See docs/designs/reproducibility_across_gpus.md for full details.
#
# Example (4 GPUs):
#   B_eff = 8 × 4 × 1 = 32 (same as 32 × 1 × 1 on single GPU)
#   max_steps = 500,000 (unchanged), LR = 8.49e-4 (unchanged)
EFFECTIVE_BATCH_SIZE=32  # Must stay constant across all benchmarking runs
DDP_BATCH_SIZE=""
USE_MIG=false
MIG_UUIDS=""
if [ "$NUM_DEVICES" -gt 1 ]; then
    # Compute per-GPU batch size to keep B_eff constant
    if (( EFFECTIVE_BATCH_SIZE % NUM_DEVICES != 0 )); then
        echo "ERROR: effective_batch_size ($EFFECTIVE_BATCH_SIZE) is not divisible by num_devices ($NUM_DEVICES)"
        echo "  Supported GPU counts for B_eff=$EFFECTIVE_BATCH_SIZE: 1, 2, 4, 8, 16, 32"
        exit 1
    fi
    DDP_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE / NUM_DEVICES))

    # Detect MIG instances (Multi-Instance GPU).
    # With MIG, each process can only access one MIG instance via the CUDA Runtime.
    # PL's built-in DDPStrategy (which uses device indices 0..N-1) fails because
    # the Runtime only exposes 1 device per MIG partition.
    # Fix: launch N separate processes, each with CUDA_VISIBLE_DEVICES set to one
    # MIG UUID, LOCAL_RANK=0, and trainer.num_nodes=N to satisfy PL validation.
    MIG_UUIDS=$(nvidia-smi -L 2>/dev/null | grep -oP 'MIG-[0-9a-f-]+' | paste -sd, 2>/dev/null || true)
    MIG_COUNT=$(echo "$MIG_UUIDS" | tr ',' '\n' | grep -c 'MIG' 2>/dev/null || echo 0)
    if [ "$MIG_COUNT" -ge "$NUM_DEVICES" ]; then
        USE_MIG=true
    fi
fi

echo "========================================"
echo "Training Benchmark Models"
echo "========================================"
echo ""
echo "Settings:"
echo "  Dataset: $DATASET"
echo "  Max steps: $MAX_STEPS"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE (constant)"
if [ "$NUM_DEVICES" -gt 1 ]; then
    echo "  DDP: ${NUM_DEVICES} GPUs (batch=${DDP_BATCH_SIZE}/GPU × ${NUM_DEVICES} GPUs = ${EFFECTIVE_BATCH_SIZE})"
    if [ "$USE_MIG" = true ]; then
        echo "  MIG: detected (manual per-process launching)"
    fi
else
    echo "  GPUs: 1 (batch=${EFFECTIVE_BATCH_SIZE})"
fi
echo "  WandB: $WANDB_ENABLED"
echo "  Output: $OUTPUT_DIR"
echo "  Skip SC/HAC: $SKIP_SC_HAC"
echo ""
echo "Tokenizers to train:"
for tok_config in "${TOKENIZERS[@]}"; do
    IFS=':' read -r tok coarse <<< "$tok_config"
    if [ "$coarse" = "none" ]; then
        echo "  - $tok"
    else
        echo "  - $tok ($coarse)"
    fi
done
echo ""

# Get coarsening strategy name
get_coarsening_name() {
    local coarse="$1"
    case $coarse in
        mc) echo "motif_community" ;;
        sc) echo "spectral" ;;
        hac) echo "hac" ;;
        mas) echo "motif_aware_spectral" ;;
        *) echo "" ;;
    esac
}

# Get short name for directory
get_short_name() {
    local tok="$1"
    local coarse="$2"
    if [ "$coarse" = "none" ]; then
        echo "${tok}"
    else
        echo "${tok}_${coarse}"
    fi
}

# Check if tokenizer supports coarsening
supports_coarsening() {
    local tok="$1"
    if [[ "$tok" == "hsent" ]] || [[ "$tok" == "hdt" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Get recommended sampling.max_length per tokenizer/dataset.
# Values derived from tokenization stats on 1000 samples with ~15% buffer,
# rounded to multiples of 128.
#
# MOSES (10-26 nodes):                COCONUT (20-100 nodes):
#   SENT:      max=121  → 128          SENT:      max=433  → 512
#   HSENT_MC:  max=358  → 384          HSENT_MC:  max=1337 → 1536
#   HSENT_SC:  max=375  → 384          HSENT_SC:  max=1413 → 1536
#   HSENT_HAC: max=474  → 512          HSENT_HAC: max=1736 → 2048
#   HDT_MC:    max=232  → 256          HDT_MC:    max=868  → 1024
#   HDT_SC:    max=234  → 256          HDT_SC:    max=840  → 1024
#   HDT_HAC:   max=272  → 384          HDT_HAC:   max=1010 → 1280
#   HDTC:      max=308  → 384          HDTC:      max=1180 → 1536
get_max_length() {
    local dataset="$1"
    local tok="$2"
    local coarse="$3"

    if [ "$dataset" = "coconut" ]; then
        case "${tok}_${coarse}" in
            sent_none)  echo 512  ;;
            hsent_mc)   echo 1536 ;;
            hsent_sc)   echo 1536 ;;
            hsent_hac)  echo 2048 ;;
            hdt_mc)     echo 1024 ;;
            hdt_sc)     echo 1024 ;;
            hdt_hac)    echo 1280 ;;
            hdtc_none)  echo 1536 ;;
            *)          echo 2048 ;;
        esac
    else
        case "${tok}_${coarse}" in
            sent_none)  echo 128  ;;
            hsent_mc)   echo 384  ;;
            hsent_sc)   echo 384  ;;
            hsent_hac)  echo 512  ;;
            hdt_mc)     echo 256  ;;
            hdt_sc)     echo 256  ;;
            hdt_hac)    echo 384  ;;
            hdtc_none)  echo 384  ;;
            *)          echo 1024 ;;
        esac
    fi
}

# Precompute tokenized data caches so DDP processes load pre-tokenized data.
# The preprocess script skips splits whose cache files already exist.
echo "========================================"
echo "Ensuring tokenized data caches exist"
echo "========================================"

# Deduplicate tokenizer configs (e.g., sent:none appears once regardless of
# how many training configs use it). Different coarsening strategies need
# separate cache files due to different config hashes.
SEEN_CONFIGS=""
for tok_config in "${TOKENIZERS[@]}"; do
    # Skip duplicates (e.g., if sent:none appears twice in different filter configs)
    if echo "$SEEN_CONFIGS" | grep -qF "$tok_config"; then
        continue
    fi
    SEEN_CONFIGS="$SEEN_CONFIGS $tok_config"

    IFS=':' read -r TOK COARSE <<< "$tok_config"
    COARSE_FULL=$(get_coarsening_name "$COARSE")
    SHORT=$(get_short_name "$TOK" "$COARSE")

    PREPROCESS_ARGS="experiment=$DATASET tokenizer=$TOK"
    if [ -n "$COARSE_FULL" ] && supports_coarsening "$TOK"; then
        PREPROCESS_ARGS="$PREPROCESS_ARGS tokenizer.coarsening_strategy=$COARSE_FULL"
    fi

    echo "  $SHORT:"
    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] python scripts/preprocess/preprocess_dataset.py $PREPROCESS_ARGS"
    else
        python scripts/preprocess/preprocess_dataset.py $PREPROCESS_ARGS 2>&1 | grep -E "already exists|✓ Cached|Saving cache|Preprocessing" | sed 's/^/    /' || true
    fi
done
echo ""

# Count total configurations
TOTAL=${#TOKENIZERS[@]}
CURRENT=0

# Train each configuration
for tok_config in "${TOKENIZERS[@]}"; do
    CURRENT=$((CURRENT + 1))
    IFS=':' read -r TOKENIZER COARSENING <<< "$tok_config"
    SHORT_NAME=$(get_short_name "$TOKENIZER" "$COARSENING")
    COARSENING_FULL=$(get_coarsening_name "$COARSENING")

    RUN_NAME="${DATASET}_${SHORT_NAME}"

    # Check for existing last.ckpt only. last.ckpt reflects furthest training progress;
    # best.ckpt may lag behind (saved on val loss improvement, not at end of training).
    # If only best.ckpt exists without last.ckpt, training was interrupted before the
    # final save — train.py's resume logic will pick it up automatically.
    EXISTING_DIR=""
    RESUME_CKPT=""
    LAST_CKPT=$(find "$OUTPUT_DIR" -path "*/${RUN_NAME}_*/last.ckpt" -type f 2>/dev/null | sort -r | head -1)

    if [ -n "$LAST_CKPT" ]; then
        EXISTING_DIR=$(dirname "$LAST_CKPT")
    else
        # No last.ckpt — check if best.ckpt exists (interrupted run, always resume)
        BEST_CKPT=$(find "$OUTPUT_DIR" -path "*/${RUN_NAME}_*/best.ckpt" -type f 2>/dev/null | sort -r | head -1)
        if [ -n "$BEST_CKPT" ]; then
            EXISTING_DIR=$(dirname "$BEST_CKPT")
        fi
    fi

    if [ "$FORCE" = false ] && [ -n "$EXISTING_DIR" ]; then
        if [ -n "$LAST_CKPT" ]; then
            # Extract global_step from last.ckpt to check if training is complete
            CKPT_STEP=$(python -c "
import torch
ckpt = torch.load('$LAST_CKPT', map_location='cpu', weights_only=False)
print(ckpt.get('global_step', 0))
" 2>/dev/null || echo "0")

            if [ "$CKPT_STEP" -ge "$MAX_STEPS" ]; then
                echo "========================================"
                echo "[$CURRENT/$TOTAL] SKIPPING: $SHORT_NAME (fully trained)"
                echo "  Checkpoint: $LAST_CKPT"
                echo "  Steps: $CKPT_STEP / $MAX_STEPS (complete)"
                echo "  Use --force to re-run"
                echo ""
                continue
            else
                echo "========================================"
                echo "[$CURRENT/$TOTAL] RESUMING: $SHORT_NAME ($CKPT_STEP / $MAX_STEPS steps)"
                echo "  Checkpoint: $LAST_CKPT"
                echo "  Output dir: $EXISTING_DIR"
                RESUME_CKPT="$EXISTING_DIR"
            fi
        else
            # Only best.ckpt exists — interrupted run, always resume
            echo "========================================"
            echo "[$CURRENT/$TOTAL] RESUMING: $SHORT_NAME (interrupted, no last.ckpt)"
            echo "  Checkpoint: $BEST_CKPT"
            echo "  Output dir: $EXISTING_DIR"
            RESUME_CKPT="$EXISTING_DIR"
        fi
    fi

    if [ -z "$RESUME_CKPT" ]; then
        echo "========================================"
        echo "[$CURRENT/$TOTAL] Training: $SHORT_NAME (from scratch)"
    fi
    echo "Dataset: $DATASET"
    echo "Tokenizer: $TOKENIZER"
    if [ -n "$COARSENING_FULL" ]; then
        echo "Coarsening: $COARSENING_FULL"
    fi
    echo "========================================"

    # Build Hydra arguments (common to all launch modes)
    # max_steps and LR/warmup are always set identically regardless of GPU count.
    # Only per-GPU batch_size changes to keep B_eff constant.
    ARGS="experiment=$DATASET"
    ARGS="$ARGS tokenizer=$TOKENIZER"
    ARGS="$ARGS trainer.max_steps=$MAX_STEPS"
    ARGS="$ARGS logs.run_name=$RUN_NAME"
    ARGS="$ARGS logs.base_dir=$OUTPUT_DIR"
    ARGS="$ARGS wandb.enabled=$WANDB_ENABLED"

    # Resume: point logs.path to existing dir so train.py finds the checkpoint
    if [ -n "$RESUME_CKPT" ]; then
        ARGS="$ARGS resume=true"
        ARGS="$ARGS logs.path=$RESUME_CKPT"
    fi

    # Add DDP settings — only device topology and per-GPU batch size change.
    # LR and warmup are NOT overridden; they come from the experiment config.
    if [ "$NUM_DEVICES" -gt 1 ]; then
        if [ "$USE_MIG" = true ]; then
            # MIG: 1 device per process, N "nodes" to satisfy PL validation
            ARGS="$ARGS trainer.devices=1"
            ARGS="$ARGS trainer.num_nodes=$NUM_DEVICES"
            ARGS="$ARGS trainer.strategy=ddp"
        else
            # Non-MIG: PL manages multi-device DDP internally
            ARGS="$ARGS trainer.devices=$NUM_DEVICES"
            ARGS="$ARGS trainer.strategy=ddp"
        fi
        ARGS="$ARGS data.batch_size=$DDP_BATCH_SIZE"
    fi

    # Add coarsening for hierarchical tokenizers
    if [ -n "$COARSENING_FULL" ] && supports_coarsening "$TOKENIZER"; then
        ARGS="$ARGS tokenizer.coarsening_strategy=$COARSENING_FULL"
    fi

    # Set per-tokenizer sampling max_length for position embeddings
    TOK_MAX_LENGTH=$(get_max_length "$DATASET" "$TOKENIZER" "$COARSENING")
    ARGS="$ARGS sampling.max_length=$TOK_MAX_LENGTH"

    if [ "$DRY_RUN" = true ]; then
        if [ -n "$RESUME_CKPT" ]; then
            echo "[DRY RUN] Would RESUME from $RESUME_CKPT:"
        elif [ "$USE_MIG" = true ] && [ "$NUM_DEVICES" -gt 1 ]; then
            echo "[DRY RUN] Would launch $NUM_DEVICES MIG DDP processes:"
        else
            echo "[DRY RUN] Would execute:"
        fi
        echo "  python scripts/train.py $ARGS"
    elif [ "$USE_MIG" = true ] && [ "$NUM_DEVICES" -gt 1 ]; then
        # MIG DDP: launch one process per MIG instance.
        # Each process gets exactly one MIG UUID in CUDA_VISIBLE_DEVICES.
        # We set LOCAL_RANK=0 (1 device per "node") and GROUP_RANK=i so
        # PL's TorchElasticEnvironment sees devices(1) × num_nodes(N) = WORLD_SIZE.
        echo "Running MIG DDP training with $NUM_DEVICES processes..."
        IFS=',' read -ra MIG_ARRAY <<< "$MIG_UUIDS"
        MASTER_PORT=$(( (RANDOM % 10000) + 20000 ))
        DDP_PIDS=()

        for i in $(seq 0 $((NUM_DEVICES - 1))); do
            CUDA_VISIBLE_DEVICES="${MIG_ARRAY[$i]}" \
            MASTER_ADDR=localhost \
            MASTER_PORT=$MASTER_PORT \
            WORLD_SIZE=$NUM_DEVICES \
            RANK=$i \
            LOCAL_RANK=0 \
            LOCAL_WORLD_SIZE=1 \
            GROUP_RANK=$i \
            python scripts/train.py $ARGS &
            DDP_PIDS+=($!)
            echo "  Launched rank $i (PID ${DDP_PIDS[-1]}, MIG: ${MIG_ARRAY[$i]:0:20}...)"
        done

        # Wait for all DDP processes.
        # NCCL may segfault during teardown on MIG even when training succeeds,
        # so we collect exit codes but verify success via checkpoint existence.
        DDP_EXIT_CODES=()
        for pid in "${DDP_PIDS[@]}"; do
            wait $pid 2>/dev/null
            DDP_EXIT_CODES+=($?)
        done

        # Check if training actually succeeded by looking for checkpoint
        SAVED_CKPT=$(find "$OUTPUT_DIR" -path "*/${RUN_NAME}_*/last.ckpt" -type f 2>/dev/null | sort -r | head -1)

        if [ -n "$SAVED_CKPT" ]; then
            echo "  DDP training completed. Checkpoint: $SAVED_CKPT"
            # Log any non-zero exit codes as warnings (likely NCCL teardown segfaults)
            for i in "${!DDP_EXIT_CODES[@]}"; do
                if [ "${DDP_EXIT_CODES[$i]}" -ne 0 ]; then
                    echo "  Warning: rank $i exited with code ${DDP_EXIT_CODES[$i]} (likely NCCL teardown)"
                fi
            done
        else
            echo "ERROR: MIG DDP training failed. No checkpoint found."
            echo "  Exit codes: ${DDP_EXIT_CODES[*]}"
            exit 1
        fi
    else
        echo "Running training..."
        python scripts/train.py $ARGS
    fi

    echo ""
done

echo "========================================"
echo "Done! Trained models saved to: $OUTPUT_DIR/"
echo "========================================"