#!/bin/bash
# Train benchmark models from scratch on MOSES or COCONUT dataset.
#
# This script trains all tokenizer variants (SENT, H-SENT, HDT, HDTC) from scratch.
# By default, trains all coarsening variants (MC, SC, HAC) for hierarchical tokenizers.
#
# Usage:
#   ./bash_scripts/train_benchmarks.sh              # Train on MOSES (default)
#   ./bash_scripts/train_benchmarks.sh --coconut    # Train on COCONUT
#   ./bash_scripts/train_benchmarks.sh --ddp        # Train with DDP on 4 GPUs
#   ./bash_scripts/train_benchmarks.sh --devices=2  # Train with DDP on 2 GPUs
#   ./bash_scripts/train_benchmarks.sh --skip-sc-hac  # Only train MC variants
#   ./bash_scripts/train_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/train_benchmarks.sh --help       # Show help
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
            NUM_DEVICES=4  # Default to 4 GPUs
            ;;
        --devices=*)
            NUM_DEVICES="${arg#*=}"
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
            echo "  --force           Re-run even if output best.ckpt already exists"
            echo "  --coconut         Train on COCONUT instead of MOSES"
            echo "  --ddp             Enable DDP multi-GPU training (default: 4 GPUs)"
            echo "  --devices=N       Set number of GPUs (implies DDP when N > 1)"
            echo "  --skip-sc-hac     Only train MC variants (skip SC and HAC coarsening)"
            echo "  --no-wandb        Disable WandB logging"
            echo "  --steps=N         Set max training steps before DDP scaling (default: 500000 MOSES, 50000 COCONUT)"
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

# DDP scaling: adjust hyperparameters for multi-GPU training
# Per-GPU batch is reduced to 16 to fit MIG instances (~12GB VRAM each).
# Effective batch = 16 × NUM_DEVICES. Scaling is relative to baseline batch=32:
#   scale_factor = sqrt(effective_batch / 32)
#   LR × scale_factor, warmup × scale_factor, steps × (32 / effective_batch)
DDP_BATCH_SIZE=16
SCALED_LR=""
SCALED_WARMUP=""
USE_MIG=false
MIG_UUIDS=""
if [ "$NUM_DEVICES" -gt 1 ]; then
    if [ "$DATASET" = "coconut" ]; then
        BASE_LR="1e-5"
    else
        BASE_LR="6e-4"
    fi
    BASE_WARMUP=1000
    EFFECTIVE_BATCH=$((DDP_BATCH_SIZE * NUM_DEVICES))

    ORIG_STEPS=$MAX_STEPS
    # Scale steps so total samples seen = original_steps × 32
    MAX_STEPS=$(awk "BEGIN {printf \"%d\", $MAX_STEPS * 32 / $EFFECTIVE_BATCH}")
    SCALED_LR=$(awk "BEGIN {printf \"%.2e\", $BASE_LR * sqrt($EFFECTIVE_BATCH / 32)}")
    SCALED_WARMUP=$(awk "BEGIN {printf \"%d\", $BASE_WARMUP * sqrt($EFFECTIVE_BATCH / 32)}")

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
if [ "$NUM_DEVICES" -gt 1 ]; then
    echo "  DDP: ${NUM_DEVICES} GPUs (batch=${DDP_BATCH_SIZE}/GPU)"
    if [ "$USE_MIG" = true ]; then
        echo "  MIG: detected (manual per-process launching)"
    fi
    echo "  Max steps: $ORIG_STEPS → $MAX_STEPS (×32/${EFFECTIVE_BATCH} for equivalent training)"
    echo "  LR: $BASE_LR → $SCALED_LR (×√(${EFFECTIVE_BATCH}/32))"
    echo "  Warmup: $BASE_WARMUP → $SCALED_WARMUP (×√(${EFFECTIVE_BATCH}/32))"
    echo "  Effective batch: ${EFFECTIVE_BATCH} (${DDP_BATCH_SIZE} × ${NUM_DEVICES})"
else
    echo "  Max steps: $MAX_STEPS"
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

    # Skip if already trained (best.ckpt exists in any timestamped output dir)
    EXISTING_CKPT=$(find "$OUTPUT_DIR" -path "*/${RUN_NAME}_*/best.ckpt" -type f 2>/dev/null | head -1)
    if [ "$FORCE" = false ] && [ -n "$EXISTING_CKPT" ]; then
        echo "========================================"
        echo "[$CURRENT/$TOTAL] SKIPPING: $SHORT_NAME (already trained)"
        echo "  Output exists: $EXISTING_CKPT"
        echo "  Use --force to re-run"
        echo ""
        continue
    fi

    echo "========================================"
    echo "[$CURRENT/$TOTAL] Training: $SHORT_NAME"
    echo "Dataset: $DATASET"
    echo "Tokenizer: $TOKENIZER"
    if [ -n "$COARSENING_FULL" ]; then
        echo "Coarsening: $COARSENING_FULL"
    fi
    echo "========================================"

    # Build Hydra arguments (common to all launch modes)
    ARGS="experiment=$DATASET"
    ARGS="$ARGS tokenizer=$TOKENIZER"
    ARGS="$ARGS trainer.max_steps=$MAX_STEPS"
    ARGS="$ARGS logs.run_name=$RUN_NAME"
    ARGS="$ARGS logs.base_dir=$OUTPUT_DIR"
    ARGS="$ARGS wandb.enabled=$WANDB_ENABLED"

    # Add DDP settings
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
        ARGS="$ARGS model.learning_rate=$SCALED_LR"
        ARGS="$ARGS model.warmup_steps=$SCALED_WARMUP"
    fi

    # Add coarsening for hierarchical tokenizers
    if [ -n "$COARSENING_FULL" ] && supports_coarsening "$TOKENIZER"; then
        ARGS="$ARGS tokenizer.coarsening_strategy=$COARSENING_FULL"
    fi

    if [ "$DRY_RUN" = true ]; then
        if [ "$USE_MIG" = true ] && [ "$NUM_DEVICES" -gt 1 ]; then
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
