#!/bin/bash
# Learning rate sweep for SC/HAC coarsening strategies.
#
# Trains HSENT and HDT with SC and/or HAC coarsening at 4 learning rates
# (6e-4, 1e-3, 2e-3, 4e-3) using task parallelism on 4 MIG GPUs.
# Each GPU trains one model independently (no DDP).
# After training, evaluates all models and generates comparison tables.
#
# Usage:
#   ./bash_scripts/train/train_lr_sweep.sh                    # Both SC+HAC on COCONUT
#   ./bash_scripts/train/train_lr_sweep.sh --sc --coconut     # SC only on COCONUT
#   ./bash_scripts/train/train_lr_sweep.sh --hac --moses      # HAC only on MOSES
#   ./bash_scripts/train/train_lr_sweep.sh --dry-run          # Show commands only
#   ./bash_scripts/train/train_lr_sweep.sh --eval-only        # Skip training, eval existing
#   ./bash_scripts/train/train_lr_sweep.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
DATASET="coconut"
COARSENING_MODE="both"  # "sc", "hac", or "both"
DRY_RUN=false
WANDB_ENABLED=true
EVAL_ONLY=false
NO_EVAL=false
FORCE=false
MAX_STEPS=25000
GRAD_ACCUM=2

# Sweep grid
TOKENIZERS=("hsent" "hdt")
LEARNING_RATES=("6e-4" "1e-3" "2e-3" "4e-3")

# Parse arguments
for arg in "$@"; do
    case $arg in
        --sc)
            COARSENING_MODE="sc"
            ;;
        --hac)
            COARSENING_MODE="hac"
            ;;
        --both)
            COARSENING_MODE="both"
            ;;
        --coconut)
            DATASET="coconut"
            ;;
        --moses)
            DATASET="moses"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --no-wandb)
            WANDB_ENABLED=false
            ;;
        --eval-only)
            EVAL_ONLY=true
            ;;
        --no-eval)
            NO_EVAL=true
            ;;
        --force)
            FORCE=true
            ;;
        --steps=*)
            MAX_STEPS="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Learning rate sweep for SC/HAC coarsening strategies."
            echo "Trains HSENT and HDT at 4 learning rates using MIG GPU parallelism."
            echo ""
            echo "Coarsening selection:"
            echo "  --sc              Only SC (spectral clustering)"
            echo "  --hac             Only HAC (hierarchical agglomerative clustering)"
            echo "  --both            Both SC and HAC (default)"
            echo ""
            echo "Dataset selection:"
            echo "  --coconut         Train on COCONUT (default)"
            echo "  --moses           Train on MOSES"
            echo ""
            echo "Execution options:"
            echo "  --dry-run         Show commands without executing"
            echo "  --no-wandb        Disable WandB logging"
            echo "  --eval-only       Skip training, only evaluate existing checkpoints"
            echo "  --no-eval         Skip evaluation after training"
            echo "  --force           Re-run even if checkpoints exist"
            echo "  --steps=N         Optimizer steps (default: 25000, with accum=2 -> 50K equivalent)"
            echo ""
            echo "Sweep grid: 2 tokenizers x 4 LRs = 8 models per coarsening type"
            echo "  Tokenizers: HSENT, HDT"
            echo "  LRs: 6e-4, 1e-3, 2e-3, 4e-3"
            echo ""
            echo "Output:"
            echo "  Training:   outputs/lr_sweep/{dataset}_{tok}_{coarse}_lr{lr}_.../"
            echo "  Evaluation: outputs/lr_sweep_eval/{dataset}_{coarse}/"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

OUTPUT_BASE="$PROJECT_ROOT/outputs/lr_sweep"
EVAL_BASE="$PROJECT_ROOT/outputs/lr_sweep_eval"

# Build coarsening list
COARSENINGS=()
case $COARSENING_MODE in
    sc)   COARSENINGS=("sc") ;;
    hac)  COARSENINGS=("hac") ;;
    both) COARSENINGS=("sc" "hac") ;;
esac

# Get coarsening strategy full name
get_coarsening_name() {
    case $1 in
        sc) echo "spectral" ;;
        hac) echo "hac" ;;
    esac
}

# Get recommended sampling.max_length per tokenizer/dataset/coarsening.
get_max_length() {
    local dataset="$1"
    local tok="$2"
    local coarse="$3"

    if [ "$dataset" = "coconut" ]; then
        case "${tok}_${coarse}" in
            hsent_sc)   echo 1536 ;;
            hsent_hac)  echo 2048 ;;
            hdt_sc)     echo 1024 ;;
            hdt_hac)    echo 1280 ;;
            *)          echo 2048 ;;
        esac
    else
        case "${tok}_${coarse}" in
            hsent_sc)   echo 384  ;;
            hsent_hac)  echo 512  ;;
            hdt_sc)     echo 256  ;;
            hdt_hac)    echo 384  ;;
            *)          echo 1024 ;;
        esac
    fi
}

# Detect MIG GPUs
detect_mig() {
    MIG_UUIDS=$(nvidia-smi -L 2>/dev/null | grep -oP 'MIG-[0-9a-f-]+' | paste -sd, 2>/dev/null || true)
    MIG_COUNT=$(echo "$MIG_UUIDS" | tr ',' '\n' | grep -c 'MIG' 2>/dev/null || echo 0)
    if [ "$MIG_COUNT" -ge 4 ]; then
        USE_MIG=true
        IFS=',' read -ra MIG_ARRAY <<< "$MIG_UUIDS"
        echo "Detected $MIG_COUNT MIG instances — will run 4 models in parallel"
    else
        USE_MIG=false
        MIG_ARRAY=()
        echo "No MIG GPUs detected (found $MIG_COUNT) — will run sequentially"
    fi
}

# Find best.ckpt for a given run name pattern (handles Hydra timestamp suffix)
find_checkpoint() {
    local run_name="$1"
    find "$OUTPUT_BASE" -path "*/${run_name}_*/best.ckpt" -type f 2>/dev/null | sort -r | head -1
}

# Print header
echo "========================================"
echo "LR Sweep: ${COARSENING_MODE^^} Coarsening"
echo "========================================"
echo ""
echo "Settings:"
echo "  Dataset:      $DATASET"
echo "  Coarsening:   $COARSENING_MODE"
echo "  Steps:        $MAX_STEPS (accum=$GRAD_ACCUM → ${GRAD_ACCUM}x equivalent base steps)"
echo "  Batch size:   32 (8 for HSENT on COCONUT)"
echo "  Warmup:       1000"
echo "  WandB:        $WANDB_ENABLED"
echo "  Eval only:    $EVAL_ONLY"
echo "  No eval:      $NO_EVAL"
echo "  Force:        $FORCE"
echo ""
echo "Sweep grid:"
for tok in "${TOKENIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "  ${tok^^} lr=$lr"
    done
done
echo ""

# Detect MIG
detect_mig
echo ""

# ============================================================
# Preprocess caches
# ============================================================
if [ "$EVAL_ONLY" = false ]; then
    echo "========================================"
    echo "Ensuring tokenized data caches exist"
    echo "========================================"

    SEEN_PREPROCESS=""
    for coarse in "${COARSENINGS[@]}"; do
        COARSE_FULL=$(get_coarsening_name "$coarse")
        for tok in "${TOKENIZERS[@]}"; do
            COMBO="${tok}:${coarse}"
            if echo "$SEEN_PREPROCESS" | grep -qF "$COMBO"; then
                continue
            fi
            SEEN_PREPROCESS="$SEEN_PREPROCESS $COMBO"

            echo "  ${tok}_${coarse}:"
            if [ "$DRY_RUN" = true ]; then
                echo "    [DRY RUN] python scripts/preprocess/preprocess_dataset.py experiment=$DATASET tokenizer=$tok tokenizer.coarsening_strategy=$COARSE_FULL"
            else
                python scripts/preprocess/preprocess_dataset.py \
                    experiment=$DATASET \
                    tokenizer=$tok \
                    tokenizer.coarsening_strategy=$COARSE_FULL \
                    2>&1 | grep -E "already exists|✓ Cached|Saving cache|Preprocessing" | sed 's/^/    /' || true
            fi
        done
    done
    echo ""
fi

# ============================================================
# Training
# ============================================================

# Train a batch of 4 models in parallel on MIG GPUs (or sequentially)
# Arguments: array of "tok:coarse:lr" specs
train_batch() {
    local specs=("$@")
    local pids=()

    for i in "${!specs[@]}"; do
        IFS=':' read -r tok coarse lr <<< "${specs[$i]}"
        COARSE_FULL=$(get_coarsening_name "$coarse")
        TOK_MAX_LENGTH=$(get_max_length "$DATASET" "$tok" "$coarse")
        RUN_NAME="${DATASET}_${tok}_${coarse}_lr${lr}"

        # Skip if checkpoint exists
        if [ "$FORCE" = false ]; then
            EXISTING_CKPT=$(find_checkpoint "$RUN_NAME")
            if [ -n "$EXISTING_CKPT" ]; then
                echo "  SKIP $RUN_NAME — checkpoint exists: $EXISTING_CKPT"
                continue
            fi
        fi

        # Reduce batch size for HSENT on COCONUT (long sequences OOM on 12GB MIG)
        local batch_size=32
        local grad_accum=$GRAD_ACCUM
        if [ "$DATASET" = "coconut" ] && [ "$tok" = "hsent" ]; then
            batch_size=8
            grad_accum=8  # Keep effective batch = 8 * 8 = 64 ≈ 32 * 2
        fi

        ARGS="experiment=$DATASET"
        ARGS="$ARGS tokenizer=$tok"
        ARGS="$ARGS tokenizer.coarsening_strategy=$COARSE_FULL"
        ARGS="$ARGS trainer.max_steps=$MAX_STEPS"
        ARGS="$ARGS trainer.accumulate_grad_batches=$grad_accum"
        ARGS="$ARGS data.batch_size=$batch_size"
        ARGS="$ARGS model.learning_rate=$lr"
        ARGS="$ARGS model.warmup_steps=1000"
        ARGS="$ARGS sampling.max_length=$TOK_MAX_LENGTH"
        ARGS="$ARGS logs.run_name=$RUN_NAME"
        ARGS="$ARGS logs.base_dir=$OUTPUT_BASE"
        ARGS="$ARGS wandb.enabled=$WANDB_ENABLED"

        if [ "$DRY_RUN" = true ]; then
            if [ "$USE_MIG" = true ]; then
                echo "  [DRY RUN] CUDA_VISIBLE_DEVICES=${MIG_ARRAY[$i]:0:20}... python scripts/train.py $ARGS"
            else
                echo "  [DRY RUN] python scripts/train.py $ARGS"
            fi
        elif [ "$USE_MIG" = true ]; then
            echo "  Launching $RUN_NAME on MIG ${MIG_ARRAY[$i]:0:20}..."
            CUDA_VISIBLE_DEVICES="${MIG_ARRAY[$i]}" python scripts/train.py $ARGS &
            pids+=($!)
        else
            echo "  Training $RUN_NAME (sequential)..."
            python scripts/train.py $ARGS
        fi
    done

    # Wait for parallel jobs
    if [ "$USE_MIG" = true ] && [ ${#pids[@]} -gt 0 ] && [ "$DRY_RUN" = false ]; then
        echo "  Waiting for ${#pids[@]} parallel jobs..."
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait $pid 2>/dev/null; then
                echo "  WARNING: PID $pid exited with non-zero status"
                failed=$((failed + 1))
            fi
        done
        if [ $failed -gt 0 ]; then
            echo "  WARNING: $failed job(s) may have failed — check logs"
        else
            echo "  All ${#pids[@]} jobs completed successfully"
        fi
    fi
}

if [ "$EVAL_ONLY" = false ]; then
    for coarse in "${COARSENINGS[@]}"; do
        echo "========================================"
        echo "Training: ${coarse^^} coarsening"
        echo "========================================"

        for tok in "${TOKENIZERS[@]}"; do
            echo ""
            echo "--- ${tok^^} at 4 learning rates ---"
            BATCH_SPECS=()
            for lr in "${LEARNING_RATES[@]}"; do
                BATCH_SPECS+=("${tok}:${coarse}:${lr}")
            done
            train_batch "${BATCH_SPECS[@]}"
        done
        echo ""
    done
fi

# ============================================================
# Evaluation
# ============================================================
if [ "$NO_EVAL" = true ]; then
    echo "Skipping evaluation (--no-eval)"
    echo ""
else
    for coarse in "${COARSENINGS[@]}"; do
        EVAL_DIR="$EVAL_BASE/${DATASET}_${coarse}"
        mkdir -p "$EVAL_DIR"

        echo "========================================"
        echo "Evaluating: ${coarse^^} coarsening"
        echo "========================================"

        RUN_NAMES=()
        COLUMN_LABELS=()

        for tok in "${TOKENIZERS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                RUN_NAME="${DATASET}_${tok}_${coarse}_lr${lr}"
                CKPT=$(find_checkpoint "$RUN_NAME")

                if [ -z "$CKPT" ]; then
                    echo "  SKIP $RUN_NAME — no checkpoint found"
                    continue
                fi

                COARSE_FULL=$(get_coarsening_name "$coarse")
                EVAL_RUN_DIR="$EVAL_DIR/$RUN_NAME"

                # Skip if already evaluated
                if [ "$FORCE" = false ] && [ -f "$EVAL_RUN_DIR/results.json" ]; then
                    echo "  SKIP $RUN_NAME — already evaluated"
                else
                    echo "  Evaluating $RUN_NAME..."
                    if [ "$DRY_RUN" = true ]; then
                        echo "    [DRY RUN] python scripts/test.py model.checkpoint_path=$CKPT tokenizer=$tok tokenizer.coarsening_strategy=$COARSE_FULL experiment=$DATASET logs.path=$EVAL_RUN_DIR"
                    else
                        CUDA_VISIBLE_DEVICES="${MIG_ARRAY[0]:-0}" python scripts/test.py \
                            model.checkpoint_path="$CKPT" \
                            tokenizer=$tok \
                            tokenizer.coarsening_strategy=$COARSE_FULL \
                            experiment=$DATASET \
                            logs.path="$EVAL_RUN_DIR" || {
                            echo "    WARNING: Evaluation failed for $RUN_NAME"
                            continue
                        }
                    fi
                fi

                RUN_NAMES+=("$RUN_NAME")
                COLUMN_LABELS+=("${tok^^} lr=$lr")
            done
        done

        # Generate comparison table
        if [ ${#RUN_NAMES[@]} -gt 0 ]; then
            echo ""
            echo "  Generating comparison table..."
            LABELS_STR=$(IFS=','; echo "${COLUMN_LABELS[*]}")

            if [ "$DRY_RUN" = true ]; then
                echo "    [DRY RUN] python scripts/comparison/compare_results.py \\"
                echo "      --test-dir $EVAL_DIR \\"
                echo "      --output $EVAL_DIR/comparison_${coarse}.png \\"
                echo "      --all --test-only \\"
                echo "      --runs ${RUN_NAMES[*]} \\"
                echo "      --column-labels \"$LABELS_STR\" \\"
                echo "      --title \"LR Sweep: ${coarse^^} Coarsening ($DATASET)\""
            else
                python scripts/comparison/compare_results.py \
                    --test-dir "$EVAL_DIR" \
                    --output "$EVAL_DIR/comparison_${coarse}.png" \
                    --all --test-only \
                    --runs "${RUN_NAMES[@]}" \
                    --column-labels "$LABELS_STR" \
                    --title "LR Sweep: ${coarse^^} Coarsening ($DATASET)" || {
                    echo "    WARNING: Comparison table generation failed"
                }
            fi
        fi
        echo ""
    done
fi

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "LR Sweep Complete"
echo "========================================"
echo ""
echo "Training outputs:   $OUTPUT_BASE/"
echo "Evaluation outputs: $EVAL_BASE/"
echo ""

for coarse in "${COARSENINGS[@]}"; do
    echo "${coarse^^} coarsening:"
    for tok in "${TOKENIZERS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            RUN_NAME="${DATASET}_${tok}_${coarse}_lr${lr}"
            CKPT=$(find_checkpoint "$RUN_NAME")
            EVAL_RESULT="$EVAL_BASE/${DATASET}_${coarse}/$RUN_NAME/results.json"
            TRAIN_STATUS="no ckpt"
            EVAL_STATUS="not evaluated"
            if [ -n "$CKPT" ]; then
                TRAIN_STATUS="trained"
            fi
            if [ -f "$EVAL_RESULT" ]; then
                EVAL_STATUS="evaluated"
            fi
            printf "  %-40s  train: %-10s  eval: %s\n" "$RUN_NAME" "$TRAIN_STATUS" "$EVAL_STATUS"
        done
    done
    TABLE="$EVAL_BASE/${DATASET}_${coarse}/comparison_${coarse}.png"
    if [ -f "$TABLE" ]; then
        echo "  Comparison table: $TABLE"
    fi
    echo ""
done
