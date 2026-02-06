#!/bin/bash
# Train benchmark models from scratch on MOSES or COCONUT dataset.
#
# This script trains all tokenizer variants (SENT, H-SENT, HDT, HDTC) from scratch.
# By default, uses motif_community (MC) coarsening for hierarchical tokenizers.
#
# Usage:
#   ./bash_scripts/train_benchmarks.sh              # Train on MOSES (default)
#   ./bash_scripts/train_benchmarks.sh --coconut    # Train on COCONUT
#   ./bash_scripts/train_benchmarks.sh --all-coarsening  # Include spectral clustering
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
ALL_COARSENING=false

# Tokenizers to train
# Format: "tokenizer:coarsening" where coarsening is "none", "mc", or "sc"
TOKENIZERS=(
    "sent:none"
    "hsent:mc"
    "hdt:mc"
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
        --all-coarsening)
            ALL_COARSENING=true
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
            echo "  --all-coarsening  Include spectral clustering variants (default: MC only)"
            echo "  --no-wandb        Disable WandB logging"
            echo "  --steps=N         Set max training steps (default: 500000 MOSES, 50000 COCONUT)"
            echo ""
            echo "Tokenizers trained:"
            echo "  - SENT (flat, no coarsening)"
            echo "  - H-SENT + MC (hierarchical SENT with motif community)"
            echo "  - HDT + MC (hierarchical DFS with motif community)"
            echo "  - HDTC (compositional, uses functional hierarchy)"
            echo ""
            echo "With --all-coarsening, also trains:"
            echo "  - H-SENT + SC (spectral clustering)"
            echo "  - HDT + SC (spectral clustering)"
            echo ""
            echo "Output directories:"
            echo "  MOSES:   outputs/benchmark/{dataset}_{tokenizer}_{coarsening}_n{steps}_{date}/"
            echo "  COCONUT: outputs/benchmark_coconut/{dataset}_{tokenizer}_{coarsening}_n{steps}_{date}/"
            exit 0
            ;;
    esac
done

# Add spectral clustering variants if requested
if [ "$ALL_COARSENING" = true ]; then
    TOKENIZERS+=(
        "hsent:sc"
        "hdt:sc"
    )
fi

cd "$PROJECT_ROOT"

echo "========================================"
echo "Training Benchmark Models"
echo "========================================"
echo ""
echo "Settings:"
echo "  Dataset: $DATASET"
echo "  Max steps: $MAX_STEPS"
echo "  WandB: $WANDB_ENABLED"
echo "  Output: $OUTPUT_DIR"
echo "  All coarsening: $ALL_COARSENING"
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

    # Build command
    CMD="python scripts/train.py"
    CMD="$CMD experiment=$DATASET"
    CMD="$CMD tokenizer=$TOKENIZER"
    CMD="$CMD trainer.max_steps=$MAX_STEPS"
    CMD="$CMD logs.run_name=$RUN_NAME"
    CMD="$CMD logs.base_dir=$OUTPUT_DIR"
    CMD="$CMD wandb.enabled=$WANDB_ENABLED"

    # Add coarsening for hierarchical tokenizers
    if [ -n "$COARSENING_FULL" ] && supports_coarsening "$TOKENIZER"; then
        CMD="$CMD tokenizer.coarsening_strategy=$COARSENING_FULL"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  $CMD"
    else
        echo "Running training..."
        eval $CMD
    fi

    echo ""
done

echo "========================================"
echo "Done! Trained models saved to: $OUTPUT_DIR/"
echo "========================================"
