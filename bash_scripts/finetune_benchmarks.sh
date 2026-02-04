#!/bin/bash
# Fine-tune benchmark checkpoints on COCONUT dataset.
#
# This script fine-tunes all pretrained models in outputs/benchmark/
# on the COCONUT complex natural products dataset.
#
# Usage:
#   ./bash_scripts/finetune_benchmarks.sh              # Fine-tune all benchmarks
#   ./bash_scripts/finetune_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/finetune_benchmarks.sh --help       # Show help

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark"
OUTPUT_DIR="$PROJECT_ROOT/outputs/finetune"

# Default settings
MAX_STEPS=50000
WANDB_ENABLED=true
DRY_RUN=false
FORCE=false
SKIP_SC=false

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
        --skip-sc)
            SKIP_SC=true
            ;;
        --steps=*)
            MAX_STEPS="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Fine-tune all benchmark models on COCONUT dataset."
            echo ""
            echo "Options:"
            echo "  --dry-run      Show what would be run without executing"
            echo "  --force        Re-run even if output best.ckpt already exists"
            echo "  --skip-sc      Skip spectral clustering checkpoints"
            echo "  --no-wandb     Disable WandB logging"
            echo "  --steps=N      Set max training steps (default: 50000)"
            echo ""
            echo "Results are saved to outputs/finetune/{model_name}/"
            exit 0
            ;;
    esac
done

# Find checkpoints in benchmark directory (prefer best.ckpt, fall back to last.ckpt)
CHECKPOINTS=""
for dir in "$BENCHMARK_DIR"/*/; do
    if [ -f "$dir/best.ckpt" ]; then
        CHECKPOINTS="${CHECKPOINTS}${dir}best.ckpt"$'\n'
    elif [ -f "$dir/last.ckpt" ]; then
        CHECKPOINTS="${CHECKPOINTS}${dir}last.ckpt"$'\n'
    fi
done
CHECKPOINTS=$(echo "$CHECKPOINTS" | sed '/^$/d')

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in $BENCHMARK_DIR"
    echo "Expected to find best.ckpt files in the benchmark directory."
    exit 1
fi

echo "Found checkpoints to fine-tune:"
echo "$CHECKPOINTS" | while read -r ckpt; do echo "  - $ckpt"; done
echo ""
echo "Settings:"
echo "  Max steps: $MAX_STEPS"
echo "  WandB: $WANDB_ENABLED"
echo ""

cd "$PROJECT_ROOT"

# Extract tokenizer type from checkpoint path
# Directory names follow pattern: moses_{tokenizer}_{variant}_... or moses_{tokenizer}_n{count}_...
get_tokenizer_type() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    # Extract tokenizer from directory name
    # Check order matters: hsent before sent, hdtc before hdt
    if [[ "$dir_name" == *"_hsent_"* ]] || [[ "$dir_name" =~ _hsent[^a-z] ]]; then
        echo "hsent"
    elif [[ "$dir_name" == *"_hdtc_"* ]] || [[ "$dir_name" =~ _hdtc[^a-z] ]]; then
        echo "hdtc"
    elif [[ "$dir_name" == *"_hdt_"* ]] || [[ "$dir_name" =~ _hdt[^a-z] ]]; then
        echo "hdt"
    elif [[ "$dir_name" == *"_sent_"* ]] || [[ "$dir_name" =~ _sent[^a-z] ]] || [[ "$dir_name" == *"_sent" ]]; then
        echo "sent"
    else
        echo "hdtc"  # Default fallback
    fi
}

# Extract coarsening strategy from checkpoint path
get_coarsening_strategy() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    if [[ "$dir_name" == *"_mc_"* ]] || [[ "$dir_name" =~ _mc[^a-z] ]]; then
        echo "motif_community"
    elif [[ "$dir_name" == *"_sc_"* ]] || [[ "$dir_name" =~ _sc[^a-z] ]]; then
        echo "spectral"
    elif [[ "$dir_name" == *"_mas_"* ]] || [[ "$dir_name" =~ _mas[^a-z] ]]; then
        echo "motif_aware_spectral"
    else
        echo "spectral"  # Default fallback
    fi
}

# Check if tokenizer supports coarsening strategy
supports_coarsening() {
    local tokenizer="$1"
    if [[ "$tokenizer" == "hsent" ]] || [[ "$tokenizer" == "hdt" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Get short name for output directory
get_short_name() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")
    # Extract key parts: tokenizer and coarsening
    echo "$dir_name" | sed 's/moses_//' | sed 's/_n[0-9]*_.*//'
}

# Count total checkpoints
TOTAL=$(echo "$CHECKPOINTS" | wc -l)
CURRENT=0

# Fine-tune each checkpoint
echo "$CHECKPOINTS" | while read -r ckpt; do
    CURRENT=$((CURRENT + 1))
    TOKENIZER=$(get_tokenizer_type "$ckpt")
    COARSENING=$(get_coarsening_strategy "$ckpt")
    SHORT_NAME=$(get_short_name "$ckpt")

    OUTPUT_NAME="coconut_${SHORT_NAME}"

    # Skip spectral clustering checkpoints if requested (only for tokenizers that use coarsening)
    if [ "$SKIP_SC" = true ] && [ "$COARSENING" = "spectral" ] && supports_coarsening "$TOKENIZER"; then
        echo "========================================"
        echo "[$CURRENT/$TOTAL] SKIPPING: $SHORT_NAME (spectral clustering, --skip-sc)"
        echo ""
        continue
    fi

    # Skip if already fine-tuned (best.ckpt exists in any timestamped output dir)
    # Hydra creates dirs like: outputs/finetune/coconut_hdtc_mc_20260203-142530/
    EXISTING_CKPT=$(find "$OUTPUT_DIR" -path "*/${OUTPUT_NAME}_*/best.ckpt" -type f 2>/dev/null | head -1)
    if [ "$FORCE" = false ] && [ -n "$EXISTING_CKPT" ]; then
        echo "========================================"
        echo "[$CURRENT/$TOTAL] SKIPPING: $SHORT_NAME (already fine-tuned)"
        echo "  Output exists: $EXISTING_CKPT"
        echo "  Use --force to re-run"
        echo ""
        continue
    fi

    echo "========================================"
    echo "[$CURRENT/$TOTAL] Fine-tuning: $SHORT_NAME"
    echo "Checkpoint: $ckpt"
    echo "Tokenizer: $TOKENIZER"

    # Build command
    CMD="python scripts/finetune.py"
    CMD="$CMD model.pretrained_path=\"$ckpt\""
    CMD="$CMD experiment=coconut"
    CMD="$CMD tokenizer=$TOKENIZER"
    CMD="$CMD trainer.max_steps=$MAX_STEPS"
    CMD="$CMD logs.run_name=$OUTPUT_NAME"
    CMD="$CMD logs.base_dir=$OUTPUT_DIR"
    CMD="$CMD wandb.enabled=$WANDB_ENABLED"

    # Add coarsening args for tokenizers that support it
    if supports_coarsening "$TOKENIZER"; then
        echo "Coarsening: $COARSENING"
        CMD="$CMD tokenizer.coarsening_strategy=$COARSENING"
    fi

    echo "========================================"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  $CMD"
    else
        echo "Running fine-tuning..."
        eval $CMD
    fi

    echo ""
done

echo "========================================"
echo "Done! Fine-tuned models saved to: $OUTPUT_DIR/"
echo "========================================"
