#!/bin/bash
# Evaluate fine-tuned checkpoints and generate comparison table.
#
# This script runs eval_finetune.py on all checkpoints in outputs/finetune/,
# then generates a comparison table using compare_finetune_results.py.
#
# Usage:
#   ./bash_scripts/eval_finetune_benchmarks.sh              # Evaluate all
#   ./bash_scripts/eval_finetune_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/eval_finetune_benchmarks.sh --compare-only  # Just generate table
#   ./bash_scripts/eval_finetune_benchmarks.sh --help       # Show help

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FINETUNE_DIR="$PROJECT_ROOT/outputs/finetune"
EVAL_OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_finetune"

# Default settings
DRY_RUN=false
COMPARE_ONLY=false
NUM_SAMPLES=1000
N_REFERENCE=1000

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --compare-only)
            COMPARE_ONLY=true
            ;;
        --samples=*)
            NUM_SAMPLES="${arg#*=}"
            ;;
        --reference=*)
            N_REFERENCE="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Evaluate fine-tuned models and generate comparison table."
            echo ""
            echo "Options:"
            echo "  --dry-run       Show what would be run without executing"
            echo "  --compare-only  Skip evaluation, just generate comparison table"
            echo "  --samples=N     Number of molecules to generate (default: 1000)"
            echo "  --reference=N   Number of reference molecules (default: 1000)"
            echo ""
            echo "Input:  outputs/finetune/*/best.ckpt"
            echo "Output: outputs/eval_finetune/*/evaluation_results.json"
            echo "        outputs/finetune/comparison.png"
            exit 0
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Extract tokenizer type from checkpoint path
get_tokenizer_type() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    # Extract tokenizer from directory name
    # Check order matters: hsent before sent, hdtc before hdt
    if [[ "$dir_name" == *"_hsent_"* ]] || [[ "$dir_name" =~ _hsent[^a-z] ]]; then
        echo "hsent"
    elif [[ "$dir_name" == *"_hdtc_"* ]] || [[ "$dir_name" =~ _hdtc[^a-z] ]] || [[ "$dir_name" == *"_hdtc" ]]; then
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

    if [[ "$dir_name" == *"_mc_"* ]] || [[ "$dir_name" =~ _mc[^a-z] ]] || [[ "$dir_name" == *"_mc" ]]; then
        echo "motif_community"
    elif [[ "$dir_name" == *"_sc_"* ]] || [[ "$dir_name" =~ _sc[^a-z] ]] || [[ "$dir_name" == *"_sc" ]]; then
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
get_output_name() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")
    echo "$dir_name"
}

if [ "$COMPARE_ONLY" = false ]; then
    # Find all best.ckpt files in finetune directory
    CHECKPOINTS=$(find "$FINETUNE_DIR" -name "best.ckpt" -type f 2>/dev/null)

    if [ -z "$CHECKPOINTS" ]; then
        echo "Error: No checkpoints found in $FINETUNE_DIR"
        echo "Expected to find best.ckpt files in the finetune directory."
        echo ""
        echo "Run fine-tuning first:"
        echo "  ./bash_scripts/finetune_benchmarks.sh"
        exit 1
    fi

    echo "Found checkpoints to evaluate:"
    echo "$CHECKPOINTS" | while read -r ckpt; do echo "  - $ckpt"; done
    echo ""
    echo "Settings:"
    echo "  Samples to generate: $NUM_SAMPLES"
    echo "  Reference molecules: $N_REFERENCE"
    echo ""

    # Count total checkpoints
    TOTAL=$(echo "$CHECKPOINTS" | wc -l)
    CURRENT=0

    # Evaluate each checkpoint
    echo "$CHECKPOINTS" | while read -r ckpt; do
        CURRENT=$((CURRENT + 1))
        TOKENIZER=$(get_tokenizer_type "$ckpt")
        COARSENING=$(get_coarsening_strategy "$ckpt")
        OUTPUT_NAME=$(get_output_name "$ckpt")

        echo "========================================"
        echo "[$CURRENT/$TOTAL] Evaluating: $OUTPUT_NAME"
        echo "Checkpoint: $ckpt"
        echo "Tokenizer: $TOKENIZER"

        # Build command
        # Use sampling.max_length=1024 to match coconut fine-tuning config
        CMD="python scripts/eval_finetune.py"
        CMD="$CMD model.checkpoint_path=\"$ckpt\""
        CMD="$CMD tokenizer=$TOKENIZER"
        CMD="$CMD sampling.max_length=1024"
        CMD="$CMD generation.num_samples=$NUM_SAMPLES"
        CMD="$CMD data.n_reference=$N_REFERENCE"
        CMD="$CMD logs.base_dir=$EVAL_OUTPUT_DIR"
        CMD="$CMD logs.path=$EVAL_OUTPUT_DIR/$OUTPUT_NAME"

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
            echo "Running evaluation..."
            eval $CMD
        fi

        echo ""
    done
fi

# Generate comparison table
echo "========================================"
echo "Generating comparison table..."
echo "========================================"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute:"
    echo "  python scripts/compare_finetune_results.py --finetune-dir $EVAL_OUTPUT_DIR"
else
    python scripts/compare_finetune_results.py --finetune-dir "$EVAL_OUTPUT_DIR" --output "$FINETUNE_DIR/comparison.png"
fi

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo "Results saved to:"
echo "  - Evaluation results:  $EVAL_OUTPUT_DIR/"
echo "  - Comparison table:    $FINETUNE_DIR/comparison.png"
echo "========================================"
