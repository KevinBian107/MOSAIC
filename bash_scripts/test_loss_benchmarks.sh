#!/bin/bash
# Compute test loss for all benchmark checkpoints.
#
# This script runs test_loss.py on all checkpoints in outputs/benchmark/,
# computing cross-entropy loss on the held-out test split.
#
# Usage:
#   ./bash_scripts/test_loss_benchmarks.sh              # Evaluate MOSES benchmarks
#   ./bash_scripts/test_loss_benchmarks.sh --coconut    # Evaluate COCONUT benchmarks
#   ./bash_scripts/test_loss_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/test_loss_benchmarks.sh --help       # Show help
#
# Output:
#   outputs/test_loss/{dataset}_{tokenizer}_test_loss_{timestamp}/results.json

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
DATASET="moses"
BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark"
TEST_LOSS_OUTPUT_DIR="outputs/test_loss"
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --coconut)
            DATASET="coconut"
            BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark_coconut"
            TEST_LOSS_OUTPUT_DIR="outputs/test_loss_coconut"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Compute test loss for all benchmark checkpoints."
            echo ""
            echo "Options:"
            echo "  --dry-run     Show commands without executing"
            echo "  --coconut     Evaluate COCONUT benchmarks (from outputs/benchmark_coconut/)"
            echo ""
            echo "Results are saved to:"
            echo "  MOSES:   outputs/test_loss/"
            echo "  COCONUT: outputs/test_loss_coconut/"
            exit 0
            ;;
    esac
done

# Find all last.ckpt files in benchmark directory
CHECKPOINTS=$(find "$BENCHMARK_DIR" -name "last.ckpt" -type f 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in $BENCHMARK_DIR"
    echo "Expected to find last.ckpt files in the benchmark directory."
    exit 1
fi

echo "========================================"
echo "Test Loss Evaluation"
echo "========================================"
echo ""
echo "Dataset: $DATASET"
echo "Benchmark dir: $BENCHMARK_DIR"
echo "Output dir: $TEST_LOSS_OUTPUT_DIR"
echo ""
echo "Found checkpoints:"
echo "$CHECKPOINTS" | while read -r ckpt; do echo "  - $ckpt"; done
echo ""

cd "$PROJECT_ROOT"

# Extract tokenizer type from checkpoint path
get_tokenizer_type() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    if [[ "$dir_name" == *"_hsent_"* ]] || [[ "$dir_name" =~ _hsent[^a-z] ]]; then
        echo "hsent"
    elif [[ "$dir_name" == *"_hdtc_"* ]] || [[ "$dir_name" =~ _hdtc[^a-z] ]]; then
        echo "hdtc"
    elif [[ "$dir_name" == *"_hdt_"* ]] || [[ "$dir_name" =~ _hdt[^a-z] ]]; then
        echo "hdt"
    elif [[ "$dir_name" == *"_sent_"* ]] || [[ "$dir_name" =~ _sent[^a-z] ]]; then
        echo "sent"
    else
        echo "hsent"  # Default fallback
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
    elif [[ "$dir_name" == *"_hac_"* ]] || [[ "$dir_name" =~ _hac[^a-z] ]]; then
        echo "hac"
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

# Evaluate each checkpoint
echo "$CHECKPOINTS" | while read -r ckpt; do
    TOKENIZER=$(get_tokenizer_type "$ckpt")
    COARSENING=$(get_coarsening_strategy "$ckpt")

    echo "========================================"
    echo "Computing test loss: $ckpt"
    echo "Tokenizer: $TOKENIZER"

    # Build coarsening args (only for tokenizers that support it)
    COARSENING_ARGS=""
    if supports_coarsening "$TOKENIZER"; then
        echo "Coarsening: $COARSENING"
        COARSENING_ARGS="tokenizer.coarsening_strategy=$COARSENING"
    fi
    echo "========================================"

    CMD="python scripts/test_loss.py"
    CMD="$CMD model.checkpoint_path=$ckpt"
    CMD="$CMD tokenizer=$TOKENIZER"
    CMD="$CMD experiment=$DATASET"
    CMD="$CMD logs.base_dir=$TEST_LOSS_OUTPUT_DIR"
    CMD="$CMD $COARSENING_ARGS"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  $CMD"
    else
        eval $CMD
    fi

    echo ""
done

echo "========================================"
echo "Done! Test loss results saved to: $TEST_LOSS_OUTPUT_DIR/"
echo "========================================"
