#!/bin/bash
# Evaluate benchmark checkpoints and generate comparison table.
#
# This script runs test.py and realistic_gen.py on all checkpoints
# in outputs/benchmark/, then generates a comparison table.
#
# Usage:
#   ./bash_scripts/eval_benchmarks.sh              # Evaluate all benchmarks
#   ./bash_scripts/eval_benchmarks.sh --test-only  # Skip realistic_gen
#   ./bash_scripts/eval_benchmarks.sh --gen-only   # Skip test, only realistic_gen

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark"

# Parse arguments
RUN_TEST=true
RUN_GEN=true

for arg in "$@"; do
    case $arg in
        --test-only)
            RUN_GEN=false
            ;;
        --gen-only)
            RUN_TEST=false
            ;;
        --help|-h)
            echo "Usage: $0 [--test-only] [--gen-only]"
            echo ""
            echo "Options:"
            echo "  --test-only   Only run test.py (skip realistic_gen.py)"
            echo "  --gen-only    Only run realistic_gen.py (skip test.py)"
            echo ""
            echo "Results are saved to outputs/test/ and outputs/realistic_gen/"
            echo "Comparison table is saved to outputs/test/comparison.png"
            exit 0
            ;;
    esac
done

# Find all best.ckpt files in benchmark directory
CHECKPOINTS=$(find "$BENCHMARK_DIR" -name "best.ckpt" -type f 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in $BENCHMARK_DIR"
    echo "Expected to find best.ckpt files in the benchmark directory."
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS" | while read -r ckpt; do echo "  - $ckpt"; done
echo ""

cd "$PROJECT_ROOT"

# Extract tokenizer type from checkpoint path
# Directory names follow pattern: moses_{tokenizer}_{variant}_...
get_tokenizer_type() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    # Extract tokenizer from directory name (e.g., moses_hsent_mc_... -> hsent)
    if [[ "$dir_name" == *"_hsent_"* ]]; then
        echo "hsent"
    elif [[ "$dir_name" == *"_hdtc_"* ]]; then
        echo "hdtc"
    elif [[ "$dir_name" == *"_hdt_"* ]]; then
        echo "hdt"
    elif [[ "$dir_name" == *"_sent_"* ]]; then
        echo "sent"
    else
        echo "hsent"  # Default fallback
    fi
}

# Extract coarsening strategy from checkpoint path
# Directory names follow pattern: moses_{tokenizer}_{coarsening}_...
# mc = motif_community, sc = spectral, mas = motif_aware_spectral
get_coarsening_strategy() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    # Extract coarsening strategy from directory name
    if [[ "$dir_name" == *"_mc_"* ]] || [[ "$dir_name" == *"_mc-"* ]]; then
        echo "motif_community"
    elif [[ "$dir_name" == *"_sc_"* ]] || [[ "$dir_name" == *"_sc-"* ]]; then
        echo "spectral"
    elif [[ "$dir_name" == *"_mas_"* ]] || [[ "$dir_name" == *"_mas-"* ]]; then
        echo "motif_aware_spectral"
    else
        echo "spectral"  # Default fallback
    fi
}

# Check if tokenizer supports coarsening strategy
supports_coarsening() {
    local tokenizer="$1"
    # Only hsent and hdt support coarsening strategies
    # sent and hdtc do not (hdtc uses FunctionalHierarchyBuilder)
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
    echo "Evaluating: $ckpt"
    echo "Tokenizer: $TOKENIZER"

    # Build coarsening args (only for tokenizers that support it)
    COARSENING_ARGS=""
    if supports_coarsening "$TOKENIZER"; then
        echo "Coarsening: $COARSENING"
        COARSENING_ARGS="tokenizer.coarsening_strategy=$COARSENING"
    fi
    echo "========================================"

    if [ "$RUN_TEST" = true ]; then
        echo "[1/2] Running test.py..."
        python scripts/test.py model.checkpoint_path="$ckpt" tokenizer=$TOKENIZER $COARSENING_ARGS
    fi

    if [ "$RUN_GEN" = true ]; then
        echo "[2/2] Running realistic_gen.py..."
        python scripts/realistic_gen.py model.checkpoint_path="$ckpt" tokenizer=$TOKENIZER $COARSENING_ARGS
    fi

    echo ""
done

# Generate comparison table
echo "========================================"
echo "Generating comparison table..."
echo "========================================"
python scripts/compare_results.py

echo ""
echo "Done! Results saved to:"
echo "  - Test results:      outputs/test/"
echo "  - Realistic gen:     outputs/realistic_gen/"
echo "  - Comparison table:  outputs/test/comparison.png"
