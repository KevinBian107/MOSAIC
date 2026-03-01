#!/bin/bash
# Evaluate benchmark checkpoints and generate comparison table.
#
# This script runs test.py and realistic_gen.py on all checkpoints
# in outputs/benchmark/, then generates a comparison table.
#
# Usage:
#   ./bash_scripts/eval/eval_benchmarks.sh              # Evaluate MOSES benchmarks
#   ./bash_scripts/eval/eval_benchmarks.sh --coconut    # Evaluate COCONUT benchmarks
#   ./bash_scripts/eval/eval_benchmarks.sh --full-ref   # Use train+test as reference population
#   ./bash_scripts/eval/eval_benchmarks.sh --test-only  # Skip realistic_gen
#   ./bash_scripts/eval/eval_benchmarks.sh --gen-only   # Skip test, only realistic_gen
#   ./bash_scripts/eval/eval_benchmarks.sh --force      # Re-evaluate even if results exist

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
DATASET="moses"
REFERENCE_SPLIT="test"

# Parse arguments
RUN_TEST=true
RUN_GEN=true
USE_COCONUT=false
USE_FULL_REF=false
FORCE_REEVAL=false

for arg in "$@"; do
    case $arg in
        --test-only)
            RUN_GEN=false
            ;;
        --gen-only)
            RUN_TEST=false
            ;;
        --coconut)
            USE_COCONUT=true
            DATASET="coconut"
            ;;
        --full-ref)
            USE_FULL_REF=true
            REFERENCE_SPLIT="full"
            ;;
        --force)
            FORCE_REEVAL=true
            ;;
        --help|-h)
            echo "Usage: $0 [--test-only] [--gen-only] [--coconut] [--full-ref] [--force]"
            echo ""
            echo "Options:"
            echo "  --test-only   Only run test.py (skip realistic_gen.py)"
            echo "  --gen-only    Only run realistic_gen.py (skip test.py)"
            echo "  --coconut     Evaluate COCONUT benchmarks (from outputs/benchmark_coconut/)"
            echo "  --full-ref    Use train+test combined as reference population for distributional metrics"
            echo "  --force       Re-evaluate checkpoints even if results already exist"
            echo ""
            echo "Results are saved to:"
            echo "  MOSES:             outputs/test/ and outputs/realistic_gen/"
            echo "  COCONUT:           outputs/test_coconut/ and outputs/realistic_gen_coconut/"
            echo "  --full-ref:        outputs/test_fullref/ and outputs/realistic_gen_fullref/"
            echo "  COCONUT+full-ref:  outputs/test_coconut_fullref/ and outputs/realistic_gen_coconut_fullref/"
            echo "Comparison tables are saved to the test output directory as comparison.png"
            exit 0
            ;;
    esac
done

# Resolve output directories after all args are parsed
BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark"
DIR_SUFFIX=""

if [ "$USE_COCONUT" = true ]; then
    BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark_coconut"
    DIR_SUFFIX="_coconut"
fi

if [ "$USE_FULL_REF" = true ]; then
    DIR_SUFFIX="${DIR_SUFFIX}_fullref"
fi

TEST_OUTPUT_DIR="outputs/test${DIR_SUFFIX}"
REALISTIC_GEN_OUTPUT_DIR="outputs/realistic_gen${DIR_SUFFIX}"
COMPARISON_OUTPUT="${TEST_OUTPUT_DIR}/comparison.png"

# Find all last.ckpt files in benchmark directory
CHECKPOINTS=$(find "$BENCHMARK_DIR" -name "last.ckpt" -type f 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in $BENCHMARK_DIR"
    echo "Expected to find last.ckpt files in the benchmark directory."
    exit 1
fi

echo "Dataset: $DATASET"
echo "Reference split: $REFERENCE_SPLIT"
echo "Benchmark dir: $BENCHMARK_DIR"
echo ""
echo "Found checkpoints:"
echo "$CHECKPOINTS" | while read -r ckpt; do echo "  - $ckpt"; done
echo ""

cd "$PROJECT_ROOT"

# Extract tokenizer type from checkpoint path
# Directory names follow pattern: {dataset}_{tokenizer}_{variant}_... or {dataset}_{tokenizer}_n{count}_...
get_tokenizer_type() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    # Extract tokenizer from directory name
    # Check order matters: hsent before sent, hdtc before hdt
    # Patterns match both _tokenizer_ and _tokenizer followed by non-alpha (e.g., _sent_n, _hdt-)
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
# Directory names follow pattern: {dataset}_{tokenizer}_{coarsening}_...
# mc = motif_community, sc = spectral, hac = hac, mas = motif_aware_spectral
get_coarsening_strategy() {
    local ckpt_path="$1"
    local dir_name=$(basename "$(dirname "$ckpt_path")")

    # Extract coarsening strategy from directory name
    # Patterns match _strategy_ or _strategy followed by non-alpha (e.g., _mc_, _mcn, _mc-)
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
    # Only hsent and hdt support coarsening strategies
    # sent and hdtc do not (hdtc uses FunctionalHierarchyBuilder)
    if [[ "$tokenizer" == "hsent" ]] || [[ "$tokenizer" == "hdt" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Check if a run directory has a completed evaluation.
# Returns 0 (true) only when both results and generated_smiles exist.
is_already_evaluated() {
    local run_dir="$1"
    [ -f "$run_dir/results.json" ] && [ -f "$run_dir/generated_smiles.txt" ] && [ -s "$run_dir/generated_smiles.txt" ]
}

# Evaluate each checkpoint
echo "$CHECKPOINTS" | while read -r ckpt; do
    TOKENIZER=$(get_tokenizer_type "$ckpt")
    COARSENING=$(get_coarsening_strategy "$ckpt")
    RUN_DIR_NAME=$(basename "$(dirname "$ckpt")")
    LOGS_PATH_TEST="${TEST_OUTPUT_DIR}/${RUN_DIR_NAME}"
    LOGS_PATH_GEN="${REALISTIC_GEN_OUTPUT_DIR}/${RUN_DIR_NAME}"

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
        if [ "$FORCE_REEVAL" = false ] && is_already_evaluated "$LOGS_PATH_TEST"; then
            echo "[1/2] Skipping test.py (already evaluated)"
        else
            echo "[1/2] Running test.py..."
            python scripts/test.py model.checkpoint_path="$ckpt" tokenizer=$TOKENIZER experiment=$DATASET logs.path="$LOGS_PATH_TEST" metrics.reference_split=$REFERENCE_SPLIT $COARSENING_ARGS
        fi
    fi

    if [ "$RUN_GEN" = true ]; then
        if [ "$FORCE_REEVAL" = false ] && is_already_evaluated "$LOGS_PATH_GEN"; then
            echo "[2/2] Skipping realistic_gen.py (already evaluated)"
        else
            echo "[2/2] Running realistic_gen.py..."
            python scripts/realistic_gen.py model.checkpoint_path="$ckpt" tokenizer=$TOKENIZER experiment=$DATASET logs.path="$LOGS_PATH_GEN" metrics.reference_split=$REFERENCE_SPLIT $COARSENING_ARGS
        fi
    fi

    echo ""
done

# Generate comparison table
echo "========================================"
echo "Generating comparison table..."
echo "========================================"
python scripts/comparison/compare_results.py --test-dir "$TEST_OUTPUT_DIR" --realistic-gen-dir "$REALISTIC_GEN_OUTPUT_DIR" --output "$COMPARISON_OUTPUT"

echo ""
echo "Done! Results saved to:"
echo "  - Test results:      $TEST_OUTPUT_DIR/"
echo "  - Realistic gen:     $REALISTIC_GEN_OUTPUT_DIR/"
echo "  - Comparison table:  $COMPARISON_OUTPUT"
