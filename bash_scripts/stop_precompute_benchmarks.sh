#!/bin/bash
# Stop (close) all screen sessions started by precompute_benchmarks.sh
#
# This script mirrors the key CLI parameters of precompute_benchmarks.sh and
# kills any matching screen sessions named:
#   preprocess_${dataset}_${tokenizer}_${coarsening_short}_chunk${i}
#
# Usage examples:
#   # Stop all HSENT + spectral MOSES precompute screens (default chunks=8)
#   ./bash_scripts/stop_precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc
#
#   # Stop HDT + spectral with custom chunks
#   ./bash_scripts/stop_precompute_benchmarks.sh --tokenizer=hdt --coarsening=sc --chunks=4
#
#   # Stop all MOSES + COCONUT screens for all tokenizers/coarsenings
#   ./bash_scripts/stop_precompute_benchmarks.sh --all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings (mirrors precompute_benchmarks.sh)
DATASET="moses"
RUN_ALL=false
NUM_CHUNKS=8
COARSENING_FILTER="all"
TOKENIZER_FILTER="all"

# Parse arguments (subset of precompute_benchmarks.sh)
for arg in "$@"; do
    case $arg in
        --coconut)
            DATASET="coconut"
            ;;
        --all)
            RUN_ALL=true
            ;;
        --coarsening=*)
            COARSENING_FILTER="${arg#*=}"
            ;;
        --tokenizer=*)
            TOKENIZER_FILTER="${arg#*=}"
            ;;
        --chunks=*)
            NUM_CHUNKS="${arg#*=}"
            ;;
        # Ignored but accepted for compatibility with precompute_benchmarks.sh
        --output-dir=*|--train-samples=*|--val-samples=*|--spectral-n-init=*|--spectral-k-min-factor=*|--spectral-k-max-factor=*)
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Stop screen sessions created by bash_scripts/precompute_benchmarks.sh."
            echo ""
            echo "Options (mirrors precompute_benchmarks.sh subset):"
            echo "  --coconut            Target COCONUT instead of MOSES"
            echo "  --all                Target both MOSES and COCONUT"
            echo "  --coarsening=STR     sc, hac, or all (default: all)"
            echo "  --tokenizer=TYPE     hsent, hdt, or all (default: all)"
            echo "  --chunks=N           Number of chunks to consider (default: 8)"
            echo ""
            echo "Examples:"
            echo "  ./bash_scripts/stop_precompute_benchmarks.sh --tokenizer=hsent --coarsening=sc"
            echo "  ./bash_scripts/stop_precompute_benchmarks.sh --all"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Map short coarsening code to full strategy name (for logging only)
get_coarsening_name() {
    local coarse="$1"
    case $coarse in
        sc) echo "spectral" ;;
        hac) echo "hac" ;;
        *) echo "" ;;
    esac
}

# Build list of tokenizer:coarsening combos (same logic as precompute_benchmarks.sh)
build_combos() {
    local combos=()

    local tokenizers=()
    if [ "$TOKENIZER_FILTER" = "all" ]; then
        tokenizers=("hsent" "hdt")
    else
        tokenizers=("$TOKENIZER_FILTER")
    fi

    local coarsenings=()
    if [ "$COARSENING_FILTER" = "all" ]; then
        coarsenings=("sc" "hac")
    else
        coarsenings=("$COARSENING_FILTER")
    fi

    for tok in "${tokenizers[@]}"; do
        for coarse in "${coarsenings[@]}"; do
            combos+=("${tok}:${coarse}")
        done
    done

    echo "${combos[@]}"
}

stop_for_dataset() {
    local dataset="$1"

    local combos
    read -ra combos <<< "$(build_combos)"

    echo "========================================"
    echo "Stop Precompute Screens: ${dataset^^}"
    echo "========================================"
    echo ""
    echo "Settings:"
    echo "  Dataset: $dataset"
    echo "  Chunks: $NUM_CHUNKS"
    echo ""
    echo "Target combos:"
    for combo in "${combos[@]}"; do
        IFS=':' read -r tok coarse <<< "$combo"
        echo "  - $tok ($(get_coarsening_name "$coarse"))"
    done
    echo ""

    local combo_count=0
    local total_combos=${#combos[@]}

    for combo in "${combos[@]}"; do
        combo_count=$((combo_count + 1))
        IFS=':' read -r TOKENIZER COARSENING_SHORT <<< "$combo"

        echo "----------------------------------------"
        echo "[$combo_count/$total_combos] ${dataset}: ${TOKENIZER} + $(get_coarsening_name "$COARSENING_SHORT")"

        for ((i=0; i<NUM_CHUNKS; i++)); do
            local screen_name="preprocess_${dataset}_${TOKENIZER}_${COARSENING_SHORT}_chunk${i}"

            # Check if screen exists
            if screen -ls | grep -q "[[:space:]]${screen_name}[[:space:]]"; then
                echo "  Killing screen: $screen_name"
                screen -S "$screen_name" -X quit || echo "    (warning) Failed to kill $screen_name"
            else
                echo "  No screen found: $screen_name"
            fi
        done
        echo ""
    done
}

if [ "$RUN_ALL" = true ]; then
    stop_for_dataset "moses"
    echo ""
    stop_for_dataset "coconut"
elif [ "$DATASET" = "coconut" ]; then
    stop_for_dataset "coconut"
else
    stop_for_dataset "moses"
fi

echo "========================================"
echo "Done stopping matching precompute screen sessions."
echo "========================================"

