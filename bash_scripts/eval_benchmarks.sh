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

# Evaluate each checkpoint
echo "$CHECKPOINTS" | while read -r ckpt; do
    echo "========================================"
    echo "Evaluating: $ckpt"
    echo "========================================"

    if [ "$RUN_TEST" = true ]; then
        echo "[1/2] Running test.py..."
        python scripts/test.py model.checkpoint_path="$ckpt"
    fi

    if [ "$RUN_GEN" = true ]; then
        echo "[2/2] Running realistic_gen.py..."
        python scripts/realistic_gen.py model.checkpoint_path="$ckpt"
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
