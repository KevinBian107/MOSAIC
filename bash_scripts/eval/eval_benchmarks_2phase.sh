#!/bin/bash
# Two-phase benchmark evaluation:
#   Phase 1 (GPU, sequential): generate molecules + save SMILES (no metrics)
#   Phase 2 (CPU, parallel):   load SMILES + run metrics_only for each model
#
# Usage:
#   ./bash_scripts/eval/eval_benchmarks_2phase.sh [--coconut] [--full-ref] [--force]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="moses"
REFERENCE_SPLIT="test"
USE_COCONUT=false
USE_FULL_REF=false
FORCE_REEVAL=false

for arg in "$@"; do
    case $arg in
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
            echo "Usage: $0 [--coconut] [--full-ref] [--force]"
            echo ""
            echo "Runs test.py in two phases:"
            echo "  1) metrics.generate_only=true  (sequential, uses GPU)"
            echo "  2) metrics.metrics_only=true   (parallel, CPU only)"
            exit 0
            ;;
    esac
done

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

CHECKPOINTS=$(find "$BENCHMARK_DIR" -name "last.ckpt" -type f 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in $BENCHMARK_DIR"
    exit 1
fi

cd "$PROJECT_ROOT"

get_tokenizer_type() {
    local ckpt_path="$1"
    local dir_name
    dir_name=$(basename "$(dirname "$ckpt_path")")
    if [[ "$dir_name" == *"_hsent_"* ]] || [[ "$dir_name" =~ _hsent[^a-z] ]]; then
        echo "hsent"
    elif [[ "$dir_name" == *"_hdtc_"* ]] || [[ "$dir_name" =~ _hdtc[^a-z] ]]; then
        echo "hdtc"
    elif [[ "$dir_name" == *"_hdt_"* ]] || [[ "$dir_name" =~ _hdt[^a-z] ]]; then
        echo "hdt"
    elif [[ "$dir_name" == *"_sent_"* ]] || [[ "$dir_name" =~ _sent[^a-z] ]]; then
        echo "sent"
    else
        echo "hsent"
    fi
}

supports_coarsening() {
    local tokenizer="$1"
    if [[ "$tokenizer" == "hsent" ]] || [[ "$tokenizer" == "hdt" ]]; then
        return 0
    else
        return 1
    fi
}

is_already_generated() {
    local run_dir="$1"
    [ -f "$run_dir/generated_smiles.txt" ] && [ -s "$run_dir/generated_smiles.txt" ]
}

echo "Dataset: $DATASET"
echo "Reference split: $REFERENCE_SPLIT"
echo "Benchmark dir: $BENCHMARK_DIR"
echo ""
echo "Found checkpoints:"
echo "$CHECKPOINTS" | while read -r ckpt; do echo "  - $ckpt"; done
echo ""

echo "========== PHASE 1: GENERATION ONLY (sequential, GPU) =========="
echo "$CHECKPOINTS" | while read -r ckpt; do
    TOKENIZER=$(get_tokenizer_type "$ckpt")
    RUN_DIR_NAME=$(basename "$(dirname "$ckpt")")
    LOGS_PATH_TEST="${TEST_OUTPUT_DIR}/${RUN_DIR_NAME}"

    COARSENING_ARGS=""
    if supports_coarsening "$TOKENIZER"; then
        # Let test.py infer coarsening from directory name via tokenizer.coarsening_strategy if desired
        :
    fi

    mkdir -p "$LOGS_PATH_TEST"

    if [ "$FORCE_REEVAL" = false ] && is_already_generated "$LOGS_PATH_TEST"; then
        echo "Skipping generation (already have generated_smiles.txt) for $RUN_DIR_NAME"
    else
        echo "Generating for $RUN_DIR_NAME ..."
        python scripts/test.py \
          model.checkpoint_path="$ckpt" \
          tokenizer="$TOKENIZER" \
          experiment="$DATASET" \
          logs.path="$LOGS_PATH_TEST" \
          metrics.reference_split="$REFERENCE_SPLIT" \
          metrics.generate_only=true \
          metrics.metrics_only=false \
          $COARSENING_ARGS
    fi
done

echo ""
echo "========== PHASE 2: METRICS ONLY (parallel, CPU, one screen per run) =========="
echo "Spawning a detached screen session for each checkpoint (attach with: screen -r <name>)"

echo "$CHECKPOINTS" | while read -r ckpt; do
    TOKENIZER=$(get_tokenizer_type "$ckpt")
    RUN_DIR_NAME=$(basename "$(dirname "$ckpt")")
    LOGS_PATH_TEST="${TEST_OUTPUT_DIR}/${RUN_DIR_NAME}"

    if ! is_already_generated "$LOGS_PATH_TEST"; then
        echo "Warning: Skipping metrics for $RUN_DIR_NAME (no generated_smiles.txt)."
        continue
    fi

    SESSION_NAME="mosaic_metrics_${RUN_DIR_NAME}"
    echo "Starting metrics_only for $RUN_DIR_NAME in screen session '$SESSION_NAME' ..."

    # Use a detached screen session running bash -lc so it inherits the env/conda activation.
    # If 'screen' is not in PATH, adjust to full path (e.g., /usr/bin/screen).
    usr/bin/screen -S "$SESSION_NAME" -dm bash -lc "
cd \"$PROJECT_ROOT\" && \
python scripts/test.py \
  model.checkpoint_path=\"$ckpt\" \
  tokenizer=\"$TOKENIZER\" \
  experiment=\"$DATASET\" \
  logs.path=\"$LOGS_PATH_TEST\" \
  metrics.reference_split=\"$REFERENCE_SPLIT\" \
  metrics.generate_only=false \
  metrics.metrics_only=true \
  $COARSENING_ARGS
"
done

echo ""
echo "Phase 2 metrics_only jobs have been started in screen sessions."
echo "Attach to a session with:  screen -r mosaic_metrics_<run_dir_name>"
echo "Logs and results are under $TEST_OUTPUT_DIR"

