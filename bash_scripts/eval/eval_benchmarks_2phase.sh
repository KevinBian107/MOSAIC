#!/bin/bash
# Two-phase benchmark evaluation with motif-only parallelization:
#   Phase 1 (GPU, sequential): run test.py WITHOUT motif metrics
#   Phase 2 (CPU, parallel):   run motif-only metrics in screen sessions
#   Optional realistic_gen + comparison chart (same style as eval_benchmarks.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="moses"
REFERENCE_SPLIT="test"
RUN_TEST=true
RUN_GEN=true
USE_COCONUT=false
USE_FULL_REF=false
FORCE_REEVAL=false
REUSE_GENERATED=false

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
        --reuse-generated)
            REUSE_GENERATED=true
            ;;
        --help|-h)
            echo "Usage: $0 [--test-only] [--gen-only] [--coconut] [--full-ref] [--force] [--reuse-generated]"
            echo ""
            echo "Two-phase behavior:"
            echo "  Phase 1 (sequential/GPU): test.py WITHOUT motif metrics"
            echo "  Phase 2 (parallel/CPU):   motif-only test.py in detached screen sessions"
            echo "Then optional realistic_gen.py and final comparison chart."
            echo ""
            echo "  --reuse-generated: run phase 1 in metrics_only mode using existing"
            echo "                     generated_smiles.txt (no generation)."
            echo "                     Also runs realistic_gen.py in reuse mode."
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
REALISTIC_GEN_OUTPUT_DIR="outputs/realistic_gen${DIR_SUFFIX}"
COMPARISON_OUTPUT="${TEST_OUTPUT_DIR}/comparison.png"

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

get_coarsening_strategy() {
    local ckpt_path="$1"
    local dir_name
    dir_name=$(basename "$(dirname "$ckpt_path")")
    if [[ "$dir_name" == *"_mc_"* ]] || [[ "$dir_name" =~ _mc[^a-z] ]]; then
        echo "motif_community"
    elif [[ "$dir_name" == *"_sc_"* ]] || [[ "$dir_name" =~ _sc[^a-z] ]]; then
        echo "spectral"
    elif [[ "$dir_name" == *"_hac_"* ]] || [[ "$dir_name" =~ _hac[^a-z] ]]; then
        echo "hac"
    elif [[ "$dir_name" == *"_mas_"* ]] || [[ "$dir_name" =~ _mas[^a-z] ]]; then
        echo "motif_aware_spectral"
    else
        echo "spectral"
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

is_test_phase1_done() {
    local run_dir="$1"
    [ -f "$run_dir/results.json" ] && [ -f "$run_dir/generated_smiles.txt" ] && [ -s "$run_dir/generated_smiles.txt" ]
}

is_motif_done() {
    local run_dir="$1"
    local results_file="$run_dir/results.json"
    if [ ! -f "$results_file" ]; then
        return 1
    fi
    python - <<PY
import json, sys
path = "$results_file"
try:
    with open(path) as f:
        r = json.load(f)
    needed = ["motif_fg_mmd", "motif_smarts_mmd", "motif_ring_mmd", "motif_brics_mmd"]
    ok = all(k in r and r[k] is not None for k in needed)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

is_realistic_done() {
    local run_dir="$1"
    [ -f "$run_dir/results.json" ]
}

echo "Dataset: $DATASET"
echo "Reference split: $REFERENCE_SPLIT"
echo "Benchmark dir: $BENCHMARK_DIR"
echo "Reuse generated: $REUSE_GENERATED"
echo ""
echo "Found checkpoints:"
while read -r ckpt; do
    [ -z "$ckpt" ] && continue
    echo "  - $ckpt"
done <<< "$CHECKPOINTS"
echo ""

if [ "$RUN_TEST" = true ]; then
    if [ "$REUSE_GENERATED" = true ]; then
        echo "========== PHASE 1: TEST (sequential), motif disabled, reusing generated_smiles =========="
    else
        echo "========== PHASE 1: TEST (sequential, GPU), motif disabled =========="
    fi
    while read -r ckpt; do
        [ -z "$ckpt" ] && continue
        TOKENIZER=$(get_tokenizer_type "$ckpt")
        COARSENING=$(get_coarsening_strategy "$ckpt")
        RUN_DIR_NAME=$(basename "$(dirname "$ckpt")")
        LOGS_PATH_TEST="${TEST_OUTPUT_DIR}/${RUN_DIR_NAME}"

        COARSENING_ARGS=""
        if supports_coarsening "$TOKENIZER"; then
            COARSENING_ARGS="tokenizer.coarsening_strategy=$COARSENING"
        fi

        if [ "$FORCE_REEVAL" = false ] && is_test_phase1_done "$LOGS_PATH_TEST"; then
            echo "Skipping phase1 test (already has results + generated_smiles): $RUN_DIR_NAME"
        else
            if [ "$REUSE_GENERATED" = true ]; then
                if [ ! -f "$LOGS_PATH_TEST/generated_smiles.txt" ]; then
                    echo "Skipping phase1 for $RUN_DIR_NAME (missing generated_smiles.txt for reuse)."
                    continue
                fi
                echo "Running phase1 test for $RUN_DIR_NAME in metrics_only mode (compute_motif=false)..."
                export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
                python scripts/test.py \
                  model.checkpoint_path="$ckpt" \
                  tokenizer="$TOKENIZER" \
                  experiment="$DATASET" \
                  logs.path="$LOGS_PATH_TEST" \
                  metrics.reference_split="$REFERENCE_SPLIT" \
                  metrics.compute_motif=false \
                  metrics.compute_fcd=true \
                  metrics.compute_pgd=true \
                  metrics.generate_only=false \
                  metrics.metrics_only=true \
                  metrics.motif_only=false \
                  $COARSENING_ARGS
            else
                echo "Running phase1 test for $RUN_DIR_NAME (compute_motif=false)..."
                export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
                python scripts/test.py \
                  model.checkpoint_path="$ckpt" \
                  tokenizer="$TOKENIZER" \
                  experiment="$DATASET" \
                  logs.path="$LOGS_PATH_TEST" \
                  metrics.reference_split="$REFERENCE_SPLIT" \
                  metrics.compute_motif=false \
                  metrics.generate_only=false \
                  metrics.metrics_only=false \
                  metrics.motif_only=false \
                  $COARSENING_ARGS
            fi
        fi
    done <<< "$CHECKPOINTS"

    echo ""
    echo "========== PHASE 2: MOTIF-ONLY (parallel CPU in screen) =========="
    declare -a SESSIONS=()
    while read -r ckpt; do
        [ -z "$ckpt" ] && continue
        TOKENIZER=$(get_tokenizer_type "$ckpt")
        COARSENING=$(get_coarsening_strategy "$ckpt")
        RUN_DIR_NAME=$(basename "$(dirname "$ckpt")")
        LOGS_PATH_TEST="${TEST_OUTPUT_DIR}/${RUN_DIR_NAME}"

        COARSENING_ARGS=""
        if supports_coarsening "$TOKENIZER"; then
            COARSENING_ARGS="tokenizer.coarsening_strategy=$COARSENING"
        fi

        if [ ! -f "$LOGS_PATH_TEST/generated_smiles.txt" ]; then
            echo "Skipping motif phase (missing generated_smiles): $RUN_DIR_NAME"
            continue
        fi

        if [ "$FORCE_REEVAL" = false ] && is_motif_done "$LOGS_PATH_TEST"; then
            echo "Skipping motif phase (already has motif metrics): $RUN_DIR_NAME"
            continue
        fi

        SESSION_NAME="mosaic_motif_${RUN_DIR_NAME}"
        SESSIONS+=("$SESSION_NAME")
        echo "Starting motif-only metrics in screen session '$SESSION_NAME'..."
        /usr/bin/screen -S "$SESSION_NAME" -dm bash -lc "
cd \"$PROJECT_ROOT\" && \
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && \
python scripts/test.py \
  model.checkpoint_path=\"$ckpt\" \
  tokenizer=\"$TOKENIZER\" \
  experiment=\"$DATASET\" \
  logs.path=\"$LOGS_PATH_TEST\" \
  metrics.reference_split=\"$REFERENCE_SPLIT\" \
  metrics.compute_motif=true \
  metrics.compute_fcd=false \
  metrics.compute_pgd=false \
  metrics.generate_only=false \
  metrics.metrics_only=true \
  metrics.motif_only=true \
  $COARSENING_ARGS
"
    done <<< "$CHECKPOINTS"

    if [ "${#SESSIONS[@]}" -gt 0 ]; then
        echo "Waiting for motif-only screen sessions to finish..."
        while true; do
            any_running=0
            for s in "${SESSIONS[@]}"; do
                if /usr/bin/screen -ls | awk '{print $1}' | sed 's/^[[:space:]]*//' | grep -q "\\.${s}$"; then
                    any_running=1
                    break
                fi
            done
            if [ "$any_running" -eq 0 ]; then
                break
            fi
            sleep 10
        done
    fi
fi

if [ "$RUN_GEN" = true ]; then
    echo ""
    echo "========== REALISTIC_GEN (sequential) =========="
    while read -r ckpt; do
        [ -z "$ckpt" ] && continue
        TOKENIZER=$(get_tokenizer_type "$ckpt")
        COARSENING=$(get_coarsening_strategy "$ckpt")
        RUN_DIR_NAME=$(basename "$(dirname "$ckpt")")
        LOGS_PATH_GEN="${REALISTIC_GEN_OUTPUT_DIR}/${RUN_DIR_NAME}"

        COARSENING_ARGS=""
        if supports_coarsening "$TOKENIZER"; then
            COARSENING_ARGS="tokenizer.coarsening_strategy=$COARSENING"
        fi

        if [ "$FORCE_REEVAL" = false ] && is_realistic_done "$LOGS_PATH_GEN"; then
            echo "Skipping realistic_gen (already has results): $RUN_DIR_NAME"
        else
            if [ "$REUSE_GENERATED" = true ]; then
                SMILES_SRC="${TEST_OUTPUT_DIR}/${RUN_DIR_NAME}/generated_smiles.txt"
                if [ ! -f "$SMILES_SRC" ]; then
                    echo "Skipping realistic_gen for $RUN_DIR_NAME (missing $SMILES_SRC for reuse)."
                    continue
                fi
                echo "Running realistic_gen for $RUN_DIR_NAME using reused generated_smiles ..."
                python scripts/realistic_gen.py \
                  model.checkpoint_path="$ckpt" \
                  tokenizer="$TOKENIZER" \
                  experiment="$DATASET" \
                  logs.path="$LOGS_PATH_GEN" \
                  metrics.reference_split="$REFERENCE_SPLIT" \
                  generation.reuse_generated_smiles=true \
                  generation.generated_smiles_path="$SMILES_SRC" \
                  $COARSENING_ARGS
            else
                echo "Running realistic_gen for $RUN_DIR_NAME ..."
                python scripts/realistic_gen.py \
                  model.checkpoint_path="$ckpt" \
                  tokenizer="$TOKENIZER" \
                  experiment="$DATASET" \
                  logs.path="$LOGS_PATH_GEN" \
                  metrics.reference_split="$REFERENCE_SPLIT" \
                  $COARSENING_ARGS
            fi
        fi
    done <<< "$CHECKPOINTS"
fi

echo ""
echo "========== GENERATING COMPARISON CHART =========="
if [ "$RUN_GEN" = true ]; then
    python scripts/comparison/compare_results.py \
      --test-dir "$TEST_OUTPUT_DIR" \
      --realistic-gen-dir "$REALISTIC_GEN_OUTPUT_DIR" \
      --output "$COMPARISON_OUTPUT"
else
    python scripts/comparison/compare_results.py \
      --test-dir "$TEST_OUTPUT_DIR" \
      --output "$COMPARISON_OUTPUT" \
      --test-only
fi

echo ""
echo "Done. Results saved to:"
echo "  - Test results:      $TEST_OUTPUT_DIR/"
if [ "$RUN_GEN" = true ]; then
    echo "  - Realistic gen:     $REALISTIC_GEN_OUTPUT_DIR/"
fi
echo "  - Comparison table:  $COMPARISON_OUTPUT"

