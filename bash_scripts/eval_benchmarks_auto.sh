#!/bin/bash
# Evaluate checkpoints discovered in a benchmark directory and generate a comparison table.
#
# Columns are set automatically from the checkpoints found (and optionally from a mapping file).
# You can choose last.ckpt or best.ckpt per run.
#
# Usage:
#   ./bash_scripts/eval_benchmarks_auto.sh MAPPING.txt OUTPUT_PATH [OPTIONS]
#   ./bash_scripts/eval_benchmarks_auto.sh MAPPING.txt outputs/eval_my_run --best
#   ./bash_scripts/eval_benchmarks_auto.sh MAPPING.txt outputs/eval_my_run --last --test-only
#
# MAPPING.txt format (one line per checkpoint, order = column order):
#   directory_name
#   directory_name	Display Label
#   # comment lines and empty lines are skipped
#
# If a line has two columns (tab or space separated), the second is used as the column label
# in the comparison table. Otherwise the directory name is used.
#
# Checkpoints are looked for under BENCHMARK_DIR (default: outputs/benchmark). Each directory
# under BENCHMARK_DIR is checked for last.ckpt or best.ckpt depending on --last/--best.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default checkpoint type
CKPT_NAME="best.ckpt"
RUN_TEST=true
RUN_GEN=true
DATASET="moses"
BENCHMARK_DIR="$PROJECT_ROOT/outputs/benchmark"

usage() {
    echo "Usage: $0 MAPPING_FILE OUTPUT_PATH [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  MAPPING_FILE   Path to a text file listing checkpoint dir names (and optional column labels)."
    echo "                 Order of lines = column order. Format: dir_name or dir_name<TAB>Label"
    echo "  OUTPUT_PATH    Base directory for test/ and realistic_gen/ outputs and comparison.png"
    echo ""
    echo "Options:"
    echo "  --last         Use last.ckpt instead of best.ckpt (default: best.ckpt)"
    echo "  --best         Use best.ckpt (default)"
    echo "  --benchmark-dir DIR   Directory to search for checkpoints (default: outputs/benchmark)"
    echo "  --dataset NAME       moses or coconut (default: moses)"
    echo "  --test-only    Only run test.py (skip realistic_gen.py)"
    echo "  --gen-only     Only run realistic_gen.py (skip test.py)"
    echo "  -h, --help     Show this help"
    echo ""
    echo "Example mapping file:"
    echo "  moses_hdtc_n100000_20260126-204311"
    echo "  moses_hsent_mc_20260122-093526	H-SENT MC"
    echo "  moses_sent_n1000000_20260123-140906	SENT 1M"
}

if [ $# -lt 2 ]; then
    echo "Error: MAPPING_FILE and OUTPUT_PATH are required."
    echo ""
    usage
    exit 1
fi

MAPPING_FILE="$1"
OUTPUT_PATH="$2"
shift 2

if [ ! -f "$MAPPING_FILE" ]; then
    echo "Error: Mapping file not found: $MAPPING_FILE"
    exit 1
fi

# Parse optional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --last)
            CKPT_NAME="last.ckpt"
            ;;
        --best)
            CKPT_NAME="best.ckpt"
            ;;
        --benchmark-dir)
            BENCHMARK_DIR="$2"
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            ;;
        --test-only)
            RUN_GEN=false
            ;;
        --gen-only)
            RUN_TEST=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# Resolve paths
if [[ "$BENCHMARK_DIR" != /* ]]; then
    BENCHMARK_DIR="$PROJECT_ROOT/$BENCHMARK_DIR"
fi
if [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT_PATH"
fi

TEST_OUTPUT_DIR="$OUTPUT_PATH/test"
REALISTIC_GEN_OUTPUT_DIR="$OUTPUT_PATH/realistic_gen"
COMPARISON_OUTPUT="$OUTPUT_PATH/comparison.png"

mkdir -p "$TEST_OUTPUT_DIR"
mkdir -p "$REALISTIC_GEN_OUTPUT_DIR"

# Parse mapping file: lines like "dir_name" or "dir_name\tLabel" or "dir_name  Label"
# Output order of (dir_name, label) and only include dirs that have a checkpoint
parse_mapping_and_find_ckpts() {
    local ckpt_name="$1"
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$line" ] && continue
        [[ "$line" =~ ^# ]] && continue
        dir_name="${line%%[[:space:]]*}"
        rest="${line#$dir_name}"
        rest=$(echo "$rest" | sed 's/^[[:space:]]*//')
        ckpt_path="$BENCHMARK_DIR/$dir_name/$ckpt_name"
        if [ -f "$ckpt_path" ]; then
            if [ -n "$rest" ]; then
                echo "${dir_name}	${rest}"
            else
                echo "${dir_name}	${dir_name}"
            fi
        else
            echo "Warning: Checkpoint not found, skipping: $ckpt_path" >&2
        fi
    done < "$MAPPING_FILE"
}

# If mapping file is empty or we got no matches, discover all checkpoints from benchmark dir
discover_all_ckpts() {
    local ckpt_name="$1"
    find "$BENCHMARK_DIR" -name "$ckpt_name" -type f 2>/dev/null | while read -r ckpt_path; do
        dir_name=$(basename "$(dirname "$ckpt_path")")
        echo "${dir_name}	${dir_name}"
    done | sort -t$'\t' -k1,1
}

# Build list of (dir_name, label) for checkpoints we will run
MAP_CONTENT=$(parse_mapping_and_find_ckpts "$CKPT_NAME")
if [ -z "$MAP_CONTENT" ]; then
    echo "No checkpoints from mapping file found; discovering all $CKPT_NAME under $BENCHMARK_DIR"
    MAP_CONTENT=$(discover_all_ckpts "$CKPT_NAME")
fi

if [ -z "$MAP_CONTENT" ]; then
    echo "Error: No checkpoints found in $BENCHMARK_DIR (looking for $CKPT_NAME)"
    exit 1
fi

# Count entries and build arrays for dir names and labels
RUN_DIRS=()
RUN_LABELS=()
while IFS= read -r line; do
    [ -z "$line" ] && continue
    dir_name="${line%%	*}"
    label="${line#*	}"
    RUN_DIRS+=("$dir_name")
    RUN_LABELS+=("$label")
done <<< "$MAP_CONTENT"

NUM_RUNS=${#RUN_DIRS[@]}
echo "Dataset: $DATASET"
echo "Benchmark dir: $BENCHMARK_DIR"
echo "Checkpoint type: $CKPT_NAME"
echo "Output path: $OUTPUT_PATH"
echo "Found $NUM_RUNS checkpoints to evaluate:"
for i in "${!RUN_DIRS[@]}"; do
    echo "  - ${RUN_DIRS[$i]} -> ${RUN_LABELS[$i]}"
done
echo ""

cd "$PROJECT_ROOT"

# Reuse tokenizer/coarsening helpers from eval_benchmarks.sh
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
        echo "hsent"
    fi
}

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

# Evaluate each checkpoint
for i in "${!RUN_DIRS[@]}"; do
    dir_name="${RUN_DIRS[$i]}"
    ckpt="$BENCHMARK_DIR/$dir_name/$CKPT_NAME"
    TOKENIZER=$(get_tokenizer_type "$ckpt")
    COARSENING=$(get_coarsening_strategy "$ckpt")

    echo "========================================"
    echo "[$((i+1))/$NUM_RUNS] Evaluating: $ckpt"
    echo "  Label: ${RUN_LABELS[$i]} | Tokenizer: $TOKENIZER"
    COARSENING_ARGS=""
    if supports_coarsening "$TOKENIZER"; then
        echo "  Coarsening: $COARSENING"
        COARSENING_ARGS="tokenizer.coarsening_strategy=$COARSENING"
    fi
    echo "========================================"

    # Use stable run name = dir_name so compare_results can match and order columns
    LOGS_PATH_TEST="$TEST_OUTPUT_DIR/$dir_name"
    LOGS_PATH_GEN="$REALISTIC_GEN_OUTPUT_DIR/$dir_name"

    if [ "$RUN_TEST" = true ]; then
        echo "[1/2] Running test.py..."
        python scripts/test.py model.checkpoint_path="$ckpt" tokenizer=$TOKENIZER experiment=$DATASET logs.path="$LOGS_PATH_TEST" $COARSENING_ARGS
    fi

    if [ "$RUN_GEN" = true ]; then
        echo "[2/2] Running realistic_gen.py..."
        python scripts/realistic_gen.py model.checkpoint_path="$ckpt" tokenizer=$TOKENIZER experiment=$DATASET logs.path="$LOGS_PATH_GEN" $COARSENING_ARGS
    fi

    echo ""
done

# Build --runs and --column-labels for compare_results
RUNS_ARG=""
LABELS_ARG=""
for i in "${!RUN_DIRS[@]}"; do
    RUNS_ARG="$RUNS_ARG ${RUN_DIRS[$i]}"
    # Escape commas in labels for CSV; for CLI pass as separate args
    LABELS_ARG="$LABELS_ARG ${RUN_LABELS[$i]}"
done
RUNS_ARG=$(echo "$RUNS_ARG" | sed 's/^ *//')
# Pass labels as comma-separated (compare_results will split)
LABELS_CSV=$(IFS=,; echo "${RUN_LABELS[*]}")

echo "========================================"
echo "Generating comparison table..."
echo "========================================"
python scripts/compare_results.py \
    --test-dir "$TEST_OUTPUT_DIR" \
    --realistic-gen-dir "$REALISTIC_GEN_OUTPUT_DIR" \
    --output "$COMPARISON_OUTPUT" \
    --all \
    --runs ${RUN_DIRS[@]} \
    --column-labels "$LABELS_CSV"

echo ""
echo "Done! Results saved to:"
echo "  - Test results:      $TEST_OUTPUT_DIR/"
echo "  - Realistic gen:     $REALISTIC_GEN_OUTPUT_DIR/"
echo "  - Comparison table:  $COMPARISON_OUTPUT"
