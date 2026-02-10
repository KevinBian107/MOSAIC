#!/bin/bash
# Precompute tokenized cache for MOSES or COCONUT dataset.
#
# This script runs parallel chunked preprocessing for hierarchical tokenizers
# (H-SENT, HDT) with all coarsening variants (SC, HAC, MC). Precomputed cache
# files speed up training startup by skipping on-the-fly tokenization.
#
# Usage:
#   ./bash_scripts/precompute_benchmarks.sh              # Precompute MOSES (default)
#   ./bash_scripts/precompute_benchmarks.sh --coconut    # Precompute COCONUT
#   ./bash_scripts/precompute_benchmarks.sh --all        # Precompute both datasets
#   ./bash_scripts/precompute_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/precompute_benchmarks.sh --help       # Show help
#
# Output:
#   data/cache/{dataset}_train_{tokenizer}_{num_samples}_{hash}.pt

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
DATASET="moses"
RUN_ALL=false
OUTPUT_DIR="$PROJECT_ROOT/data/cache"
NUM_CHUNKS=8
DRY_RUN=false
FORCE=false
SKIP_SC_HAC=false
COARSENING_FILTER="all"
TOKENIZER_FILTER="all"

# Dataset sample counts (from configs/experiment/*.yaml)
MOSES_SAMPLES=1000000
COCONUT_SAMPLES=5000

# Small dataset threshold — below this, run directly (no screen sessions)
SMALL_DATASET_THRESHOLD=10000

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --force)
            FORCE=true
            ;;
        --coconut)
            DATASET="coconut"
            ;;
        --all)
            RUN_ALL=true
            ;;
        --skip-sc-hac)
            SKIP_SC_HAC=true
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
        --output-dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Precompute tokenized cache for faster training startup."
            echo ""
            echo "Options:"
            echo "  --dry-run            Show commands without executing"
            echo "  --force              Re-run even if cache files exist"
            echo "  --coconut            Precompute COCONUT instead of MOSES"
            echo "  --all                Precompute both MOSES and COCONUT"
            echo "  --skip-sc-hac        Only precompute MC variants"
            echo "  --coarsening=STRATEGY  Filter coarsening: sc, hac, mc, or all (default: all)"
            echo "  --tokenizer=TYPE     Filter tokenizer: hsent, hdt, or all (default: all)"
            echo "  --chunks=N           Number of parallel chunks (default: 8)"
            echo "  --output-dir=PATH    Cache directory (default: data/cache)"
            echo ""
            echo "Datasets:"
            echo "  MOSES:   ${MOSES_SAMPLES} training samples (parallel screen sessions)"
            echo "  COCONUT: ${COCONUT_SAMPLES} training samples (runs directly, ~4 min)"
            echo ""
            echo "Tokenizer/coarsening combos (default: all 6):"
            echo "  hsent:sc  hsent:hac  hsent:mc"
            echo "  hdt:sc    hdt:hac    hdt:mc"
            echo ""
            echo "With --skip-sc-hac: only hsent:mc and hdt:mc"
            echo ""
            echo "After precomputing, use in training with:"
            echo "  data.use_cache=true"
            exit 0
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Map short coarsening code to full strategy name
get_coarsening_name() {
    local coarse="$1"
    case $coarse in
        mc) echo "motif_community" ;;
        sc) echo "spectral" ;;
        hac) echo "hac" ;;
        mas) echo "motif_aware_spectral" ;;
        *) echo "" ;;
    esac
}

# Get short name for file prefixes
get_short_name() {
    local tok="$1"
    local coarse="$2"
    echo "${tok}_${coarse}"
}

# Build list of tokenizer:coarsening combos
build_combos() {
    local combos=()

    # Determine which tokenizers
    local tokenizers=()
    if [ "$TOKENIZER_FILTER" = "all" ]; then
        tokenizers=("hsent" "hdt")
    else
        tokenizers=("$TOKENIZER_FILTER")
    fi

    # Determine which coarsening strategies
    local coarsenings=()
    if [ "$SKIP_SC_HAC" = true ]; then
        coarsenings=("mc")
    elif [ "$COARSENING_FILTER" = "all" ]; then
        coarsenings=("sc" "hac" "mc")
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

# Run preprocessing for a single dataset
run_dataset() {
    local dataset="$1"
    local total_samples

    if [ "$dataset" = "moses" ]; then
        total_samples=$MOSES_SAMPLES
    else
        total_samples=$COCONUT_SAMPLES
    fi

    local combos
    read -ra combos <<< "$(build_combos)"

    echo "========================================"
    echo "Precompute Cache: ${dataset^^}"
    echo "========================================"
    echo ""
    echo "Settings:"
    echo "  Dataset: $dataset"
    echo "  Total samples: $total_samples"
    echo "  Chunks: $NUM_CHUNKS"
    echo "  Output: $OUTPUT_DIR"
    echo "  Force: $FORCE"
    echo ""
    echo "Combos to precompute:"
    for combo in "${combos[@]}"; do
        IFS=':' read -r tok coarse <<< "$combo"
        echo "  - $tok ($(get_coarsening_name "$coarse"))"
    done
    echo ""

    local chunk_size=$((total_samples / NUM_CHUNKS))
    local combo_count=0
    local total_combos=${#combos[@]}

    for combo in "${combos[@]}"; do
        combo_count=$((combo_count + 1))
        IFS=':' read -r TOKENIZER COARSENING_SHORT <<< "$combo"
        COARSENING_FULL=$(get_coarsening_name "$COARSENING_SHORT")
        SHORT_NAME=$(get_short_name "$TOKENIZER" "$COARSENING_SHORT")

        echo "========================================"
        echo "[$combo_count/$total_combos] ${dataset}: ${TOKENIZER} + ${COARSENING_FULL}"
        echo "========================================"

        # Check if combined cache file already exists
        # Cache files follow: {dataset}_train_{tokenizer}_{num_samples}_{hash}.pt
        existing_cache=$(find "$OUTPUT_DIR" -name "${dataset}_train_${TOKENIZER}_${total_samples}_*.pt" -type f 2>/dev/null | while read -r f; do
            # Verify the coarsening strategy matches by checking the file
            python -c "
import torch, sys
d = torch.load('$f', weights_only=False)
cfg = d.get('tokenizer_config', {})
if cfg.get('coarsening_strategy') == '$COARSENING_FULL':
    print('$f')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null && break
        done)

        if [ "$FORCE" = false ] && [ -n "$existing_cache" ]; then
            echo "  SKIP: Cache already exists: $existing_cache"
            echo "  Use --force to re-run"
            echo ""
            continue
        fi

        # Build dataset-specific args for preprocess_chunk.py
        local dataset_args="--dataset $dataset"
        if [ "$dataset" = "coconut" ]; then
            dataset_args="$dataset_args --data-file data/coconut_complex.smi"
            dataset_args="$dataset_args --min-atoms 20 --max-atoms 100 --min-rings 3"
        fi

        if [ $total_samples -le $SMALL_DATASET_THRESHOLD ]; then
            # Small dataset: run directly (no screen sessions needed)
            echo "  Mode: direct (small dataset, $total_samples samples)"
            echo ""

            local output_file="$OUTPUT_DIR/${TOKENIZER}_${COARSENING_FULL}_chunk_0_${total_samples}.pt"

            local cmd="python scripts/preprocess/preprocess_chunk.py"
            cmd="$cmd --tokenizer $TOKENIZER"
            cmd="$cmd $dataset_args"
            cmd="$cmd --coarsening-strategy $COARSENING_FULL"
            cmd="$cmd --start 0"
            cmd="$cmd --end $total_samples"
            cmd="$cmd --output $output_file"

            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] Would execute:"
                echo "    $cmd"
            else
                echo "  Running preprocessing..."
                eval $cmd
            fi

            # Auto-combine (single chunk)
            local combine_cmd="python scripts/preprocess/combine_chunks.py"
            combine_cmd="$combine_cmd --tokenizer $TOKENIZER"
            combine_cmd="$combine_cmd --coarsening-strategy $COARSENING_FULL"
            combine_cmd="$combine_cmd --chunk_dir $OUTPUT_DIR"
            combine_cmd="$combine_cmd --split train"
            combine_cmd="$combine_cmd --dataset $dataset"

            if [ "$DRY_RUN" = true ]; then
                echo "    $combine_cmd"
            else
                echo "  Combining chunks..."
                eval $combine_cmd

                # Clean up chunk file after combining
                rm -f "$output_file"
                echo "  Cleaned up chunk file"
            fi
            echo ""

        else
            # Large dataset: launch parallel screen sessions
            if ! command -v screen &> /dev/null; then
                echo "  ERROR: 'screen' is not installed (needed for parallel chunks)"
                echo "  Install with: sudo apt-get install screen"
                echo "  Or use a smaller --chunks=1 to run sequentially"
                continue
            fi

            echo "  Mode: parallel ($NUM_CHUNKS screen sessions)"
            echo ""

            for ((i=0; i<NUM_CHUNKS; i++)); do
                local start=$((i * chunk_size))
                local end=$(((i + 1) * chunk_size))

                # Last chunk gets remaining samples
                if [ $i -eq $((NUM_CHUNKS - 1)) ]; then
                    end=$total_samples
                fi

                local screen_name="preprocess_${dataset}_${TOKENIZER}_${COARSENING_SHORT}_chunk${i}"
                local output_file="$OUTPUT_DIR/${TOKENIZER}_${COARSENING_FULL}_chunk_${start}_${end}.pt"

                local cmd="python scripts/preprocess/preprocess_chunk.py"
                cmd="$cmd --tokenizer $TOKENIZER"
                cmd="$cmd $dataset_args"
                cmd="$cmd --coarsening-strategy $COARSENING_FULL"
                cmd="$cmd --start $start"
                cmd="$cmd --end $end"
                cmd="$cmd --output $output_file"

                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] screen -dmS $screen_name: $cmd"
                else
                    screen -dmS "$screen_name" bash -c "
                        cd '$PROJECT_ROOT'
                        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
                        conda activate mosaic
                        $cmd
                        EXIT_CODE=\$?
                        if [ \$EXIT_CODE -eq 0 ]; then
                            echo 'Chunk $i completed successfully!'
                        else
                            echo 'Chunk $i failed with exit code '\$EXIT_CODE
                        fi
                        echo 'Press any key to close...'
                        read -n 1
                    "
                    sleep 0.5
                    echo "  Started screen: $screen_name [$start:$end]"
                fi
            done

            echo ""
            echo "  Monitor progress:"
            echo "    screen -ls"
            echo "    watch -n 5 'ls -lh $OUTPUT_DIR/${TOKENIZER}_${COARSENING_FULL}_chunk_*.pt | wc -l'"
            echo ""
            echo "  After all chunks complete, combine:"
            echo "    python scripts/preprocess/combine_chunks.py \\"
            echo "        --tokenizer $TOKENIZER \\"
            echo "        --coarsening-strategy $COARSENING_FULL \\"
            echo "        --chunk_dir $OUTPUT_DIR \\"
            echo "        --split train \\"
            echo "        --dataset $dataset"
            echo ""
        fi
    done
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run for selected dataset(s)
if [ "$RUN_ALL" = true ]; then
    run_dataset "moses"
    echo ""
    run_dataset "coconut"
elif [ "$DATASET" = "coconut" ]; then
    run_dataset "coconut"
else
    run_dataset "moses"
fi

echo "========================================"
echo "Done! Cache files saved to: $OUTPUT_DIR/"
echo "Use in training with: data.use_cache=true"
echo "========================================"
