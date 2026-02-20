#!/bin/bash
# Precompute tokenized cache for MOSES or COCONUT dataset.
#
# This script runs parallel chunked preprocessing for hierarchical tokenizers
# (H-SENT, HDT) with SC and HAC coarsening. Precomputed cache files speed up
# training startup by skipping on-the-fly tokenization.
# Note: Only SC and HAC benefit from precomputation. Other coarsening methods
# (MC, MAS) are fast enough to run on-the-fly during training.
#
# Usage:
#   ./bash_scripts/precompute_benchmarks.sh              # Precompute MOSES (default)
#   ./bash_scripts/precompute_benchmarks.sh --coconut    # Precompute COCONUT
#   ./bash_scripts/precompute_benchmarks.sh --all        # Precompute both datasets
#   ./bash_scripts/precompute_benchmarks.sh --dry-run    # Show what would be run
#   ./bash_scripts/precompute_benchmarks.sh --help       # Show help
#
# Output:
#   data/cache/{dataset}_{split}_{tokenizer}_{num_samples}_{hash}.pt

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
COARSENING_FILTER="all"
TOKENIZER_FILTER="all"

# Dataset sample counts (from configs/experiment/*.yaml)
MOSES_SAMPLES=1000000
MOSES_VAL_SAMPLES=0
COCONUT_SAMPLES=5000
COCONUT_VAL_SAMPLES=500

# Spectral (SC) speed knobs (only used when --coarsening=sc / spectral)
# "Aggressive speed up" defaults (kmeans is used inside spectral.py):
# - n_init=1
# - tighter K search range (matches our benchmark sweet spot)
SPECTRAL_N_INIT=1
SPECTRAL_K_MIN_FACTOR=0.9
SPECTRAL_K_MAX_FACTOR=1.1

# Small dataset threshold — below this, run directly (no screen sessions)
SMALL_DATASET_THRESHOLD=10000

# Precomputed SMILES file (faster than CSV)
USE_PRECOMPUTED_SMILES=false
PRECOMPUTED_SMILES_DIR="$PROJECT_ROOT/data/moses_smiles"

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
        --coarsening=*)
            COARSENING_FILTER="${arg#*=}"
            ;;
        --tokenizer=*)
            TOKENIZER_FILTER="${arg#*=}"
            ;;
        --chunks=*)
            NUM_CHUNKS="${arg#*=}"
            ;;
        --train-samples=*)
            MOSES_SAMPLES="${arg#*=}"
            ;;
        --val-samples=*)
            MOSES_VAL_SAMPLES="${arg#*=}"
            ;;
        --spectral-n-init=*)
            SPECTRAL_N_INIT="${arg#*=}"
            ;;
        --spectral-k-min-factor=*)
            SPECTRAL_K_MIN_FACTOR="${arg#*=}"
            ;;
        --spectral-k-max-factor=*)
            SPECTRAL_K_MAX_FACTOR="${arg#*=}"
            ;;
        --output-dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --use-precomputed-smiles)
            USE_PRECOMPUTED_SMILES=true
            ;;
        --precomputed-smiles-dir=*)
            PRECOMPUTED_SMILES_DIR="${arg#*=}"
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
            echo "  --coarsening=STRATEGY  Filter coarsening: sc, hac, or all (default: all)"
            echo "  --tokenizer=TYPE     Filter tokenizer: hsent, hdt, or all (default: all)"
            echo "  --chunks=N           Number of parallel chunks (default: 8)"
            echo "  --train-samples=N    MOSES train samples (default: ${MOSES_SAMPLES})"
            echo "  --val-samples=N      MOSES val samples (default: ${MOSES_VAL_SAMPLES})"
            echo "  --spectral-n-init=N  Spectral n_init (default: ${SPECTRAL_N_INIT})"
            echo "  --spectral-k-min-factor=F  Spectral k_min_factor (default: ${SPECTRAL_K_MIN_FACTOR})"
            echo "  --spectral-k-max-factor=F  Spectral k_max_factor (default: ${SPECTRAL_K_MAX_FACTOR})"
            echo "  --output-dir=PATH    Cache directory (default: data/cache)"
            echo "  --use-precomputed-smiles  Use precomputed moses_smiles.txt (faster than CSV)"
            echo "  --precomputed-smiles-dir=PATH  Directory with moses_smiles.txt (default: data/moses_smiles)"
            echo ""
            echo "Datasets:"
            echo "  MOSES:   ${MOSES_SAMPLES} training samples (parallel screen sessions)"
            echo "  COCONUT: ${COCONUT_SAMPLES} train + ${COCONUT_VAL_SAMPLES} val samples (runs directly)"
            echo ""
            echo "Tokenizer/coarsening combos (default: all 4):"
            echo "  hsent:sc  hsent:hac"
            echo "  hdt:sc    hdt:hac"
            echo ""
            echo "Note: Only SC and HAC need precomputation. Other coarsening"
            echo "methods (MC, MAS) are fast enough to tokenize on-the-fly."
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
        sc) echo "spectral" ;;
        hac) echo "hac" ;;
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

    # Determine which coarsening strategies (only SC and HAC need precomputation)
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

# Run preprocessing for a single dataset
run_dataset() {
    local dataset="$1"
    local total_samples

    if [ "$dataset" = "moses" ]; then
        total_samples=$MOSES_SAMPLES
    else
        total_samples=$COCONUT_SAMPLES
    fi

    # Determine val samples for this dataset
    local val_samples=0
    if [ "$dataset" = "moses" ]; then
        val_samples=$MOSES_VAL_SAMPLES
    elif [ "$dataset" = "coconut" ]; then
        val_samples=$COCONUT_VAL_SAMPLES
    fi

    local combos
    read -ra combos <<< "$(build_combos)"

    echo "========================================"
    # Uppercase dataset name in a POSIX-compatible way
    DATASET_UPPER=$(printf '%s' "$dataset" | tr '[:lower:]' '[:upper:]')
    echo "Precompute Cache: $DATASET_UPPER"
    echo "========================================"
    echo ""
    echo "Settings:"
    echo "  Dataset: $dataset"
    echo "  Train samples: $total_samples"
    if [ $val_samples -gt 0 ]; then
        echo "  Val samples: $val_samples"
    fi
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

        # Extra args for preprocess_chunk.py (strategy-specific)
        SPECTRAL_ARGS=""
        if [ "$COARSENING_FULL" = "spectral" ]; then
            SPECTRAL_ARGS="--spectral-n-init $SPECTRAL_N_INIT --spectral-k-min-factor $SPECTRAL_K_MIN_FACTOR --spectral-k-max-factor $SPECTRAL_K_MAX_FACTOR"
            echo "  Spectral overrides: n_init=$SPECTRAL_N_INIT, k=[$SPECTRAL_K_MIN_FACTOR,$SPECTRAL_K_MAX_FACTOR] (kmeans)"
        fi

        # Helper: check if a combined cache file exists for a given split/size
        check_cache() {
            local check_split="$1"
            local check_size="$2"
            for f in $(find "$OUTPUT_DIR" -name "${dataset}_${check_split}_${TOKENIZER}_${check_size}_*.pt" -type f 2>/dev/null); do
                match=$(python -c "
import torch
d = torch.load('$f', weights_only=False)
cfg = d.get('tokenizer_config', {})
ok = (cfg.get('coarsening_strategy') == '$COARSENING_FULL')
if ok and '$COARSENING_FULL' == 'spectral':
    ok = (
        str(cfg.get('spectral_n_init')) == str($SPECTRAL_N_INIT)
        and str(cfg.get('spectral_k_min_factor')) == str($SPECTRAL_K_MIN_FACTOR)
        and str(cfg.get('spectral_k_max_factor')) == str($SPECTRAL_K_MAX_FACTOR)
    )
if ok:
    print('$f')
" 2>/dev/null || true)
                if [ -n "$match" ]; then
                    echo "$match"
                    return
                fi
            done
        }

        # Build dataset-specific args for preprocess_chunk.py
        local dataset_args="--dataset $dataset"
        if [ "$dataset" = "coconut" ]; then
            dataset_args="$dataset_args --data-file data/coconut_complex.smi"
            dataset_args="$dataset_args --min-atoms 20 --max-atoms 100 --min-rings 3"
        fi
        
        # Add precomputed SMILES flag if file exists or explicitly requested
        local precomputed_smiles_args=""
        if [ "$dataset" = "moses" ]; then
            local precomputed_file="$PRECOMPUTED_SMILES_DIR/moses_smiles.txt"
            if [ "$USE_PRECOMPUTED_SMILES" = true ] || [ -f "$precomputed_file" ]; then
                precomputed_smiles_args="--use-precomputed-smiles --precomputed-smiles-dir $PRECOMPUTED_SMILES_DIR"
                if [ -f "$precomputed_file" ]; then
                    echo "  Using precomputed SMILES file: $precomputed_file"
                fi
            fi
        fi

        if [ $total_samples -le $SMALL_DATASET_THRESHOLD ]; then
            # Small dataset: run directly (no screen sessions needed)
            echo "  Mode: direct (small dataset)"
            echo ""

            # --- Train split ---
            local existing_train
            existing_train=$(check_cache "train" "$total_samples")

            if [ "$FORCE" = false ] && [ -n "$existing_train" ]; then
                echo "  SKIP train: cache exists: $existing_train"
            else
                local output_file="$OUTPUT_DIR/${TOKENIZER}_${COARSENING_FULL}_chunk_0_${total_samples}.pt"

                local cmd="python scripts/preprocess/preprocess_chunk.py"
                cmd="$cmd --tokenizer $TOKENIZER"
                cmd="$cmd $dataset_args"
                cmd="$cmd --coarsening-strategy $COARSENING_FULL"
                cmd="$cmd $SPECTRAL_ARGS"
                cmd="$cmd $precomputed_smiles_args"
                cmd="$cmd --start 0"
                cmd="$cmd --end $total_samples"
                cmd="$cmd --output $output_file"

                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] Train precompute:"
                    echo "    $cmd"
                else
                    echo "  Running train preprocessing..."
                    eval $cmd
                fi

                local combine_cmd="python scripts/preprocess/combine_chunks.py"
                combine_cmd="$combine_cmd --tokenizer $TOKENIZER"
                combine_cmd="$combine_cmd --coarsening-strategy $COARSENING_FULL"
                combine_cmd="$combine_cmd --chunk_dir $OUTPUT_DIR"
                combine_cmd="$combine_cmd --split train"
                combine_cmd="$combine_cmd --dataset $dataset"

                if [ "$DRY_RUN" = true ]; then
                    echo "    $combine_cmd"
                else
                    echo "  Combining train chunks..."
                    eval $combine_cmd

                    rm -f "$output_file"
                    echo "  Cleaned up train chunk file"
                fi
            fi
            echo ""

            # --- Val split ---
            if [ $val_samples -gt 0 ]; then
                local existing_val
                existing_val=$(check_cache "val" "$val_samples")

                if [ "$FORCE" = false ] && [ -n "$existing_val" ]; then
                    echo "  SKIP val: cache exists: $existing_val"
                else
                    local val_start=$total_samples
                    local val_end=$((total_samples + val_samples))
                    local val_output_file="$OUTPUT_DIR/${TOKENIZER}_${COARSENING_FULL}_chunk_${val_start}_${val_end}.pt"

                    local val_cmd="python scripts/preprocess/preprocess_chunk.py"
                    val_cmd="$val_cmd --tokenizer $TOKENIZER"
                    val_cmd="$val_cmd $dataset_args"
                    val_cmd="$val_cmd --coarsening-strategy $COARSENING_FULL"
                    val_cmd="$val_cmd $SPECTRAL_ARGS"
                    val_cmd="$val_cmd $precomputed_smiles_args"
                    val_cmd="$val_cmd --start $val_start"
                    val_cmd="$val_cmd --end $val_end"
                    val_cmd="$val_cmd --output $val_output_file"

                    if [ "$DRY_RUN" = true ]; then
                        echo "  [DRY RUN] Val precompute:"
                        echo "    $val_cmd"
                    else
                        echo "  Running val preprocessing..."
                        eval $val_cmd
                    fi

                    local val_combine_cmd="python scripts/preprocess/combine_chunks.py"
                    val_combine_cmd="$val_combine_cmd --tokenizer $TOKENIZER"
                    val_combine_cmd="$val_combine_cmd --coarsening-strategy $COARSENING_FULL"
                    val_combine_cmd="$val_combine_cmd --chunk_dir $OUTPUT_DIR"
                    val_combine_cmd="$val_combine_cmd --split val"
                    val_combine_cmd="$val_combine_cmd --dataset $dataset"

                    if [ "$DRY_RUN" = true ]; then
                        echo "    $val_combine_cmd"
                    else
                        echo "  Combining val chunks..."
                        eval $val_combine_cmd

                        rm -f "$val_output_file"
                        echo "  Cleaned up val chunk file"
                    fi
                fi
                echo ""
            fi

        else
            # Large dataset: launch parallel screen sessions
            local existing_train
            existing_train=$(check_cache "train" "$total_samples")

            if [ "$FORCE" = false ] && [ -n "$existing_train" ]; then
                echo "  SKIP train: cache exists: $existing_train"
                echo "  Use --force to re-run"
                echo ""
                continue
            fi

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
                cmd="$cmd $SPECTRAL_ARGS"
                cmd="$cmd $precomputed_smiles_args"
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

            # Val split (always run directly; typically small)
            if [ $val_samples -gt 0 ]; then
                echo ""
                echo "  Val precompute: direct (${val_samples} samples)"

                local existing_val
                existing_val=$(check_cache "val" "$val_samples")

                if [ "$FORCE" = false ] && [ -n "$existing_val" ]; then
                    echo "  SKIP val: cache exists: $existing_val"
                else
                    local val_start=$total_samples
                    local val_end=$((total_samples + val_samples))
                    local val_output_file="$OUTPUT_DIR/${TOKENIZER}_${COARSENING_FULL}_chunk_${val_start}_${val_end}.pt"

                    local val_cmd="python scripts/preprocess/preprocess_chunk.py"
                    val_cmd="$val_cmd --tokenizer $TOKENIZER"
                    val_cmd="$val_cmd $dataset_args"
                    val_cmd="$val_cmd --coarsening-strategy $COARSENING_FULL"
                    val_cmd="$val_cmd $SPECTRAL_ARGS"
                    val_cmd="$val_cmd $precomputed_smiles_args"
                    val_cmd="$val_cmd --start $val_start"
                    val_cmd="$val_cmd --end $val_end"
                    val_cmd="$val_cmd --output $val_output_file"

                    if [ "$DRY_RUN" = true ]; then
                        echo "  [DRY RUN] Val precompute:"
                        echo "    $val_cmd"
                    else
                        echo "  Running val preprocessing..."
                        eval $val_cmd
                    fi

                    local val_combine_cmd="python scripts/preprocess/combine_chunks.py"
                    val_combine_cmd="$val_combine_cmd --tokenizer $TOKENIZER"
                    val_combine_cmd="$val_combine_cmd --coarsening-strategy $COARSENING_FULL"
                    val_combine_cmd="$val_combine_cmd --chunk_dir $OUTPUT_DIR"
                    val_combine_cmd="$val_combine_cmd --split val"
                    val_combine_cmd="$val_combine_cmd --dataset $dataset"

                    if [ "$DRY_RUN" = true ]; then
                        echo "    $val_combine_cmd"
                    else
                        echo "  Combining val chunks..."
                        eval $val_combine_cmd

                        rm -f "$val_output_file"
                        echo "  Cleaned up val chunk file"
                    fi
                fi
            fi

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
