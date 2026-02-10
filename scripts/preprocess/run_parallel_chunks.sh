#!/bin/bash
# Automatically run parallel chunked preprocessing in screen sessions
#
# Usage:
#   ./scripts/preprocess/run_parallel_chunks.sh <tokenizer> [num_chunks] [output_dir]
#
# Example:
#   ./scripts/preprocess/run_parallel_chunks.sh hsent 8 data/cache
#   ./scripts/preprocess/run_parallel_chunks.sh hdt 16 /path/to/cache

set -e  # Exit on error

TOKENIZER=$1
NUM_CHUNKS=${2:-8}
OUTPUT_DIR=${3:-data/cache}
TOTAL_SAMPLES=500000

if [ -z "$TOKENIZER" ]; then
    echo "Error: Tokenizer not specified"
    echo "Usage: $0 <tokenizer> [num_chunks] [output_dir]"
    echo "Example: $0 hsent 8 data/cache"
    exit 1
fi

if [ "$TOKENIZER" != "hsent" ] && [ "$TOKENIZER" != "hdt" ]; then
    echo "Error: Tokenizer must be 'hsent' or 'hdt'"
    exit 1
fi

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "Error: 'screen' is not installed"
    echo "Install with: sudo apt-get install screen"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_CHUNKS))

echo "=========================================="
echo "PARALLEL CHUNKED PREPROCESSING"
echo "=========================================="
echo "Tokenizer: $TOKENIZER"
echo "Total samples: $TOTAL_SAMPLES"
echo "Number of chunks: $NUM_CHUNKS"
echo "Chunk size: $CHUNK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Creating $NUM_CHUNKS screen sessions..."
echo "=========================================="
echo ""

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Start each chunk in a separate screen session
for ((i=0; i<NUM_CHUNKS; i++)); do
    START=$((i * CHUNK_SIZE))
    END=$(((i + 1) * CHUNK_SIZE))

    # Last chunk gets any remaining samples
    if [ $i -eq $((NUM_CHUNKS - 1)) ]; then
        END=$TOTAL_SAMPLES
    fi

    SCREEN_NAME="preprocess_${TOKENIZER}_chunk${i}"
    OUTPUT_FILE="$OUTPUT_DIR/${TOKENIZER}_chunk_${START}_${END}.pt"

    echo "Starting chunk $i in screen '$SCREEN_NAME'"
    echo "  Range: [$START:$END]"
    echo "  Output: $OUTPUT_FILE"

    # Create screen session and run preprocessing
    screen -dmS "$SCREEN_NAME" bash -c "
        cd '$PROJECT_ROOT'
        source /home/andrew/miniconda3/bin/activate mosaic

        echo '=========================================='
        echo 'Chunk $i: Processing [$START:$END]'
        echo '=========================================='

        python scripts/preprocess/preprocess_chunk.py \
            --tokenizer $TOKENIZER \
            --start $START \
            --end $END \
            --output '$OUTPUT_FILE'

        EXIT_CODE=\$?

        if [ \$EXIT_CODE -eq 0 ]; then
            echo ''
            echo '✓ Chunk $i completed successfully!'
            echo 'Press any key to close this screen...'
        else
            echo ''
            echo '✗ Chunk $i failed with exit code '\$EXIT_CODE
            echo 'Press any key to close this screen...'
        fi

        read -n 1
    "

    # Small delay to avoid race conditions
    sleep 0.5
done

echo ""
echo "=========================================="
echo "All chunks started!"
echo "=========================================="
echo ""
echo "Screen sessions created:"
for ((i=0; i<NUM_CHUNKS; i++)); do
    echo "  - preprocess_${TOKENIZER}_chunk${i}"
done
echo ""
echo "Useful commands:"
echo "  screen -ls                           # List all screen sessions"
echo "  screen -r preprocess_${TOKENIZER}_chunk0  # Attach to chunk 0"
echo "  screen -X -S preprocess_${TOKENIZER}_chunk0 quit  # Kill chunk 0"
echo "  pkill -f 'preprocess_chunk.py'       # Kill all preprocessing (emergency)"
echo ""
echo "Monitor progress:"
echo "  watch -n 5 'ls -lh $OUTPUT_DIR/${TOKENIZER}_chunk_*.pt | wc -l'"
echo ""
echo "After all chunks complete, combine them:"
echo "  python scripts/preprocess/combine_chunks.py \\"
echo "      --tokenizer $TOKENIZER \\"
echo "      --chunk_dir $OUTPUT_DIR \\"
echo "      --split train \\"
echo "      --dataset moses"
echo ""
echo "=========================================="
echo "ESTIMATED TIME"
echo "=========================================="
echo "Speed: ~19 it/s (optimized spectral)"
echo "Per chunk: ~$((CHUNK_SIZE / 19 / 60)) minutes"
echo "With $NUM_CHUNKS parallel chunks: ~$((CHUNK_SIZE / 19 / 60)) minutes total!"
echo ""
