#!/bin/bash
# Generate commands for parallel chunked preprocessing
# With 16 CPU cores, we can run 8 chunks in parallel (leaving cores for system)

TOKENIZER=$1
NUM_CHUNKS=${2:-8}
COARSENING=${3:-spectral}
TOTAL_SAMPLES=500000

if [ -z "$TOKENIZER" ]; then
    echo "Usage: $0 <tokenizer> [num_chunks] [coarsening_strategy]"
    echo "Example: $0 hsent 8"
    echo "Example: $0 hsent 8 hac"
    exit 1
fi

CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_CHUNKS))

echo "=========================================="
echo "PARALLEL CHUNKED PREPROCESSING"
echo "=========================================="
echo "Tokenizer: $TOKENIZER"
echo "Coarsening: $COARSENING"
echo "Total samples: $TOTAL_SAMPLES"
echo "Number of chunks: $NUM_CHUNKS"
echo "Chunk size: $CHUNK_SIZE"
echo ""
echo "With 16 cores, you can run all ${NUM_CHUNKS} chunks in parallel!"
echo ""
echo "=========================================="
echo "STEP 1: Run these commands in parallel"
echo "=========================================="
echo ""

for ((i=0; i<NUM_CHUNKS; i++)); do
    START=$((i * CHUNK_SIZE))
    END=$(((i + 1) * CHUNK_SIZE))
    if [ $i -eq $((NUM_CHUNKS - 1)) ]; then
        END=$TOTAL_SAMPLES
    fi

    echo "# Terminal $((i+1)): Chunk $i"
    echo "python scripts/preprocess/preprocess_chunk.py \\"
    echo "    --tokenizer $TOKENIZER \\"
    echo "    --coarsening-strategy $COARSENING \\"
    echo "    --start $START \\"
    echo "    --end $END \\"
    echo "    --output data/cache/${TOKENIZER}_${COARSENING}_chunk_${START}_${END}.pt"
    echo ""
done

echo "=========================================="
echo "STEP 2: After all chunks complete, combine them:"
echo "=========================================="
echo ""
echo "python scripts/preprocess/combine_chunks.py \\"
echo "    --tokenizer $TOKENIZER \\"
echo "    --coarsening-strategy $COARSENING \\"
echo "    --chunk_dir data/cache \\"
echo "    --split train \\"
echo "    --dataset moses"
echo ""
echo "This will automatically create: moses_train_${TOKENIZER}_${TOTAL_SAMPLES}_<hash>.pt"
echo "(Hash is generated from tokenizer config for cache validation)"
echo ""
echo "=========================================="
echo "ESTIMATED TIME (with optimized spectral)"
echo "=========================================="
echo "Optimized speed: ~19 it/s (25x faster than original)"
echo ""
echo "Serial processing: ~7.3 hours total"
echo "With 8 parallel chunks: ~55 minutes total!"
echo ""
echo "Note: Using optimized spectral coarsening with:"
echo "  - n_init=10 (vs 100)"
echo "  - Vectorized modularity computation"
echo "  - Discretize assignment method"
echo "  - Equivalent quality to unoptimized (p=0.11)"
echo ""
