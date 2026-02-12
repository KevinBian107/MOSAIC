#!/bin/bash
# DDP Speedup Comparison Test
# Compares 1 GPU vs 2 GPU DDP training with same epochs and data size
#
# Usage:
#   bash configs/test_ddp_speedup.sh

set -e

echo "=========================================="
echo "DDP Speedup Comparison Test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Data size: $DATA_SIZE samples"
echo "  - Batch size: $BATCH_SIZE (per GPU for DDP)"
echo "  - Epochs: $EPOCHS"
echo "  - 1 GPU: effective batch = $BATCH_SIZE"
echo "  - 2 GPU DDP: effective batch = $((BATCH_SIZE * 2)) (${BATCH_SIZE} × 2)"
echo ""
echo "⚠️  WARNING: If you have cached data from smaller runs, delete data/cache/ first!"
echo "  Run: rm -rf data/cache/*.pt"
echo ""

# Common settings
DATA_SIZE=500000
BATCH_SIZE=64
EPOCHS=4
BASE_LR=6e-4

# Calculate steps per epoch
# 1 GPU: 10000 / 64 = 156 steps per epoch
# 2 GPU: 10000 / 128 = 78 steps per epoch (data split between GPUs)
STEPS_1GPU=$((DATA_SIZE / BATCH_SIZE * EPOCHS))      # 312 steps total
STEPS_2GPU=$((DATA_SIZE / (BATCH_SIZE * 2) * EPOCHS)) # 156 steps total

echo "Steps calculation:"
echo "  1 GPU: $DATA_SIZE samples ÷ $BATCH_SIZE batch = $((DATA_SIZE / BATCH_SIZE)) steps/epoch × $EPOCHS epochs = $STEPS_1GPU total steps"
echo "  2 GPU: $DATA_SIZE samples ÷ ($BATCH_SIZE × 2) = $((DATA_SIZE / (BATCH_SIZE * 2))) steps/epoch × $EPOCHS epochs = $STEPS_2GPU total steps"
echo ""

# Test 1: Single GPU baseline
echo "[1/2] Single GPU (batch=64)..."
echo "Running $EPOCHS epochs ($STEPS_1GPU steps)..."
echo "DEBUG: data.num_train=$DATA_SIZE, trainer.max_steps=$STEPS_1GPU"
START_1GPU=$(date +%s.%N)

# Run without grep first to see full output
python scripts/train.py \
  trainer.devices=1 \
  data.batch_size=$BATCH_SIZE \
  model.learning_rate=$BASE_LR \
  data.num_train=$DATA_SIZE \
  data.use_cache=false \
  trainer.max_steps=$STEPS_1GPU \
  trainer.limit_val_batches=0 \
  wandb.enabled=false \
  wandb.eval_every_n_val=0 \
  sampling.num_samples=0 \
  logs.run_name=speedup_1gpu_bs64 \
  2>&1 | tee /tmp/ddp_test_output.log | grep -E "(Training dataset size:|Steps per epoch:|Epoch.*it/s)" | tail -5

END_1GPU=$(date +%s.%N)
TIME_1GPU=$(python3 -c "print(f'{$END_1GPU - $START_1GPU:.2f}')")
THROUGHPUT_1GPU=$(python3 -c "print(f'{($DATA_SIZE * $EPOCHS) / ($END_1GPU - $START_1GPU):.2f}')")

echo "✓ 1 GPU completed: ${TIME_1GPU}s (${THROUGHPUT_1GPU} samples/sec)"
echo ""

# Test 2: 2 GPU DDP (if available)
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "[2/2] DDP 2 GPUs (batch=64 per GPU, effective=128)..."
    echo "Running $EPOCHS epochs ($STEPS_2GPU steps)..."

    # For DDP with larger effective batch, scale learning rate
    DDP_LR=$(python3 -c "import math; print(f'{$BASE_LR * math.sqrt(2):.6f}')")
    echo "Learning rate scaled: $BASE_LR → $DDP_LR (sqrt(2) scaling)"

    START_2GPU=$(date +%s.%N)

    python scripts/train.py \
      trainer.devices=2 \
      trainer.strategy=ddp \
      data.batch_size=$BATCH_SIZE \
      model.learning_rate=$DDP_LR \
      data.num_train=$DATA_SIZE \
      data.use_cache=false \
      trainer.max_steps=$STEPS_2GPU \
      trainer.limit_val_batches=0 \
      wandb.enabled=false \
      wandb.eval_every_n_val=0 \
      sampling.num_samples=0 \
      logs.run_name=speedup_2gpu_bs64 \
      2>&1 | grep -E "Epoch.*it/s" | tail -1

    END_2GPU=$(date +%s.%N)
    TIME_2GPU=$(python3 -c "print(f'{$END_2GPU - $START_2GPU:.2f}')")
    THROUGHPUT_2GPU=$(python3 -c "print(f'{($DATA_SIZE * $EPOCHS) / ($END_2GPU - $START_2GPU):.2f}')")

    echo "✓ 2 GPU DDP completed: ${TIME_2GPU}s (${THROUGHPUT_2GPU} samples/sec)"
    echo ""

    # Calculate speedup
    SPEEDUP=$(python3 -c "print(f'{$TIME_1GPU / $TIME_2GPU:.2f}')")
    TIME_SAVED=$(python3 -c "print(f'{$TIME_1GPU - $TIME_2GPU:.2f}')")
    SPEEDUP_PERCENT=$(python3 -c "print(f'{(($TIME_1GPU - $TIME_2GPU) / $TIME_1GPU * 100):.1f}')")
    THROUGHPUT_GAIN=$(python3 -c "print(f'{$THROUGHPUT_2GPU / $THROUGHPUT_1GPU:.2f}')")

    echo "=========================================="
    echo "RESULTS SUMMARY"
    echo "=========================================="
    echo ""
    echo "Training Configuration:"
    echo "  - Dataset: $DATA_SIZE samples"
    echo "  - Epochs: $EPOCHS"
    echo "  - Total samples processed: $((DATA_SIZE * EPOCHS))"
    echo ""
    echo "Performance:"
    echo "  ┌─────────────┬────────────┬──────────────┬────────────────┐"
    echo "  │ Setup       │ Time (s)   │ Samples/sec  │ Effective Batch│"
    echo "  ├─────────────┼────────────┼──────────────┼────────────────┤"
    printf "  │ 1 GPU       │ %-10s │ %-12s │ %-14s │\n" "$TIME_1GPU" "$THROUGHPUT_1GPU" "64"
    printf "  │ 2 GPU DDP   │ %-10s │ %-12s │ %-14s │\n" "$TIME_2GPU" "$THROUGHPUT_2GPU" "128"
    echo "  └─────────────┴────────────┴──────────────┴────────────────┘"
    echo ""
    echo "Speedup Metrics:"
    echo "  • Wall-clock speedup: ${SPEEDUP}x"
    echo "  • Time saved: ${TIME_SAVED} seconds (${SPEEDUP_PERCENT}% reduction)"
    echo "  • Throughput increase: ${THROUGHPUT_GAIN}x"
    echo ""

    # Efficiency analysis
    EFFICIENCY=$(python3 -c "print(f'{$SPEEDUP / 2 * 100:.1f}')")
    echo "Efficiency Analysis:"
    echo "  • Ideal speedup (2 GPUs): 2.0x"
    echo "  • Actual speedup: ${SPEEDUP}x"
    echo "  • Scaling efficiency: ${EFFICIENCY}%"

    if (( $(echo "$SPEEDUP > 1.5" | python3 -c "import sys; print(float(input()))" ) )); then
        echo "  • ✅ Good scaling! DDP is effective for this workload."
    else
        echo "  • ⚠️ Suboptimal scaling. Consider larger batch sizes or models."
    fi

else
    echo "[2/2] Skipping DDP test (only $NUM_GPUS GPU available)"
    echo ""
    echo "To test DDP speedup, you need at least 2 GPUs."
fi

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
