#!/bin/bash
# Quick test script to validate DDP setup before full training
#
# Usage:
#   bash configs/test_ddp.sh

set -e

echo "=========================================="
echo "DDP Setup Validation Tests"
echo "=========================================="
echo ""

# Test 1: Single GPU baseline (should always work)
echo "[Test 1/4] Single GPU baseline (batch=32)..."
python scripts/train.py \
  trainer.devices=1 \
  data.batch_size=32 \
  data.num_train=1000 \
  trainer.max_steps=50 \
  trainer.val_check_interval=25 \
  wandb.enabled=false \
  logs.run_name=test_single_gpu

echo "✓ Single GPU test passed"
echo ""

# Test 2: Single GPU large batch
echo "[Test 2/4] Single GPU large batch (batch=128)..."
python scripts/train.py \
  trainer.devices=1 \
  data.batch_size=128 \
  model.learning_rate=1.2e-3 \
  model.warmup_steps=100 \
  data.num_train=1000 \
  trainer.max_steps=50 \
  trainer.val_check_interval=25 \
  wandb.enabled=false \
  logs.run_name=test_large_batch

echo "✓ Large batch test passed"
echo ""

# Test 3: DDP 2 GPU (only if 2+ GPUs available)
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "[Test 3/4] DDP 2 GPUs (batch=32 per GPU)..."
    python scripts/train.py \
      trainer.devices=2 \
      trainer.strategy=ddp \
      data.batch_size=32 \
      model.learning_rate=8.5e-4 \
      model.warmup_steps=100 \
      data.num_train=1000 \
      trainer.max_steps=50 \
      trainer.val_check_interval=25 \
      wandb.enabled=false \
      logs.run_name=test_ddp_2gpu

    echo "✓ DDP 2 GPU test passed"
    echo ""
else
    echo "[Test 3/4] Skipping DDP test (only $NUM_GPUS GPU available)"
    echo ""
fi

# Test 4: Throughput benchmark
echo "[Test 4/4] Throughput benchmark..."
echo "Measuring samples/sec for different batch sizes..."
echo ""

for BS in 32 64 128 256; do
    echo "  Testing batch_size=$BS..."
    START=$(date +%s.%N)

    python scripts/train.py \
      trainer.devices=1 \
      data.batch_size=$BS \
      data.num_train=1000 \
      trainer.max_steps=100 \
      wandb.enabled=false \
      logs.run_name=test_throughput_bs${BS} \
      2>&1 | grep -E "(it/s|s/it)" | head -1 || true

    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    SAMPLES=$((BS * 100))
    THROUGHPUT=$(echo "scale=2; $SAMPLES / $ELAPSED" | bc)

    echo "    Batch $BS: ~$THROUGHPUT samples/sec"
done

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Single GPU training works"
echo "  ✓ Large batch size works"
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "  ✓ DDP multi-GPU works"
else
    echo "  - DDP not tested (need 2+ GPUs)"
fi
echo "  ✓ Throughput benchmarked"
echo ""
echo "You're ready for DDP training! See configs/train_ddp_examples.yaml for recommended settings."
