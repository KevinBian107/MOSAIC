#!/bin/bash
# DDP Training Performance Test - NO generation/testing, pure training throughput
#
# Usage:
#   bash configs/test_ddp.sh

set -e

echo "=========================================="
echo "DDP Training Performance Test"
echo "=========================================="
echo ""

# Test 1: Throughput benchmark - find optimal batch size
echo "[1/4] Throughput benchmark (finding optimal batch size)..."
echo "Testing different batch sizes on single GPU..."
echo ""

for BS in 32 64 128 256; do
    echo "  Testing batch_size=$BS..."
    START=$(date +%s.%N)

    python scripts/train.py \
      trainer.devices=1 \
      data.batch_size=$BS \
      data.num_train=2000 \
      trainer.max_steps=100 \
      trainer.limit_val_batches=0 \
      wandb.enabled=false \
      wandb.eval_every_n_val=0 \
      sampling.num_samples=0 \
      logs.run_name=throughput_bs${BS}

    END=$(date +%s.%N)
    SAMPLES=$((BS * 100))

    # Use Python for calculation instead of bc
    THROUGHPUT=$(python3 -c "print(f'{$SAMPLES / ($END - $START):.2f}')")

    echo "    Batch $BS: ~$THROUGHPUT samples/sec"
    echo ""
done

echo "Compare samples/sec above - higher is better!"
echo ""

# Test 2: Single GPU baseline
echo "[2/4] Single GPU baseline (batch=32)..."
time python scripts/train.py \
  trainer.devices=1 \
  data.batch_size=32 \
  data.num_train=5000 \
  trainer.max_steps=500 \
  trainer.limit_val_batches=0 \
  wandb.enabled=false \
  wandb.eval_every_n_val=0 \
  sampling.num_samples=0 \
  logs.run_name=perf_1gpu_bs32

echo ""

# Test 3: Single GPU large batch
echo "[3/4] Single GPU large batch (batch=128)..."
time python scripts/train.py \
  trainer.devices=1 \
  data.batch_size=128 \
  model.learning_rate=1.2e-3 \
  data.num_train=5000 \
  trainer.max_steps=125 \
  trainer.limit_val_batches=0 \
  wandb.enabled=false \
  wandb.eval_every_n_val=0 \
  sampling.num_samples=0 \
  logs.run_name=perf_1gpu_bs128

echo ""

# Test 4: 2 GPU DDP (if available)
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "[4/4] DDP 2 GPUs (batch=64 per GPU, effective=128)..."
    echo "NOTE: data.batch_size is PER-GPU. Effective batch = batch_size × num_GPUs"
    time python scripts/train.py \
      trainer.devices=2 \
      trainer.strategy=ddp \
      data.batch_size=64 \
      model.learning_rate=1.2e-3 \
      data.num_train=5000 \
      trainer.max_steps=125 \
      trainer.limit_val_batches=0 \
      wandb.enabled=false \
      wandb.eval_every_n_val=0 \
      sampling.num_samples=0 \
      logs.run_name=perf_2gpu_bs64
else
    echo "[4/4] Skipping (only $NUM_GPUS GPU available)"
fi

echo ""
echo "=========================================="
echo "Performance test complete!"
echo "=========================================="
echo ""
echo "How to read results:"
echo "  - 'it/s' = iterations per second (higher is better)"
echo "  - 'time' output = wall-clock time (lower is better)"
echo ""
echo "Expected speedups:"
echo "  - 1 GPU batch=128 vs batch=32:  ~2-2.5x faster"
echo "  - 2 GPU DDP vs 1 GPU batch=128:  ~1.7-1.9x faster"
echo ""
echo "DDP Batch Size Notes:"
echo "  - data.batch_size is PER-GPU (not total)"
echo "  - 2 GPUs × batch=64 = effective batch of 128"
echo "  - 2 GPUs × batch=128 = effective batch of 256"
echo "  - If batch=128 fits on 1 GPU, you can run batch=128 per GPU on DDP!"
echo ""
