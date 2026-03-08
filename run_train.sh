#!/bin/bash
# ============================================================================
# Quiet-STAR Training Launch Script (NVIDIA H200)
# ============================================================================
# Usage:  bash run_train.sh
#
# Pipeline:
#   Phase 0: SFT Warmup on GSM8K (500 steps, teacher-forced CoT)
#   Phase 1: RL Training on FineWeb-Edu + GSM8K mix (10K steps)
#            - REINFORCE policy gradient
#            - Verifiable reward (GSM8K answers)
#            - Reward warmup (ramp from 0 → target over 200 steps)
# ============================================================================

set -e

echo "========================================"
echo "  Quiet-STAR Training (Qwen2.5-3B)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

python train.py \
    --n_ahead 8 \
    --n_ahead_talk 4 \
    --n_passes 4 \
    --batch_size 2 \
    --full_batch_size 16 \
    --learning_rate 1e-5 \
    --max_steps 10000 \
    --warmup_steps 20 \
    --max_length 1024 \
    --n_examples 10000 \
    --save_steps 100 \
    --eval_steps 200 \
    --logging_steps 1 \
    --gumbel_temperature 1.0 \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --use_sft_warmup \
    --sft_warmup_steps 500 \
    --use_verifiable_reward \
    --verifiable_reward_weight 0.5 \
    --gsm8k_mix_ratio 0.3 \
    --reward_warmup_steps 200

echo ""
echo "========================================"
echo "  ✅ Training complete!"
echo "========================================"
