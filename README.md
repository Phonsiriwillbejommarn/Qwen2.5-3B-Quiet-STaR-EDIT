# Quiet-STAR: Language Models Can Teach Themselves to Think Before Speaking

Extended implementation of **[Quiet-STAR](https://arxiv.org/abs/2403.09629)** on **Qwen2.5-3B**, optimized for a single **NVIDIA H200 GPU** (141 GB HBM3e).

---

## Table of Contents

- [Architecture](#architecture)
- [Training Pipeline](#training-pipeline)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Inference](#inference)
- [Testing](#testing)
- [File Structure](#file-structure)
- [Known Limitations](#known-limitations)

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Quiet-STAR Forward Pass                       │
│                                                                     │
│  Input: "The cat sat on the mat"                                    │
│         [tok₁] [tok₂] [tok₃] [tok₄] [tok₅] [tok₆]                │
│           │      │      │      │      │      │                      │
│    ┌──────▼──────▼──────▼──────▼──────▼──────▼──────┐               │
│    │         Base Forward Pass (Qwen2.5-3B)          │               │
│    │         → base_hidden_states [B, T, H]          │               │
│    │         → base_logits [B, T, V]                 │               │
│    └──────┬──────┬──────┬──────┬──────┬──────┬──────┘               │
│           │      │      │      │      │      │                      │
│    ┌──────▼──────▼──────▼──────▼──────▼──────▼──────┐               │
│    │          Thinking Gate (2-layer MLP)             │               │
│    │     score = σ(W₂·ReLU(W₁·hidden + b₁) + b₂)   │               │
│    │              ↓                                  │               │
│    │     [0.1] [0.9] [0.2] [0.8] [0.1] [0.95]      │               │
│    │     skip  THINK  skip  THINK  skip  THINK       │               │
│    └──────┬──────┬──────┬──────┬──────┬──────┬──────┘               │
│           │      │      │      │      │      │                      │
│    ┌──────│──────▼──────│──────▼──────│──────▼──────┐               │
│    │      │  Think Loop │ (n_ahead   │ steps)       │               │
│    │      │  ┌────────┐ │ ┌────────┐ │              │               │
│    │      │  │<start> │ │ │<start> │ │              │               │
│    │      │  │thought₁│ │ │thought₁│ │              │               │
│    │      │  │thought₂│ │ │thought₂│ │              │               │
│    │      │  │  ...   │ │ │  ...   │ │              │               │
│    │      │  │ <end>  │ │ │ <end>  │ │              │               │
│    │      │  └───┬────┘ │ └───┬────┘ │              │               │
│    │      │      │      │     │      │              │               │
│    │      │  Talk Loop  │ (n_ahead_talk steps)      │               │
│    │      │  → thought_logits                       │               │
│    └──────┬──────┬──────┬──────┬──────┬──────┬──────┘               │
│           │      │      │      │      │      │                      │
│    ┌──────▼──────▼──────▼──────▼──────▼──────▼──────┐               │
│    │              Mixing Head                        │               │
│    │  final = gate × thought + (1-gate) × base      │               │
│    └──────┬──────┬──────┬──────┬──────┬──────┬──────┘               │
│           │      │      │      │      │      │                      │
│         [out₁] [out₂] [out₃] [out₄] [out₅] [out₆]                │
│         improved next-token predictions                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Thinking Gate (`ThinkingGate`)

A lightweight 2-layer MLP that decides **per token position** whether thinking is needed.

```
Input:  base_hidden_states [B, T, H]
          │
          ▼
    Linear(H → 128) + ReLU
          │
          ▼
    Linear(128 → 1) + Sigmoid
          │
          ▼
Output: gate_scores [B, T, 1]  ∈ (0, 1)
```

**Key design decisions:**
- **Negative bias init (`-2.0`)**: Gate starts in "don't think" mode → learns to think only when needed (curriculum-style)
- **L1 sparsity penalty**: `loss += β × mean(gate)` discourages thinking everywhere
- **Training**: Soft mask (differentiable) — gradients flow through gate to backbone
- **Inference**: Hard binarization at threshold (default 0.5)

#### 2. Dynamic n_ahead (Inference Only)

Instead of binary on/off, gate confidence maps to **5 levels of thinking depth**:

```
┌──────────────┬───────────────┬────────────────────────┐
│  Confidence  │  Think Depth  │  Example (n_ahead=8)   │
├──────────────┼───────────────┼────────────────────────┤
│  0.0 – 0.2   │    0% (skip)  │  0 think steps         │
│  0.2 – 0.4   │   25%         │  2 think steps         │
│  0.4 – 0.6   │   50%         │  4 think steps         │
│  0.6 – 0.8   │   75%         │  6 think steps         │
│  0.8 – 1.0   │  100% (full)  │  8 think steps         │
└──────────────┴───────────────┴────────────────────────┘
```

Easy inputs skip thinking entirely. Hard inputs use full depth. **No manual tuning per prompt.**

#### 3. Chunk-level Thinking

Instead of thinking at every single token (T rounds), the sequence is split into windows:

```
Standard Quiet-STAR:     T think rounds  (slow)
Chunk Thinking:          T/C think rounds (fast)

Sequence: [tok₁ tok₂ tok₃ tok₄ | tok₅ tok₆ tok₇ tok₈ | tok₉ ...]
                 chunk 1         │       chunk 2        │   chunk 3
              1 think round      │    1 think round     │  1 think round
```

- Each thought sees `chunk_size` tokens of context (wider than per-token)
- Optional overlap between chunks for continuity
- Inter-chunk KV cache threading at inference

#### 4. Token-Space Thinking (o1-style)

Optional mode that generates **actual token sequences** as thoughts:

```
Input tokens → <think> "let me calculate 24×3 = 72" </think> → Output tokens
                        ↑ real tokens, decodable         ↑
                        generated via Gumbel-Softmax
                        (straight-through gradient)
```

**Best-of-N**: Generate N thought paths → select best via:
1. Self-consistency voting (majority answer)
2. PRM tiebreak (Process Reward Model scores)

#### 5. Reward Pipeline

```
                         ┌──────────────────────────────┐
                         │     Training Loss             │
                         │                              │
  Cross-Entropy ─────────┤  base next-token loss        │
                         │                              │
  REINFORCE ─────────────┤  policy gradient on thoughts │
                         │  reward = Δloss (think helps?)│
                         │                              │
  Verifiable Reward ─────┤  extract "#### answer"       │
  (GSM8K gold answers)   │  compare to gold → 0/1      │
                         │                              │
  PRM Loss ──────────────┤  per-step reasoning quality  │
                         │  (ProcessRewardModel)        │
                         │                              │
  LLM-as-a-Judge ────────┤  Qwen2.5-7B (4-bit) judges  │
  (optional)             │  "is this reasoning good?"   │
                         │  → YES/NO probability        │
                         └──────────────────────────────┘
```

All signals combined with configurable weights + reward warmup scheduler.

---

## Training Pipeline

### Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                           │
│                                                                 │
│  ┌─── Phase 0: SFT Warmup (optional) ────────────────────────┐ │
│  │  Dataset: GSM8K (teacher-forced)                           │ │
│  │  Mode:    original_mode=True (no thought generation)       │ │
│  │  Goal:    Learn <think> CoT format                         │ │
│  │  LR:     2e-5                                              │ │
│  │  Steps:  500                                               │ │
│  │  Time:   ~2-3 hours on H200                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌─── Phase 1: RL Training ──────────────────────────────────┐ │
│  │  Dataset: FineWeb-Edu (70%) + GSM8K (30%)                  │ │
│  │  Mode:    Quiet-STAR (think + talk + REINFORCE)            │ │
│  │  LR:     1e-5                                              │ │
│  │  Steps:  10,000                                            │ │
│  │  Time:   ~2-4 days on H200                                 │ │
│  │                                                            │ │
│  │  Reward Warmup (first 200 steps):                          │ │
│  │    verifiable_reward_weight: 0.0 → 0.5                     │ │
│  │    prm_loss_weight:         0.0 → target                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌─── Output ────────────────────────────────────────────────┐ │
│  │  Checkpoint saved to: ./outputs/quietstar_qwen25_3b_final │ │
│  │  WandB dashboard: training metrics, loss curves           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Pipeline?

| Phase | Purpose | Without It |
|---|---|---|
| SFT Warmup | Model learns `<think>` format before RL | RL has to discover format from scratch (slow) |
| FineWeb-Edu | Broad reasoning on educational text | Model only learns math |
| GSM8K Mix | Provides gold answers for verifiable reward | Reward signal has nothing to verify against |
| Reward Warmup | Prevents reward signal from destabilizing early training | High reward weight + untrained model = divergence |

---

## Quick Start

### 1. Prerequisites

```bash
# System requirements
- NVIDIA H200 (141 GB) or any GPU ≥ 40 GB VRAM
- CUDA 12.0+
- Python 3.10+

# Install dependencies
pip install -r requirements.txt

# Login to WandB (required for training logs)
wandb login
```

### 2. Run Training

```bash
# One command — runs the full pipeline
bash run_train.sh
```

This executes:
```bash
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
```

### 3. Monitor Training

```bash
# WandB dashboard shows:
#   - train/total_loss
#   - train/policy_reward
#   - train/gate_confidence (if gate enabled)
#   - eval/perplexity
```

### 4. Resume from Checkpoint

```bash
# Training auto-resumes from latest checkpoint in output_dir
python train.py <same args>
# The script detects existing checkpoints and continues
```

### 5. Run Inference

```bash
# Interactive chat
python inference.py --model_path ./outputs/quietstar_qwen25_3b_final

# Single prompt
python inference.py \
    --model_path ./outputs/quietstar_qwen25_3b_final \
    --prompt "What is 24 × 3?"

# Override think depth at inference
python inference.py \
    --model_path ./outputs/quietstar_qwen25_3b_final \
    --n_ahead 4 --n_passes 1
```

### Alternative Training Configurations

```bash
# Minimal — baseline REINFORCE only (fastest, no GSM8K)
python train.py

# With Thinking Gate (adds selective thinking)
python train.py --use_thinking_gate --thinking_gate_sparsity_beta 0.01

# With Chunk-level Thinking (faster per step)
python train.py --use_chunk_thinking --chunk_size 8

# With LLM-as-a-Judge (needs ~4 GB extra VRAM for 7B judge)
python train.py \
    --use_judge_reward \
    --judge_model_name Qwen/Qwen2.5-7B-Instruct

# Full pipeline with all features
python train.py \
    --use_sft_warmup \
    --use_thinking_gate \
    --use_verifiable_reward \
    --use_judge_reward \
    --use_chunk_thinking --chunk_size 8
```

---

## Configuration Reference

### Core Training

| Parameter | Default | Description |
|---|---|---|
| `--n_ahead` | 8 | Think tokens per position |
| `--n_ahead_talk` | 4 | Talk tokens after thought |
| `--n_passes` | 4 | Parallel thought paths (Best-of-N) |
| `--batch_size` | 2 | Per-device micro-batch |
| `--full_batch_size` | 16 | Effective batch (grad accum = 8) |
| `--learning_rate` | 1e-5 | Learning rate |
| `--max_steps` | 10,000 | Optimizer steps |
| `--max_length` | 1024 | Sequence length |
| `--gumbel_temperature` | 1.0 | Gumbel-Softmax temperature |

### Thinking Gate

| Parameter | Default | Description |
|---|---|---|
| `--use_thinking_gate` | off | Enable selective thinking |
| `--thinking_gate_hidden_dim` | 128 | Gate MLP hidden size |
| `--thinking_gate_sparsity_beta` | 0.01 | L1 penalty (↑ = thinks less) |
| `--thinking_gate_threshold` | 0.5 | Inference binarization cutoff |
| `--thinking_gate_bias_init` | -2.0 | Initial bias (neg = "don't think") |

### Chunk Thinking

| Parameter | Default | Description |
|---|---|---|
| `--use_chunk_thinking` | off | Enable chunk-level thinking |
| `--chunk_size` | 8 | Tokens per chunk |
| `--chunk_overlap` | 0 | Overlap between chunks |

### SFT & Reward Warmup

| Parameter | Default | Description |
|---|---|---|
| `--use_sft_warmup` | off | Teacher-forced CoT on GSM8K before RL |
| `--sft_warmup_steps` | 500 | SFT phase duration |
| `--sft_learning_rate` | 2e-5 | Learning rate during SFT |
| `--use_verifiable_reward` | off | Enable answer verification reward |
| `--verifiable_reward_weight` | 0.5 | Weight of verifiable reward |
| `--gsm8k_mix_ratio` | 0.3 | Fraction of GSM8K in mixed dataset |
| `--reward_warmup_steps` | 200 | Steps to ramp reward weights 0 → target |

### Estimated Training Time (H200)

```
┌──────────────┬──────────────┬────────────┬──────────────┐
│    Steps     │     Time     │  Cost ~$3/hr │   Includes   │
├──────────────┼──────────────┼────────────┼──────────────┤
│   500 (SFT)  │  2-3 hours   │   ~$8      │  Phase 0     │
│  1,000 (RL)  │  12 hours    │   ~$36     │  Phase 1     │
│ 10,000 (RL)  │  2-4 days    │  ~$150-290 │  Phase 1     │
│              │              │            │              │
│ Full default │  2-4 days    │  ~$160-300 │  Phase 0+1   │
└──────────────┴──────────────┴────────────┴──────────────┘

Compute multiplier: n_ahead(8) × n_passes(4) = 32x normal training
```

### H200 Memory Budget

```
┌──────────────────────────────────┬───────────┐
│         Component                │  Memory   │
├──────────────────────────────────┼───────────┤
│ Qwen2.5-3B (bf16)               │    ~6 GB  │
│ AdamW optimizer states           │   ~18 GB  │
│ Activations + gradients          │ ~30-50 GB │
│ KV cache + thought embeddings    │  ~5-10 GB │
│ ThinkingGate MLP                 │    <1 MB  │
│ Judge model (7B, 4-bit, optional)│    ~4 GB  │
├──────────────────────────────────┼───────────┤
│ Total                            │ ~59-88 GB │
│ H200 VRAM                        │   141 GB  │
│ Headroom                         │  ~53+ GB  │
└──────────────────────────────────┴───────────┘
```

---

## Inference

### Interactive Chat

```bash
python inference.py --model_path ./outputs/quietstar_qwen25_3b_final
```

```
You: What is 24 × 3?
Model: 24 × 3 = 72

You: ppl: The quick brown fox jumps over the lazy dog
Perplexity: 12.34

You: quit
```

### Programmatic Usage

```python
from inference import load_model, generate_text

model, tokenizer = load_model("./outputs/quietstar_qwen25_3b_final")
response = generate_text(model, tokenizer, "What is 2 + 3 * 4?")
print(response)
```

### Tuning Dynamic Depth at Inference

```python
# Adjust depth levels
model.thinking_gate_levels = [
    (0.0,  0.20, 0.00),   # very easy → skip
    (0.20, 0.40, 0.25),   # easy      → 25%
    (0.40, 0.60, 0.50),   # medium    → 50%
    (0.60, 0.80, 0.75),   # hard      → 75%
    (0.80, 1.01, 1.00),   # very hard → full
]
model.thinking_gate_min_ahead = 0  # allow complete skip

# Inspect after forward pass
print(model._last_gate_confidence)    # e.g. 0.23
print(model._last_effective_n_ahead)  # e.g. 2 (out of 8)
```

---

## Testing

```bash
# Quick import check
python test_import.py

# Full test suite (8 tests)
python test_forward.py
```

Tests cover:
1. Baseline (no chunk) forward pass
2. Chunk thinking shape and loss
3. Chunk vs token-level logits differ
4. Various chunk sizes (2, 4, 8, 16)
5. Chunk overlap (0, 1, 2)
6. Full sequence coverage (no gaps)
7. Chunk + Gate + KV Cache combined
8. `use_chunk_thinking` flag restore after exception

---

## File Structure

```
Qwen2.5-3b-Quiet-StaR-Edit/
│
├── config.py                  # QuietStarConfig — all model hyperparameters
│                              #   gate, chunk, PRM, verifiable reward params
│
├── modeling_quiet_star.py     # Core model (95K lines)
│   ├── ThinkingGate           #   per-position gate MLP
│   ├── ProcessRewardModel     #   per-step reasoning scorer
│   ├── VerifiableRewardComputer#  answer extraction + verification
│   ├── ReasoningJudge         #   LLM-as-a-Judge (Qwen2.5-7B 4-bit)
│   ├── _chunk_forward()       #   chunk-level thinking
│   ├── _generate_thought_tokens()  # Gumbel-Softmax token generation
│   ├── _token_space_forward() #   o1-style token-space thinking
│   ├── _run_best_of_n()       #   Best-of-N candidate selection
│   ├── _self_consistency_select()  # majority vote + PRM tiebreak
│   └── forward()              #   main forward pass (REINFORCE + all rewards)
│
├── train.py                   # Training script
│   ├── SFT Warmup phase       #   teacher-forced CoT on GSM8K
│   ├── RewardWarmupCallback   #   ramp reward weights 0 → target
│   ├── GSM8K + FineWeb mixing #   verifiable reward dataset
│   └── main()                 #   full pipeline orchestration
│
├── eval_helpers.py            # Evaluation
│   ├── preprocess_function    #   FineWeb-Edu tokenization + chunking
│   ├── preprocess_gsm8k_sft   #   GSM8K → <think> format
│   └── compute_metrics        #   perplexity (PyTorch native, no scipy OOM)
│
├── inference.py               # Inference with thought token suppression
├── test_forward.py            # 8 unit tests
├── test_import.py             # Import sanity check
├── run_train.sh               # One-command cloud launch script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Known Limitations

1. **Train/Inference Gap (Gate)**: Soft mask during training, hard binary at inference — behavior may differ at boundary cases
2. **Dynamic n_ahead is sequence-level**: Uses `mean(gate_scores)` across all positions, not per-position adaptive depth
3. **No inter-chunk context during training**: KV cache disabled during training to avoid gradient checkpointing conflicts; chunks train independently
4. **Reward signal conflicts**: REINFORCE + PRM + Verifiable + Judge use static weights — no adaptive conflict resolution when signals disagree
5. **Gumbel temperature is constant**: `gumbel_temperature=1.0` throughout training; annealing from high→low would improve exploration→exploitation

---

## Citation

```bibtex
@article{zelikman2024quiet,
  title={Quiet-{STaR}: Language Models Can Teach Themselves to Think Before Speaking},
  author={Zelikman, Eric and Harik, Georges and Shao, Yijia and Jayasiri, Varuna and Haber, Nick and Goodman, Noah D.},
  journal={arXiv preprint arXiv:2403.09629},
  year={2024}
}
```

## License

Apache License 2.0