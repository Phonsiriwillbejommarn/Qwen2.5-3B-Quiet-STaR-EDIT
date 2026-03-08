"""
Quiet-STAR Training Script
Optimized for a single NVIDIA H200 GPU (141 GB HBM3e)
Using Qwen2.5-3B as base model.

Based on: https://arxiv.org/abs/2403.09629

Usage:
    # First run
    python train.py --hf_repo_id your-username/quiet-star-qwen2.5-3b

    # Resume after GPU crash
    python train.py --hf_repo_id your-username/quiet-star-qwen2.5-3b \
                    --resume_from_checkpoint ./outputs/quietstar_XXXXX/checkpoint-500
"""

import os
import sys
import time
import random
import argparse
import logging

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_dataset

from config import QuietStarConfig
from modeling_quiet_star import QuietStarQwen2ForCausalLM
from eval_helpers import (
    set_tokenizer,
    preprocess_function,
    preprocess_gsm8k_sft,
    preprocess_eval_function_gsm,
    preprocess_eval_function_csqa,
    compute_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Default Hyperparameters
# ============================================================================

DEFAULT_CONFIG = {
    # Model — Qwen2.5-3B (much lighter than Mistral-7B, fits easily on H200)
    "model_name": "Qwen/Qwen2.5-3B",

    # Thought parameters
    "n_ahead": 8,            # Number of thought tokens (reduced from 32 for 3B stability)
    "n_ahead_talk": 4,      # Tokens ahead to predict after thought
    "n_passes": 4,          # Number of parallel thought paths to generate and evaluate

    # Training — optimize memory for H200
    "batch_size": 2,        # Per-device batch size (increased to utilize the 141GB VRAM)
    "full_batch_size": 16,  # Total effective batch size (uses gradient accumulation)
    "learning_rate": 1e-5,
    "max_steps": 10000,
    "warmup_steps": 20,
    "weight_decay": 0.001,
    "max_grad_norm": 1.0,
    "max_length": 1024,     # Sequence length (can be larger with 3B model)

    # Dataset — FineWeb-Edu: high-quality educational web text
    # Best for Quiet-STAR because:
    # 1. Educational text contains implicit reasoning steps (proofs, explanations)
    # 2. Filtered for educational quality (score >= 3)
    # 3. Higher quality than C4, broader than OpenWebMath
    # Alternative: "open-web-math/open-web-math" for math-focused training
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_subset": "default",
    "n_examples": 10000,    # Number of training examples

    # Evaluation & checkpointing
    "logging_steps": 1,             # Log training metrics every step
    "eval_steps": 50,               # Evaluate on GSM8K/CSQA every 50 steps (prevents slowing down)
    "save_steps": 10,               # Save checkpoint every 20 steps

    # Quiet-STAR specific
    "gumbel_temperature": 1.0,
    "use_start_thought_token": True,
    "use_end_thought_token": True,
    "include_policy_loss": True,
    "gumbel_detach": True,
    "merged_talk_heads": True,
    "residual_think_head": False,
    "optimize_lm_head_only_at_start": False,

    # Paths
    "output_dir": "./outputs",
    "cache_dir": "./cache",

    # HuggingFace Hub (for checkpoint backup)
    "hf_repo_id": "Phonsiri/Qwen2.5-3b-Quiet-STaR-Edit",    # HuggingFace Hub repo for checkpoints
    "resume_from_checkpoint": None,  # Path to checkpoint dir to resume from

    # API Keys (set these to your keys)
    "hf_token": None,       # HuggingFace API token
    "wandb_key": None,      # Weights & Biases API key

    # Wandb
    "use_wandb": True,
    "wandb_project": "quiet-star-qwen2.5-3b",

    # Random seed
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quiet-STAR Training (Qwen2.5-3B)")

    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--n_ahead", type=int, default=DEFAULT_CONFIG["n_ahead"])
    parser.add_argument("--n_ahead_talk", type=int, default=DEFAULT_CONFIG["n_ahead_talk"])
    parser.add_argument("--n_passes", type=int, default=DEFAULT_CONFIG["n_passes"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--full_batch_size", type=int, default=DEFAULT_CONFIG["full_batch_size"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_CONFIG["dataset_name"])
    parser.add_argument("--dataset_subset", type=str, default=DEFAULT_CONFIG["dataset_subset"])
    parser.add_argument("--n_examples", type=int, default=DEFAULT_CONFIG["n_examples"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG["logging_steps"])
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_CONFIG["eval_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"])
    parser.add_argument("--gumbel_temperature", type=float, default=DEFAULT_CONFIG["gumbel_temperature"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--use_wandb", action="store_true", default=DEFAULT_CONFIG["use_wandb"])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_CONFIG["wandb_project"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--use_start_thought_token", action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG["use_start_thought_token"])
    parser.add_argument("--use_end_thought_token", action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG["use_end_thought_token"])

    # Thinking Gate
    parser.add_argument("--use_thinking_gate", action="store_true", default=False,
                        help="Enable selective thinking gate")
    parser.add_argument("--thinking_gate_sparsity_beta", type=float, default=0.01,
                        help="L1 sparsity penalty weight for thinking gate")
    parser.add_argument("--thinking_gate_hidden_dim", type=int, default=128,
                        help="Hidden dimension of thinking gate MLP")
    parser.add_argument("--thinking_gate_threshold", type=float, default=0.5,
                        help="Hard threshold for gate binarisation at inference")
    parser.add_argument("--thinking_gate_bias_init", type=float, default=-2.0,
                        help="Initial bias for gate output layer")
    parser.add_argument("--thought_chunk_size", type=int, default=1,
                        help="Amount of base tokens per thought (Chunk-level thinking)")

    # Token-Space Thinking (o1-style)
    parser.add_argument("--use_token_space_thinking", action="store_true", default=False,
                        help="Enable token-space thinking (o1-style)")
    parser.add_argument("--token_thought_length", type=int, default=64,
                        help="Number of thought tokens to generate")

    # GRPO
    parser.add_argument("--use_grpo", action="store_true", default=False,
                        help="Enable GRPO policy gradient")
    parser.add_argument("--grpo_group_size", type=int, default=8,
                        help="Group size for GRPO normalization")

    # Best-of-N + PRM
    parser.add_argument("--use_best_of_n", action="store_true", default=False,
                        help="Enable Best-of-N thought selection with PRM")
    parser.add_argument("--best_of_n", type=int, default=4,
                        help="Number of candidates for Best-of-N")

    # Verifiable Reward
    parser.add_argument("--use_verifiable_reward", action="store_true", default=False,
                        help="Enable verifiable (correctness) reward")
    parser.add_argument("--verifiable_reward_weight", type=float, default=1.0,
                        help="Weight for verifiable reward loss")

    # Self-consistency Voting
    parser.add_argument("--use_self_consistency", action="store_true", default=False,
                        help="Enable self-consistency voting in Best-of-N")
    parser.add_argument("--self_consistency_threshold", type=float, default=0.5,
                        help="Minimum vote fraction for majority")

    # LLM-as-a-Judge Reward
    parser.add_argument("--use_judge_reward", action="store_true", default=False,
                        help="Use a larger model to score reasoning quality")
    parser.add_argument("--judge_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for the judge")
    parser.add_argument("--judge_reward_weight", type=float, default=1.0,
                        help="Weight for the judge reward signal")

    # SFT Warmup
    parser.add_argument("--use_sft_warmup", action="store_true", default=False,
                        help="Run SFT warmup on GSM8K with <think> tags before RL")
    parser.add_argument("--sft_warmup_steps", type=int, default=500,
                        help="Number of SFT warmup steps")
    parser.add_argument("--sft_learning_rate", type=float, default=2e-5,
                        help="Learning rate for SFT warmup phase")

    # Reward Warmup
    parser.add_argument("--reward_warmup_steps", type=int, default=200,
                        help="Steps to linearly ramp verifiable_reward_weight and prm_loss_weight from 0 to target")

    # GSM8K mixing
    parser.add_argument("--gsm8k_mix_ratio", type=float, default=0.3,
                        help="Fraction of GSM8K examples mixed into training after SFT warmup")

    # HuggingFace Hub & Resume
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_CONFIG["hf_repo_id"],
                        help="HuggingFace Hub repo to push checkpoints (e.g. your-name/quiet-star)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=DEFAULT_CONFIG["resume_from_checkpoint"],
                        help="Path to checkpoint directory to resume training from")
    # API Keys
    parser.add_argument("--hf_token", type=str, default=DEFAULT_CONFIG["hf_token"],
                        help="HuggingFace API token for pushing checkpoints")
    parser.add_argument("--wandb_key", type=str, default=DEFAULT_CONFIG["wandb_key"],
                        help="Weights & Biases API key for logging")

    args = parser.parse_args()
    if args.no_wandb:
        args.use_wandb = False
    return args


def model_init(args, tokenizer):
    """
    Initialize the Quiet-STAR model with Qwen2.5-3B as base.
    Returns a function compatible with HuggingFace Trainer's model_init.
    """
    def _init(params=None):
        if params is not None:
            params = params.params
        else:
            params = {}

        n_ahead = params.get("n_ahead", args.n_ahead)
        n_ahead_talk = params.get("n_ahead_talk", args.n_ahead_talk)
        n_passes = params.get("n_passes", args.n_passes)

        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"  n_ahead={n_ahead}, n_ahead_talk={n_ahead_talk}, n_passes={n_passes}")

        # Load base Qwen2 config
        base_config = AutoConfig.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            token=args.hf_token if args.hf_token else None,
        )

        # Create QuietStarConfig from base Qwen2 config
        # Use getattr for attributes that may not exist in all transformers versions
        config_kwargs = dict(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            intermediate_size=base_config.intermediate_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
            hidden_act=getattr(base_config, 'hidden_act', 'silu'),
            max_position_embeddings=base_config.max_position_embeddings,
            initializer_range=getattr(base_config, 'initializer_range', 0.02),
            rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
            use_cache=False,
            max_thoughts=n_ahead + n_ahead_talk + 1,
            merged_talk_heads=True,
            merged_lm_and_talk_heads=False,
            merged_lm_and_think_heads=True,
            use_concat_talk_head=True,
            use_shallow_think=True,
            use_shallow_talk=False,
            use_complex_think_head=False,
            use_complex_talk_head=True,
            use_weighted_talk_head=True,
        )
        # Add optional attributes if they exist
        for attr in ['rope_theta', 'attention_dropout', 'attn_implementation',
                     'sliding_window', 'use_sliding_window', 'max_window_layers',
                     'tie_word_embeddings']:
            if hasattr(base_config, attr):
                config_kwargs[attr] = getattr(base_config, attr)
        if 'attn_implementation' not in config_kwargs:
            config_kwargs['attn_implementation'] = 'sdpa'

        # Thinking Gate parameters
        config_kwargs['use_thinking_gate'] = args.use_thinking_gate
        config_kwargs['thinking_gate_hidden_dim'] = args.thinking_gate_hidden_dim
        config_kwargs['thinking_gate_sparsity_beta'] = args.thinking_gate_sparsity_beta
        config_kwargs['thinking_gate_threshold'] = args.thinking_gate_threshold
        config_kwargs['thinking_gate_bias_init'] = args.thinking_gate_bias_init
        config_kwargs['thought_chunk_size'] = args.thought_chunk_size

        quiet_config = QuietStarConfig(**config_kwargs)

        # Load pretrained weights into Quiet-STAR architecture
        logger.info(f"Loading pretrained weights from {args.model_name}...")
        try:
            # Try to load directly (works perfectly if it's our trained HF Repo checkpoint)
            # This preserves talk_head, start_embedding, and end_embedding
            model = QuietStarQwen2ForCausalLM.from_pretrained(
                args.model_name,
                config=quiet_config,
                torch_dtype=torch.bfloat16,
                cache_dir=args.cache_dir,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="sdpa",
                ignore_mismatched_sizes=True,
                token=args.hf_token if args.hf_token else None,
            )
            logger.info("✓ Successfully loaded weights directly into Quiet-STAR architecture (resuming heads/embeddings if present)")
            
        except Exception as e:
            logger.warning(f"Direct load failed ({e}), falling back to manual weight transfer...")
            from transformers import AutoModelForCausalLM as BaseAutoModel
            
            base_model = BaseAutoModel.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                cache_dir=args.cache_dir,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="sdpa",
                token=args.hf_token if args.hf_token else None,
            )
            
            model = QuietStarQwen2ForCausalLM(quiet_config)
            
            logger.info("Transferring weights to Quiet-STAR model...")
            base_state_dict = base_model.state_dict()
            model_state_dict = model.state_dict()
            
            transferred = 0
            for key in base_state_dict:
                if key in model_state_dict and base_state_dict[key].shape == model_state_dict[key].shape:
                    model_state_dict[key] = base_state_dict[key]
                    transferred += 1
            model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"Transferred {transferred} weight tensors from pretrained base model")
            
            del base_model
            torch.cuda.empty_cache()

        # Copy embed_tokens weights INTO lm_head (not tie) to avoid shared tensor save crash.
        # Setting tie_word_embeddings=False so transformers doesn't try to re-tie them.
        if getattr(quiet_config, "tie_word_embeddings", False):
            logger.info("Copying embed_tokens weights to lm_head (untied for Quiet-STAR training)")
            model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.data.clone())
            model.config.tie_word_embeddings = False

        # Convert to bfloat16 and move to GPU
        model = model.to(dtype=torch.bfloat16)
        model = model.cuda()

        # Add special thought tokens
        special_tokens_to_add = []
        if args.use_start_thought_token:
            special_tokens_to_add.append("<|startthought|>")
        if args.use_end_thought_token:
            special_tokens_to_add.append("<|endthought|>")

        if special_tokens_to_add:
            num_added = tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens_to_add
            })
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Added {num_added} special tokens, resized embeddings to {len(tokenizer)}")

        # Initialize thought embeddings from pretrained embedding statistics
        # The default init (std=0.02) is FAR too small compared to real embeddings,
        # causing NaN in attention when processed in bfloat16
        with torch.no_grad():
            embed_mean = model.model.embed_tokens.weight.float().mean(dim=0)
            embed_std = model.model.embed_tokens.weight.float().std(dim=0).clamp(min=1e-6)
            # start_embedding shape: [2, hidden_size] — [0] is the embedding, [1] is for reparam
            model.start_embedding.data[0] = embed_mean.to(model.start_embedding.dtype)
            model.start_embedding.data[1] = (embed_std * 0.1).to(model.start_embedding.dtype)
            model.end_embedding.data[0] = embed_mean.to(model.end_embedding.dtype)
            model.end_embedding.data[1] = (embed_std * 0.1).to(model.end_embedding.dtype)
            logger.info(f"Initialized thought embeddings from pretrained embed stats (mean={embed_mean.norm():.4f})")

        # Set model attributes
        model.tokenizer = tokenizer
        model.n_ahead = n_ahead
        model.n_ahead_talk = n_ahead_talk
        model.n_passes = n_passes
        model.gumbel_temperature = args.gumbel_temperature
        model.gumbel_detach = True
        model.include_policy_loss = True
        model.use_start_thought_token = args.use_start_thought_token
        model.use_end_thought_token = args.use_end_thought_token
        model.residual_think_head = False
        model.optimize_lm_head_only_at_start = False
        model.wandb_enabled = args.use_wandb

        # Token-Space Thinking flags
        model.use_token_space_thinking   = getattr(args, 'use_token_space_thinking', False)
        model.token_thought_length       = getattr(args, 'token_thought_length', 64)
        model.use_grpo                   = getattr(args, 'use_grpo', False)
        model.grpo_group_size            = getattr(args, 'grpo_group_size', 8)
        model.use_best_of_n              = getattr(args, 'use_best_of_n', False)
        model.best_of_n                  = getattr(args, 'best_of_n', 4)
        model.use_verifiable_reward       = getattr(args, 'use_verifiable_reward', False)
        model.verifiable_reward_weight    = getattr(args, 'verifiable_reward_weight', 1.0)
        model.use_self_consistency        = getattr(args, 'use_self_consistency', False)
        model.self_consistency_threshold  = getattr(args, 'self_consistency_threshold', 0.5)
        model.use_judge_reward            = getattr(args, 'use_judge_reward', False)
        model.judge_reward_weight         = getattr(args, 'judge_reward_weight', 1.0)

        # Set <think> / </think> token IDs
        if model.use_token_space_thinking:
            think_id = tokenizer.convert_tokens_to_ids("<think>")
            end_think_id = tokenizer.convert_tokens_to_ids("</think>")
            if think_id != tokenizer.unk_token_id:
                model.think_token_id = think_id
            if end_think_id != tokenizer.unk_token_id:
                model.end_think_token_id = end_think_id
            logger.info(f"Token-space thinking: think_id={model.think_token_id}, end_think_id={model.end_think_token_id}")
        model.original_mode = False
        model.run_start = int(time.time())
        model.kill_after = None

        # Set thought token IDs
        if args.use_start_thought_token:
            model.start_token_id = tokenizer.convert_tokens_to_ids("<|startthought|>")
            model.tokenizer_has_start_thought_token = True
        if args.use_end_thought_token:
            model.end_token_id = tokenizer.convert_tokens_to_ids("<|endthought|>")
            model.tokenizer_has_end_thought_token = True

        # Pre-compute banned thought tokens mask to save time during training
        logger.info("Building banned token mask for thought generation...")
        banned = torch.zeros(model.vocab_size, dtype=torch.bool)
        
        # Fast decode all tokens to check for emoji/unicode/newlines
        for token_id in range(model.vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
            except Exception:
                banned[token_id] = True
                continue
                
            # Ban special tokens that shouldn't appear mid-thought 
            # (let the model think in Thai and any other language it knows!)
            if token_str.startswith("<|") and token_str.endswith("|>"):
                banned[token_id] = True
                continue
                
            # Ban pure whitespace tokens (except single space)
            if token_str.strip() == '' and token_str != ' ':
                banned[token_id] = True
                continue
                
        # Register as a buffer so it automatically moves to the GPU and saves/loads correctly
        model.register_buffer('_banned_thought_tokens_mask', banned, persistent=False)
        logger.info(f"Banned {banned.sum().item()} cheap tokens from thought generation")

        # Gradient accumulation
        gradient_accumulation_steps = max(1, args.full_batch_size // args.batch_size)
        model.gradient_accumulation_steps = gradient_accumulation_steps
        model.n_tokens_print = gradient_accumulation_steps

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated after model load: {allocated:.2f} GB")

        logger.info("✓ Quiet-STAR Qwen2.5-3B model ready for training")
        model.train()
        return model

    return _init


# ============================================================================
# Reward Warmup Callback
# ============================================================================
class RewardWarmupCallback(TrainerCallback):
    """
    Linearly ramp verifiable_reward_weight and prm_loss_weight from 0 to target
    over `warmup_steps` training steps.

    This prevents the RL reward signals from destabilizing early training
    when the model hasn't yet learned the <think> format.
    """
    def __init__(self, model, warmup_steps: int):
        self.model = model
        self.warmup_steps = max(1, warmup_steps)
        self.target_vr_weight = model.verifiable_reward_weight
        self.target_prm_weight = model.prm_loss_weight
        # Start at zero
        model.verifiable_reward_weight = 0.0
        model.prm_loss_weight = 0.0
        logger.info(f"RewardWarmupCallback: ramping over {warmup_steps} steps "
                     f"(vr_weight → {self.target_vr_weight}, prm_weight → {self.target_prm_weight})")

    def on_step_end(self, args, state, control, **kwargs):
        progress = min(1.0, state.global_step / self.warmup_steps)
        self.model.verifiable_reward_weight = progress * self.target_vr_weight
        self.model.prm_loss_weight = progress * self.target_prm_weight


def main():
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ================================================================
    # Login to HuggingFace & WandB
    # ================================================================

    # HuggingFace login
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            logger.info("✓ HuggingFace logged in successfully!")
        except Exception as e:
            logger.warning(f"Failed to login to HuggingFace: {e}")
    elif args.hf_repo_id:
        logger.warning("⚠️  hf_repo_id is set but no hf_token provided. Push may fail.")

    # WandB login & setup
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_key:
            try:
                import wandb
                wandb.login(key=args.wandb_key)
                logger.info("✓ W&B logged in successfully!")
            except Exception as e:
                logger.warning(f"Failed to login to W&B: {e}")
                args.use_wandb = False
        else:
            try:
                import wandb
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                args.use_wandb = False

    # ================================================================
    # Load tokenizer
    # ================================================================
    # Always load tokenizer from the original Qwen repository to avoid corrupted tokenizer saves on Hub
    original_base_model = "Qwen/Qwen2.5-3B"
    logger.info(f"Loading tokenizer from {original_base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        original_base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        token=args.hf_token if args.hf_token else None,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add <think> / </think> tokens for token-space thinking
    if args.use_token_space_thinking or args.use_sft_warmup:
        added = tokenizer.add_special_tokens({
            "additional_special_tokens": ["<think>", "</think>"]
        })
        if added > 0:
            logger.info(f"Added {added} special tokens: <think>, </think>")

    set_tokenizer(tokenizer, max_length=args.max_length)

    # ================================================================
    # Load datasets
    # ================================================================
    logger.info(f"Loading training dataset: {args.dataset_name}")
    logger.info("  ➤ Using STREAMING mode (no need to download entire dataset)")
    logger.info(f"  ➤ Only fetching {args.n_examples} examples")

    # Use streaming to avoid downloading the entire dataset (FineWeb-Edu is ~1.5TB!)
    streamed_dataset = load_dataset(
        args.dataset_name,
        args.dataset_subset,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Take only n_examples and convert to regular dataset
    from datasets import Dataset
    examples = list(streamed_dataset.take(args.n_examples))
    dataset = Dataset.from_list(examples)
    logger.info(f"  ✓ Fetched {len(dataset)} examples via streaming")

    train_dataset = dataset.shuffle(seed=args.seed).map(
        preprocess_function,
        batched=True,
        writer_batch_size=200,
        remove_columns=dataset.column_names,
    )

    logger.info(f"Training dataset: {len(train_dataset)} examples")

    # Load GSM8K for SFT warmup and/or verifiable reward
    gsm8k_sft_dataset = None
    if args.use_sft_warmup or args.use_verifiable_reward:
        logger.info("Loading GSM8K dataset for SFT warmup / verifiable reward...")
        try:
            gsm8k_raw = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
            gsm8k_sft_dataset = gsm8k_raw.shuffle(seed=args.seed).map(
                preprocess_gsm8k_sft,
                batched=True,
                writer_batch_size=200,
                remove_columns=gsm8k_raw.column_names,
            )
            logger.info(f"  ✓ GSM8K SFT dataset: {len(gsm8k_sft_dataset)} examples")
        except Exception as e:
            logger.warning(f"Failed to load GSM8K: {e}. SFT warmup will be skipped.")
            args.use_sft_warmup = False

    # ── Mix GSM8K into main training dataset ──────────────────────────────
    # Without this, verifiable reward has no gold_answers to compare against
    # because FineWeb-Edu is pure text with no math answers.
    if gsm8k_sft_dataset is not None and args.use_verifiable_reward:
        from datasets import concatenate_datasets

        # Add empty gold_answers column to FineWeb so columns match
        train_dataset = train_dataset.map(
            lambda batch: {"gold_answers": [""] * len(batch["input_ids"])},
            batched=True,
        )

        # Sample a proportional number of GSM8K examples based on mix ratio
        n_gsm8k = min(
            len(gsm8k_sft_dataset),
            int(len(train_dataset) * args.gsm8k_mix_ratio / (1 - args.gsm8k_mix_ratio + 1e-8)),
        )
        gsm8k_subset = gsm8k_sft_dataset.shuffle(seed=args.seed).select(range(n_gsm8k))

        train_dataset = concatenate_datasets([train_dataset, gsm8k_subset]).shuffle(seed=args.seed)
        logger.info(
            f"  ✓ Mixed dataset: {len(train_dataset)} total "
            f"(FineWeb: {len(train_dataset) - n_gsm8k}, GSM8K: {n_gsm8k}, "
            f"ratio: {args.gsm8k_mix_ratio:.0%})"
        )
    # ─────────────────────────────────────────────────────────────────────

    # Evaluation skipped for early training phase
    eval_datasets = None

    # ================================================================
    # Training Arguments (optimized for Qwen2.5-3B on H200)
    # ================================================================
    gradient_accumulation_steps = max(1, args.full_batch_size // args.batch_size)
    run_id = int(time.time())

    # Use consistent output_dir for resume support
    if args.resume_from_checkpoint:
        # Extract output_dir from checkpoint path: .../outputs/quietstar_XXXXX/checkpoint-500 → .../outputs/quietstar_XXXXX
        output_dir = os.path.dirname(args.resume_from_checkpoint)
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        logger.info(f"Using output dir: {output_dir}")
    else:
        output_dir = os.path.join(args.output_dir, f"quietstar_{run_id}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_names=["labels"],
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_datasets else "no",
        save_steps=args.save_steps,
        save_total_limit=5,         # Keep last 5 checkpoints
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True, # Enable memory savings
        eval_accumulation_steps=4,   # PREVENT OOM DURING EVALUATION (offloads predictions to CPU regularly)
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen2.5-3b_n={args.n_ahead}_nt={args.n_ahead_talk}_np={args.n_passes}",
        auto_find_batch_size=False,  # Disable to prevent crash loop when memory is tight
        remove_unused_columns=False, # Required for our custom dataset mapping
        # HuggingFace Hub — push checkpoints for backup
        push_to_hub=args.hf_repo_id is not None,
        hub_model_id=args.hf_repo_id,
        hub_strategy="checkpoint",   # Push every save_steps
        hub_private_repo=True,       # Keep repo private
    )

    # ================================================================
    # Load LLM-as-a-Judge (if enabled)
    # ================================================================
    judge_instance = None
    if args.use_judge_reward:
        logger.info("=" * 60)
        logger.info(f"Loading Judge Model: {args.judge_model_name} in 4-bit...")
        from transformers import AutoModelForCausalLM as BaseAutoModel, BitsAndBytesConfig
        from modeling_quiet_star import ReasoningJudge
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        judge_hf_model = BaseAutoModel.from_pretrained(
            args.judge_model_name,
            device_map="auto",
            quantization_config=quant_config,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            token=args.hf_token if args.hf_token else None,
        )
        judge_hf_model.eval()
        judge_tokenizer = AutoTokenizer.from_pretrained(
            args.judge_model_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            token=args.hf_token if args.hf_token else None,
        )
        judge_instance = ReasoningJudge(judge_hf_model, judge_tokenizer)
        logger.info("✓ Judge model loaded successfully")

    # ================================================================
    # ================================================================
    init_fn = model_init(args, tokenizer)
    model = init_fn()

    # Inject judge model
    if judge_instance is not None:
        model.judge_model = judge_instance

    # ================================================================
    # Phase 0: SFT Warmup (teacher-forced CoT with <think> tags)
    # ================================================================
    if args.use_sft_warmup and gsm8k_sft_dataset is not None:
        logger.info("=" * 60)
        logger.info("Phase 0: SFT Warmup on GSM8K with <think> tags")
        logger.info(f"  Steps: {args.sft_warmup_steps}")
        logger.info(f"  LR: {args.sft_learning_rate}")
        logger.info("=" * 60)

        sft_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"sft_warmup_{int(time.time())}"),
            learning_rate=args.sft_learning_rate,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.sft_warmup_steps,
            warmup_steps=min(50, args.sft_warmup_steps // 10),
            weight_decay=args.weight_decay,
            label_names=["labels"],
            logging_steps=args.logging_steps,
            save_steps=args.sft_warmup_steps,  # save once at end
            save_total_limit=1,
            bf16=True,
            dataloader_num_workers=4,
            gradient_checkpointing=True,
            report_to="wandb" if args.use_wandb else "none",
            run_name=f"sft_warmup_qwen2.5-3b",
            remove_unused_columns=False,
        )

        # Disable RL features during SFT
        model.original_mode = True  # pure next-token prediction, no thought generation

        sft_trainer = Trainer(
            model=model,
            args=sft_args,
            train_dataset=gsm8k_sft_dataset,
        )
        sft_trainer.train()
        logger.info("✓ SFT Warmup complete — model has learned <think> format")

        # Re-enable RL features
        model.original_mode = False

    # ================================================================
    # RewardWarmupCallback (ramp reward weights from 0 → target)
    # ================================================================
    callbacks = []
    if args.reward_warmup_steps > 0 and (args.use_verifiable_reward or args.use_best_of_n):
        callbacks.append(RewardWarmupCallback(model, args.reward_warmup_steps))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if eval_datasets else None,
        compute_metrics=compute_metrics if eval_datasets else None,
        callbacks=callbacks if callbacks else None,
    )

    # ================================================================
    # Train!
    # ================================================================
    logger.info("=" * 60)
    logger.info("Starting Quiet-STAR training")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  n_ahead: {args.n_ahead}")
    logger.info(f"  n_ahead_talk: {args.n_ahead_talk}")
    logger.info(f"  n_passes: {args.n_passes}")
    logger.info(f"  Batch size: {args.batch_size} x {gradient_accumulation_steps} = {args.batch_size * gradient_accumulation_steps}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Sequence length: {args.max_length}")
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  GPU Memory: {gpu_mem:.1f} GB")
    if args.resume_from_checkpoint:
        logger.info(f"⚡ RESUMING from {args.resume_from_checkpoint}")
    if args.hf_repo_id:
        logger.info(f"☁️  Pushing checkpoints to: https://huggingface.co/{args.hf_repo_id}")
    logger.info("=" * 60)

    # Train (with optional resume)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_output = os.path.join(args.output_dir, f"quietstar_qwen25_3b_final_{run_id}")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info(f"✓ Model saved to {final_output}")

    # Push final model to Hub
    if args.hf_repo_id:
        logger.info(f"Pushing final model to HuggingFace Hub: {args.hf_repo_id}")
        trainer.push_to_hub(commit_message="Final model after training")
        logger.info(f"✓ Final model pushed to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
