"""
Quiet-STAR: Patched Qwen2 Model with Thought Generation
Based on: https://arxiv.org/abs/2403.09629
Adapted for Qwen2.5-3B architecture.

This module patches Qwen2ForCausalLM to add:
1. Tokenwise parallel thought generation with Gumbel-Softmax
2. Mixing heads for combining base/thought predictions
3. REINFORCE / GRPO policy gradient for thought quality
4. Learnable start/end thought token embeddings
5. Selective Thinking Gate (per-position, learned)
6. Dynamic n_ahead based on gate confidence
7. Incremental KV Cache for think steps
8. Chunk-level Thinking — think over N-token windows
9. Process Reward Model + Best-of-N selection
10. Verifiable Reward — correctness signal from GSM8K math problems
"""

import math
import re
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import GenerationMixin
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2Model,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging as hf_logging

from config import QuietStarConfig

logger = hf_logging.get_logger(__name__)


# ============================================================================
# Utility functions
# ============================================================================

# ============================================================================
# Selective Thinking Gate
# ============================================================================

class ThinkingGate(nn.Module):
    """
    A lightweight MLP that predicts, per token position, whether the model
    should spend thought tokens on that position.

    Architecture
    ------------
    hidden_states [B, T, H]  →  Linear(H, gate_dim)  →  ReLU
                             →  Linear(gate_dim, 1)   →  Sigmoid
                             →  gate_scores [B, T, 1]   ∈ (0, 1)

    Training behaviour
    ------------------
    The gate is applied as a *soft* multiplicative mask on the thought
    contribution so the whole system remains differentiable:

        final_logits = gate * thought_logits + (1 - gate) * base_logits

    An optional L1 sparsity penalty  β · mean(gate)  discourages the model
    from thinking everywhere; it is added to the total loss.

    Inference behaviour
    -------------------
    When ``hard_threshold`` is provided (or ``model.thinking_gate_threshold``
    is set) the gate output is binarised:

        gate_hard = (gate_scores >= threshold).float()

    This allows positions below the threshold to skip the thought pass
    entirely during decoding (no grad needed at inference so cost is low).

    Initialisation
    --------------
    The output layer bias is set to ``bias_init`` (default −2.0) so the gate
    starts near 0 (rarely think), forcing the model to *learn* when thinking
    helps rather than burning compute thinking everywhere from step 0.
    """

    def __init__(self, hidden_size: int, gate_hidden_dim: int = 128, bias_init: float = -2.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, gate_hidden_dim, bias=True)
        self.fc2 = nn.Linear(gate_hidden_dim, 1, bias=True)
        # Initialise output bias negative → gate starts near 0 (don't think)
        nn.init.constant_(self.fc2.bias, bias_init)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : Tensor[B, T, H]

        Returns
        -------
        gate_scores : Tensor[B, T, 1]  values in (0, 1)
        """
        # Let gradients flow back through the backbone so the gate learns from downstream tasks,
        # unless prevent_backbone_update is handled at the caller level.
        x = F.relu(self.fc1(hidden_states))
        return torch.sigmoid(self.fc2(x))             # [B, T, 1]


# ============================================================================
# Process Reward Model (PRM)
# ============================================================================

class ProcessRewardModel(nn.Module):
    """
    Lightweight Process Reward Model that scores the quality of each thought
    step given the hidden states produced by the backbone.

    Input  : hidden states from the backbone at each think step  [B, T, H]
    Output : scalar reward score per (batch, position)           [B, T]

    Architecture: 3-layer MLP with LayerNorm + residual
        H → hidden_dim → hidden_dim → 1
    """

    def __init__(self, hidden_size: int, hidden_dim: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1   = nn.Linear(hidden_size, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.fc3   = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm1(hidden_states)
        x = F.gelu(self.fc1(x))
        x = self.norm2(x)
        x = F.gelu(self.fc2(x))
        return self.fc3(x).squeeze(-1)   # [B, T]

    def score_mean(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return a single scalar score per batch item (mean over T)."""
        return self.forward(hidden_states).mean(dim=-1)   # [B]


# ============================================================================
# Verifiable Reward Computer
# ============================================================================

class VerifiableRewardComputer:
    """
    Computes correctness-based rewards for math problems (GSM8K-style).

    reward = +1.0  if model's final answer == gold answer
           =  0.0  if wrong or unparseable
           = -0.5  (optional) if answer present but wrong (configurable)
    """

    _NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")

    @classmethod
    def extract_answer(cls, text: str) -> Optional[float]:
        if not text:
            return None
        # Strategy 1: GSM8K canonical format "#### 42"
        if "####" in text:
            after = text.split("####")[-1].strip()
            nums = cls._NUMBER_RE.findall(after)
            if nums:
                return cls._to_float(nums[0])
        # Strategy 2: "the answer is X" / "answer: X"
        for marker in ("the answer is", "answer is", "answer:"):
            if marker in text.lower():
                after = text.lower().split(marker)[-1].strip()
                nums = cls._NUMBER_RE.findall(after)
                if nums:
                    return cls._to_float(nums[0])
        # Strategy 3: last number anywhere in text
        nums = cls._NUMBER_RE.findall(text)
        if nums:
            return cls._to_float(nums[-1])
        return None

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @classmethod
    def answers_match(cls, predicted: Optional[float], gold: Optional[float],
                      tol: float = 1e-3) -> bool:
        if predicted is None or gold is None:
            return False
        return abs(predicted - gold) <= tol + abs(gold) * tol

    @classmethod
    def compute_batch_rewards(
        cls,
        predicted_texts: List[str],
        gold_texts: List[str],
        wrong_penalty: float = 0.0,
    ) -> torch.Tensor:
        rewards = []
        for pred_text, gold_text in zip(predicted_texts, gold_texts):
            pred_val = cls.extract_answer(pred_text)
            gold_val = cls.extract_answer(gold_text)
            if gold_val is None:
                rewards.append(0.0)
            elif pred_val is None:
                rewards.append(wrong_penalty)
            elif cls.answers_match(pred_val, gold_val):
                rewards.append(1.0)
            else:
                rewards.append(wrong_penalty)
        return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# LLM-as-a-Judge (Reasoning Evaluator)
# ============================================================================
class ReasoningJudge:
    """
    Wraps a larger language model (e.g., Qwen2.5-7B) to evaluate reasoning steps.
    Returns a continuous score [0, 1] based on YES/NO logits.

    Prompt format:
    Evaluate the following reasoning for solving the problem. Is it logical and on track? Answer only YES or NO.
    Problem: {Q}
    Reasoning: {R}
    Answer:
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Cache YES/NO token IDs to grab logits
        self.yes_id = self.tokenizer.encode("YES", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("NO", add_special_tokens=False)[0]

    @torch.no_grad()
    def score_reasoning(self, questions: List[str], reasonings: List[str]) -> torch.Tensor:
        """
        Evaluate a batch of reasonings. Returns a tensor of scores [0, 1].
        """
        import functools
        
        @functools.lru_cache(maxsize=1024)
        def _format_prompt(q, r):
            prompt = (
                "Evaluate the following reasoning for solving the problem. "
                "Is it logical and on track? Answer only YES or NO.\n\n"
                f"Problem: {q}\nReasoning: {r}\nAnswer:"
            )
            try:
                msg = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            except Exception:
                return prompt

        if not questions or not reasonings:
            return torch.tensor([], dtype=torch.float32)

        prompts = [_format_prompt(q, r) for q, r in zip(questions, reasonings)]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Get logits for the last token position
        last_logits = outputs.logits[:, -1, :]  # [B, V]

        yes_logits = last_logits[:, self.yes_id]
        no_logits = last_logits[:, self.no_id]

        # Softmax over just YES/NO to get a [0, 1] probability of YES
        scores = torch.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)[:, 0]
        return scores.cpu()

def soft_cap_logits(logits: torch.Tensor, cap: float = 30.0) -> torch.Tensor:
    """
    Apply Tanh soft-capping to prevent exploding logits while maintaining gradient flow.
    Replaces hard torch.clamp(..., min=-cap, max=cap) which causes zero gradients.
    Used in Gemma 2 / PaLM 2 to fix training instabilities.
    """
    return cap * torch.tanh(logits / cap)

def nonzero_mean(x, axis=None):
    """Compute mean of non-zero elements along an axis."""
    if axis is not None:
        den = (x != 0).float().sum(axis)
        den = torch.where(den == 0, torch.ones_like(den), den)
        return x.sum(axis) / den
    
    den = (x != 0).float().sum()
    if den == 0:
        return x.sum() * 0.0 # Return strong 0 rather than NaN
    return x.sum() / den


def loss_mean(x):
    """Compute mean of non-zero loss values."""
    den = (x != 0).float().sum()
    if den == 0:
        return x.sum() * 0.0
    return x.sum() / den


# ============================================================================
# Quiet-STAR Qwen2 Model
# ============================================================================

class QuietStarQwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    """
    Qwen2ForCausalLM patched with Quiet-STAR thought generation.

    During training, the model:
    1. Runs a base forward pass to get base logits
    2. Generates thought tokens using Gumbel-Softmax sampling
    3. Runs the model again with thought tokens to get thought-augmented logits
    4. Mixes base and thought logits using a learned mixing head
    5. Uses REINFORCE to optimize the quality of generated thoughts

    During inference:
    - n_ahead_talk is set to 1 and n_passes to 1 for standard generation
    - Start/end thought tokens should be masked during generation
    """

    def __init__(self, config: QuietStarConfig):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_thoughts = config.max_thoughts

        # Head configuration
        self.merged_lm_and_talk_heads = config.merged_lm_and_talk_heads
        self.use_concat_talk_head = config.use_concat_talk_head
        self.use_shallow_talk = config.use_shallow_talk
        self.use_complex_talk_head = config.use_complex_talk_head
        self.use_weighted_talk_head = config.use_weighted_talk_head
        self.merged_lm_and_think_heads = config.merged_lm_and_think_heads
        self.use_shallow_think = config.use_shallow_think
        self.use_complex_think_head = config.use_complex_think_head
        self.merged_talk_heads = config.merged_talk_heads

        # --- Talk head ---
        if self.use_weighted_talk_head:
            talk_input_dim = config.hidden_size * 2 if self.use_concat_talk_head else config.hidden_size
            talk_output_dim = config.hidden_size
            if self.merged_talk_heads:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                ])
            else:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                    for _ in range(self.max_thoughts)
                ])
        elif self.use_complex_talk_head:
            talk_input_dim = config.hidden_size * 2 if self.use_concat_talk_head else config.hidden_size
            if self.use_shallow_talk:
                talk_output_dim = config.hidden_size
            else:
                talk_output_dim = config.vocab_size
            if self.merged_talk_heads:
                self.talk_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(talk_input_dim, talk_input_dim),
                        nn.ReLU(),
                        nn.Linear(talk_input_dim, talk_output_dim, bias=False),
                    )
                ])
            else:
                self.talk_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(talk_input_dim, talk_input_dim),
                        nn.ReLU(),
                        nn.Linear(talk_input_dim, talk_output_dim, bias=False),
                    )
                    for _ in range(self.max_thoughts)
                ])
        else:
            talk_input_dim = config.hidden_size * 2 if self.use_concat_talk_head else config.hidden_size
            if self.use_shallow_talk:
                talk_output_dim = config.hidden_size
            else:
                talk_output_dim = config.vocab_size
            if self.merged_talk_heads:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                ])
            else:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                    for _ in range(self.max_thoughts)
                ])

        # --- Thought embeddings (learnable start/end tokens) ---
        self.start_embedding = nn.Parameter(torch.zeros(2, config.hidden_size))
        self.end_embedding = nn.Parameter(torch.zeros(2, config.hidden_size))
        nn.init.normal_(self.start_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.end_embedding, mean=0.0, std=0.02)

        # --- Selective Thinking Gate ---
        # When use_thinking_gate=True the gate decides per token position
        # whether thought augmentation is applied (soft during train,
        # optionally hard-thresholded at inference).
        self.use_thinking_gate = getattr(config, "use_thinking_gate", False)
        if self.use_thinking_gate:
            self.thinking_gate = ThinkingGate(
                hidden_size=config.hidden_size,
                gate_hidden_dim=getattr(config, "thinking_gate_hidden_dim", 128),
                bias_init=getattr(config, "thinking_gate_bias_init", -2.0),
            )
        else:
            self.thinking_gate = None
        # Sparsity penalty weight (β · mean(gate) added to loss during training)
        self.thinking_gate_sparsity_beta = getattr(config, "thinking_gate_sparsity_beta", 0.01)
        # Hard threshold used at inference (gate_score < threshold → skip thoughts)
        self.thinking_gate_threshold = getattr(config, "thinking_gate_threshold", 0.5)

        # --- Dynamic n_ahead ---
        # Configurable confidence → depth mapping (see _resolve_effective_n_ahead)
        # Override via model.thinking_gate_levels = [...] after construction.
        self.thinking_gate_levels = [
            (0.0,  0.20, 0.00),   # very easy  → skip thinking
            (0.20, 0.40, 0.25),   # easy       → 25 % of n_ahead steps
            (0.40, 0.60, 0.50),   # medium     → 50 %
            (0.60, 0.80, 0.75),   # hard       → 75 %
            (0.80, 1.01, 1.00),   # very hard  → full n_ahead
        ]
        # Minimum think steps even on easiest inputs (set 0 to allow full skip)
        self.thinking_gate_min_ahead = 0
        # NOT applied during training — gradients must flow through the full
        # attention graph for REINFORCE to work correctly.
        self.use_kv_cache_for_thoughts = False

        # --- Chunk-level Thinking ---
        # Instead of thinking at every single token position (T think rounds),
        # the sequence is split into non-overlapping windows of chunk_size tokens.
        # A single think-talk round is run per window, giving each thought a
        # wider context window (chunk_size tokens) to reason over.
        #
        # Benefits:
        #   • Fewer think rounds: T/chunk_size instead of T
        #   • Each thought sees chunk_size tokens → deeper reasoning
        #   • Memory proportional to T/chunk_size instead of T
        #   • Works with gate + KV cache (applied per chunk)
        #
        # chunk_overlap: number of tokens from the previous chunk prepended to
        # the current chunk as read-only context (not predicted).
        self.use_chunk_thinking = getattr(config, "use_chunk_thinking", False)
        self.chunk_size = getattr(config, "chunk_size", 8)
        self.chunk_overlap = getattr(config, "chunk_overlap", 0)
        self.thought_chunk_size = getattr(config, "thought_chunk_size", 1)

        # --- GRPO (Group Relative Policy Optimisation) ---
        self.use_grpo = False
        self.grpo_group_size = 8

        # --- Cross-chunk Memory ---
        self.use_cross_chunk_memory = False
        self.cross_chunk_memory_size = 1
        self._cross_chunk_state = None

        # --- Semantic Chunk Boundaries ---
        self.use_semantic_boundaries = False
        self.semantic_boundary_window = 8

        # --- Adaptive Thought Length ---
        self.use_adaptive_thought_length = False
        self.adaptive_thought_min = 1

        # --- Process Reward Model (PRM) + Best-of-N ---
        self.use_best_of_n       = False
        self.best_of_n           = 4
        self.prm_hidden_dim      = 256
        self.prm_loss_weight     = 0.1
        self.prm: Optional[ProcessRewardModel] = None
        self._prm_initialised    = False

        # --- Verifiable Reward ---
        self.use_verifiable_reward      = False
        self.verifiable_reward_weight   = 0.1
        self.verifiable_wrong_penalty   = 0.0
        self._verifiable_reward_computer = VerifiableRewardComputer()
        self._current_gold_answers: Optional[List[str]] = None

        # --- LLM-as-a-Judge Reward ---
        self.use_judge_reward     = False
        self.judge_reward_weight  = 1.0
        self.judge_model: Optional[ReasoningJudge] = None

        # --- Runtime parameters (set by training script) ---
        self.n_ahead = 1
        self.n_ahead_talk = 1
        self.n_passes = 1
        self.n_tokens_print = 1
        self.gradient_accumulation_steps = 1
        self.training_steps = 0
        self.wandb_enabled = False
        self.original_mode = False
        self.gumbel_temperature = 1.0
        self.gumbel_detach = True
        self.include_policy_loss = True
        self.use_end_thought_token = True
        self.use_start_thought_token = True
        self.residual_think_head = False
        self.optimize_lm_head_only_at_start = False
        self.optimize_model_only_at_start = False
        self.use_reparam_for_thought_embeddings = False
        self.train_only_thinking_embedding = False
        self.use_thought_prefix = False
        self.thought_prefix = None
        self.tokenized_thought_prefix = None
        self.reinforce_temperature = 3.0
        self.base_loss_beta = 1.0
        self.entropy_reg_beta = 0.03
        self.repetition_penalty = 1.2

        # Residual mode flags (exactly one should be True)
        self.cumulative_residual = False
        self.clever_residual = False
        self.skip_residual = False
        self.no_residual = True

        # First and last mode for REINFORCE
        self.first_and_last_mode = True

        # Token IDs (set after tokenizer is assigned)
        self.start_token_id = None
        self.end_token_id = None
        self.tokenizer = None
        self.tokenizer_has_start_thought_token = False
        self.tokenizer_has_end_thought_token = False

        # Kill switch
        self.kill_after = None
        self.run_start = 0

        # Logging
        self.log_dict = {}
        self.eval_log_dict = {}
        self.config_params = {}

        # Banned token mask for thought sampling (built lazily, registered as buffer for auto device sync)
        self.register_buffer('_banned_thought_tokens_mask', None, persistent=False)

        # --- KV Cache Reuse for Think Steps ---
        self.use_kv_cache_for_thoughts = False

        # --- Token-Space Thinking (o1-style) ---
        self.use_token_space_thinking   = False
        self.token_thought_length       = 64
        self.token_thought_temperature  = 1.0
        self.think_token_id             = None
        self.end_think_token_id         = None

        # --- Self-consistency Voting ---
        # Majority vote over decoded answers from Best-of-N paths.
        # PRM score breaks ties within the majority group.
        # Requires use_best_of_n=True and use_token_space_thinking=True.
        self.use_self_consistency       = False
        self.self_consistency_threshold = 0.5  # min vote fraction for majority

        # Post init
        self.post_init()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """
        Resize token embeddings and synchronize self.vocab_size to prevent index out of bounds
        during token masking and one_hot generation.
        """
        result = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)
        
        # Sync vocab sizes directly from model properties now that it's updated
        self.vocab_size = self.config.vocab_size
        return result

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _resolve_effective_n_ahead(self, gate_scores: torch.Tensor) -> int:
        """
        Map mean gate confidence → effective number of think steps.

        Called once per forward pass (inference only) after the base pass.
        The result replaces ``self.n_ahead`` for the duration of this call so
        the think loop terminates early on easy inputs.

        Confidence levels (configurable via ``self.thinking_gate_levels``):

            [0.0, 0.2)  → 0  think steps  (skip thinking entirely)
            [0.2, 0.4)  → 25 % of n_ahead
            [0.4, 0.6)  → 50 % of n_ahead
            [0.6, 0.8)  → 75 % of n_ahead
            [0.8, 1.0]  → 100% of n_ahead  (full thinking)

        A minimum of 1 think step is always kept so the talk-head still
        receives thought-augmented hidden states (otherwise the blending
        in Phase 2 degenerates).  Pass ``self.thinking_gate_min_ahead = 0``
        to allow full skip (returns base logits immediately in forward).

        Parameters
        ----------
        gate_scores : Tensor[B, T, 1]
            Soft or hard gate scores produced by ``ThinkingGate``.

        Returns
        -------
        effective_n_ahead : int  ∈ [min_ahead, self.n_ahead]
        """
        confidence = gate_scores.mean().item()   # scalar in [0, 1]

        # Allow caller to override levels via instance attribute
        levels = getattr(
            self,
            "thinking_gate_levels",
            [
                (0.0,  0.20, 0.00),   # (low, high, fraction_of_n_ahead)
                (0.20, 0.40, 0.25),
                (0.40, 0.60, 0.50),
                (0.60, 0.80, 0.75),
                (0.80, 1.01, 1.00),
            ],
        )

        fraction = 1.0  # default: full thinking
        for lo, hi, frac in levels:
            if lo <= confidence < hi:
                fraction = frac
                break

        raw = round(self.n_ahead * fraction)
        min_ahead = getattr(self, "thinking_gate_min_ahead", 1)
        effective = max(min_ahead, min(raw, self.n_ahead))

        # Log so callers / wandb can track thinking depth
        self._last_effective_n_ahead = effective
        self._last_gate_confidence = confidence

        return effective

    def get_kv_cache_speedup_estimate(self) -> dict:
        """Estimate compute saved by incremental KV cache reuse."""
        if not self.use_kv_cache_for_thoughts or self.training:
            return {"use_kv_cache_for_thoughts": False}
        eff = getattr(self, "_last_effective_n_ahead", self.n_ahead)
        think_steps_cached = max(0, eff - 1)
        saved_frac = think_steps_cached * 0.4 / max(eff, 1)
        speedup = 1.0 / max(1.0 - saved_frac, 0.01)
        return {
            "use_kv_cache_for_thoughts": True,
            "cache_mode": "incremental",
            "think_steps_total": eff,
            "think_steps_cached": think_steps_cached,
            "estimated_speedup_x": round(speedup, 2),
        }

    def _apply_head(self, head, states, detach=False):
        """Apply a linear head (lm_head-style) to hidden states."""
        if detach:
            head_weight = head.weight.detach()
        else:
            head_weight = head.weight
        head_weight = head_weight.to(states.device)
        return (head_weight @ states.transpose(-1, -2)).transpose(-1, -2).contiguous()

    def _none_repeat_interleave(self, x, n):
        """Repeat interleave while handling None."""
        if x is None:
            return x
        return x.repeat_interleave(n, dim=0)

    # ------------------------------------------------------------------ #
    #  Verifiable Reward helpers                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _decode_greedy_answers(self, logits: torch.Tensor) -> List[str]:
        """Greedy-decode the model's prediction for each batch item."""
        if self.tokenizer is None:
            return [""] * logits.shape[0]
        pred_ids = logits.argmax(dim=-1).cpu()
        texts = []
        for b in range(pred_ids.shape[0]):
            ids = pred_ids[b].tolist()
            if self.tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(self.tokenizer.eos_token_id) + 1]
            try:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                text = ""
            texts.append(text)
        return texts

    def _compute_verifiable_reward_loss(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        action_loglikelihoods_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute REINFORCE loss weighted by verifiable (correctness) reward."""
        if self._current_gold_answers is None or len(action_loglikelihoods_list) == 0:
            return torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()

        B = logits.shape[0]
        gold = self._current_gold_answers
        if len(gold) < B:
            gold = gold + [""] * (B - len(gold))
        gold = gold[:B]

        pred_texts = self._decode_greedy_answers(logits)
        rewards = self._verifiable_reward_computer.compute_batch_rewards(
            predicted_texts=pred_texts, gold_texts=gold,
            wrong_penalty=self.verifiable_wrong_penalty,
        ).to(logits.device).to(logits.dtype)

        r_mean = rewards.mean()
        r_std  = rewards.std().clamp(min=1e-6)
        rewards_norm = (rewards - r_mean) / r_std

        verifiable_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()
        for action_loglik in action_loglikelihoods_list:
            T_log = action_loglik.shape[-1]
            r_expanded = rewards_norm.unsqueeze(-1).expand(-1, T_log)
            cur_loss = -action_loglik * r_expanded
            verifiable_loss = verifiable_loss + loss_mean(cur_loss)

        correct_rate = (rewards == 1.0).float().mean().item()
        if self.wandb_enabled:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "train/verifiable_correct_rate": correct_rate,
                        "train/verifiable_reward_mean": rewards.mean().item(),
                    }, step=self.training_steps)
            except ImportError:
                pass
        return verifiable_loss

    # ------------------------------------------------------------------ #
    #  Process Reward Model helpers                                       #
    # ------------------------------------------------------------------ #

    def _init_prm(self):
        """Lazily create the PRM and move it to the same device/dtype as the backbone."""
        if self._prm_initialised:
            return
        H = self.config.hidden_size
        self.prm = ProcessRewardModel(
            hidden_size=H, hidden_dim=self.prm_hidden_dim,
        ).to(device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype)
        self._prm_initialised = True
        logger.info(f"[PRM] Initialised (H={H}, hidden_dim={self.prm_hidden_dim}, "
                     f"params={sum(p.numel() for p in self.prm.parameters()):,})")

    def _run_best_of_n(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate n independent thought sequences, score with PRM, return best."""
        self._init_prm()
        B, T, H = inputs_embeds.shape
        dtype = inputs_embeds.dtype
        exp_embeds = inputs_embeds.unsqueeze(0).expand(n, -1, -1, -1).reshape(n * B, T, H)
        exp_mask = (attention_mask.unsqueeze(0).expand(n, -1, -1).reshape(n * B, T)
                    if attention_mask is not None else None)
        exp_pos = (position_ids.unsqueeze(0).expand(n, -1, -1).reshape(n * B, T)
                   if position_ids is not None else None)

        with torch.set_grad_enabled(self.training):
            think_out = self.model(
                inputs_embeds=exp_embeds, attention_mask=exp_mask,
                position_ids=exp_pos, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )
        hidden = think_out.hidden_states[-1]
        raw_scores = self.prm.score_mean(hidden)
        prm_scores = raw_scores.view(n, B)

        best_idx = prm_scores.argmax(dim=0)
        hidden_4d = hidden.view(n, B, T, H)
        idx_expanded = best_idx.view(B, 1, 1).expand(B, T, H)
        best_hidden = hidden_4d.permute(1, 0, 2, 3).gather(1, idx_expanded.unsqueeze(1)).squeeze(1)

        embed_w = self.model.embed_tokens.weight.to(dtype)
        soft_logits = best_hidden @ embed_w.T
        soft_probs = torch.softmax(soft_logits / 0.5, dim=-1)
        best_embeds = soft_probs @ embed_w
        return best_embeds, prm_scores

    def _compute_prm_loss(
        self,
        prm_scores: torch.Tensor,
        base_loss_per_pos: torch.Tensor,
        thought_loss_per_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Train PRM to predict normalised CE improvement."""
        with torch.no_grad():
            improvement = (base_loss_per_pos - thought_loss_per_pos).detach()
            imp_mean = improvement.mean()
            imp_std = improvement.std().clamp(min=1e-6)
            target = (improvement - imp_mean) / imp_std
        best_idx = prm_scores.detach().argmax(dim=0)
        best_score = prm_scores.gather(0, best_idx.unsqueeze(0)).squeeze(0)
        return F.mse_loss(best_score, target)

    # ------------------------------------------------------------------ #
    #  Token-Space Thinking (o1-style)                                   #
    # ------------------------------------------------------------------ #

    def _self_consistency_select(
        self,
        candidate_logits: List[torch.Tensor],
        prm_scores: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Select best candidate per batch item via majority vote + PRM tiebreak.

        For each batch item:
          1. Decode greedy answer from each candidate's logits
          2. Extract numeric answer
          3. Count votes per unique answer
          4. If majority (>= threshold × N) → pick candidate with that answer + highest PRM
          5. No majority → fall back to PRM argmax

        Returns: LongTensor[B] — selected candidate index per batch item
        """
        N = len(candidate_logits)
        B = prm_scores.shape[1]

        if N == 0:
            return torch.zeros(B, dtype=torch.long, device=prm_scores.device)

        # Decode answer for each (candidate, batch item)
        answers: List[List[Optional[float]]] = []
        for n in range(N):
            texts = self._decode_greedy_answers(candidate_logits[n])
            batch_answers = [
                self._verifiable_reward_computer.extract_answer(t) for t in texts
            ]
            answers.append(batch_answers)

        best_idx = prm_scores.argmax(dim=0)  # [B] — fallback

        for b in range(B):
            vote_count: Dict[float, int] = {}
            for n in range(N):
                val = answers[n][b]
                if val is not None:
                    key = round(val, 3)
                    vote_count[key] = vote_count.get(key, 0) + 1

            if not vote_count:
                continue  # no parseable answers → PRM fallback

            top_answer, top_votes = max(vote_count.items(), key=lambda x: x[1])
            if top_votes / N >= threshold:
                # Majority found — pick candidate with this answer + highest PRM
                majority_scores = torch.full(
                    (N,), fill_value=-1e9,
                    device=prm_scores.device, dtype=prm_scores.dtype,
                )
                for n in range(N):
                    val = answers[n][b]
                    if val is not None and abs(round(val, 3) - top_answer) < 1e-3:
                        majority_scores[n] = prm_scores[n, b]
                best_idx[b] = majority_scores.argmax()

        return best_idx

    def _generate_thought_tokens(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        n_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Autoregressively generate n_tokens thought tokens.
        Training: straight-through Gumbel-softmax.
        Inference: greedy argmax.
        """
        B, T, H = inputs_embeds.shape
        dev   = inputs_embeds.device
        dtype = inputs_embeds.dtype
        embed_w = self.model.embed_tokens.weight.to(dtype)

        cur_embeds = inputs_embeds
        cur_mask   = attention_mask
        thought_logprobs: List[torch.Tensor] = []

        for step in range(n_tokens):
            out = self.model(
                inputs_embeds=cur_embeds, attention_mask=cur_mask,
                use_cache=False, return_dict=True,
            )
            last_hidden = out.last_hidden_state[:, -1, :]
            logits = self.lm_head(last_hidden)

            if self.training and self._banned_thought_tokens_mask is not None:
                mask = self._banned_thought_tokens_mask.to(dev)
                logits = logits.masked_fill(mask.unsqueeze(0), -1e10)

            thought_logprobs.append(F.log_softmax(logits.float(), dim=-1).to(dtype))

            if self.training:
                one_hot = F.gumbel_softmax(
                    logits / max(self.token_thought_temperature, 1e-3),
                    tau=1.0, hard=True,
                )
                next_embed = one_hot @ embed_w
            else:
                token_ids = logits.argmax(dim=-1)
                next_embed = self.model.embed_tokens(token_ids)

            cur_embeds = torch.cat([cur_embeds, next_embed.unsqueeze(1)], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones(B, 1, device=dev, dtype=cur_mask.dtype)], dim=1,
            )

        return cur_embeds, cur_mask, thought_logprobs

    def _token_space_forward(
        self, input_ids, attention_mask, position_ids, labels,
        inputs_embeds, output_attentions, output_hidden_states, return_dict,
    ) -> CausalLMOutputWithPast:
        """Token-Space Thinking forward pass (o1-style)."""
        dev = input_ids.device if input_ids is not None else inputs_embeds.device
        dtype = self.lm_head.weight.dtype

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(dtype)
        B, T, H = inputs_embeds.shape
        loss = torch.zeros(1, device=dev, dtype=dtype).squeeze()

        # Base pass
        base_out = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            use_cache=False, output_hidden_states=True, return_dict=True,
        )
        base_hidden = base_out.last_hidden_state
        base_logits = self.lm_head(base_hidden)

        # Thinking Gate
        gate_scores = None
        do_think = True
        if self.use_thinking_gate and self.thinking_gate is not None:
            gate_scores = self.thinking_gate(base_hidden)
            if self.training:
                if self.thinking_gate_sparsity_beta > 0:
                    loss = loss + self.thinking_gate_sparsity_beta * gate_scores.mean()
            else:
                gate_scores = (gate_scores >= self.thinking_gate_threshold).float()
                do_think = gate_scores.mean().item() > 0.0

        # Determine thought length
        n_think = self.token_thought_length
        if self.use_adaptive_thought_length and gate_scores is not None:
            complexity = float(gate_scores.mean().item())
            n_think = max(self.adaptive_thought_min,
                          round(self.adaptive_thought_min +
                                complexity * (self.token_thought_length - self.adaptive_thought_min)))

        # Generate thought tokens
        thought_logprobs: List[torch.Tensor] = []
        final_logits = base_logits

        if do_think and n_think > 0:
            think_embeds = inputs_embeds.clone()
            if self.think_token_id is not None:
                think_tok = torch.tensor([self.think_token_id], device=dev)
                think_embed = self.model.embed_tokens(think_tok)
                think_prefix = think_embed.unsqueeze(0).expand(B, -1, -1)
                think_embeds = torch.cat([think_embeds, think_prefix], dim=1)
                think_mask = torch.cat([
                    attention_mask,
                    torch.ones(B, 1, device=dev, dtype=attention_mask.dtype)
                ], dim=1)
            else:
                think_mask = attention_mask

            # Best-of-N with PRM
            _bon_prm_scores = None
            if self.use_best_of_n and self.best_of_n > 1 and self.training:
                self._init_prm()
                all_prm_scores = []
                all_ext_embeds = []
                all_ext_masks = []
                all_logprobs = []
                all_candidate_logits = []  # for self-consistency
                for n_idx in range(self.best_of_n):
                    ext_e, ext_m, t_lp = self._generate_thought_tokens(
                        think_embeds, think_mask, n_think)
                    with torch.no_grad():
                        score_out = self.model(
                            inputs_embeds=ext_e, attention_mask=ext_m,
                            use_cache=False, return_dict=True)
                        score = self.prm.score_mean(score_out.last_hidden_state)
                        # Decode candidate logits for self-consistency
                        cand_logits = self.lm_head(score_out.last_hidden_state[:, :T, :])
                    all_prm_scores.append(score)
                    all_ext_embeds.append(ext_e)
                    all_ext_masks.append(ext_m)
                    all_logprobs.append(t_lp)
                    all_candidate_logits.append(cand_logits)

                _bon_prm_scores = torch.stack(all_prm_scores, dim=0)  # [N, B]

                # Select best candidate per batch item
                if self.use_self_consistency and self.tokenizer is not None:
                    best_idx = self._self_consistency_select(
                        all_candidate_logits, _bon_prm_scores,
                        threshold=self.self_consistency_threshold,
                    )
                else:
                    best_idx = _bon_prm_scores.argmax(dim=0)  # [B] — PRM-only

                # Gather selected candidate per batch item
                extended_embeds_list = []
                extended_mask_list = []
                thought_logprobs_list = []
                
                for b in range(B):
                    idx = best_idx[b].item()
                    extended_embeds_list.append(all_ext_embeds[idx][b:b+1])
                    extended_mask_list.append(all_ext_masks[idx][b:b+1])
                    # Bug Fix 5: Ensure logprobs are gathered per batch item, not locked to item 0
                    thought_logprobs_list.append(all_logprobs[idx][b:b+1])
                    
                extended_embeds = torch.cat(extended_embeds_list, dim=0)
                extended_mask = torch.cat(extended_mask_list, dim=0)
                thought_logprobs = torch.cat(thought_logprobs_list, dim=0)
            else:
                extended_embeds, extended_mask, thought_logprobs = \
                    self._generate_thought_tokens(think_embeds, think_mask, n_think)
                _bon_prm_scores = None

            # Append </think>
            if self.end_think_token_id is not None:
                end_tok = torch.tensor([self.end_think_token_id], device=dev)
                end_embed = self.model.embed_tokens(end_tok).unsqueeze(0).expand(B, -1, -1)
                extended_embeds = torch.cat([extended_embeds, end_embed], dim=1)
                extended_mask = torch.cat([
                    extended_mask,
                    torch.ones(B, 1, device=dev, dtype=extended_mask.dtype)
                ], dim=1)

            # Final pass on [input + thoughts]
            final_out = self.model(
                inputs_embeds=extended_embeds, attention_mask=extended_mask,
                use_cache=False, output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            final_logits = self.lm_head(final_out.last_hidden_state[:, :T, :])

            if gate_scores is not None and self.training:
                final_logits = gate_scores * final_logits + (1 - gate_scores) * base_logits

        # CE Loss
        base_loss = None
        if labels is not None:
            shift_logits = final_logits[..., :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(dev)
            if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
                shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            ce_loss = CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
            if not torch.isnan(ce_loss) and not torch.isinf(ce_loss):
                base_loss = ce_loss
                loss = loss + ce_loss

        # GRPO / REINFORCE on thought log-probs
        if self.include_policy_loss and self.training and len(thought_logprobs) > 0 and labels is not None:
            with torch.no_grad():
                base_ce_per_pos = CrossEntropyLoss(
                    ignore_index=-100, reduction="none"
                )(
                    base_logits[..., :-1, :].contiguous().view(-1, self.vocab_size),
                    labels[..., 1:].contiguous().view(-1).to(dev),
                ).view(B, -1).mean(-1)
                thought_ce_per_pos = CrossEntropyLoss(
                    ignore_index=-100, reduction="none"
                )(
                    final_logits[..., :-1, :].contiguous().view(-1, self.vocab_size),
                    labels[..., 1:].contiguous().view(-1).to(dev),
                ).view(B, -1).mean(-1)
                
                # Baseline CE improvement reward
                reward = (base_ce_per_pos - thought_ce_per_pos).detach()

                # LLM-as-a-Judge Reward
                if self.use_judge_reward and self.judge_model is not None:
                    # Get question texts and reasoning texts
                    if input_ids is not None:
                        q_texts = self._decode_greedy_answers(input_ids)
                    else:
                        q_texts = [""] * B  # fallback if inputs_embeds provided directly
                    r_texts = self._decode_greedy_answers(extended_embeds)
                    
                    # Compute judge score [0, 1]
                    judge_scores = self.judge_model.score_reasoning(q_texts, r_texts).to(dev)
                    
                    # Add judge reward (weighted)
                    reward = reward + (self.judge_reward_weight * judge_scores)

            if self.use_grpo and self.grpo_group_size > 1 and B % self.grpo_group_size == 0:
                G = self.grpo_group_size
                rg = reward.view(-1, G)
                reward = ((rg - rg.mean(1, keepdim=True)) /
                          rg.std(1, keepdim=True).clamp(min=1e-6)).view(B)
            else:
                reward = (reward - reward.mean()) / reward.std().clamp(min=1e-6)

            for t_lp in thought_logprobs:
                action_lp = t_lp.max(dim=-1).values
                policy_loss = -(action_lp * reward).mean()
                if not torch.isnan(policy_loss):
                    loss = loss + policy_loss

        # PRM Loss
        if (self.training and self.use_best_of_n and _bon_prm_scores is not None
                and base_loss is not None):
            with torch.no_grad():
                improvement = (base_ce_per_pos - thought_ce_per_pos).detach()
                imp_std = improvement.std().clamp(min=1e-6)
                target = (improvement - improvement.mean()) / imp_std
            best_idx = _bon_prm_scores.detach().argmax(dim=0)
            best_score = _bon_prm_scores.gather(0, best_idx.unsqueeze(0)).squeeze(0)
            prm_loss = F.mse_loss(best_score, target)
            if not torch.isnan(prm_loss):
                loss = loss + self.prm_loss_weight * prm_loss

        # Verifiable Reward Loss
        if (self.training and self.use_verifiable_reward and len(thought_logprobs) > 0):
            vr_loss = self._compute_verifiable_reward_loss(
                logits=final_logits, input_ids=input_ids,
                action_loglikelihoods_list=[
                    t_lp.max(dim=-1).values.unsqueeze(-1) for t_lp in thought_logprobs
                ],
            )
            if not torch.isnan(vr_loss) and not torch.isinf(vr_loss):
                loss = loss + self.verifiable_reward_weight * vr_loss

        # Logging
        if self.training:
            self.training_steps += 1
            if self.wandb_enabled and self.training_steps % self.n_tokens_print == 0:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "train/total_loss": loss.item(),
                            "train/base_loss": base_loss.item() if base_loss else 0,
                            "train/thought_length": n_think,
                            "train/do_think": float(do_think),
                        }, step=self.training_steps)
                except ImportError:
                    pass

        if not return_dict:
            return (loss, final_logits)
        return CausalLMOutputWithPast(
            loss=loss, logits=final_logits, past_key_values=None,
            hidden_states=base_out.hidden_states if output_hidden_states else None,
            attentions=None,
        )

    # ------------------------------------------------------------------ #
    #  Chunk-level Thinking helpers                                       #
    # ------------------------------------------------------------------ #

    def _chunk_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        labels,
        inputs_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        """
        Chunk-level Thinking forward pass.

        Splits the sequence into windows of ``chunk_size`` tokens and runs
        one full think-talk round per window.  The key advantage over the
        original token-level approach is that each thought sees an entire
        chunk of context rather than a single token position, enabling
        deeper and more coherent reasoning.

        Layout
        ------
        For a sequence of length T with chunk_size=C and overlap=V:

            chunk 0 : tokens [0   : C]
            chunk 1 : tokens [C-V : 2C-V]
            chunk 2 : tokens [2C-2V : 3C-2V]
            ...

        Each chunk's think-talk round is fully self-contained.  Loss values
        are averaged across chunks; logits are pasted back into the full
        sequence tensor with overlap regions averaged.

        Returns
        -------
        CausalLMOutputWithPast  (aggregated over all chunks)
        """
        C      = self.chunk_size
        V      = min(self.chunk_overlap, C - 1)
        stride = max(C - V, 1)

        T   = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        B   = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        dev = input_ids.device   if input_ids is not None else inputs_embeds.device
        dtype = self.lm_head.weight.dtype

        chunk_starts = list(range(0, T, stride)) or [0]

        total_loss   = torch.zeros(1, device=dev, dtype=dtype).squeeze()
        logits_accum = torch.zeros(B, T, self.config.vocab_size, device=dev, dtype=dtype)
        logits_count = torch.zeros(B, T, 1, device=dev, dtype=dtype)
        n_valid = 0

        # Temporarily disable chunk thinking to avoid recursion in forward()
        self.use_chunk_thinking = False
        
        # Bug Fix 4: Inter-chunk KV Cache
        # Thread the past_key_values out of chunk i into chunk i+1
        chunk_past_kv = None

        try:
            for c_start in chunk_starts:
                c_end = min(c_start + C, T)
                if c_end <= c_start:
                    continue
    
                c_out = self(
                    input_ids      = input_ids[:, c_start:c_end]          if input_ids      is not None else None,
                    attention_mask = attention_mask[:, c_start:c_end]      if attention_mask is not None else None,
                    position_ids   = position_ids[:, c_start:c_end]        if position_ids   is not None else None,
                    past_key_values= chunk_past_kv, # Thread cached context forward
                    inputs_embeds  = inputs_embeds[:, c_start:c_end, :]    if inputs_embeds  is not None else None,
                    labels         = labels[:, c_start:c_end]              if labels         is not None else None,
                    use_cache      = not self.training, # Cache only at inference; training + grad checkpointing conflict
                    output_attentions    = output_attentions,
                    output_hidden_states = output_hidden_states,
                    return_dict    = True,
                )
    
                if c_out.loss is not None and not torch.isnan(c_out.loss):
                    total_loss = total_loss + c_out.loss
                    n_valid   += 1
    
                if c_out.logits is not None:
                    # The length of the returned chunk logits
                    out_len = c_out.logits.shape[1]
                    
                    # The safe length to add back into the accumulator
                    # Min of: (Expected chunk end - start), Returned logit length, and Space left in accumulator
                    valid_len = min(c_end - c_start, out_len, T - c_start)
                    
                    if valid_len > 0:
                        # Explicitly slice both sides to be exactly valid_len
                        lhs_slice = logits_accum[:, c_start:c_start+valid_len, :]
                        rhs_slice = c_out.logits[:, :valid_len, :]
                        
                        # Use out-of-place addition to be absolutely safe against inplace memory errors
                        logits_accum[:, c_start:c_start+valid_len, :] = lhs_slice + rhs_slice
                        
                        lhs_count_slice = logits_count[:, c_start:c_start+valid_len, :]
                        logits_count[:, c_start:c_start+valid_len, :] = lhs_count_slice + 1.0
                    
                # Store the updated cache for the next chunk
                chunk_past_kv = c_out.past_key_values
        finally:
            # Bug Fix 1: Ensure this flag resets EVEN on OOM or Exception
            self.use_chunk_thinking = True   # re-enable

        if n_valid > 0:
            total_loss = total_loss / n_valid

        logits_count = logits_count.clamp(min=1.0)
        full_logits  = logits_accum / logits_count

        if not return_dict:
            return (total_loss, full_logits)

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=full_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Quiet-STAR forward pass.

        Flow:
        1. If original_mode (no thoughts): standard causal LM forward
        2. Otherwise:
           a. Phase 1 (Think): Generate n_ahead thought tokens via Gumbel-Softmax
              - Token 0: <|startthought|>
              - Tokens 1..n_ahead-3: sampled thought tokens
              - Token n_ahead-2: <|endthought|>
           b. Phase 2 (Talk): Predict next tokens using thought-augmented hidden states
              - Mixing head combines base & thought predictions
           c. Compute losses:
              - Standard CE loss on base predictions
              - CE loss on thought-augmented predictions
              - REINFORCE policy loss to optimize thought quality
        """
        # ── Chunk-level Thinking dispatch ────────────────────────────────────
        # When enabled the sequence is split into chunks of self.chunk_size
        # tokens and _chunk_forward() handles the rest.  The flag is temporarily
        # set to False inside _chunk_forward to avoid infinite recursion.
        if self.use_chunk_thinking:
            return self._chunk_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict if return_dict is not None else self.config.use_return_dict,
            )
        # ─────────────────────────────────────────────────────────────────────

        log_dict = self.log_dict if self.training else self.eval_log_dict

        if self.training and self.kill_after is not None:
            if self.training_steps // self.gradient_accumulation_steps > self.kill_after:
                raise ValueError("Killed after specified training steps")

        if not self.training:
            n_ahead_talk_to_restore = self.n_ahead_talk
            n_passes_to_restore = self.n_passes
            self.n_ahead_talk = 1
            self.n_passes = 1

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate exactly one residual mode is set
        residual_count = sum([self.cumulative_residual, self.clever_residual,
                             self.skip_residual, self.no_residual])
        if residual_count != 1:
            raise ValueError(
                f"Exactly one residual mode must be enabled, but {residual_count} are set. "
                f"cumulative={self.cumulative_residual}, clever={self.clever_residual}, "
                f"skip={self.skip_residual}, no={self.no_residual}"
            )
        if self.skip_residual and self.include_policy_loss:
            raise ValueError(
                "skip_residual and include_policy_loss cannot both be True — "
                "skip_residual discards thought hidden states that policy loss needs."
            )

        if self.tokenized_thought_prefix is None and self.use_thought_prefix:
            self.tokenized_thought_prefix = self.tokenizer(
                self.thought_prefix, return_tensors="pt", add_special_tokens=False
            )["input_ids"]

        # Multi-pass: replicate inputs
        if self.n_passes > 1:
            input_ids = self._none_repeat_interleave(input_ids, self.n_passes)
            attention_mask = self._none_repeat_interleave(attention_mask, self.n_passes)
            position_ids = self._none_repeat_interleave(position_ids, self.n_passes)
            labels = self._none_repeat_interleave(labels, self.n_passes)
            if inputs_embeds is not None:
                inputs_embeds = self._none_repeat_interleave(inputs_embeds, self.n_passes)

        # Total number of ahead steps: think steps + talk steps
        n_ahead_total = self.n_ahead + self.n_ahead_talk

        # ---- KV Cache ----
        # Two modes of caching:
        #
        # (A) Incremental decoding (generate → past_key_values provided)
        #     Skip thinking, base pass only + cached KV.  O(1) per new token.
        #
        # (B) Thought KV reuse (use_kv_cache_for_thoughts + inference + n_ahead>1)
        #     Cache grows each think step so thoughts see prior thoughts:
        #       base(T) → +think1(2T) → +think2(3T) → …
        #     Base K/V never recomputed.  Talk passes run normally (no cache).
        _incremental_decoding = (
            not self.training and past_key_values is not None
        )
        _thought_kv_enabled = (
            not self.training
            and self.use_kv_cache_for_thoughts
            and self.n_ahead > 1
            and not _incremental_decoding
        )
        _base_past_key_values = None   # Returned to generate()
        _thought_past_kv = None        # Incremental cache across think steps

        if _incremental_decoding:
            n_ahead_total = 1 + self.n_ahead_talk  # base + talk only

        # BUG FIX 3: Loss initialized with correct dtype matching model weights
        loss = torch.zeros(1, device=input_ids.device if input_ids is not None else "cuda",
                          dtype=self.lm_head.weight.dtype).squeeze()
        
        policy_reward = None
        action_loglikelihoods_list = []
        sampled_token_history = []

        # Get initial embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Save original for later
        original_input_ids = input_ids.clone() if input_ids is not None else None
        original_attention_mask = attention_mask.clone() if attention_mask is not None else None
        original_position_ids = position_ids.clone() if position_ids is not None else None

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Initialize variables for the loop
        base_hidden_states = None
        base_logits = None
        hidden_states = None
        logits = None
        prev_hidden_states = None
        rm_logits = None
        cur_rm_tokens = None
        prev_rm_logits = None
        prev_rm_tokens = None
        probabilities_2d = None
        prev_probabilities_2d = None
        sample_probs = None
        prev_sample_probs = None
        initial_loss_logits = None
        previous_loss = None
        gate_scores = None  # Thinking gate scores [B, T, 1]; set after base pass

        # Thought token embeddings
        start_embedding = self.start_embedding
        end_embedding = self.end_embedding

        # ============================================================
        # Main Think-Talk Loop
        # NOTE: n_ahead_total may be reduced mid-loop by the dynamic gate
        # (inference only), so we use a while loop instead of for-range.
        # ============================================================
        ahead_idx = 0
        while ahead_idx < n_ahead_total:

            # ================================================================
            # KV Cache routing — 3 cases
            # ================================================================
            is_base_pass = (ahead_idx == 0)
            is_think_step = (0 < ahead_idx < self.n_ahead)
            is_talk_step  = (ahead_idx >= self.n_ahead)

            # --- Case A: Base pass ---
            if is_base_pass:
                past_key_values_step = past_key_values if _incremental_decoding else None
                use_cache_for_step = (
                    not self.training
                    and (use_cache is True or _incremental_decoding or _thought_kv_enabled)
                )
                # Reset masks to originals (first iteration)
                cur_attention_mask = attention_mask
                cur_position_ids = position_ids

            # --- Case B: Think step + incremental cache (inference) ---
            elif is_think_step and _thought_kv_enabled and _thought_past_kv is not None:
                past_key_values_step = _thought_past_kv
                use_cache_for_step = True
                # Extended attention mask: [B, cached_len + T]
                cached_len = _thought_past_kv.get_seq_length() if hasattr(_thought_past_kv, 'get_seq_length') else _thought_past_kv[0][0].shape[2]
                ext_mask = torch.ones(
                    batch_size, cached_len + seq_len,
                    device=inputs_embeds.device, dtype=torch.long,
                )
                cur_attention_mask = ext_mask
                # Position IDs: [offset .. offset+T-1]
                offset = cached_len
                cur_position_ids = torch.arange(
                    offset, offset + seq_len,
                    device=inputs_embeds.device, dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)

            # --- Case C: Normal (training, talk steps, or no cache) ---
            else:
                past_key_values_step = None
                use_cache_for_step = False
                cur_attention_mask = original_attention_mask
                cur_position_ids = original_position_ids

            # ----- Forward through the model -----
            outputs = self.model(
                attention_mask=cur_attention_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values_step,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache_for_step,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            prev_hidden_states = hidden_states
            hidden_states = outputs[0]

            # ---- Selective Checkpointing / Detach ----
            # If a token's gate is exactly 0 (e.g., from chunk masking or hard threshold),
            # it will not contribute to the final blended output. We can detach it
            # right here to cut the computation graph for this token's think step,
            # saving significant memory during training.
            if self.training and ahead_idx > 0 and gate_scores is not None:
                # gate_scores is [B, T, 1]. hidden_states is [B, seq_len, H]
                # During training, seq_len == T (no caching), so we can match shapes directly.
                if gate_scores.shape[1] == hidden_states.shape[1]:
                    detach_mask = (gate_scores == 0.0)
                    if detach_mask.any():
                        hidden_states = torch.where(detach_mask, hidden_states.detach(), hidden_states)

            # ---- Capture / grow KV cache ----
            if use_cache_for_step:
                step_cache = getattr(outputs, 'past_key_values', None)
                if is_base_pass:
                    _base_past_key_values = step_cache
                    if _thought_kv_enabled:
                        _thought_past_kv = step_cache  # seed for think steps
                elif is_think_step and _thought_kv_enabled:
                    _thought_past_kv = step_cache  # grown cache

            prev_rm_logits = rm_logits
            prev_rm_tokens = cur_rm_tokens

            # ============ Phase 1: Base Pass ============
            if ahead_idx == 0:
                hidden_states_lm = hidden_states
                logits = self.lm_head(hidden_states_lm)
                base_hidden_states = hidden_states.clone()
                initial_loss_logits = logits.clone()

                if self.optimize_lm_head_only_at_start or self.optimize_model_only_at_start:
                    logits = logits.detach()
                    base_hidden_states = base_hidden_states.detach()
                if self.optimize_model_only_at_start:
                    hidden_states = hidden_states.detach()

                base_logits = logits.clone()

                # ---- Chunk-level Thinking Mask ----
                if self.thought_chunk_size > 1:
                    chunk_size = self.thought_chunk_size
                    # True only at the end of each chunk (e.g., pos 7, 15, 23 for chunk=8)
                    positions = torch.arange(seq_len, device=inputs_embeds.device)
                    chunk_mask_1d = ((positions % chunk_size) == (chunk_size - 1)).float()
                    chunk_mask = chunk_mask_1d.view(1, seq_len, 1)
                else:
                    chunk_mask = None

                # ---- Selective Thinking Gate + Dynamic n_ahead ----
                # Compute gate scores from base hidden states so the gate can
                # use the model's first-pass representation to decide which
                # positions need additional thought.
                if self.use_thinking_gate and self.thinking_gate is not None:
                    # gate_scores : [B, T, 1] in (0, 1)
                    gate_scores = self.thinking_gate(base_hidden_states)

                    if chunk_mask is not None:
                        gate_scores = gate_scores * chunk_mask


                    if not self.training:
                        # ---- Dynamic n_ahead (inference only) ----
                        # Use raw mean gate confidence (before binarization) to resolve how many think
                        # steps this particular input actually needs.
                        effective_n_ahead = self._resolve_effective_n_ahead(gate_scores)

                        # Hard binarisation at inference: skip thinking below threshold for masking
                        gate_scores = (gate_scores >= self.thinking_gate_threshold).float()

                        n_ahead_total = effective_n_ahead + self.n_ahead_talk

                        # (Metrics can be returned in a dict to be logged by the trainer later)

                    # Add sparsity penalty during training to encourage the gate
                    # to skip thinking for positions that don't need it.
                    if self.training and self.thinking_gate_sparsity_beta > 0:
                        sparsity_loss = self.thinking_gate_sparsity_beta * gate_scores.mean()
                        loss = loss + sparsity_loss
                else:
                    # Gate disabled
                    if chunk_mask is not None:
                        gate_scores = chunk_mask.expand(batch_size, -1, -1)
                    else:
                        gate_scores = None

            # ============ Phase 2: Think/Talk Passes ============
            else:
                talk_hidden_states = hidden_states

                if self.merged_lm_and_talk_heads:
                    assert self.no_residual
                    residual_logits = self.lm_head(hidden_states)
                else:
                    if ahead_idx > self.n_ahead - 1:
                        cur_base_hidden = torch.cat([
                            base_hidden_states[..., ahead_idx - self.n_ahead + 1:, :],
                            base_hidden_states[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                    else:
                        cur_base_hidden = base_hidden_states

                    if self.use_concat_talk_head:
                        head_input_hidden_states = torch.cat(
                            [cur_base_hidden, talk_hidden_states], dim=-1
                        )
                    else:
                        head_input_hidden_states = talk_hidden_states

                    residual_logits = self.talk_head[0](head_input_hidden_states)

                    if self.use_shallow_talk:
                        residual_logits = self._apply_head(
                            self.lm_head, residual_logits,
                            detach=self.optimize_lm_head_only_at_start
                        )

                    residual_logits = residual_logits.to(logits.device)

                    if self.use_weighted_talk_head:
                        thought_weight = torch.sigmoid(residual_logits)
                        residual_logits = (
                            cur_base_hidden * (1 - thought_weight) +
                            talk_hidden_states * thought_weight
                        )
                        residual_logits = self._apply_head(
                            self.lm_head, residual_logits,
                            detach=self.optimize_lm_head_only_at_start
                        )

                # Soft-cap residual logits to prevent NaN propagation while preserving gradients
                residual_logits = soft_cap_logits(residual_logits, cap=1e4)

                # Apply residual connection
                if self.no_residual:
                    logits = residual_logits
                elif self.cumulative_residual:
                    logits = logits + residual_logits
                elif self.clever_residual:
                    if ahead_idx >= self.n_ahead - 1:
                        cur_base_logits = torch.cat([
                            base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                            base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                        logits = cur_base_logits + residual_logits
                    else:
                        logits = residual_logits
                elif self.skip_residual:
                    logits = base_logits + residual_logits

                # ---- Apply Selective Thinking Gate ----
                # After the final talk pass (ahead_idx == n_ahead + n_ahead_talk - 1)
                # blend thought-augmented logits with base logits using the gate.
                # For intermediate think steps the gate is not applied (those are
                # internal thought steps, not output predictions).
                if (
                    gate_scores is not None
                    and ahead_idx >= self.n_ahead - 1  # Only on talk/output passes
                ):
                    # gate_scores : [B, T, 1] — broadcast over vocab dim
                    # Shift gate_scores to align with the shifted logits window
                    shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                    if shift_amount > 0:
                        # Use the gate for the visible (non-shifted) portion
                        cur_gate = gate_scores[:, shift_amount:, :]   # [B, T-shift, 1]
                        cur_base  = initial_loss_logits[:, shift_amount:, :]
                    else:
                        cur_gate = gate_scores                         # [B, T, 1]
                        cur_base  = initial_loss_logits

                    # Soft blend: gate=1 → full thought, gate=0 → pure base
                    # We align over the logits sequence dimension (may differ by 1
                    # due to next-token prediction shift; take the minimum).
                    min_len = min(logits.shape[1], cur_base.shape[1], cur_gate.shape[1])
                    logits_slice    = logits[:, :min_len, :]
                    base_slice      = cur_base[:, :min_len, :]
                    gate_slice      = cur_gate[:, :min_len, :]
                    blended = gate_slice * logits_slice + (1.0 - gate_slice) * base_slice
                    # Write blended result back; keep original shape intact
                    if min_len == logits.shape[1]:
                        logits = blended
                    else:
                        logits = torch.cat([blended, logits[:, min_len:, :]], dim=1)

            # Safety soft-cap for logits before loss functions
            logits = soft_cap_logits(logits, cap=1e4)

            # ============ Compute Loss ============
            if labels is not None and ahead_idx >= self.n_ahead - 1:
                shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                shift_logits = logits[..., shift_amount:-1, :].contiguous()
                shift_labels = labels[..., 1 + shift_amount:].contiguous()

                loss_fct = CrossEntropyLoss(reduction="none")
                shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
                shift_labels_flat = shift_labels.view(-1).clone()

                if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
                    shift_labels_flat[shift_labels_flat == self.tokenizer.pad_token_id] = -100
                shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)

                # Validations: PyTorch requires labels in [0, vocab_size-1] or ignore_index=-100
                invalid_labels = (shift_labels_flat < 0) & (shift_labels_flat != -100)
                if invalid_labels.any():
                    shift_labels_flat[invalid_labels] = -100
                
                out_of_bounds = shift_labels_flat >= self.config.vocab_size
                if out_of_bounds.any():
                    shift_labels_flat[out_of_bounds] = -100

                # BUG 5 FIX: Use soft_cap_logits (not clamp or nan_to_num) to preserve gradient flow.
                # hard clamp zeroes gradients when values exceed max, soft cap allows continuous flow.
                if torch.isnan(shift_logits_flat).any() or torch.isinf(shift_logits_flat).any():
                    shift_logits_flat = soft_cap_logits(shift_logits_flat, cap=30.0)

                unreduced_loss = loss_fct(
                    shift_logits_flat, shift_labels_flat
                ).reshape(logits.shape[0], -1)

                # If still NaN (e.g. all labels were -100), zero out but warn.
                if torch.isnan(unreduced_loss).any():
                    unreduced_loss = torch.zeros_like(unreduced_loss)
                
                if ahead_idx == self.n_ahead - 1:
                    previous_loss = unreduced_loss.clone().detach()

                cur_loss = loss_mean(unreduced_loss)
                loss = loss + cur_loss

                # ============ REINFORCE Policy Gradient ============
                if self.include_policy_loss and ahead_idx > 0 and not self.original_mode:
                    if ahead_idx < self.n_ahead - 1:
                        shift_amount = 0
                        original_dqn_reward = (previous_loss - unreduced_loss).detach()
                        if self.first_and_last_mode:
                            original_dqn_reward = original_dqn_reward * 0.0
                    else:
                        shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                        cur_policy_shift_logits = initial_loss_logits[..., shift_amount:-1, :].contiguous().detach()
                        cur_policy_shift_labels = labels[..., 1 + shift_amount:].contiguous()

                        cur_policy_loss_fct = CrossEntropyLoss(reduction="none")
                        cur_policy_shift_logits = cur_policy_shift_logits.view(-1, self.config.vocab_size)
                        cur_policy_shift_labels = cur_policy_shift_labels.view(-1).clone()
                        if self.tokenizer is not None:
                            cur_policy_shift_labels[cur_policy_shift_labels == self.tokenizer.pad_token_id] = -100
                        cur_policy_shift_labels = cur_policy_shift_labels.to(cur_policy_shift_logits.device)

                        cur_policy_reward_base_loss = cur_policy_loss_fct(
                            cur_policy_shift_logits, cur_policy_shift_labels
                        ).reshape(logits.shape[0], -1)
                        original_dqn_reward = cur_policy_reward_base_loss.detach() - unreduced_loss

                    if prev_probabilities_2d is not None and prev_sample_probs is not None and prev_probabilities_2d.dim() == 2:
                        # Only compute action log-likelihoods for thought phase (2D probability distributions)
                        # Skip for talk phase where prev_probabilities_2d is 1D token indices
                        nonzero_indices = prev_probabilities_2d.nonzero()
                        if nonzero_indices.shape[0] > 0 and nonzero_indices.dim() == 2 and nonzero_indices.shape[1] >= 2:
                            # Sanitize NaNs before soft capping because soft_cap_logits(nan) == nan
                            safe_prev_sample = torch.nan_to_num(prev_sample_probs, nan=0.0)
                            soft_capped_sample_probs = soft_cap_logits(safe_prev_sample, cap=1e4)
                            # BUG FIX 2: Float32 casting for numerical stability during REINFORCE log_softmax
                            action_loglikelihoods = F.log_softmax(
                                soft_capped_sample_probs.float() / self.reinforce_temperature, dim=-1
                            )[nonzero_indices[:, 0], nonzero_indices[:, 1]].to(soft_capped_sample_probs.dtype)
                            
                            action_loglikelihoods_2d = action_loglikelihoods.reshape(
                                batch_size, -1
                            )[:, :-1 - shift_amount] if shift_amount > 0 else action_loglikelihoods.reshape(batch_size, -1)[:, :-1]
                            action_loglikelihoods_list.append(action_loglikelihoods_2d)

                    if policy_reward is None:
                        if self.n_ahead_talk > shift_amount:
                            policy_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                        else:
                            policy_reward = original_dqn_reward
                    else:
                        if self.n_ahead_talk > shift_amount:
                            added_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                        else:
                            added_reward = original_dqn_reward
                        policy_reward = policy_reward + added_reward

            # ============ Sample Next Token for Thought ============
            rm_logits = self.lm_head(hidden_states)
            
            # Sanitize NaN rm_logits from thought token processing and soft-cap
            if torch.isnan(rm_logits).any():
                rm_logits = soft_cap_logits(torch.nan_to_num(rm_logits, nan=0.0), cap=30.0)

            # === ALL modifications use NON-INPLACE ops to preserve autograd ===
            # Build additive bias (detached, no gradient needed for masks)
            logit_bias = torch.zeros(self.vocab_size, device=rm_logits.device, dtype=rm_logits.dtype)

            # Ban cheap tokens
            if self.training and ahead_idx < self.n_ahead - 1:
                if self._banned_thought_tokens_mask is not None:
                    logit_bias.masked_fill_(self._banned_thought_tokens_mask.to(rm_logits.device), -1e10)

            # Mask start/end thought tokens
            if self.tokenizer_has_start_thought_token:
                logit_bias[self.start_token_id] = -1e10
            if self.tokenizer_has_end_thought_token:
                logit_bias[self.end_token_id] = -1e10

            # Non-inplace addition preserves gradient graph
            rm_logits = rm_logits + logit_bias

            # Repetition penalty as detached additive adjustment
            if self.training and ahead_idx > 0 and len(sampled_token_history) > 0:
                last_tokens = sampled_token_history[-1].to(rm_logits.device)
                flat_logits_detached = rm_logits.detach().view(-1, rm_logits.size(-1))
                if last_tokens.shape[0] == flat_logits_detached.shape[0]:
                    token_logits = flat_logits_detached.gather(1, last_tokens.unsqueeze(1)).squeeze(1)
                    penalized = torch.where(
                        token_logits > 0,
                        token_logits / self.repetition_penalty,
                        token_logits * self.repetition_penalty
                    )
                    diff = penalized - token_logits
                    rep_bias = torch.zeros_like(flat_logits_detached)
                    rep_bias.scatter_(1, last_tokens.unsqueeze(1), diff.unsqueeze(1))
                    rm_logits = rm_logits + rep_bias.view_as(rm_logits)

            probabilities = rm_logits
            if probabilities_2d is not None:
                prev_probabilities_2d = probabilities_2d.clone()
            probabilities_2d = probabilities.view(-1, probabilities.size(-1))

            skip_sampling = False

            if ahead_idx == 0 and self.use_start_thought_token:
                override_token = self.start_token_id
            elif self.use_thought_prefix and self.tokenized_thought_prefix is not None and ahead_idx < self.tokenized_thought_prefix.shape[-1]:
                override_token = self.tokenized_thought_prefix[..., ahead_idx]
            elif ahead_idx == self.n_ahead - 2 and self.use_end_thought_token:
                override_token = self.end_token_id
            else:
                override_token = None

            if override_token is not None and self.n_ahead > 1:
                probabilities_2d = torch.zeros_like(probabilities_2d)
                probabilities_2d[:, override_token] = 1.0
                skip_sampling = True
            elif ahead_idx >= self.n_ahead - 1:
                if labels is not None:
                    cur_talk_n = ahead_idx - (self.n_ahead - 1) + 1
                    shift_labels_talk = labels[..., cur_talk_n:].contiguous().to(probabilities_2d.device)
                    padding = torch.full_like(
                        labels[..., :cur_talk_n],
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=shift_labels_talk.device
                    )
                    new_rm_tokens = torch.cat([shift_labels_talk, padding], dim=-1)
                    # Instead of creating a massive one-hot tensor [..., 151665] which OOMs/freezes,
                    # we just keep the indices and will look up embeddings directly later
                    probabilities_2d = new_rm_tokens.reshape(-1)
                    skip_sampling = True
                else:
                    ahead_idx += 1
                    continue

            temperature = self.gumbel_temperature if self.training else 0.001
            prev_sample_probs = sample_probs
            if not skip_sampling or (skip_sampling and probabilities_2d.dim() > 1):
                sample_probs = probabilities_2d
            # If skip_sampling and dim == 1, probabilities_2d is just direct token indices for talk phase

            if ahead_idx < self.n_ahead - 1 and not skip_sampling:
                probabilities_2d = F.gumbel_softmax(
                    sample_probs, tau=temperature, hard=True, dim=-1
                )
                
                # Safety check: if Gumbel-Softmax produces NaNs, fallback to standard argmax
                if torch.isnan(probabilities_2d).any():
                    fallback_idx = torch.nan_to_num(sample_probs, nan=0.0).argmax(dim=-1)
                    probabilities_2d = F.one_hot(fallback_idx, num_classes=self.vocab_size).to(sample_probs.dtype)

                # Entropy regularization: encourage diverse thought tokens
                if self.training and self.entropy_reg_beta > 0:
                    soft_probs = F.softmax(sample_probs.float(), dim=-1)
                    entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1).mean()
                    # Prevent entropy penalty from pushing total loss below 0
                    penalty = self.entropy_reg_beta * entropy
                    loss = torch.clamp(loss - penalty, min=0.0)

                if self.gumbel_detach:
                    probabilities_2d = probabilities_2d.detach()

            if probabilities_2d.dim() == 1:
                # In talk phase, probabilities_2d is a 1D tensor of token indices
                sampled_token_history.append(probabilities_2d.detach().cpu())
                contains_start = self.use_start_thought_token and (probabilities_2d == self.start_token_id).any()
                contains_end = self.use_end_thought_token and (probabilities_2d == self.end_token_id).any()
            else:
                sampled_token_history.append(probabilities_2d.argmax(dim=-1).detach().cpu())
                contains_start = (
                    self.use_start_thought_token and
                    (probabilities_2d[..., self.start_token_id].sum() > 0)
                )
                contains_end = (
                    self.use_end_thought_token and
                    (probabilities_2d[..., self.end_token_id].sum() > 0)
                )
            contains_thought = contains_start or contains_end

            if not contains_thought:
                with torch.set_grad_enabled(not self.train_only_thinking_embedding):
                    if ahead_idx >= self.n_ahead - 1 and probabilities_2d.dim() == 1:
                        # Direct lookup for talk phase (avoids massive 151k one-hot matmul)
                        inputs_embeds = self.model.embed_tokens(probabilities_2d)
                    else:
                        if torch.isnan(probabilities_2d).any():
                            probabilities_2d = torch.nan_to_num(probabilities_2d, nan=0.0)
                            
                        inputs_embeds = probabilities_2d @ (
                            self.model.embed_tokens.weight.to(probabilities.device).to(probabilities.dtype)
                        )
            else:
                cur_thought_embedding = start_embedding if contains_start else end_embedding

                if self.use_reparam_for_thought_embeddings:
                    inputs_embeds = torch.randn(
                        batch_size, seq_len, self.model.config.hidden_size,
                        device=inputs_embeds.device, dtype=cur_thought_embedding.dtype
                    )
                    inputs_embeds = inputs_embeds * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
                else:
                    inputs_embeds = cur_thought_embedding[0].unsqueeze(0).unsqueeze(0).expand(
                        batch_size, seq_len, -1
                    )

            inputs_embeds = inputs_embeds.view(batch_size, seq_len, -1).to(self.model.embed_tokens.weight.dtype)

            ahead_idx += 1  # manual increment for while loop

        # ============================================================
        # Compute Final Policy Loss
        # ============================================================
        if (
            self.include_policy_loss and
            self.training and
            policy_reward is not None and
            len(action_loglikelihoods_list) > 0
        ):
            policy_reward = policy_reward.detach()
            reward_mean = policy_reward.mean()
            reward_std = policy_reward.std().clamp(min=1e-6)
            if torch.isnan(reward_std) or reward_std == 0:
                reward_std = torch.tensor(1e-6, device=policy_reward.device)
            policy_reward = (policy_reward - reward_mean) / reward_std

            for action_loglik in action_loglikelihoods_list:
                min_len = min(action_loglik.shape[-1], policy_reward.shape[-1])
                cur_policy_loss = -action_loglik[:, :min_len] * policy_reward[:, :min_len]
                policy_loss = loss_mean(cur_policy_loss)
                loss = loss + policy_loss

        # ============================================================
        # Base Loss (standard next-token prediction)
        # ============================================================
        base_loss = None
        if labels is not None and initial_loss_logits is not None:
            shift_logits_base = initial_loss_logits[..., :-1, :].contiguous()
            shift_labels_base = labels[..., 1:].contiguous()

            shift_logits_base = shift_logits_base.view(-1, self.config.vocab_size)
            shift_labels_base = shift_labels_base.view(-1).to(shift_logits_base.device)
            
            # Mask pad_token_id to -100
            if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
                shift_labels_base[shift_labels_base == self.tokenizer.pad_token_id] = -100

            loss_fct_base = CrossEntropyLoss(ignore_index=-100)
            base_loss_raw = loss_fct_base(shift_logits_base, shift_labels_base)
            
            # BUG 6 FIX: Never use torch.tensor(0.0, requires_grad=True) as a fallback.
            # A freshly created leaf tensor has NO connection to the computation graph —
            # it provides zero gradient to any model parameter, silently killing training.
            # Instead, just skip adding base_loss when it is invalid.
            if torch.isnan(base_loss_raw) or torch.isinf(base_loss_raw):
                pass  # Skip — don't add invalid loss
            else:
                base_loss = base_loss_raw
                loss = loss + self.base_loss_beta * base_loss

        # ============================================================
        # Logging
        # ============================================================
        if self.training:
            self.training_steps += 1

            if self.training_steps % self.n_tokens_print == 0:
                if base_loss is not None:
                    log_dict["train/base_loss"] = base_loss.item()
                log_dict["train/total_loss"] = loss.item()
                log_dict["train/training_steps"] = self.training_steps

                # (Wandb logging removed to avoid race conditions with HF Trainer)

        # Restore eval settings
        if not self.training:
            self.n_ahead_talk = n_ahead_talk_to_restore
            self.n_passes = n_passes_to_restore

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=_base_past_key_values,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past