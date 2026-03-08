"""
Quiet-STAR Configuration
Custom Qwen2Config with thought-specific parameters.
Based on: https://arxiv.org/abs/2403.09629
Adapted for Qwen2.5-3B architecture.
"""

from transformers import Qwen2Config


class QuietStarConfig(Qwen2Config):
    """
    Extended Qwen2Config with Quiet-STAR thought generation parameters.

    Additional Args:
        max_thoughts (`int`, *optional*, defaults to 16):
            Maximum number of thought tokens (n_ahead + n_ahead_talk + 1).
        merged_talk_heads (`bool`, *optional*, defaults to True):
            Whether talk heads are merged into a single head.
        merged_lm_and_talk_heads (`bool`, *optional*, defaults to False):
            Whether the LM head and talk head share weights.
        merged_lm_and_think_heads (`bool`, *optional*, defaults to True):
            Whether the LM head and think head share weights.
        use_concat_talk_head (`bool`, *optional*, defaults to True):
            Concatenate base + thought hidden states as input to the talk head.
        use_shallow_think (`bool`, *optional*, defaults to True):
            Use a shallow (single-layer) think head.
        use_shallow_talk (`bool`, *optional*, defaults to False):
            Use a shallow (single-layer) talk head.
        use_complex_think_head (`bool`, *optional*, defaults to False):
            Use a complex (multi-layer) think head.
        use_complex_talk_head (`bool`, *optional*, defaults to True):
            Use a complex (multi-layer) talk head.
        use_weighted_talk_head (`bool`, *optional*, defaults to True):
            Use weighted mixing of base and thought predictions.

        --- Selective Thinking Gate ---
        use_thinking_gate (`bool`, *optional*, defaults to False):
            Enable a learned gate that decides per token position whether to
            apply thought augmentation. When disabled the model thinks at every
            position (original Quiet-STAR behaviour).
        thinking_gate_hidden_dim (`int`, *optional*, defaults to 128):
            Hidden dimension of the two-layer ThinkingGate MLP.
        thinking_gate_sparsity_beta (`float`, *optional*, defaults to 0.01):
            Weight for the L1 sparsity penalty on gate activations during
            training. Higher values → the gate will skip more positions.
        thinking_gate_threshold (`float`, *optional*, defaults to 0.5):
            Hard threshold used at *inference* time: positions with gate
            probability < threshold do not receive thought augmentation.
        thinking_gate_bias_init (`float`, *optional*, defaults to -2.0):
            Initial bias for the gate's output layer. Negative values make the
            gate start in the "don't think" state, allowing the model to learn
            when thinking is beneficial rather than thinking everywhere from
            the beginning.
    """

    model_type = "qwen2"

    def __init__(
        self,
        # Quiet-STAR specific parameters
        max_thoughts=16,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        # Selective thinking gate
        use_thinking_gate=False,
        thinking_gate_hidden_dim=128,
        thinking_gate_sparsity_beta=0.01,
        thinking_gate_threshold=0.5,
        thinking_gate_bias_init=-2.0,
        use_chunk_thinking=False,
        chunk_size=8,
        chunk_overlap=0,
        thought_chunk_size=1,
        **kwargs,
    ):
        self.max_thoughts = max_thoughts
        self.merged_talk_heads = merged_talk_heads
        self.merged_lm_and_talk_heads = merged_lm_and_talk_heads
        self.merged_lm_and_think_heads = merged_lm_and_think_heads
        self.use_concat_talk_head = use_concat_talk_head
        self.use_shallow_think = use_shallow_think
        self.use_shallow_talk = use_shallow_talk
        self.use_complex_think_head = use_complex_think_head
        self.use_complex_talk_head = use_complex_talk_head
        self.use_weighted_talk_head = use_weighted_talk_head
        # Selective thinking gate
        self.use_thinking_gate = use_thinking_gate
        self.thinking_gate_hidden_dim = thinking_gate_hidden_dim
        self.thinking_gate_sparsity_beta = thinking_gate_sparsity_beta
        self.thinking_gate_threshold = thinking_gate_threshold
        self.thinking_gate_bias_init = thinking_gate_bias_init
        
        # Chunk-level thinking
        self.use_chunk_thinking = use_chunk_thinking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.thought_chunk_size = thought_chunk_size

        super().__init__(**kwargs)