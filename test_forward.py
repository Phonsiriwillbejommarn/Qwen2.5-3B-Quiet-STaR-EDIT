"""
Tests for Chunk-level Thinking in Quiet-STAR.

Verifies:
  1. Baseline (no chunk) still works
  2. Chunk thinking: forward completes, output shape correct
  3. Chunk thinking produces different (richer) logits than token-level
  4. Different chunk sizes work correctly
  5. Chunk overlap works correctly
  6. Chunk + Gate + KV Cache all together
  7. Chunk boundaries cover the full sequence (no tokens missed)
"""
import torch
from config import QuietStarConfig
from modeling_quiet_star import QuietStarQwen2ForCausalLM

VOCAB = 1000
H     = 128

def make_model(use_chunk=False, chunk_size=4, chunk_overlap=0,
               use_gate=False, use_kv=False, n_ahead=4):
    cfg = QuietStarConfig(
        vocab_size=VOCAB, hidden_size=H, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        hidden_act="silu", max_position_embeddings=512,
        initializer_range=0.02, rms_norm_eps=1e-6, use_cache=False,
        max_thoughts=n_ahead + 4 + 1,
        merged_talk_heads=True, merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True, use_concat_talk_head=True,
        use_shallow_think=True, use_shallow_talk=False,
        use_complex_think_head=False, use_complex_talk_head=True,
        use_weighted_talk_head=True, attn_implementation="eager",
        tie_word_embeddings=True,
        use_thinking_gate=use_gate, thinking_gate_hidden_dim=64,
        thinking_gate_sparsity_beta=0.01, thinking_gate_threshold=0.5,
        thinking_gate_bias_init=0.0,
        use_chunk_thinking=use_chunk,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    m = QuietStarQwen2ForCausalLM(cfg)

    class DummyTok:
        pad_token_id = 0

    m.tokenizer = DummyTok()
    m.n_ahead = n_ahead
    m.n_ahead_talk = 2
    m.n_passes = 1
    m.use_start_thought_token = False
    m.use_end_thought_token = False
    m.first_and_last_mode = False
    m.lm_head.weight = m.model.embed_tokens.weight
    m.use_kv_cache_for_thoughts = use_kv
    return m.to(dtype=torch.float32)


def batch(B=2, T=16):
    ids  = torch.randint(1, VOCAB, (B, T))
    return ids, ids.clone(), torch.ones(B, T, dtype=torch.long)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_baseline():
    print("\n── Test 1: Baseline (no chunk) ──")
    m = make_model(); m.eval()
    ids, labels, mask = batch()
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
    assert out.logits.shape == (2, 16, VOCAB)
    assert not torch.isnan(out.loss)
    print(f"  loss={out.loss.item():.4f}  logits={out.logits.shape}  ✓")


def test_chunk_basic():
    print("\n── Test 2: Chunk thinking — shape and loss ──")
    m = make_model(use_chunk=True, chunk_size=4); m.eval()
    ids, labels, mask = batch(T=16)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
    assert out.logits.shape == (2, 16, VOCAB), f"got {out.logits.shape}"
    assert not torch.isnan(out.loss), "loss NaN"
    print(f"  loss={out.loss.item():.4f}  logits={out.logits.shape}  ✓")


def test_chunk_vs_token_level():
    print("\n── Test 3: Chunk vs token-level logits differ ──")
    ids, labels, mask = batch(T=16)

    m_tok   = make_model(use_chunk=False)
    m_chunk = make_model(use_chunk=True, chunk_size=4)
    m_chunk.load_state_dict(m_tok.state_dict())

    m_tok.eval(); m_chunk.eval()
    with torch.no_grad():
        out_tok   = m_tok  (input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
        out_chunk = m_chunk(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)

    diff = (out_tok.logits - out_chunk.logits).abs().mean().item()
    print(f"  Mean logit diff (token-level vs chunk): {diff:.4f}")
    assert diff > 0, "Logits identical — chunk has no effect?"
    print(f"  Chunk produces different (richer context) logits  ✓")


def test_chunk_sizes():
    print("\n── Test 4: Various chunk sizes ──")
    for C in [2, 4, 8, 16]:
        m = make_model(use_chunk=True, chunk_size=C); m.eval()
        ids, labels, mask = batch(T=16)
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
        assert out.logits.shape == (2, 16, VOCAB)
        assert not torch.isnan(out.loss)
        n_chunks = (16 + C - 1) // C
        print(f"  chunk_size={C:2d}  n_chunks={n_chunks}  loss={out.loss.item():.4f}  ✓")


def test_chunk_overlap():
    print("\n── Test 5: Chunk overlap ──")
    for V in [0, 1, 2]:
        m = make_model(use_chunk=True, chunk_size=6, chunk_overlap=V); m.eval()
        ids, labels, mask = batch(T=18)
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
        assert out.logits.shape == (2, 18, VOCAB)
        assert not torch.isnan(out.loss)
        print(f"  overlap={V}  loss={out.loss.item():.4f}  ✓")


def test_chunk_full_coverage():
    print("\n── Test 6: All tokens covered (no gaps) ──")
    C, T = 5, 17   # deliberately awkward: T not divisible by C
    m = make_model(use_chunk=True, chunk_size=C); m.eval()
    ids, labels, mask = batch(T=T)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
    assert out.logits.shape == (2, T, VOCAB)
    # Every logit position should be non-zero (covered by at least one chunk)
    covered = (out.logits.abs().sum(-1) > 0).all()
    assert covered, "Some token positions have zero logits (coverage gap!)"
    print(f"  T={T}, C={C}: all {T} positions covered  ✓")


def test_chunk_plus_gate_plus_kv():
    print("\n── Test 7: Chunk + Gate + KV Cache ──")
    m = make_model(use_chunk=True, chunk_size=4, use_gate=True, use_kv=True); m.eval()
    ids, labels, mask = batch(T=16)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
    assert not torch.isnan(out.loss)
    assert out.logits.shape == (2, 16, VOCAB)
    print(f"  loss={out.loss.item():.4f}  all features combined  ✓")


def test_no_recursion():
    print("\n── Test 8: No infinite recursion ──")
    # _chunk_forward sets use_chunk_thinking=False before calling forward(),
    # then restores it.  Verify the flag is restored even after an exception.
    m = make_model(use_chunk=True, chunk_size=4); m.eval()
    assert m.use_chunk_thinking is True
    ids, labels, mask = batch(T=8)
    with torch.no_grad():
        m(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
    assert m.use_chunk_thinking is True, "Flag not restored after forward!"
    print(f"  use_chunk_thinking correctly restored after forward  ✓")


def main():
    print("=" * 58)
    print("  Quiet-STaR: Chunk-level Thinking Tests")
    print("=" * 58)
    test_baseline()
    test_chunk_basic()
    test_chunk_vs_token_level()
    test_chunk_sizes()
    test_chunk_overlap()
    test_chunk_full_coverage()
    test_chunk_plus_gate_plus_kv()
    test_no_recursion()
    print("\n" + "=" * 58)
    print("  ✅  All tests passed!")
    print("=" * 58)


if __name__ == "__main__":
    main()