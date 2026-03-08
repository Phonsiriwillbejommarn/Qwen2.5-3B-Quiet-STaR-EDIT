"""
Quiet-STAR Inference Script (Qwen2.5-3B)
Run inference with a trained Quiet-STAR model.

IMPORTANT: The model generates <|startthought|> and <|endthought|> tokens
during generation. These must be masked out for clean output.

Usage:
    python inference.py --model_path ./outputs/quietstar_qwen25_3b_final_XXXXX
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoConfig

from config import QuietStarConfig
from modeling_quiet_star import QuietStarQwen2ForCausalLM


def load_model(model_path, device="cuda", n_ahead=None, n_passes=None):
    """Load trained Quiet-STAR model and tokenizer."""

    print(f"Loading tokenizer from Qwen/Qwen2.5-3B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    
    # Add special thought tokens to tokenizer just like in training
    special_tokens_to_add = ['<|startthought|>', '<|endthought|>', '<think>', '</think>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

    tokenizer.padding_side = "left"  # For generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {model_path}...")
    try:
        config = QuietStarConfig.from_pretrained(model_path)
        model = QuietStarQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
        )
    except Exception as e:
        print(f"Could not load as QuietStarConfig ({e}), trying standard load...")
        model = QuietStarQwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    # Allow custom n_ahead and n_passes for inference
    if n_ahead is not None:
        model.n_ahead = n_ahead
        print(f"  Explicitly set n_ahead={model.n_ahead}")
    elif getattr(model.config, "use_thinking_gate", False):
        trained_n_ahead = getattr(model.config, "max_thoughts", 8) - 4 - 1  # max_thoughts = n_ahead + n_ahead_talk + 1
        model.n_ahead = max(2, trained_n_ahead)  # at least 2 for start+end tokens
        print(f"  Thinking Gate enabled — n_ahead={model.n_ahead} (gate will reduce dynamically)")
    else:
        # Default to whatever the config says, or 1
        model.n_ahead = getattr(model.config, "n_ahead", 1)
        print(f"  Defaulting n_ahead={model.n_ahead}")
        
    model.n_ahead_talk = getattr(model.config, "n_ahead_talk", 1)
    
    if n_passes is not None:
        model.n_passes = n_passes
    else:
        model.n_passes = getattr(model.config, "n_passes", 1)
    print(f"  Using n_passes={model.n_passes}")
    model.original_mode = False
    model.tokenizer = tokenizer

    # Set thought token IDs if they exist
    if "<|startthought|>" in tokenizer.get_vocab():
        model.start_token_id = tokenizer.convert_tokens_to_ids("<|startthought|>")
        model.tokenizer_has_start_thought_token = True
    if "<|endthought|>" in tokenizer.get_vocab():
        model.end_token_id = tokenizer.convert_tokens_to_ids("<|endthought|>")
        model.tokenizer_has_end_thought_token = True

    # Enable KV cache for autoregressive generation
    # This lets generate() reuse the base-pass cache across tokens,
    # giving a major speedup (O(1) per token instead of O(T)).
    model.config.use_cache = True

    model.eval()
    print("✓ Model loaded successfully (KV cache enabled for generation)")
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    suppress_thought_tokens=True,
):
    """Generate text with the Quiet-STAR model, suppressing thought tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    suppress_tokens = []
    if suppress_thought_tokens:
        if hasattr(model, "start_token_id") and model.start_token_id is not None:
            suppress_tokens.append(model.start_token_id)
        if hasattr(model, "end_token_id") and model.end_token_id is not None:
            suppress_tokens.append(model.end_token_id)

    # Save initial talk steps for restoration
    initial_n_ahead_talk = None
    if hasattr(model, "n_ahead_talk"):
        initial_n_ahead_talk = model.n_ahead_talk
        
    try:
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    
        if suppress_tokens:
            generate_kwargs["suppress_tokens"] = suppress_tokens
    
        output_ids = model.generate(**inputs, **generate_kwargs)
    finally:
        # Restore n_ahead_talk cleanly
        if initial_n_ahead_talk is not None:
            model.n_ahead_talk = initial_n_ahead_talk

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


@torch.no_grad()
def compute_perplexity(model, tokenizer, text, max_length=512):
    """Compute perplexity of text."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
    )

    perplexity = torch.exp(outputs.loss).item()
    return perplexity


def interactive_chat(model, tokenizer, args):
    """Run interactive chat with the model."""
    print("\n" + "=" * 60)
    print("Quiet-STAR Interactive Chat (Qwen2.5-3B)")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'ppl: <text>' to compute perplexity")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if prompt.lower().startswith("ppl:"):
            text = prompt[4:].strip()
            if text:
                ppl = compute_perplexity(model, tokenizer, text)
                print(f"Perplexity: {ppl:.2f}\n")
            else:
                print("Please provide text after 'ppl:'\n")
            continue

        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )
        print(f"Model: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Quiet-STAR Inference (Qwen2.5-3B)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--n_ahead", type=int, default=None,
                        help="Override n_ahead (thoughts) for inference. Defaults to config.")
    parser.add_argument("--n_passes", type=int, default=None,
                        help="Override n_passes for inference. Defaults to config.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_model(
        args.model_path, 
        device=args.device,
        n_ahead=args.n_ahead,
        n_passes=args.n_passes,
    )

    if args.prompt:
        response = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        interactive_chat(model, tokenizer, args)


if __name__ == "__main__":
    main()
