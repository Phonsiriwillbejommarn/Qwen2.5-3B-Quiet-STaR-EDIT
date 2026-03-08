"""
Quiet-STAR Evaluation Helpers
Functions for preprocessing evaluation datasets (GSM8K, CommonsenseQA)
and computing metrics during training.
"""

import numpy as np
from transformers import AutoTokenizer


# Tokenizer for preprocessing (will be set by training script)
_tokenizer = None
_max_length = 512


def set_tokenizer(tokenizer, max_length=512):
    """Set the global tokenizer for preprocessing functions."""
    global _tokenizer, _max_length
    _tokenizer = tokenizer
    _max_length = max_length


def truncate_or_pad(ids, max_length, pad_token_id):
    """Truncate or pad a sequence of token IDs to a fixed length."""
    if len(ids) > max_length:
        ids = ids[:max_length]
    elif len(ids) < max_length:
        ids = ids + [pad_token_id] * (max_length - len(ids))
    return ids


def preprocess_function(examples):
    """
    Tokenize training examples and split long articles into chunks.

    Instead of truncating at max_length (losing data), this splits each article
    into multiple sequential chunks so all content is used for training:
      Article (5000 tokens) → [chunk1: 1024] [chunk2: 1024] [chunk3: 1024] [chunk4: 1024] [chunk5: 928+pad]
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = examples.get("text", examples.get("content", [""]))
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize without truncation first to get all tokens
    tokenized_full = _tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_tensors=None,
    )

    # Split into chunks of max_length
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for input_ids in tokenized_full["input_ids"]:
        # Split this article's tokens into chunks
        for start in range(0, len(input_ids), _max_length):
            chunk = input_ids[start:start + _max_length]

            # Skip very short chunks (less than 64 tokens) to avoid noise
            if len(chunk) < 64:
                continue

            # Pad if needed
            attention_mask = [1] * len(chunk)
            label = chunk.copy()
            if len(chunk) < _max_length:
                pad_len = _max_length - len(chunk)
                chunk = chunk + [_tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                label = label + [-100] * pad_len

            all_input_ids.append(chunk)
            all_attention_masks.append(attention_mask)
            all_labels.append(label)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def preprocess_gsm8k_sft(examples):
    """
    Preprocess GSM8K examples for SFT warmup with <think> tag format.

    Formats each example as:
        Q: {question}
        <think> {reasoning_steps} </think>
        #### {final_answer}

    This teaches the model the token-space thinking format during SFT phase
    so it knows HOW to use <think> tokens before RL training begins.

    GSM8K 'answer' field format:
        "Step 1: ... Step 2: ... #### 42"
    We split at '####' to separate reasoning from answer.
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = []
    gold_answers = []
    for q, a in zip(examples["question"], examples["answer"]):
        # Split reasoning and final answer at ####
        if "####" in a:
            reasoning, final_answer = a.rsplit("####", 1)
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = a.strip()
            final_answer = ""

        # Format with <think> tags
        text = f"Q: {q}\n<think> {reasoning} </think>\n#### {final_answer}"
        texts.append(text)
        gold_answers.append(final_answer)

    tokenized = _tokenizer(
        texts,
        truncation=True,
        max_length=_max_length,
        padding="max_length",
        return_tensors=None,
    )

    # Create labels and mask padding with -100 so loss ignores it
    # We also mask the Question prefix so the model only trains on generating the response
    labels = []
    pad_id = _tokenizer.pad_token_id
    think_id = _tokenizer.encode("<think>", add_special_tokens=False)[0]

    for ids in tokenized["input_ids"]:
        label = [tok if tok != pad_id else -100 for tok in ids]
        
        # Find where <think> starts to mask the question prefix
        try:
            start_idx = ids.index(think_id)
            # Mask everything before <think> with -100
            for i in range(start_idx):
                label[i] = -100
        except ValueError:
            pass # If <think> not found (shouldn't happen), just leave it
            
        labels.append(label)
        
    tokenized["labels"] = labels
    tokenized["gold_answers"] = gold_answers

    return tokenized


def preprocess_eval_function_gsm(examples):
    """
    Preprocess GSM8K examples for evaluation.
    GSM8K has 'question' and 'answer' fields.
    Format: "Q: {question}\nA: {answer}"
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        text = f"Q: {q}\nA: {a}"
        texts.append(text)

    tokenized = _tokenizer(
        texts,
        truncation=True,
        max_length=_max_length,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def preprocess_eval_function_csqa(examples):
    """
    Preprocess CommonsenseQA examples for evaluation.
    CommonsenseQA has 'question', 'choices', and 'answerKey' fields.
    Format: "Q: {question}\nChoices: A) ... B) ... C) ...\nAnswer: {answer}"
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = []
    for q, choices, answer_key in zip(
        examples["question"], examples["choices"], examples["answerKey"]
    ):
        # Format choices
        choice_texts = []
        labels = choices["label"]
        choice_text_list = choices["text"]
        for label, text in zip(labels, choice_text_list):
            choice_texts.append(f"{label}) {text}")
        choices_str = " ".join(choice_texts)

        text = f"Q: {q}\nChoices: {choices_str}\nAnswer: {answer_key}"
        texts.append(text)

    tokenized = _tokenizer(
        texts,
        truncation=True,
        max_length=_max_length,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def compute_metrics(eval_preds):
    """
    Compute metrics for evaluation.
    For language modeling, we compute perplexity and accuracy.
    """
    logits, labels = eval_preds

    if isinstance(logits, tuple):
        logits = logits[0]

    # Convert to torch tensor if they are numpy arrays
    import torch
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Create mask for valid labels
    mask = shift_labels != -100
    if _tokenizer is not None and hasattr(_tokenizer, 'pad_token_id') and _tokenizer.pad_token_id is not None:
        mask = mask & (shift_labels != _tokenizer.pad_token_id)

    # Compute accuracy (top-1)
    predictions = torch.argmax(shift_logits, dim=-1)
    correct = (predictions == shift_labels) & mask
    accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0

    # Avoid CPU OOM on large vocabs: use PyTorch CrossEntropyLoss directly instead of manual log_softmax arrays
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    # Flatten just like training
    shift_logits_flat = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels_flat = shift_labels.view(-1).clone()
    shift_labels_flat[~mask.view(-1)] = -100

    # Calculate loss
    losses = loss_fct(shift_logits_flat, shift_labels_flat)
    
    # Perplexity is exp(avg_loss)
    avg_loss = losses[shift_labels_flat != -100].mean().item() if (shift_labels_flat != -100).any() else 0.0
    perplexity = float(np.exp(avg_loss))

    return {
        "accuracy": float(accuracy),
        "perplexity": perplexity,
    }
