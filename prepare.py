"""Data preparation for discovery lab.

Downloads alpaca_cleaned, tokenizes, creates deterministic train/val split, caches.
Also provides evaluate_val_loss() used by train.py.
"""

import argparse
import hashlib
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, default_data_collator


# ---------- constants ----------
DATASET_NAME = "yahma/alpaca-cleaned"
SPLIT_SEED = 42  # fixed across all phases
VAL_RATIO = 0.1
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def format_example(example: dict) -> str:
    """Format an alpaca example into a single string for SFT."""
    if example.get("input", "").strip():
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )


def tokenize_dataset(model_name: str = DEFAULT_MODEL, max_length: int = 512,
                     max_examples: int = 0, max_val_examples: int = 0):
    """Download, tokenize, split, and cache the dataset.

    Returns (train_dataset, val_dataset) as lists of {input_ids, attention_mask, labels}.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(DATASET_NAME, split="train")

    # Deterministic split: hash-based for reproducibility across machines
    indices = list(range(len(ds)))
    val_indices = []
    train_indices = []
    for idx in indices:
        h = hashlib.md5(f"{SPLIT_SEED}_{idx}".encode()).hexdigest()
        if int(h, 16) % 1000 < int(VAL_RATIO * 1000):
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    # Deterministic subsetting (always take first N from sorted indices)
    if max_examples > 0:
        train_indices = sorted(train_indices)[:max_examples]
    val_indices = sorted(val_indices)
    if max_val_examples > 0:
        val_indices = val_indices[:max_val_examples]

    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")

    def tokenize_indices(idxs):
        results = []
        for idx in idxs:
            text = format_example(ds[idx])
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            results.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        return results

    train_data = tokenize_indices(train_indices)
    val_data = tokenize_indices(val_indices)

    print(f"Tokenized: {len(train_data)} train, {len(val_data)} val (max_length={max_length})")
    return train_data, val_data


def evaluate_val_loss(model, val_data: list, batch_size: int = 4) -> float:
    """Compute mean val loss over the held-out split."""
    model.eval()
    device = next(model.parameters()).device

    loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                        collate_fn=default_data_collator)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else float("inf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and cache dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Max training examples (0 = all)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    train_data, val_data = tokenize_dataset(
        model_name=args.model,
        max_length=args.max_length,
        max_examples=args.max_examples,
    )
    print(f"Done. {len(train_data)} train, {len(val_data)} val examples ready.")
