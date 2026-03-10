"""Single LoRA training run for discovery lab.

Hyperparameter block at top is agent-mutable.
Loads 4-bit NF4 model, applies LoRA, trains under time budget, evaluates val_loss.
Appends results to results.tsv.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from transformers import default_data_collator

from prepare import tokenize_dataset, evaluate_val_loss

# ============================================================
# HYPERPARAMETERS (agent-mutable)
# ============================================================
LEARNING_RATE = 2e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SCHEDULER = "cosine"  # "cosine" or "linear"
TARGET_MODULES = ["q_proj", "v_proj"]  # options: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
GRADIENT_ACCUMULATION_STEPS = 4
BATCH_SIZE = 4

# ============================================================
# FIXED (do not change during discovery)
# ============================================================
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TIME_BUDGET = 300  # seconds (5 minutes)
MAX_LENGTH = 512
MAX_EXAMPLES = 500
EVAL_BATCH_SIZE = 4
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")


def parse_args():
    parser = argparse.ArgumentParser(description="Single LoRA training run")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--scheduler", default=SCHEDULER, choices=["cosine", "linear"])
    parser.add_argument("--target_modules", nargs="+", default=TARGET_MODULES)
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--time_budget", type=int, default=TIME_BUDGET)
    parser.add_argument("--max_examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--max_val_examples", type=int, default=500,
                        help="Max val examples (0 = all, default 500 for discovery)")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", default="", help="Free-text note for results.tsv")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"=== Discovery Lab Training Run ===")
    print(f"Model: {args.model}")
    print(f"lr={args.lr}, rank={args.rank}, alpha={args.alpha}, dropout={args.dropout}")
    print(f"wd={args.weight_decay}, warmup={args.warmup_ratio}, scheduler={args.scheduler}")
    print(f"targets={args.target_modules}, grad_accum={args.grad_accum}, batch={args.batch_size}")
    print(f"time_budget={args.time_budget}s, max_examples={args.max_examples}")
    print()

    # --- Data ---
    print("Loading data...")
    train_data, val_data = tokenize_dataset(
        model_name=args.model,
        max_length=args.max_length,
        max_examples=args.max_examples,
        max_val_examples=args.max_val_examples,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=default_data_collator,
    )

    # --- Model ---
    print("Loading model (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Estimate total steps from time budget (rough: assume we get through data)
    steps_per_epoch = len(train_loader) // args.grad_accum
    estimated_epochs = max(1, args.time_budget // 60)  # rough: 1 epoch per minute
    total_steps = steps_per_epoch * estimated_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Training Loop ---
    print(f"\nTraining (budget: {args.time_budget}s)...")
    model.train()
    device = next(model.parameters()).device

    start_time = time.time()
    global_step = 0
    micro_step = 0
    epoch = 0
    total_train_loss = 0.0
    loss_count = 0

    while True:
        epoch += 1
        for batch in train_loader:
            if time.time() - start_time >= args.time_budget:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            total_train_loss += outputs.loss.item()
            loss_count += 1
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_train_loss / loss_count
                    print(f"  step {global_step} | loss {avg_loss:.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if time.time() - start_time >= args.time_budget:
            break

    elapsed = time.time() - start_time
    avg_train_loss = total_train_loss / loss_count if loss_count > 0 else float("inf")
    print(f"\nTraining done: {global_step} steps, {epoch} epochs, {elapsed:.1f}s")
    print(f"Avg train loss: {avg_train_loss:.4f}")

    # --- Evaluation ---
    print("Evaluating val loss...")
    val_loss = evaluate_val_loss(model, val_data, batch_size=EVAL_BATCH_SIZE)
    print(f"Val loss: {val_loss:.4f}")

    # --- Results ---
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "lr": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": args.scheduler,
        "target_modules": "+".join(args.target_modules),
        "grad_accum": args.grad_accum,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_examples": args.max_examples,
        "max_length": args.max_length,
        "seed": args.seed,
        "steps": global_step,
        "epochs": epoch,
        "elapsed_s": round(elapsed, 1),
        "train_loss": round(avg_train_loss, 6),
        "val_loss": round(val_loss, 6),
        "trainable_params": trainable,
        "note": args.note,
    }

    # Parseable output
    print(f"\n=== RESULT ===")
    for k, v in result.items():
        print(f"{k}={v}")
    print(f"=== END ===")

    # Append to TSV
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys(), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"\nAppended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
