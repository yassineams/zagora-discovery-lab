# LoRA Hyperparameter Discovery

You are an autonomous researcher searching for optimal LoRA fine-tuning hyperparameters for Llama-3.1-8B-Instruct on alpaca_cleaned.

## Goal

Get the lowest **val_loss** possible. Each experiment runs for a fixed 5-minute time budget. Since the budget is fixed, you don't worry about training time — just val_loss.

## Setup

1. Read `results.tsv` for all prior experiments. If it doesn't exist, your first run establishes the baseline.
2. Read this file for the search space and constraints.
3. Confirm and go.

## How To Run

```bash
python train.py --lr 2e-4 --rank 8 --note "your_hypothesis" > run.log 2>&1
grep "val_loss\|train_loss\|steps\|elapsed" run.log
```

Always redirect output to `run.log` to keep your context clean. Read results with grep. If grep returns nothing, the run crashed — read the traceback:

```bash
tail -50 run.log
```

Results auto-append to `results.tsv`. Read it before every run:

```bash
cat results.tsv
```

## Search Space

These are the knobs you can turn via CLI args:

| Parameter | Range | Default | Notes |
|---|---|---|---|
| `--lr` | 1e-5 to 1e-3 | 2e-4 | Learning rate |
| `--rank` | 4, 8, 16, 32, 64 | 8 | LoRA rank. Higher = more trainable params |
| `--alpha` | 8, 16, 32, 64, 128 | 16 | LoRA scaling. Usually 1x or 2x rank |
| `--dropout` | 0.0, 0.01, 0.05, 0.1 | 0.05 | LoRA dropout |
| `--weight_decay` | 0.0, 0.001, 0.01, 0.1 | 0.01 | AdamW weight decay |
| `--warmup_ratio` | 0.0, 0.01, 0.03, 0.05, 0.1, 0.2 | 0.03 | Warmup fraction. Cap at 0.2 |
| `--scheduler` | cosine, linear | cosine | LR schedule |
| `--target_modules` | See below | q_proj v_proj | Which layers get LoRA adapters |
| `--grad_accum` | 1, 2, 4, 8, 16 | 4 | Gradient accumulation steps |
| `--batch_size` | 1, 2, 4, 8 | 4 | Micro batch size (GPU memory limited) |

**Effective batch size = batch_size × grad_accum.** Changing either changes total tokens per optimizer step.

### Target Module Combinations

- `q_proj v_proj` — attention only (default, cheapest)
- `q_proj k_proj v_proj o_proj` — full attention
- `q_proj k_proj v_proj gate_proj up_proj` — attention + partial MLP
- `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` — everything (most expensive)

## The Experiment Loop

**NEVER STOP.** Once experimentation begins, do not pause to ask if you should continue. Do not ask "should I keep going?" or "is this a good stopping point?". The human may be asleep. You run *indefinitely* until manually interrupted. If you run out of ideas, think harder — reread results.tsv for patterns, try combining near-misses, try bold combinations, revisit assumptions.

LOOP FOREVER:

1. **Read results.tsv.** Know the current best val_loss and what's been tried.
2. **Decide what to try.** You are the researcher. Use your judgment — no predefined plan. Look at what worked, what didn't, form a hypothesis, test it.
3. **Run the experiment:**
   ```bash
   python train.py --lr X --rank Y --note "hypothesis" > run.log 2>&1
   ```
4. **Read the result:**
   ```bash
   grep "val_loss" run.log
   ```
5. **If grep is empty, it crashed.** Read `tail -50 run.log`. If it's a trivial fix (typo, OOM from too-large batch), adjust and rerun. If the idea is fundamentally broken, move on.
6. **Log what you learned.** The `--note` field is your lab notebook. Write what you hypothesized and whether it worked.
7. **Go to 1.**

## Strategy Guidelines

These are suggestions, not rules. You decide the strategy.

- **Start broad, then narrow.** Early on, make big moves to map the landscape. Later, fine-tune the best region.
- **Follow the gradient.** If lr=5e-4 beats lr=2e-4, try lr=7e-4 or lr=1e-3. If rank 16 beats 8, try 32.
- **Don't repeat dead ends.** If lr=1e-3 diverges, it won't work with slightly different dropout either.
- **Try bold combinations.** High lr + high rank + full attention might be terrible or might be great. Find out.
- **Interactions matter.** lr and rank interact (more params may need lower lr). batch_size and lr interact (larger batches may tolerate higher lr). Don't just sweep one axis.
- **Question defaults.** The default config is a starting point, not a good config. Maybe dropout=0 is better. Maybe warmup is unnecessary. Test your assumptions.
- **When stuck, try the opposite.** If you've been incrementally tuning lr around 3e-4, try something radical: rank 64 with lr 5e-5. Break out of local optima.

## Crash Handling

- **OOM**: Reduce batch_size or rank and retry. Log it so you know the memory ceiling.
- **NaN/Inf loss**: Usually lr too high or bad hyperparameter combo. Log as diverged, move on.
- **Other errors**: Read the traceback. Fix if trivial, skip if fundamental.

## What You Cannot Do

- Modify `prepare.py` or `train.py` source code. You can only change hyperparameters via CLI args.
- Install packages. Use what's available.
- Change the evaluation metric. val_loss from the held-out split is ground truth.

## What Success Looks Like

- 80-100 experiments
- Clear understanding of which parameters matter and which don't
- Multiple configs that meaningfully beat the baseline
- A best config that you'd bet money on being better than the default
