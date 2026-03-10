# LoRA Hyperparameter Discovery Program

You are searching for optimal LoRA fine-tuning hyperparameters for Llama-3.1-8B-Instruct on alpaca_cleaned.

## Goal

Find the configuration that minimizes **val_loss** within a 5-minute training budget per run.

## How To Run

```bash
python train.py --lr 2e-4 --rank 8 --alpha 16 --dropout 0.05 --weight_decay 0.01 --warmup_ratio 0.03 --scheduler cosine --target_modules q_proj v_proj --grad_accum 4 --batch_size 4
```

Results are appended to `results.tsv`. Read it before each run to see all prior findings.

## Search Space

| Parameter | Range | Default | Notes |
|---|---|---|---|
| `--lr` | 1e-5 to 1e-3 | 2e-4 | Learning rate. Try: 5e-5, 1e-4, 2e-4, 3e-4, 5e-4 |
| `--rank` | 4, 8, 16, 32, 64 | 8 | LoRA rank. Higher = more params |
| `--alpha` | 8, 16, 32, 64, 128 | 16 | LoRA alpha. Usually 1x or 2x rank |
| `--dropout` | 0.0, 0.01, 0.05, 0.1 | 0.05 | LoRA dropout |
| `--weight_decay` | 0.0, 0.001, 0.01, 0.1 | 0.01 | AdamW weight decay |
| `--warmup_ratio` | 0.0, 0.01, 0.03, 0.05, 0.1, 0.2 | 0.03 | Fraction of steps for warmup. Cap at 0.2 |
| `--scheduler` | cosine, linear | cosine | LR schedule |
| `--target_modules` | See below | q_proj v_proj | Which layers get LoRA |
| `--grad_accum` | 1, 2, 4, 8, 16 | 4 | Gradient accumulation steps |
| `--batch_size` | 1, 2, 4, 8 | 4 | Micro batch size (limited by GPU memory) |

### Target Module Options

Pick a combination:
- Attention only: `q_proj v_proj` (default, cheapest)
- Attention full: `q_proj k_proj v_proj o_proj`
- Attention + MLP: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (most expensive)
- Attention QKV + MLP gate/up: `q_proj k_proj v_proj gate_proj up_proj` (middle ground)

## Rules

1. **Change 1-2 parameters at a time.** This lets you attribute improvements to specific changes.
2. **Read results.tsv before every run.** Know what's been tried. Don't repeat experiments.
3. **Start with the default config** to establish a baseline (if results.tsv doesn't exist yet).
4. **Track what you learn.** Add a `--note` to each run describing your hypothesis.
5. **Follow the gradient.** If lowering lr helps, try lowering it more. If rank 16 beats 8, try 32.
6. **Explore then exploit.** First 20 runs: broad sweeps. After that: refine the best region.
7. **Don't waste runs on known-bad configs.** If lr=1e-3 diverges, don't try it again with small changes.
8. **Effective batch size = batch_size × grad_accum.** Consider this when changing either.

## Strategy Suggestions

### Phase 1: Baselines (runs 1-3)
1. Default config (baseline)
2. Higher lr (5e-4) to see if default is conservative
3. More target modules (attention full) to see if capacity helps

### Phase 2: Learning Rate Sweep (runs 4-8)
Sweep lr across 5e-5, 1e-4, 2e-4, 3e-4, 5e-4 with best rank/targets from Phase 1.

### Phase 3: Rank/Alpha Exploration (runs 9-15)
Try rank 4, 16, 32 with the best lr. For each rank, try alpha = rank and alpha = 2×rank.

### Phase 4: Regularization (runs 16-22)
Best config so far + vary dropout (0.0, 0.01, 0.1) and weight_decay (0.0, 0.001, 0.1).

### Phase 5: Target Module Search (runs 23-30)
Best regularization config + try all 4 target module combinations.

### Phase 6: Refinement (runs 31+)
Fine-tune the best config: small lr adjustments, warmup tuning, scheduler comparison.

## Output Format

Each run prints a parseable block:
```
=== RESULT ===
val_loss=X.XXXX
lr=...
rank=...
...
=== END ===
```

And appends a row to `results.tsv`.

## What Success Looks Like

- 80-100 experiments over ~8 hours
- Clear ranking of hyperparameter importance
- Top 5 candidates with val_loss meaningfully below baseline
- Understanding of which parameters matter most and which don't
