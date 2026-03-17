# Can cheap local LoRA search compress the distributed fine-tuning search space?

**Llama 8B → 70B proxy-to-target hyperparameter transfer**

A proxy-discovered LoRA recipe beat the default both locally and in a single controlled 70B distributed screen. The gap compressed under stricter local confirmation (4.14% → 1.48%), then re-expanded in the 70B screen (3.35%).

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): 3 files, fixed budget, autonomous search.

---

## The Question

Default LoRA configurations (lr=2e-4, rank=8, q_proj+v_proj) are the most common starting point, that rarely get questioned in practice. Can autonomous hyperparameter search on a cheap proxy model (Llama-3.1-8B-Instruct, 1 GPU) find recipes that transfer to an expensive target (Llama-3.1-70B-Instruct, distributed GPUs), compressing ~100 proxy experiments into a single paired 70B screen after local confirmation?

## Methodology

Three-phase pipeline: **Discovery → Confirmation → Cross-Scale Validation**.

| Phase | Model | Hardware | Budget | Train Examples |
|-------|-------|----------|--------|----------------|
| Discovery | Llama-3.1-8B-Instruct | 1× RTX 4090 24GB | 5 min × 100 runs | 500 |
| Confirmation | Llama-3.1-8B-Instruct | 1× RTX 4090 24GB | 10 min × 18 runs | 2000 |
| Cross-Scale | Llama-3.1-70B-Instruct | 2× RTX 4090 48GB + coordinator | 20 steps, EBS=64 | 1600 train / 500 val subset |

Cross-scale validation used distributed pipeline parallelism to split 70B across 2 GPU workers (40 transformer blocks each). We used [Zagora](https://zagora.ai) for this step, but any distributed fine-tuning setup that supports LoRA on 70B-class models works — the discovered recipe is just hyperparameters.

**Design choice**: only hyperparameters transfer across scales. Batch size was overridden to a fixed EBS of 64 at 70B, since batch-dependent effects from discovery (EBS=16) may not hold at different data scales and sequence lengths.

All models quantized to NF4 (4-bit) at both scales. Dataset: [alpaca_cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (instruction SFT).

## Results

### Discovery Phase

An autonomous agent ran [`program.md`](program.md) — a search protocol defining the hyperparameter space and experiment loop — executing 100 experiments on Llama-3.1-8B-Instruct, each with a 5-minute wall-clock training budget.

**Baseline val_loss**: 1.1984 | **Best val_loss**: 1.1488 (**4.14%** relative improvement)

**Key findings from 100 experiments:**

1. **Target modules dominate**: All 7 modules > full attention (4) > q_proj+v_proj (2). Widening which layers get LoRA adapters matters more than increasing rank.
2. **Lower rank wins under time budgets**: Rank 4 beats 8/16/32 — cheaper per step, more optimizer steps in the same wall-clock time.
3. **Linear scheduler > cosine**: Consistent across all configurations tested.
4. **Minimal regularization at low rank**: Rank 4 with all modules needs no dropout and only tiny weight decay. Low parameter count prevents overfitting naturally.
5. **Learning rate scales inversely with trainable params**: ~1.55e-4 for everything-r4, ~1.3e-4 for everything-r8, ~2e-4 for full-attention-r16.
6. **Warmup unnecessary at short horizons**: With ~50 steps, warmup wastes budget. Zero warmup was optimal.
7. **Batch config matters at the margin**: bs=8/ga=2 slightly outperformed bs=4/ga=4 at the same EBS of 16.

**Top 5 diverse candidates** (grouped by rank × target_modules × batch_size):

| # | Val Loss | lr | Rank | Target Modules | Scheduler | Warmup | WD | Dropout |
|---|---------|-----|------|---------------|-----------|--------|------|---------|
| C1 | 1.148844 | 1.55e-4 | 4 | everything (7) | linear | 0.0 | 0.001 | 0.0 |
| C2 | 1.149438 | 1.7e-4 | 4 | everything (7) | linear | 0.0 | 0.0 | 0.0 |
| C3 | 1.150137 | 1.3e-4 | 8 | everything (7) | linear | 0.01 | 0.001 | 0.1 |
| C4 | 1.151491 | 2e-4 | 16 | full attn (4) | linear | 0.1 | 0.001 | 0.1 |
| C5 | 1.151963 | 1.5e-4 | 8 | everything (7) | linear | 0.1 | 0.001 | 0.05 |

### Local Confirmation

5 candidates × 3 seeds {42, 123, 777} + 3 baseline runs = 18 total runs. Budget increased to 10 minutes with 2000 examples (4× data, 2× time vs discovery).

| Config | Seed 42 | Seed 123 | Seed 777 | Mean | Improvement |
|--------|---------|----------|----------|------|-------------|
| **Baseline** | 1.147215 | 1.146074 | 1.145887 | **1.146392** | — |
| **C1** (r4 everything, bs=8) | 1.129864 | 1.129066 | 1.129767 | **1.129566** | **1.47%** |
| **C2** (r4 everything, bs=4) | 1.129932 | 1.129790 | 1.128441 | **1.129388** | **1.48%** |
| **C3** (r8 everything, do=0.1) | 1.130762 | 1.130088 | 1.130536 | **1.130462** | **1.39%** |
| **C4** (r16 full attn, do=0.1) | 1.138220 | 1.139063 | 1.139720 | **1.139001** | **0.64%** |
| **C5** (r8 everything, do=0.05) | 1.139170 | 1.139411 | 1.140726 | **1.139769** | **0.58%** |

**0 of 5 candidates cleared the pre-registered >3% promotion threshold.** The threshold was calibrated for discovery-phase variance. With 4× more data and 2× more time, the baseline improved dramatically (1.198 → 1.146, a 4.3% self-improvement), leaving less headroom. The absolute gap (~0.017 val_loss) remained consistent — all seeds agreed — but the relative percentage shrank against a stronger baseline.

**Strong directional consistency**: C1–C3 beat the baseline in every individual run (9/9 aggregate across 3 candidates × 3 seeds). C2 had the highest mean improvement (1.48%) with the simplest config (no weight decay, no dropout).

The next step proceeded as a single proof-of-concept cross-scale test with C2, the best consistent candidate.

### Cross-Scale Validation (70B)

A single controlled proof-of-concept: one baseline run and one C2 run on Llama-3.1-70B-Instruct. 70B doesn't fit on a single consumer GPU even in NF4 quantization — distributed training is required.

We used [Zagora](https://zagora.ai), a distributed LoRA fine-tuning platform using Petals pipeline parallelism, to split the 80 transformer blocks across 2 workers (40 blocks each) with a VPS coordinator. Any distributed fine-tuning tool capable of LoRA on 70B+ models can be used to reproduce this step.

**Training curves (eval every 5 steps):**

| Step | Baseline Val Loss | C2 Val Loss | Δ | Relative |
|------|------------------|-------------|---|----------|
| 5 | 1.5713 | 1.3215 | 0.2498 | 15.9% |
| 10 | 1.3175 | 1.2529 | 0.0646 | 4.9% |
| 15 | 1.2371 | 1.2095 | 0.0276 | 2.2% |
| 20 | 1.2059 | 1.1655 | 0.0404 | **3.35%** |

**Loss curve interpretation** (hypothesis, not proven mechanism):

1. **Steps 1–5: C2 leads large** — likely due to zero warmup (full lr from step 1 vs baseline ramping). Partly a warmup artifact.
2. **Steps 5–15: Gap narrows** — baseline's lr catches up post-warmup.
3. **Steps 15–20: Gap widens again** — one interpretation is that C2's adaptation pattern (rank 4 across all 7 modules, linear decay) produces a more durable advantage once the warmup artifact washes out. The late-training re-expansion is the more interesting signal.


**Controlled conditions**: same seed (42), same dataset subset (1600 train / 500 val, deterministic split), 20 optimizer steps, EBS=64 (bs=1, ga=64), NF4 quantization, max_length=1024.

### Improvement Trajectory Across All Phases

| Phase | Scale | Examples | Budget | Relative Improvement |
|-------|-------|----------|--------|---------------------|
| Discovery | 8B | 500 | 5 min | 4.14% |
| Confirmation | 8B | 2000 | 10 min | 1.48% |
| Cross-Scale | 70B | 1600 | 20 steps | 3.35% |

The gap compressed under stricter local confirmation, then re-expanded in the 70B screen. This is evidence that the discovered recipe transfers across scale — not proof of a universal principle.

## The Discovered Recipe

| Parameter | Baseline (Default) | C2 (Discovered) | Notes |
|-----------|----------|-----------------|-------|
| target_modules | q_proj, v_proj (2) | all 7 modules | Most impactful change in discovery |
| rank | 8 | 4 | Traded per-adapter capacity for coverage |
| alpha | 16 | 8 | Maintained 2:1 ratio |
| lr | 2e-4 | 1.7e-4 | Slightly lower for more trainable params |
| dropout | 0.05 | 0.0 | No explicit regularization at low rank |
| weight_decay | 0.01 | 0.0 | Same reasoning |
| warmup_ratio | 0.03 | 0.0 | No warmup under fixed budgets |
| scheduler | cosine | linear | Part of the C2 recipe |

**The pattern**: distribute adaptation widely with minimal per-adapter capacity — low rank (4) across all 7 module types, linear schedule, no explicit regularization. This full recipe beat the baseline at both 8B and 70B in this experiment. Whether individual parameter choices (e.g., linear vs cosine in isolation) generalize is an open question — the 70B evidence validates the recipe as a whole, not its components separately.

## Reproduce

### Phase 1 & 2: Discovery + Confirmation (single GPU)

```bash
pip install -r requirements.txt

# Prepare data (downloads alpaca_cleaned, tokenizes, caches)
python prepare.py

# Run baseline (5-min budget, 500 examples)
python train.py

# Run with discovered recipe
python train.py --lr 1.7e-4 --rank 4 --alpha 8 --dropout 0 \
  --weight_decay 0 --warmup_ratio 0 --scheduler linear \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

# Run with more data/time (confirmation phase)
python train.py --max_examples 2000 --time_budget 600 --seed 42

# Full CLI options
python train.py --help
```

The autonomous agent used [`program.md`](program.md) as its search protocol — it defines the hyperparameter space, experiment loop, and strategy guidelines.

### Phase 3: Cross-Scale Validation (distributed, 70B)

70B requires distributed training. The recipe to validate:

```
lr=1.7e-4, rank=4, alpha=8, dropout=0, weight_decay=0,
warmup_ratio=0, scheduler=linear,
target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

vs the baseline:

```
lr=2e-4, rank=8, alpha=16, dropout=0.05, weight_decay=0.01,
warmup_ratio=0.03, scheduler=cosine,
target_modules=q_proj,v_proj
```

**Both runs should share**: same seed, same dataset subset, same step budget, same EBS, same quantization, same max_length.

You can run this with any distributed fine-tuning tool that supports LoRA on 70B-class models. Our run used [Zagora](https://zagora.ai) (distributed LoRA via Petals pipeline parallelism, 2 GPU workers, VPS coordinator), but alternatives include DeepSpeed ZeRO, FSDP, or multi-node setups with enough aggregate VRAM.

## Structured Findings

```json
{
  "methodology": "proxy-to-target LoRA hyperparameter transfer",
  "proxy_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "target_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "discovery": {
    "hardware": "1x RTX 4090 24GB",
    "experiments": 100,
    "time_budget_per_run_s": 300,
    "train_examples": 500,
    "baseline_val_loss": 1.198448,
    "best_val_loss": 1.148844,
    "improvement_pct": 4.14
  },
  "confirmation": {
    "candidates_tested": 5,
    "seeds_per_candidate": 3,
    "train_examples": 2000,
    "time_budget_per_run_s": 600,
    "baseline_mean_val_loss": 1.146392,
    "best_candidate_mean_val_loss": 1.129388,
    "improvement_pct": 1.48,
    "candidates_promoted_3pct_threshold": 0
  },
  "cross_scale_validation": {
    "hardware": "2x RTX 4090 48GB + VPS coordinator",
    "steps": 20,
    "effective_batch_size": 64,
    "dataset_subset_train": 1600,
    "dataset_subset_val": 500,
    "examples_consumed_per_run": 1280,
    "baseline_val_loss": 1.2059,
    "c2_val_loss": 1.1655,
    "improvement_pct": 3.35,
    "type": "single paired proof-of-concept"
  },
  "validated_recipe": {
    "lr": 1.7e-4,
    "rank": 4,
    "alpha": 8,
    "dropout": 0.0,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "scheduler": "linear",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  },
  "dead_ends": []
}
```

## Limitations

1. **One model family** — Llama 3.1 only. Transfer may differ for other models, or architecturally distinct models.
2. **One dataset** — alpaca_cleaned (instruction SFT). May not generalize to domain-specific, coding, or long-context tasks.
3. **SFT only** — Not tested with DPO, RLHF, or other alignment methods.
4. **Proxy and target differ** in sequence length (512 vs 1024), data scale (500 vs 1600 train examples), and step budget. Transfer is approximate.
5. **Single paired run at 70B** — Directional evidence, not a replicated result. One pair cannot establish statistical significance.
6. **EBS overridden** to 64 for both 70B runs. Batch-dependent effects from discovery (EBS=16) are lost.
7. **Fixed step budget (20 steps), not full convergence** — Val_loss at step 20 is a ranking signal, not a final quality measure.
8. **No candidate cleared the pre-registered >3% threshold** — The 70B screen was a proof-of-concept, not a validated promotion.

## Takeaways

1. A proxy-discovered LoRA recipe beat the default both locally (C2: 1.48%, 3/3 seeds; C1–C3: 9/9 aggregate) and in a single controlled 70B screen (3.35%). The gap compressed under stricter confirmation, then re-expanded at 70B.
2. The discovery phase identified (rank 4, all 7 modules) as consistently better than (rank 8, 2 modules) across 100 experiments on 8B — the most impactful finding.
3. Local confirmation is essential — compressed the discovery signal from 4.14% to 1.48%, revealing which improvements are durable under more data and time.
4. The C2 recipe beat the baseline at both scales in this experiment.
5. This is candidate evidence for proxy-to-target transfer within a model family, not a universal result. Replication on other families and datasets is needed.

## Files

| File | Purpose |
|---|---|
| [`train.py`](train.py) | Single training run: load 4-bit model, apply LoRA, train under time budget, evaluate val_loss |
| [`prepare.py`](prepare.py) | Download alpaca_cleaned, tokenize, deterministic hash-based train/val split |
| [`program.md`](program.md) | Search space definition and autonomous agent instructions (the "research protocol") |
| `requirements.txt` | Python dependencies |

## Environment

```
Python: 3.10.13
torch: 2.10.0+cu128
CUDA: NVIDIA GeForce RTX 4090, CUDA 12.8
transformers: 5.3.0
peft: 0.18.1
bitsandbytes: 0.49.2
datasets: 4.7.0
accelerate: 1.13.0
```

## License

MIT
