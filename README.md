# zagora-discovery-lab

Can cheap autonomous LoRA search on a proxy model compress the distributed search space for its larger sibling?

## What This Is

A standalone experiment testing whether hyperparameters discovered on **Llama-3.1-8B** (1 GPU, 5-minute runs) transfer to **Llama-3.1-70B** (4 GPUs, distributed LoRA via [Petals](https://github.com/bigscience-workshop/petals) pipeline parallelism).

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): 3 files, fixed budget, autonomous search.

## Methodology

1. **Discovery** — Autonomous hyperparameter search on 8B proxy (single 4090, ~100 experiments overnight)
2. **Local Confirmation** — Top 5 candidates rerun 3x with more data/time to filter noise
3. **Distributed Validation** — Confirmed candidates tested on 70B target via distributed pipeline parallelism (4x4090)
4. **Reproducibility** — Paired-seed design, adapter verification, parity certification

## Files

| File | Purpose |
|---|---|
| `train.py` | Single training run: load 4-bit model, apply LoRA, train, evaluate val_loss |
| `prepare.py` | Download alpaca_cleaned, tokenize, deterministic train/val split |
| `program.md` | Search space and instructions for autonomous agent |
| `results.tsv` | Accumulated experiment results (created by train.py) |

## Quick Start

```bash
pip install -r requirements.txt

# Prepare data (downloads + tokenizes + caches)
python prepare.py

# Single training run with defaults
python train.py

# Training run with custom hyperparams
python train.py --lr 3e-4 --rank 16 --alpha 32 --dropout 0.05
```

## Environment

Record before running experiments:

```
GPU: (e.g., RTX 4090 24GB)
CUDA: (e.g., 12.1)
Python: (e.g., 3.10)
torch: (e.g., 2.2.0)
transformers: (e.g., 4.44.0)
peft: (e.g., 0.12.0)
bitsandbytes: (e.g., 0.43.0)
```

## License

MIT
