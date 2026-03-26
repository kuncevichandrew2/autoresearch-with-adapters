# autoresearch

Autonomous ML research system. An AI agent iterates on `train.py` to minimize `val_loss` (cross-entropy, nats/token) within a fixed 5-minute training budget. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

**This fork adds:** WandB experiment tracking · Kaggle P100 adapter · RunPod adapter · per-experiment markdown reports · simplified vanilla GPT + AdamW baseline

---

## How it works

```mermaid
flowchart LR
    A[Read train.py] --> B[Make a change]
    B --> C[Git commit + push]
    C --> D[Train 5 min on GPU]
    D --> E{val_loss improved?}
    E -->|Yes| F[keep]
    E -->|No| G[git reset]
    F --> H[Save report]
    G --> H
    H --> A
```

One mutable file — `train.py`. Everything else is fixed. The agent runs forever until you stop it.

---

## Quick start

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU — local, Kaggle (free), or RunPod (paid)
- WandB account (optional, for experiment tracking)

### 1. Clone and configure

```bash
git clone https://github.com/kuncevichandrew2/autoresearch-with-adapters
cd autoresearch-with-adapters

cp .env.example .env
# Edit .env and fill in your keys
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Prepare data (one-time, ~5 min)

```bash
uv run prepare.py
```

Data is cached in `~/.cache/autoresearch/`. Re-running is safe.

### 4. Run a single experiment

```bash
uv run train.py
```

Trains for 5 minutes, prints a summary block:

```
---
val_loss:         5.129095
training_seconds: 300.2
total_seconds:    343.0
peak_vram_mb:     6324.1
mfu_percent:      28.03
total_tokens_M:   11.6
num_steps:        355
num_params_M:     16.9
depth:            6
```

### 5. Run the agent (autonomous loop)

```bash
claude  # then: "read program.md and start experimenting"
```

The agent modifies `train.py`, runs training, checks `val_loss`, keeps or reverts, saves a report, and repeats.

---

## Project structure

```
autoresearch/
├── train.py              # Model + optimizer + training loop  ← AGENT MODIFIES
├── prepare.py            # Data, tokenizer, evaluation        ← READ-ONLY
├── program.md            # Agent instructions                 ← HUMAN EDITS
├── pyproject.toml        # Python dependencies
├── .env.example          # Environment variable template
├── CLAUDE.md             # Agent context (workflow, learnings)
├── reports/              # Per-experiment markdown reports
├── analysis.ipynb        # Result visualization
└── adapters/
    ├── kaggle/
    │   ├── notebook.py          # Kaggle script (clones repo, runs training)
    │   ├── adapter.py           # Local helper (setup, prepare, train)
    │   ├── kernel-metadata.json # Kaggle push config
    │   └── DESCRIPTION.md
    └── runpod/
        ├── run.py               # Pod script (clones repo, runs training)
        ├── adapter.py           # Local: create pod, poll, fetch log, terminate
        ├── pod-config.json      # GPU type, image, resource settings
        └── DESCRIPTION.md
```

---

## Model architecture

Vanilla GPT transformer:

- Token embedding → RMS Norm
- N × [CausalSelfAttention (RoPE) + RMS Norm + MLP (GELU) + residuals]
- Final RMS Norm → LM Head

**GPU compatibility:** Flash Attention 3 on SM80+ (H100/A100), PyTorch SDPA fallback on older GPUs (P100/T4/V100). `fp16` + GradScaler on SM < 80, `bfloat16` otherwise. `torch.compile` on SM70+.

---

## WandB integration

Set `WANDB_API_KEY` in `.env`. Per-step metrics logged every 10 steps; final summary logged at end of run.

To disable: leave `WANDB_API_KEY` unset — logging is silently skipped.

---

## Metric: val_loss

Cross-entropy loss in nats/token, evaluated on 100 validation batches after training. Lower is better. Vocab-size-independent.

---

## GPU adapters

Two adapters are available for running training remotely. Both follow the same experiment cycle: commit `train.py` → push to GitHub → run adapter → get results.

---

### Adapter 1: Kaggle (free P100/T4)

**When to use:** Free GPU quota, P100 16GB.

#### Environment variables

| Variable | Description |
|----------|-------------|
| `KAGGLE_API_TOKEN` | From kaggle.com → Settings → API → Create New Token |
| `WANDB_API_KEY` | Optional. From wandb.ai/authorize |
| `WANDB_PROJECT` | Optional. WandB project name (default: `autoresearch`) |

#### One-time setup

1. Fill in `.env` with `KAGGLE_API_TOKEN` and optionally `WANDB_API_KEY`
2. In `adapters/kaggle/kernel-metadata.json`, set `"id"` to `"<your-username>/autoresearch-p100"`
3. Add `WANDB_API_KEY` as a Kaggle Secret: Notebook → Add-ons → Secrets

#### Experiment cycle

```bash
# 1. Modify train.py, commit, push
git add train.py && git commit -m "expNNN: description"
git push origin autoresearch/mar26

# 2. Push notebook to Kaggle and run
source .env
kaggle kernels push -p adapters/kaggle/

# 3. Poll for completion (~10-15 min total)
for i in $(seq 1 30); do
  STATUS=$(kaggle kernels status <your-username>/autoresearch-p100 2>&1)
  echo "[$(date +%H:%M:%S)] $STATUS"
  echo "$STATUS" | grep -qiE "complete|error|cancel" && break
  sleep 30
done

# 4. Pull and parse results
kaggle kernels output <your-username>/autoresearch-p100 -p /tmp/kout
python3 -c "
import json
events = json.loads(open('/tmp/kout/autoresearch-p100.log').read())
out = ''.join(e['data'] for e in events if e['stream_name']=='stdout')
idx = out.find('---')
print(out[idx:idx+300])
"
```

#### P100-specific train.py settings

```python
DEPTH = 6                  # smaller model = more optimizer steps in 5 min
DEVICE_BATCH_SIZE = 16     # fits in 16GB VRAM
TOTAL_BATCH_SIZE = 2**15   # grad_accum=1, ~355 steps per run
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.95)
WARMDOWN_RATIO = 0.2
```

---

### Adapter 2: RunPod (paid RTX 4090 / A100 / H100)

**When to use:** Faster GPU, Kaggle quota exhausted, or need more VRAM.

#### Environment variables

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | From runpod.io/console/user/settings → API Keys |
| `WANDB_API_KEY` | Optional. From wandb.ai/authorize |
| `WANDB_PROJECT` | Optional. WandB project name (default: `autoresearch`) |

#### One-time setup

1. Fill in `.env` with `RUNPOD_API_KEY`
2. `pip install runpod`

#### Experiment cycle

```bash
# 1. Modify train.py, commit, push
git add train.py && git commit -m "expNNN: description"
git push origin autoresearch/mar26

# 2. Run adapter — creates pod, trains, prints results, terminates pod automatically
source .env
python adapters/runpod/adapter.py

# Optional: override GPU type
python adapters/runpod/adapter.py --gpu "NVIDIA A100 80GB PCIe"
```

The adapter creates a fresh pod, curls `run.py` from GitHub, trains for 5 minutes, SSH-fetches the log, prints the `---` summary, and terminates the pod. Total time ~15 min.

#### GPU options (pod-config.json)

| `gpu_type_id` | VRAM | Notes |
|---------------|------|-------|
| `NVIDIA GeForce RTX 4090` | 24GB | Default, cheapest |
| `NVIDIA A100 80GB PCIe` | 80GB | Fastest for large models |
| `NVIDIA H100 80GB HBM3` | 80GB | Latest gen |

---

## License

MIT
