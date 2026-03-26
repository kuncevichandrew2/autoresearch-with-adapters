# CLAUDE.md — autoresearch project notes

## Project overview

Autonomous ML research loop. Agent modifies `train.py` to minimize `val_loss` (cross-entropy, nats/token) within a fixed 5-minute training budget. **Only `train.py` is modified during experiments.** Everything else is infrastructure.

## Experiment workflow (Kaggle P100)

No local GPU. All training runs on Kaggle via the P100 adapter.

### Full cycle for one experiment

```bash
# 1. Modify train.py with your idea

# 2. Commit (pre-commit hook auto-updates notebook.py — but we now use git clone so no hook needed)
git add train.py && git commit -m "expNNN: description"

# 3. Push experiment branch to GitHub (Kaggle clones from here)
git push adapters autoresearch/mar26

# 4. Push notebook to Kaggle
KAGGLE_API_TOKEN=KGAT_229c3e774cda8080facedc39a8dd37fc kaggle kernels push -p adapters/kaggle/

# 5. Poll until complete (~10-15 min total: setup + 5 min train)
for i in $(seq 1 30); do
  STATUS=$(KAGGLE_API_TOKEN=KGAT_229c3e774cda8080facedc39a8dd37fc kaggle kernels status andrewk444/autoresearch-p100 2>&1)
  echo "[$(date +%H:%M:%S)] $STATUS"
  echo "$STATUS" | grep -qiE "complete|error|cancel" && break
  sleep 30
done

# 6. Pull logs
rm -rf /tmp/kout && KAGGLE_API_TOKEN=KGAT_229c3e774cda8080facedc39a8dd37fc kaggle kernels output andrewk444/autoresearch-p100 -p /tmp/kout

# 7. Parse result
python3 -c "
import json
events = json.loads(open('/tmp/kout/autoresearch-p100.log').read())
out = ''.join(e['data'] for e in events if e['stream_name']=='stdout')
idx = out.find('---')
print(out[idx:idx+300] if idx>=0 else 'NO SUMMARY\n'+out[-500:])
"
```

### Keep or discard

```bash
# If val_loss improved — keep commit as-is

# If val_loss worse — revert train.py to best commit
git checkout <best_commit> -- train.py
git add train.py && git commit -m "revert expNNN: <reason>"
```

### Log to results.tsv

```bash
echo -e "<commit7>\t<val_loss>\t<vram_gb>\t<keep|discard>\t<description>" >> results.tsv
```

## Kaggle setup

- **Kernel:** `andrewk444/autoresearch-p100` (P100 16GB, internet enabled)
- **Kernel metadata:** `adapters/kaggle/kernel-metadata.json`
- **Notebook:** `adapters/kaggle/notebook.py` — clones `autoresearch/mar26` branch from GitHub, runs `prepare.py` then `executor(open("train.py").read())`
- **API token:** `KGAT_229c3e774cda8080facedc39a8dd37fc` (also in `adapters/kaggle/.env`)
- **WandB project:** `autoresearch` (key in `adapters/kaggle/.env`)
- **Branch pushed to Kaggle:** always `autoresearch/mar26` (hardcoded in `notebook.py`)

## P100 constraints and learnings

- **CUDA capability:** SM 6.0 — no torch.compile, no Flash Attention 3, fp16+GradScaler, requires `torch==2.4.1+cu118`
- **VRAM:** 16GB, but small models (DEPTH=6) only use ~6GB
- **Throughput:** ~38K tok/sec, MFU ~28-30%
- **Key insight:** number of optimizer steps dominates over model size in 5-min budget
  - DEPTH=6 (16.9M params) → 355 steps → val_loss 5.13 ✓
  - DEPTH=8 (33.6M params) → 190 steps → val_loss 5.43 ✗
- **Best config found so far (exp007):**
  - `DEPTH = 6`, `DEVICE_BATCH_SIZE = 16`
  - `TOTAL_BATCH_SIZE = 2**15` (grad_accum=1, max steps)
  - `LEARNING_RATE = 1e-3`, `ADAM_BETAS = (0.9, 0.95)`
  - `WARMDOWN_RATIO = 0.2`
  - → **val_loss = 5.129**, 355 steps

## Log format

Kaggle output is JSON. Parse stdout for the `---` summary block:

```
---
val_loss:         5.129095
training_seconds: 300.2     # pure train loop time (steps 11+)
total_seconds:    374.1     # includes setup, prepare, eval
peak_vram_mb:     6324.1
mfu_percent:      28.03
total_tokens_M:   11.6
num_steps:        355
num_params_M:     16.9
depth:            6
```

`training_seconds` does NOT include pip install, git clone, prepare.py, or final eval.

## File rules

- **Only modify:** `train.py`
- **Never modify:** `prepare.py` (read-only, provides `make_dataloader`, `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `Tokenizer`)
- **Auto-updated:** `adapters/kaggle/notebook.py` — update manually when changing Kaggle setup (not per-experiment)
- **Don't commit:** `results.tsv`, `.env`

## Remote repos

- `origin` — original upstream
- `adapters` — `https://github.com/kuncevichandrew2/autoresearch-with-adapters` (main working remote)

Always push to `adapters`, not `origin`.

## Experiment progress

| Exp | val_loss | steps | status |
|-----|----------|-------|--------|
| 002 | 7.846 | 33 | baseline |
| 003 | 6.796 | 98 | keep |
| 004 | 5.781 | 184 | keep |
| 005 | 5.140 | 355 | keep |
| 006 | 5.774 | 201 | discard (larger batch → fewer steps) |
| 007 | **5.129** | 355 | keep ← current best |
| 008 | 5.430 | 190 | discard (larger model → fewer steps) |
