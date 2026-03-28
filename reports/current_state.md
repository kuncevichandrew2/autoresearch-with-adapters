# Research Current State

> Agent: read this at the start of every session. Update after every experiment.

---

## Current best

| val_loss | commit  | steps | config                            |
|----------|---------|-------|-----------------------------------|
| **3.623**| 2d430fd | 7927  | DEPTH=4, LR=1e-3, BETAS=(0.9,0.95) |

Hardware: RunPod RTX 3090 24GB · fp16 · torch.compile · 5-min budget · ~87K tok/sec · MFU ~49%

---

## Experiment history

| # | val_loss | Δ      | steps | status  | what changed                        |
|---|----------|--------|-------|---------|-------------------------------------|
| 001 | —      | —      | 33    | keep    | baseline (old metric val_bpb)       |
| 002 | 7.846  | —      | 33    | keep    | baseline — new val_loss metric      |
| 003 | 6.796  | -1.050 | 98    | keep    | TOTAL_BATCH 2^20→2^18, LR 3e-4→1e-3|
| 004 | 5.781  | -1.015 | 184   | keep    | TOTAL_BATCH 2^18→2^17               |
| 005 | 5.140  | -0.641 | 355   | keep    | TOTAL_BATCH 2^17→2^15 (grad_acc=1)  |
| 006 | 5.774  | +0.634 | 201   | discard | DEVICE_BATCH 16→32 (fewer steps)    |
| 007 | 5.129  | -0.011 | 355   | keep    | ADAM_BETAS (0.9,0.999)→(0.9,0.95)  |
| 008 | 5.430  | +0.301 | 190   | discard | DEPTH 6→8 (fewer steps)             |
| 009 | 3.623  | -1.506 | 7927  | keep    | DEPTH 6→4 + **RTX 3090** (new HW)  |

> **Note:** Exp 009 marks hardware switch to RunPod RTX 3090. Direct comparison with P100 experiments is not valid. 7927 steps vs 355 on P100 explains most of the gain.

---

## Current train.py config

```python
DEPTH             = 4
ASPECT_RATIO      = 64
HEAD_DIM          = 128
TOTAL_BATCH_SIZE  = 2**15   # 32768 — grad_accum=1, ~7927 steps/run on RTX 3090
DEVICE_BATCH_SIZE = 16
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 0.1
ADAM_BETAS        = (0.9, 0.95)
WARMUP_RATIO      = 0.05
WARMDOWN_RATIO    = 0.2
```

7.3M params · ~7927 steps · ~4GB VRAM · MFU ~49%

---

## Key learnings

### Hardware: RunPod RTX 3090 (new baseline as of exp009)
- **torch.compile enabled** (SM 8.6 ≥ 7.0) — significant speedup vs P100
- **MFU ~49%** vs P100's 28% — ~2.3x faster throughput
- ~87K tok/sec; DEPTH=4 gives 7927 steps in 5-min budget
- VRAM 24GB available; current config uses only ~4GB → lots of headroom
- Pod kept alive with `--no-terminate` between experiments (saves ~3 min setup)

### Hardware: Kaggle P100 (deprecated baseline)
- SM 6.0 — no torch.compile, ~38K tok/sec, MFU ~28%
- Step count dominated: DEPTH=6 → 355 steps → 5.129

### General learnings
- **Step count still dominates** — maximize steps within budget
- BETAS=(0.9,0.95) marginally better than (0.9,0.999)
- With RTX 3090 + DEPTH=4: we now have capacity for larger models without losing many steps

---

## Next ideas to try

RTX 3090 with ~87K tok/sec changes the tradeoff. Re-evaluate model size:

- **DEPTH=6 on RTX 3090** — 16.9M params, estimate ~3000-4000 steps — worth trying?
- **DEPTH=3** — even smaller, more steps, risk of underfitting
- **TOTAL_BATCH_SIZE=2^16** — double batch, ~half steps, better gradient estimates?
- **LEARNING_RATE tuning** — with 7927 steps, current LR schedule might be suboptimal
- **WARMDOWN_RATIO** — try 0.3 or 0.4 for longer cosine decay
- Running pod: xi7o6y4svrf0lt — reuse with `--pod-id xi7o6y4svrf0lt`
