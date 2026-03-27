# Research Current State

> Agent: read this at the start of every session. Update after every experiment.

---

## Current best

| val_loss | commit  | steps | config                            |
|----------|---------|-------|-----------------------------------|
| **5.129**| f1413e6 | 355   | DEPTH=6, LR=1e-3, BETAS=(0.9,0.95) |

Hardware: Kaggle P100 16GB · fp16 · 5-min budget · ~38K tok/sec

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

---

## Current train.py config

```python
DEPTH             = 6
ASPECT_RATIO      = 64
HEAD_DIM          = 128
TOTAL_BATCH_SIZE  = 2**15   # 32768 — grad_accum=1, ~355 steps/run
DEVICE_BATCH_SIZE = 16
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 0.1
ADAM_BETAS        = (0.9, 0.95)
WARMUP_RATIO      = 0.05
WARMDOWN_RATIO    = 0.2
```

16.9M params · ~355 steps · ~6.2GB VRAM · MFU ~28%

---

## Key learnings

- **Step count dominates** on P100 in 5-min budget — maximize steps, minimize model size
- TOTAL_BATCH_SIZE=2^15 with grad_accum=1 is optimal for P100 throughput
- Larger model (DEPTH=8) → 190 steps → worse
- Larger batch (DEVICE_BATCH=32) → 201 steps → worse
- BETAS=(0.9,0.95) marginally better than (0.9,0.999)

---

## Next ideas to try

- DEPTH=4 — even smaller/faster, more steps (estimated ~500+ steps)
- WARMDOWN_RATIO: try 0.1 or 0.3
- HEAD_DIM: try 64 (smaller attention heads, potentially faster)
- LEARNING_RATE: try 3e-3 or 5e-4
- muP (maximal update parametrization) for better LR transfer
