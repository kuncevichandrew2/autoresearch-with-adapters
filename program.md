# autoresearch

Autonomous ML research: iterate on `train.py` to minimize `val_loss` (eval cross-entropy, nats/token).

## Setup

1. Agree on a run tag (e.g. `mar5`), create branch `autoresearch/<tag>`.
2. Read: `prepare.py` (read-only), `train.py` (you modify this), this file.
3. Read `reports/current_state.md` — current best, full experiment history, key learnings, ideas queue.
4. Verify `~/.cache/autoresearch/` has data. If not: `uv run prepare.py`.
5. Create `results.tsv` with header row. Run baseline first: `uv run train.py`.

## Rules

- Only modify `train.py`. `prepare.py` is read-only. No new dependencies.
- Goal: **lowest `val_loss`** (cross-entropy, nats/token). Time budget is fixed at 5 min per experiment.
- Simpler is better at equal performance. Removing code for same result = win.
- VRAM is a soft constraint — some increase is OK for meaningful gains.

## Experiment Loop

Run forever until manually stopped. Never pause to ask the human.

1. Edit `train.py` with an idea, `git commit`.
2. `uv run train.py > run.log 2>&1`
3. `grep "^val_loss:\|^peak_vram_mb:" run.log`
4. If empty: crashed. `tail -n 50 run.log` to debug. Fix if trivial, else skip.
5. Log to `results.tsv` (do not commit this file).
6. If `val_loss` improved → keep commit. If worse → `git reset` to previous best.

Kill runs exceeding 10 minutes. Treat as failure.

7. Save a mini-report and update the state file (see below).

## Experiment Reports

After each experiment:

**a) Save a mini-report** to `reports/YYYY-MM-DD-<provider-model>/NNN_HHMMSS.md`
- `YYYY-MM-DD` = session date
- `<provider-model>` = e.g. `anthropic-sonnet-4-6` (model running the agent)
- `NNN` = experiment number zero-padded to 3 digits
- `HHMMSS` = time of experiment completion

Example: `reports/2026-03-26-anthropic-sonnet-4-6/009_143200.md`

**b) Update `reports/current_state.md`** — add a row to the experiment history table and update "current best" and "current config" if improved.

Report template:

```markdown
# Experiment 033

**Date:** 2026-03-26 14:15:00
**Commit:** a1b2c3d
**Status:** keep | discard | crash

## Hypothesis

What you expected to happen and why you tried this change.

## Changes

Concrete diff summary: what was modified in train.py (hyperparams, architecture, etc).

## Metrics

| Metric           | Value    |
|------------------|----------|
| val_loss         | 2.1234   |
| prev_best_loss   | 2.1891   |
| delta_loss       | -0.0657  |
| peak_vram_mb     | 45060.2  |
| training_seconds | 300.1    |
| mfu_percent      | 39.80    |
| total_tokens_M   | 499.6    |
| num_steps        | 953      |
| num_params_M     | 50.3     |

## Result

One-line verdict: kept/discarded/crashed and why.

## Notes

Any observations, surprises, or ideas for follow-up experiments.
```

## results.tsv

Tab-separated (not commas). Header + 5 columns:

```
commit	val_loss	memory_gb	status	description
a1b2c3d	2.189100	44.0	keep	baseline
b2c3d4e	0.000000	0.0	crash	double model width (OOM)
```

- `commit`: short hash (7 chars)
- `val_loss`: cross-entropy nats/token (0.000000 for crashes)
- `memory_gb`: peak VRAM in GB (0.0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: what this experiment tried

## Output Format

The script prints a `---` summary block at the end:

```
---
val_loss:         2.189100
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract metric: `grep "^val_loss:" run.log`
