# AI Research Project — Organization Guide

A practical guide for structuring autonomous ML research projects: how to lay out the repository, track experiments, and eventually bridge research artifacts into production systems.

---

## 1. The Core Tension: Research vs. Production

Research and production have opposite priorities:

| | Research | Production |
|---|---|---|
| **Goal** | Discover what works | Reliably deliver what works |
| **Code** | Mutable, experimental | Stable, versioned |
| **Reproducibility** | Nice to have | Required |
| **Failure mode** | "Interesting failure" | Outage |
| **Iteration speed** | Fast | Careful |

A good project structure keeps these worlds separate while making it easy to graduate discoveries from one to the other.

---

## 2. Logical Structure of a Research Project

Every ML research project has the same logical layers regardless of size:

```
Problem definition
    └── Hypothesis space
            └── Experiments (iterate)
                    └── Findings / learnings
                            └── Best config / artifact
                                    └── Production candidate
```

**The agent's job** is to traverse this graph efficiently: form hypotheses, run experiments, record findings, update beliefs, repeat.

### What to track

| Layer | What | Where |
|---|---|---|
| Problem | Goal metric, constraints, budget | `program.md` |
| State | Current best, history, ideas queue | `reports/current_state.md` |
| Experiments | Individual run details | `reports/<session>/NNN_HHMMSS.md` |
| Code | One mutable file per experiment | `train.py` (git history) |
| Metrics | Compact log for analysis | `results.tsv` |
| Insights | Cross-experiment patterns | `CLAUDE.md` |

---

## 3. Folder Structure (This Project)

```
autoresearch/
│
├── train.py                    ← THE experiment file (agent modifies this)
├── prepare.py                  ← Data pipeline (read-only)
├── program.md                  ← Agent instructions: goal, rules, loop
├── CLAUDE.md                   ← Agent memory: learnings, workflow, setup
├── AI_RESEARCH_PROJECT.md      ← This file
│
├── reports/
│   ├── current_state.md        ← Living summary: best config, full history, ideas
│   └── YYYY-MM-DD-<model>/     ← One folder per agent session
│       ├── 001_HHMMSS.md       ← Experiment report: hypothesis, changes, metrics
│       ├── 002_HHMMSS.md
│       └── ...
│
├── results.tsv                 ← Tab-separated: commit, val_loss, vram, status, desc
├── analysis.ipynb              ← Visualization and trend analysis
│
├── adapters/
│   ├── kaggle/                 ← Free P100 GPU adapter
│   └── runpod/                 ← Paid RTX 4090/A100/H100 adapter
│
└── pyproject.toml              ← Dependencies
```

### Navigation guide

| Question | Where to look |
|---|---|
| What's the current best result? | `reports/current_state.md` → "Current best" |
| What has been tried? | `reports/current_state.md` → "Experiment history" |
| Why was X discarded? | `reports/<session>/NNN_HHMMSS.md` → "Notes" |
| What is the agent supposed to do? | `program.md` |
| How does training/eval work? | `prepare.py` (read-only) |
| What are the P100 constraints? | `CLAUDE.md` → "P100 constraints" |
| How do I run on Kaggle/RunPod? | `CLAUDE.md` → experiment workflow sections |
| Trend plots across experiments | `analysis.ipynb` |

---

## 4. Experiment Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│  Session start                                               │
│  1. Read program.md  →  2. Read reports/current_state.md    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Form hypothesis    │  (from ideas queue or new insight)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Edit train.py      │  (smallest change that tests the idea)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  git commit + push  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Run training       │  (5-min budget, on GPU adapter)
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    ┌─────────▼─────────┐             ┌─────────▼─────────┐
    │  val_loss improved│             │  val_loss worse   │
    │  → keep commit    │             │  → git reset      │
    └─────────┬─────────┘             └─────────┬─────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Write report       │  reports/<session>/NNN.md
                    │  Update state       │  reports/current_state.md
                    │  Log results.tsv    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Next hypothesis    │  ← repeat forever
                    └─────────────────────┘
```

### Experiment report template

```markdown
# Experiment NNN

**Date:** YYYY-MM-DD HH:MM:SS
**Commit:** abc1234
**Status:** keep | discard | crash

## Hypothesis
What you expected to happen and why.

## Changes
Concrete diff: what changed in train.py (hyperparams, architecture, etc).

## Metrics

| Metric           | Value    |
|------------------|----------|
| val_loss         | 5.129    |
| prev_best_loss   | 5.140    |
| delta_loss       | -0.011   |
| peak_vram_mb     | 6324.1   |
| training_seconds | 300.2    |
| mfu_percent      | 28.03    |
| total_tokens_M   | 11.6     |
| num_steps        | 355      |
| num_params_M     | 16.9     |

## Result
One-line verdict: kept/discarded/crashed and why.

## Notes
Observations, surprises, follow-up ideas.
```

---

## 5. `current_state.md` — The Agent's Working Memory

This file is the single most important artifact for multi-session research. It answers: "where are we right now?"

**Updated after every experiment.** Read at the start of every session.

### Structure

```markdown
# Research Current State

## Current best
| val_loss | commit | steps | config |
...

## Experiment history
| # | val_loss | Δ | steps | status | what changed |
...

## Current train.py config
(exact hyperparams block)

## Key learnings
(cross-experiment patterns, confirmed hypotheses)

## Next ideas to try
(ordered queue of hypotheses, most promising first)
```

**Why it works:**
- Agent doesn't need to re-read all individual reports on each session start
- The ideas queue prevents forgetting good hypotheses when context is cleared
- Key learnings encode hard-won insights (e.g. "step count dominates over model size")

---

## 6. ML Production Project Structure

When a research finding is ready to ship, the project structure changes significantly.

```
ml-service/
│
├── src/
│   ├── model/
│   │   ├── architecture.py     ← Frozen (came from research train.py)
│   │   ├── config.py           ← Typed dataclass, validated
│   │   └── checkpoint.py       ← Load/save, versioning
│   │
│   ├── data/
│   │   ├── pipeline.py         ← Deterministic, tested
│   │   ├── tokenizer.py        ← Pinned version
│   │   └── loaders.py
│   │
│   ├── training/
│   │   ├── trainer.py          ← Orchestration (multi-GPU, checkpointing)
│   │   ├── optimizer.py        ← Exact config from best research run
│   │   └── scheduler.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py          ← val_loss + downstream task metrics
│   │   └── benchmark.py        ← Regression tests against known baselines
│   │
│   └── serving/
│       ├── inference.py        ← Batching, quantization, caching
│       └── api.py              ← REST/gRPC interface
│
├── configs/
│   ├── base.yaml               ← Canonical best config
│   ├── small.yaml              ← Smaller variant for cheaper deployment
│   └── large.yaml
│
├── scripts/
│   ├── train.py                ← Entry point (CLI args, config loading)
│   ├── evaluate.py
│   └── export.py               ← ONNX / TorchScript export
│
├── tests/
│   ├── unit/                   ← Model forward pass, loss shapes, etc.
│   ├── integration/            ← Full train loop (smoke test, 10 steps)
│   └── regression/             ← val_loss must stay below known threshold
│
├── notebooks/
│   └── analysis.ipynb          ← Visualization (read-only in CI)
│
├── Dockerfile
├── pyproject.toml              ← Pinned deps, no ranges
└── .github/
    └── workflows/
        ├── test.yml
        └── train.yml           ← Triggered on config change
```

### Key differences from research structure

| Aspect | Research | Production |
|---|---|---|
| Config | Inline constants in train.py | Typed YAML + dataclass |
| Dependencies | Ranges (`>=0.1`) | Pinned (`==2.4.1`) |
| Testing | None | Unit + integration + regression |
| Checkpointing | Optional | Required, versioned |
| Reproducibility | Git hash | Git hash + seed + pinned deps + Dockerfile |
| Serving | N/A | Batching, quantization, API |
| Monitoring | WandB (optional) | WandB + alerting + drift detection |

---

## 7. Bridging Research → Production

The transition is not a rewrite — it's a series of graduated steps.

### Stage 1: Research (this project)
- One file (`train.py`), constants at top, no abstractions
- Goal: find what works
- Artifact: best commit hash + val_loss

### Stage 2: Consolidation
- Extract `architecture.py` from train.py (no logic change, just move)
- Write a typed `Config` dataclass replacing inline constants
- Add smoke test: 10 steps must complete without crash
- Pin all dependencies

### Stage 3: Hardening
- Add checkpointing and resume logic
- Multi-GPU support (if needed)
- Integration tests: full eval must match research val_loss ±0.001
- Regression baseline: commit the known-good val_loss as a test threshold

### Stage 4: Production
- Add serving layer (inference.py, API)
- Dockerfile + CI pipeline
- Monitoring: alert if live eval degrades beyond threshold

### What to preserve verbatim from research

These should be copied, not redesigned:
- Model architecture (exact layer order, norms, activations)
- Optimizer config (LR, betas, weight decay, scheduler)
- Data pipeline logic (`prepare.py` → `data/pipeline.py`)
- Batch sizes and sequence lengths that hit the val_loss target

### What to redesign for production

- Config management (inline constants → validated YAML)
- Logging (print statements → structured logging)
- Error handling (crash-on-error → retry + checkpoint)
- Dependency management (loose → pinned)

---

## 8. Experiment Tracking at Scale

As the number of experiments grows, the flat `current_state.md` becomes insufficient. At ~50+ experiments, consider:

### Option A: Structured TSV + notebook
Keep `results.tsv` as the source of truth. Use `analysis.ipynb` for:
- Loss curves across experiments
- Correlation: which hyperparams matter most
- Pareto front: val_loss vs. VRAM vs. steps

### Option B: WandB sweeps
Replace the manual loop with `wandb sweep`:
- Define search space in `sweep.yaml`
- WandB agent calls `train.py` with different configs
- WandB UI shows parallel coordinates, importance plots

### Option C: Optuna / Ray Tune
For large hyperparameter spaces:
- Define the objective function (val_loss after N steps)
- Let the optimizer suggest configs (Bayesian, TPE, etc.)
- Requires wrapping `train.py` to accept CLI args

**For this project:** Option A is sufficient. Move to B/C when the ideas queue is exhausted and you want systematic search.

---

## 9. Quick Reference

### Start a new session
```bash
# 1. Read program.md and current_state.md
# 2. Form a hypothesis from the ideas queue
# 3. Edit train.py, commit, push, run adapter
# 4. Parse val_loss from output
# 5. Keep or reset
# 6. Write report + update current_state.md
```

### Add a new adapter (e.g., Lambda Cloud, Vast.ai)
Mirror `adapters/runpod/`: `run.py` (on-pod) + `adapter.py` (local lifecycle).

### Graduate a finding to production
1. Identify the best commit hash
2. Extract model config to `configs/best.yaml`
3. Copy architecture verbatim to `src/model/architecture.py`
4. Write regression test: `assert val_loss < 5.13`
5. Pin dependencies, add Dockerfile
