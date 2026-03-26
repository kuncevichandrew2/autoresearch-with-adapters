# RunPod Adapter

Runs autoresearch on RunPod cloud GPUs (RTX 4090, A100, H100, etc.).

## Prerequisites

1. [RunPod account](https://www.runpod.io/) with API key
2. `pip install runpod`
3. `RUNPOD_API_KEY` in root `.env`

## Experiment cycle

```bash
# 1. Modify train.py, commit, push
git add train.py && git commit -m "expNNN: description"
git push adapters autoresearch/mar26

# 2. Run adapter — creates pod, trains, streams result, terminates
source .env
python adapters/runpod/adapter.py

# 3. Optionally override GPU
python adapters/runpod/adapter.py --gpu "NVIDIA A100 80GB PCIe"
```

The adapter:
1. Creates a pod with the GPU from `pod-config.json`
2. Pod curls `run.py` from GitHub and executes it
3. `run.py` clones the repo, runs `prepare.py`, runs `train.py`
4. All output is tee'd to `/workspace/output.log`
5. Adapter polls via SSH until the `---` summary block appears
6. Prints metrics, terminates pod

## Configuration

Edit `pod-config.json` to change GPU type or Docker image.

Available GPU type IDs (examples):
- `NVIDIA GeForce RTX 4090` — 24GB, cheapest option
- `NVIDIA A100 80GB PCIe` — 80GB, fastest for large models
- `NVIDIA H100 80GB HBM3` — latest gen

## Notes

- Pod is always terminated after training (even on error)
- If termination fails, a warning prints with a link to the RunPod console
- SSH port 22 must be open (set in `pod-config.json` as `"ports": "22/tcp"`)
- `run.py` does NOT reinstall PyTorch — RunPod images have it pre-installed
- For RTX 4090 (SM 8.9): bfloat16, torch.compile, Flash Attention 3 all work
