"""
RunPod adapter for autoresearch.
Creates a GPU pod, SSH-launches training, streams logs, terminates pod.

Usage:
    source .env
    python adapters/runpod/adapter.py [--gpu "NVIDIA A100 80GB PCIe"]

Requires:
    pip install runpod

Design:
    - Pod starts WITHOUT docker_args so the image's /start.sh runs normally
      (RunPod images use supervisord to start sshd + jupyter etc.)
    - Once SSH is confirmed connectable, we launch training via SSH + nohup
    - Poll /workspace/output.log via SSH until '---' summary block appears
    - Fetch full log, print metrics, terminate pod
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pod-config.json")

REPO_URL    = "https://github.com/kuncevichandrew2/autoresearch-with-adapters"
REPO_BRANCH = "autoresearch/mar26"
RUN_PY_RAW  = (
    f"https://raw.githubusercontent.com/kuncevichandrew2/"
    f"autoresearch-with-adapters/{REPO_BRANCH}/adapters/runpod/run.py"
)

POLL_INTERVAL   = 15    # seconds between polls
SSH_TIMEOUT     = 10    # seconds per SSH connection attempt
SSH_RETRY_LIMIT = 40    # max attempts to establish first SSH (~10 min)
MAX_WAIT        = 1800  # 30 min hard timeout for training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env():
    env_path = os.path.join(REPO_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def ssh_cmd(host: str, port: int, cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "LogLevel=ERROR",
            "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-p", str(port), f"root@{host}",
            cmd,
        ],
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------

def create_pod(config: dict, gpu_type: str | None = None) -> str:
    """Create a pod WITHOUT docker_args so /start.sh runs and sshd starts."""
    import runpod

    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    env_vars = {
        "REPO_URL":      REPO_URL,
        "REPO_BRANCH":   REPO_BRANCH,
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "autoresearch"),
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    pod = runpod.create_pod(
        name="autoresearch",
        image_name=config["image_name"],
        gpu_type_id=gpu_type or config["gpu_type_id"],
        cloud_type=config.get("cloud_type", "ALL"),
        gpu_count=1,
        volume_in_gb=config.get("volume_in_gb", 10),
        container_disk_in_gb=config.get("container_disk_in_gb", 20),
        ports=config.get("ports", "22/tcp"),
        env=env_vars,
    )
    pod_id = pod["id"]
    print(f"[runpod] Created pod {pod_id}", flush=True)
    return pod_id


def wait_for_api_ready(pod_id: str) -> tuple[str, int]:
    """Poll API until runtime.ports has SSH. Returns (host, port)."""
    import runpod
    print("[runpod] Waiting for pod API to report SSH port...", flush=True)
    elapsed = 0
    while elapsed < MAX_WAIT:
        pod = runpod.get_pod(pod_id)
        if pod is None:
            raise RuntimeError(f"Pod {pod_id} disappeared from API")
        status = pod.get("desiredStatus") or "UNKNOWN"
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        print(f"  [{elapsed:4d}s] status={status}  ports={len(ports)}", flush=True)
        for p in ports:
            if p.get("privatePort") == 22:
                return p["ip"], int(p["publicPort"])
        if status in ("FAILED", "DEAD", "TERMINATED"):
            raise RuntimeError(f"Pod entered terminal state: {status}")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(f"Pod SSH port never appeared in API after {MAX_WAIT}s")


def wait_for_ssh(host: str, port: int) -> None:
    """Actually verify SSH connectivity — try until it works."""
    print(f"[runpod] Waiting for SSH at {host}:{port}...", flush=True)
    for attempt in range(SSH_RETRY_LIMIT):
        result = ssh_cmd(host, port, "echo ok")
        if result.returncode == 0 and "ok" in result.stdout:
            print(f"  SSH ready after {attempt * POLL_INTERVAL}s", flush=True)
            return
        elapsed = attempt * POLL_INTERVAL
        print(f"  [{elapsed:4d}s] SSH not yet ready (rc={result.returncode})", flush=True)
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"SSH never became connectable at {host}:{port}")


def start_training(host: str, port: int) -> None:
    """Launch training in background via SSH (nohup + disown)."""
    print("[runpod] Starting training via SSH...", flush=True)
    # nohup + disown ensures the process survives SSH session end
    launch_cmd = (
        f"mkdir -p /workspace && "
        f"nohup bash -c 'curl -fsSL {RUN_PY_RAW} -o /tmp/run.py && "
        f"python /tmp/run.py' > /workspace/output.log 2>&1 & disown"
    )
    result = ssh_cmd(host, port, launch_cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start training.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print("[runpod] Training launched in background.", flush=True)


def wait_for_training(host: str, port: int) -> None:
    """Poll via SSH until '---' summary block appears in /workspace/output.log."""
    print("[runpod] Waiting for training to complete (polling log)...", flush=True)
    elapsed = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        result = ssh_cmd(host, port,
            "grep -c '^---' /workspace/output.log 2>/dev/null || echo 0")
        count = result.stdout.strip()
        # Also show last training line for visibility
        tail = ssh_cmd(host, port,
            "grep 'step ' /workspace/output.log 2>/dev/null | tail -1 || echo ''")
        step_line = tail.stdout.strip()[-80:] if tail.stdout.strip() else "..."
        print(f"  [{elapsed:4d}s] '---' blocks: {count or '(ssh err)'}  |  {step_line}", flush=True)
        if count.isdigit() and int(count) >= 1:
            return
        if result.returncode != 0 and elapsed > 120:
            print(f"    SSH error: rc={result.returncode} stderr={result.stderr[:100]}", flush=True)
    raise TimeoutError(f"Training did not finish within {MAX_WAIT}s")


def fetch_log(host: str, port: int) -> str:
    result = ssh_cmd(host, port, "cat /workspace/output.log")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch log: {result.stderr}")
    return result.stdout


def parse_summary(log: str) -> dict:
    idx = log.find("---")
    if idx < 0:
        return {}
    block = log[idx + 3:].strip()
    metrics = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("---"):
            break
        if ":" in line:
            key, val = line.split(":", 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                metrics[key.strip()] = val.strip()
    return metrics


def terminate_pod(pod_id: str) -> None:
    import runpod
    runpod.terminate_pod(pod_id)
    print(f"[runpod] Pod {pod_id} terminated.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run autoresearch on RunPod")
    parser.add_argument("--gpu", default=None,
        help="GPU type override (e.g. 'NVIDIA A100 80GB PCIe')")
    args = parser.parse_args()

    load_env()

    if "RUNPOD_API_KEY" not in os.environ:
        print("ERROR: RUNPOD_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    pod_id = None
    try:
        pod_id = create_pod(config, gpu_type=args.gpu)
        host, port = wait_for_api_ready(pod_id)
        print(f"[runpod] Pod API reports SSH at {host}:{port}", flush=True)
        wait_for_ssh(host, port)
        start_training(host, port)
        wait_for_training(host, port)

        log = fetch_log(host, port)
        metrics = parse_summary(log)

        print("\n" + "=" * 40)
        if metrics:
            for k, v in metrics.items():
                print(f"{k}: {v}")
        else:
            print("WARNING: no summary block found.")
            print("--- last 500 chars ---")
            print(log[-500:])
        print("=" * 40)

    finally:
        if pod_id:
            try:
                terminate_pod(pod_id)
            except Exception as e:
                print(f"WARNING: failed to terminate {pod_id}: {e}", file=sys.stderr)
                print("  Manually terminate at https://www.runpod.io/console/pods",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
