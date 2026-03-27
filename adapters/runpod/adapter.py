"""
RunPod adapter for autoresearch.
Creates a GPU pod, runs train.py, streams logs, terminates pod.

Usage:
    source .env
    python adapters/runpod/adapter.py [--gpu "NVIDIA GeForce RTX 4090"]

Requires:
    pip install runpod
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

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pod-config.json")

# Raw URL to run.py on GitHub — adapter.py curls this and runs it on the pod
REPO_URL    = "https://github.com/kuncevichandrew2/autoresearch-with-adapters"
REPO_BRANCH = "autoresearch/mar26"
RUN_PY_RAW  = (
    f"https://raw.githubusercontent.com/kuncevichandrew2/"
    f"autoresearch-with-adapters/{REPO_BRANCH}/adapters/runpod/run.py"
)

POLL_INTERVAL = 15   # seconds between status polls
SSH_TIMEOUT   = 10   # seconds for SSH connection timeout
MAX_WAIT      = 1800 # 30 min max before giving up


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------

def load_env():
    """Load .env from repo root into os.environ."""
    env_path = os.path.join(REPO_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------

def create_pod(config: dict, gpu_type: str | None = None) -> str:
    """Create a RunPod pod and return its ID."""
    import runpod

    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    env_vars = {
        "REPO_URL":      REPO_URL,
        "REPO_BRANCH":   REPO_BRANCH,
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "autoresearch"),
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # docker_args: the command the container runs at startup.
    # NOTE: RunPod SDK embeds docker_args inside a double-quoted GraphQL string,
    # so docker_args must NOT contain double-quote characters.
    docker_cmd = (
        f"bash -c 'curl -fsSL {RUN_PY_RAW} -o /tmp/run.py && "
        f"python /tmp/run.py 2>&1; tail -f /dev/null'"
    )

    pod = runpod.create_pod(
        name="autoresearch",
        image_name=config["image_name"],
        gpu_type_id=gpu_type or config["gpu_type_id"],
        cloud_type=config.get("cloud_type", "SECURE"),
        docker_args=docker_cmd,
        gpu_count=1,
        volume_in_gb=config.get("volume_in_gb", 10),
        container_disk_in_gb=config.get("container_disk_in_gb", 20),
        ports=config.get("ports", "22/tcp"),
        env=env_vars,
    )
    pod_id = pod["id"]
    print(f"[runpod] Created pod {pod_id}", flush=True)
    return pod_id


def wait_for_running(pod_id: str) -> dict:
    """Poll until pod is RUNNING and has an IP. Returns pod info."""
    import runpod
    print("[runpod] Waiting for pod to start...", flush=True)
    elapsed = 0
    while elapsed < MAX_WAIT:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus") or pod.get("status") or "UNKNOWN"
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        print(f"  [{elapsed:4d}s] status={status}  ports={ports}", flush=True)
        if status == "RUNNING" and ports:
            return pod
        if status in ("FAILED", "DEAD", "TERMINATED"):
            raise RuntimeError(f"Pod entered terminal state: {status}")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(f"Pod did not start within {MAX_WAIT}s")


def get_ssh_info(pod: dict) -> tuple[str, int]:
    """Extract (host, port) for SSH from pod info."""
    ports = pod.get("runtime", {}).get("ports", [])
    for p in ports:
        if p.get("privatePort") == 22:
            return p["ip"], int(p["publicPort"])
    raise RuntimeError(f"No SSH port found in pod info: {pod}")


def wait_for_training(host: str, port: int) -> None:
    """Poll via SSH until the training summary block appears in the log."""
    print("[runpod] Waiting for training to complete (polling log)...", flush=True)
    elapsed = 0
    ssh_base = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={SSH_TIMEOUT}",
        "-p", str(port), f"root@{host}",
    ]
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        result = subprocess.run(
            ssh_base + ["grep -c '^---' /workspace/output.log 2>/dev/null || echo 0"],
            capture_output=True, text=True,
        )
        count = result.stdout.strip()
        print(f"  [{elapsed:4d}s] '---' blocks in log: {count}", flush=True)
        if count.isdigit() and int(count) >= 1:
            return
    raise TimeoutError(f"Training did not finish within {MAX_WAIT}s")


def fetch_log(host: str, port: int) -> str:
    """SSH-cat the output log from the pod."""
    result = subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-p", str(port), f"root@{host}",
            "cat /workspace/output.log",
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_summary(log: str) -> dict:
    """Extract the --- summary block from training output."""
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
    """Terminate the pod."""
    import runpod
    runpod.terminate_pod(pod_id)
    print(f"[runpod] Pod {pod_id} terminated.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run autoresearch on RunPod")
    parser.add_argument("--gpu", default=None, help="GPU type override (e.g. 'NVIDIA A100 80GB PCIe')")
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
        pod = wait_for_running(pod_id)
        host, port = get_ssh_info(pod)
        print(f"[runpod] Pod running at {host}:{port}", flush=True)

        wait_for_training(host, port)

        log = fetch_log(host, port)
        metrics = parse_summary(log)

        print("\n" + "=" * 40)
        if metrics:
            for k, v in metrics.items():
                print(f"{k}: {v}")
        else:
            print("WARNING: no summary block found in output.")
            print("--- last 500 chars of log ---")
            print(log[-500:])
        print("=" * 40)

    finally:
        if pod_id:
            try:
                terminate_pod(pod_id)
            except Exception as e:
                print(f"WARNING: failed to terminate pod {pod_id}: {e}", file=sys.stderr)
                print(f"  Manually terminate at https://www.runpod.io/console/pods", file=sys.stderr)


if __name__ == "__main__":
    main()
