"""
RunPod adapter for autoresearch.
Creates a GPU pod, SSH-launches training, streams logs, optionally keeps pod alive.

Usage:
    source .env
    python adapters/runpod/adapter.py [--gpu "NVIDIA RTX 5090"] [--no-terminate]
    python adapters/runpod/adapter.py --pod-id ID --no-terminate   # reuse existing pod

Requires:
    pip install runpod

Design:
    - Pod starts WITHOUT docker_args so the image's /start.sh runs normally
      (RunPod images use supervisord to start sshd + jupyter etc.)
    - SSH uses direct IP:port from runtime.ports (RunPod exposes 22/tcp publicly)
    - Once SSH is confirmed connectable, we launch training via SSH + nohup
    - Poll /workspace/output.log via SSH until '---' summary block appears
    - Fetch full log, print metrics
    - Terminate pod (unless --no-terminate)

--no-terminate keeps the pod alive so the next experiment can reuse it.
Use --pod-id to attach to an already-running pod (skips create + wait_for_api_ready).
The next-run command is printed at exit when --no-terminate is used.
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

POLL_INTERVAL    = 15    # seconds between polls
SSH_TIMEOUT      = 10    # seconds per SSH connection attempt
SSH_RETRY_LIMIT  = 40    # max attempts to establish first SSH (~10 min)
SSH_FAIL_LIMIT   = 5     # consecutive failures during training before giving up
MAX_WAIT         = 1800  # 30 min hard timeout for training


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
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    print("[runpod] Waiting for pod API to report SSH port...", flush=True)
    elapsed = 0
    while elapsed < MAX_WAIT:
        pod = runpod.get_pod(pod_id)
        if pod is None:
            print(f"  [{elapsed:4d}s] get_pod returned None — retrying...", flush=True)
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue
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


def get_ssh_address(pod_id: str) -> tuple[str, int]:
    """Get current SSH host:port from API (for reuse with --pod-id)."""
    import runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pod = runpod.get_pod(pod_id)
    if pod is None:
        raise RuntimeError(f"Pod {pod_id} not found")
    runtime = pod.get("runtime") or {}
    for p in (runtime.get("ports") or []):
        if p.get("privatePort") == 22:
            return p["ip"], int(p["publicPort"])
    raise RuntimeError(f"Pod {pod_id} has no SSH port mapped yet")


def wait_for_ssh(host: str, port: int) -> None:
    """Verify SSH connectivity — retry until it works."""
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
    """Launch training in background via SSH (nohup + disown).

    RunPod's SSH session does not inherit the container's env vars, so we
    write them to /tmp/runenv.sh and source it before running the script.
    """
    print("[runpod] Starting training via SSH...", flush=True)

    # Build env file content
    wandb_key     = os.environ.get("WANDB_API_KEY", "")
    wandb_project = os.environ.get("WANDB_PROJECT", "autoresearch")
    env_lines = [
        f"export REPO_URL='{REPO_URL}'",
        f"export REPO_BRANCH='{REPO_BRANCH}'",
        f"export WANDB_PROJECT='{wandb_project}'",
    ]
    if wandb_key:
        env_lines.append(f"export WANDB_API_KEY='{wandb_key}'")

    # Write env file to pod
    env_content = "\n".join(env_lines)
    write_env = f"printf '%s\\n' {chr(39)}{env_content}{chr(39)} > /tmp/runenv.sh"
    r = ssh_cmd(host, port, write_env)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to write env file: {r.stderr}")

    # Launch training
    launch_cmd = (
        f"mkdir -p /workspace && "
        f"nohup bash -c 'source /tmp/runenv.sh && "
        f"curl -fsSL {RUN_PY_RAW} -o /tmp/run.py && "
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
    ssh_fails = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        result = ssh_cmd(host, port,
            "grep -c '^---' /workspace/output.log 2>/dev/null || echo 0")
        if result.returncode != 0:
            ssh_fails += 1
            print(f"  [{elapsed:4d}s] SSH failed (rc={result.returncode}, consecutive={ssh_fails})"
                  f" stderr={result.stderr[:80]}", flush=True)
            if ssh_fails >= SSH_FAIL_LIMIT:
                raise RuntimeError(f"SSH failed {ssh_fails} consecutive times — pod likely dead")
            continue
        ssh_fails = 0
        count = result.stdout.strip().split('\n')[0]
        tail = ssh_cmd(host, port,
            "grep 'step ' /workspace/output.log 2>/dev/null | tail -1 || echo ''")
        step_line = tail.stdout.strip()[-80:] if tail.stdout.strip() else "..."
        print(f"  [{elapsed:4d}s] '---' blocks: {count}  |  {step_line}", flush=True)
        if count.isdigit() and int(count) >= 1:
            return
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
        help="GPU type override (e.g. 'NVIDIA RTX 5090')")
    parser.add_argument("--pod-id", default=None,
        help="Reuse an already-running pod (skips create + wait_for_api_ready)")
    parser.add_argument("--no-terminate", action="store_true",
        help="Keep pod alive after training so the next experiment can reuse it")
    args = parser.parse_args()

    load_env()

    if "RUNPOD_API_KEY" not in os.environ:
        print("ERROR: RUNPOD_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    pod_id = args.pod_id
    try:
        if pod_id:
            print(f"[runpod] Reusing existing pod {pod_id}", flush=True)
            host, port = get_ssh_address(pod_id)
            print(f"[runpod] SSH at {host}:{port}", flush=True)
        else:
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
            if args.no_terminate:
                print(f"\n[runpod] Pod {pod_id} kept alive (--no-terminate).", flush=True)
                print(f"  Next run: python adapters/runpod/adapter.py --pod-id {pod_id} --no-terminate",
                      flush=True)
                print(f"  Terminate: python -c \"import runpod,os; "
                      f"runpod.api_key=os.environ['RUNPOD_API_KEY']; "
                      f"runpod.terminate_pod('{pod_id}')\"", flush=True)
            else:
                try:
                    terminate_pod(pod_id)
                except Exception as e:
                    print(f"WARNING: failed to terminate {pod_id}: {e}", file=sys.stderr)
                    print("  Manually terminate at https://www.runpod.io/console/pods",
                          file=sys.stderr)


if __name__ == "__main__":
    main()
