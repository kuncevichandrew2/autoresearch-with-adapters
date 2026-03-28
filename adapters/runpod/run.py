"""
Autoresearch — RunPod training script.
This runs ON the RunPod pod. Do not run locally.

Env vars injected by adapter.py:
  REPO_URL       — GitHub repo URL to clone
  REPO_BRANCH    — branch to clone (e.g. autoresearch/mar26)
  WANDB_API_KEY  — optional, WandB logging
  WANDB_PROJECT  — WandB project name
"""
import subprocess, sys, os, types

REPO_URL    = os.environ["REPO_URL"]
REPO_BRANCH = os.environ["REPO_BRANCH"]

os.environ.setdefault("WANDB_PROJECT", "autoresearch")

# Redirect all output to /workspace/output.log so adapter.py can fetch it via SSH
log_path = "/workspace/output.log"
os.makedirs("/workspace", exist_ok=True)
log_file = open(log_path, "w", buffering=1)

import sys as _sys

class Tee:
    """Write to both original stream and log file."""
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log
    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def fileno(self):
        return self.stream.fileno()

_sys.stdout = Tee(_sys.stdout, log_file)
_sys.stderr = Tee(_sys.stderr, log_file)

print("=== Installing dependencies ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install",
    "rustbpe>=0.1.0", "tiktoken>=0.11.0", "wandb>=0.19.0",
    "pyarrow>=16.0.0", "-q",
], check=True)

print("\n=== Cloning repo ===", flush=True)
import shutil
if os.path.exists("repo"):
    shutil.rmtree("repo")
subprocess.run([
    "git", "clone", "--branch", REPO_BRANCH, "--depth", "1", REPO_URL, "repo",
], check=True)
os.chdir("repo")
sys.path.insert(0, os.getcwd())

print("\n=== Preparing data ===", flush=True)
subprocess.run([sys.executable, "prepare.py", "--num-shards", "4"], check=True)


def executor(code: str) -> None:
    module = types.ModuleType("__main__")
    module.__file__ = "train.py"
    exec(compile(code, "train.py", "exec"), module.__dict__)


print("\n=== Training ===", flush=True)
executor(open("train.py").read())
