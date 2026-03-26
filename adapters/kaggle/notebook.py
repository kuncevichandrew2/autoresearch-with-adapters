"""
Autoresearch P100 — Kaggle notebook.
Clones the repo, installs deps, prepares data, runs train.py via executor.
"""
import subprocess, sys, os, types

REPO_URL    = "https://github.com/kuncevichandrew2/autoresearch-with-adapters"
REPO_BRANCH = "autoresearch/mar26"

os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_PROJECT"] = "autoresearch"

print("=== Installing PyTorch 2.4.1+cu118 (P100 compatible) ===")
subprocess.run([sys.executable, "-m", "pip", "install",
    "torch==2.4.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118", "-q",
], check=True)

print("\n=== Installing dependencies ===")
subprocess.run([sys.executable, "-m", "pip", "install",
    "rustbpe>=0.1.0", "tiktoken>=0.11.0", "wandb>=0.19.0", "pyarrow>=16.0.0", "-q",
], check=True)

print("\n=== Cloning repo ===")
subprocess.run([
    "git", "clone", "--branch", REPO_BRANCH, "--depth", "1", REPO_URL, ".",
], check=True)

print("\n=== Preparing data ===")
subprocess.run([sys.executable, "prepare.py", "--num-shards", "4"], check=True)


def executor(code: str) -> None:
    module = types.ModuleType("__main__")
    module.__file__ = "train.py"
    exec(compile(code, "train.py", "exec"), module.__dict__)


print("\n=== Training ===")
executor(open("train.py").read())
