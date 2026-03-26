"""
Kaggle adapter for autoresearch.
Sets up the environment to run train.py on Kaggle GPU notebooks.
"""

import os
import subprocess


def setup():
    """Install dependencies and load environment variables."""
    # Install deps (torch is pre-installed on Kaggle)
    subprocess.run([
        "pip", "install", "kernels>=0.11.7", "rustbpe>=0.1.0",
        "tiktoken>=0.11.0", "pyarrow>=21.0.0", "wandb>=0.19.0",
    ], check=True)

    # Load .env from repo root if present
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def run_prepare(num_shards=4):
    """Download data shards and train tokenizer."""
    subprocess.run(["python", "prepare.py", "--num-shards", str(num_shards)], check=True)


def run_train():
    """Run the training script."""
    subprocess.run(["python", "train.py"], check=True)


if __name__ == "__main__":
    setup()
    run_prepare()
    run_train()
