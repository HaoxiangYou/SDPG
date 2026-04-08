# ApproximateFoRL

Official implmentation for Approximate First Order RL.

## Installation

```bash
conda create -n AFRL python=3.11
conda activate AFRL

pip install -e ".[dev]"

# Optional: install Genesis in editable mode 
pip install -e "externals/Genesis[dev]"
```


For developer, install the git hooks so ruff runs on commit:

```bash
pre-commit install        # enable hooks on git commit
```

Run manually on all files: `pre-commit run --all-files`

### Third-party packages

Includes packages for running most baselines. See [`externals/README.md`](externals/README.md) for details.

## Training

### Quick start

Training is launched via Hydra. Override `task` and `agent` to select the environment and algorithm:

```bash
python scripts/run.py task=genesis/hopper agent=afrl/genesis_hopper
```

Logs are written to `logs/<backend>/<task>/<agent>/train/<timestamp>/`.

### 

## Evaluation

Set `train=False` and point to a checkpoint:

```bash
python scripts/run.py task=genesis/hopper agent=afrl/genesis_hopper train=False checkpoint=<path_to_checkpoint>
```

## Custom Environments

See [`envs/genesis_env/README.md`](envs/genesis_env/README.md) for a step-by-step guide on adding new environments.