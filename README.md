# ApproximateFoRL

# Installation

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

## Third-party packages
See README in externals folder 