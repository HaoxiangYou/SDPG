# ApproximateFoRL

# Setup

```bash
conda create -n AFRL python=3.10
conda activate AFRL

pip install .

# --- Third-party physical simulators ---
# --- rewarped (Note support to rewarped sim is limited and unstable) ---
pip install gym==0.23.1
pip install rewarped
# --- genesis ---
TODO

# --- Third-party rl supports ---
# --- rl-games ---
pip install rl-games