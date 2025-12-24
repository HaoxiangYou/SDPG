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

# NOTE 

# 1. rl-games's load function does not support torch>=2.6
# You may need manually add weights_only to safe_load() 
# in  rl_games/algos_torch/torch_ext.py, e.g.
# `def safe_load(filename):
#    return safe_filesystem_op(torch.load, filename, weights_only=False)`


# 2. Genesis initialization may conflict with rl-games during evaluation
# You may need to change 
# `if self.is_tensor_obses:
#    return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos`
# to 
# `if self.is_tensor_obses:
#    return self.obs_to_torch(obs), rewards, dones, infos`
# in rl_games/common/player.py BasePlayer.env_step

# TODO
# Make a copy of rl-games in third-party dir?

pip install rl-games
