# ApproximateFoRL

# Setup

```bash
conda create -n AFRL python=3.10
conda activate AFRL

pip install .
pip install gym==0.23.1
pip install rewarped

# --- Third-party physical simulators ---

# --- rewarped (Note support to rewarped sim is limited and unstable) ---

# --- genesis ---
TODO

```

## Genesis
See Readme in genesis readme

## RL-games
### Commit id
 `208b9f9464b8a4ae6fcb17a2d8ee7b6ee44a417b`
### Modification

TODO: both modification are due to gs.init(); may can solve this more elegant

- **CPU/GPU**

    In `rl_games/common/player.py`, `BasePlayer`, `env_step` function
    
    change 
    ```
    if self.is_tensor_obses:
        return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
    ```
    to
    ```
    if self.is_tensor_obses:
        return self.obs_to_torch(obs), rewards, dones, infos
    ```
- **torch/numpy mixture **
    In `rl_games/rl_games/algos_torch/models.py` Line 322-325
    change
    
    ```
    @torch.compile()
        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size(-1) \
                + logstd.sum(dim=-1)
    ```

    to

    ```
    @torch.compile()
        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * torch.log(2.0 * torch.tensor(math.pi, device=x.device, dtype=x.dtype)) * x.size(-1) \
                + logstd.sum(dim=-1)
    ```