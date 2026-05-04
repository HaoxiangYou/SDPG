# Custom Genesis Environments

This guide explains how to add a new environment to the Genesis simulator backend.

## 1. Create the Environment

### 1.1 Overview

Create a new file `envs/genesis_env/<env_name>.py` with a class that inherits from `GenesisEnv`.
The class name must be the PascalCase version of the file name (e.g., `my_robot.py` → `MyRobot`).
Once the environment is working, create the corresponding configs under `cfgs/task/genesis/`.

### 1.2 Examples

| File | Features |
|---|---|
| `hopper.py` | Simplest env — loading a robot, defining observation/action spaces, `vis_obs` setup, and image history buffer |
| `allegro_hand.py` | Goal-conditioned reward, mixed observation keys (proprioception + ego-centric camera) |
| `walker_hurtle.py` | Terrain generation, heightfield observations, RGB/depth camera switching |
| `go2_terrain.py` | User commands, curriculum learning, domain randomization |
|`aloha_insertion.py`|Multiple cameras|

### 1.3 Abstract methods

You must implement the following abstract methods:

| Method | Purpose |
|---|---|
| `init_scene()` | Load robot URDF/MJCF, configure sensors, other initialization |
| `build_scene()` | Call `self._scene.build(n_envs=...)` and initalization after scene is build|
| `_reset_idx(env_ids)` | Reset specified environments to initial state |
| `_set_actions(actions)` | Apply actions to the robot.|
| `get_states(env_ids)` | Return a dict with `robot_states` and `progress_buf` |
| `set_states(states, env_ids)` | Restore environment from a states dict |
| `compute_observations(states)` | Compute and return observation dict |
| `compute_reward(states, actions)` | Compute scalar reward per environment |
| `compute_termination(states)` | Return boolean termination mask |
| `_post_physics_step()` | Post-step logic (e.g., update commands, do rendering) |

Minimal skeleton:

<details>

```python
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import torch
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class MyRobot(GenesisEnv):
    _num_actions = 6
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))

    def __init__(self, num_envs, vis_obs=False, seed=0, **kwargs):
        self._vis_obs = vis_obs
        self._observation_space = spaces.Dict({
            "observations": spaces.Box(low=-float("inf"), high=float("inf"), shape=(17,)),
        })
        super().__init__(num_envs=num_envs, episode_length=1000, seed=seed, **kwargs)

    def init_scene(self):
        self._robot = self._scene.add_entity(gs.morphs.URDF(file="path/to/robot.urdf"))
        self._scene.add_entity(gs.morphs.Plane())

    def build_scene(self):
        self._scene.build(n_envs=self._num_envs)

    def _reset_idx(self, env_ids): ...
    def _set_actions(self, actions): ...
    def get_states(self, env_ids=None): ...
    def set_states(self, states, env_ids=None): ...
    def compute_observations(self, states): ...
    def compute_reward(self, states, actions): ...
    def compute_termination(self, states): ...
    def _post_physics_step(self): ...
```

</details>


### 1.4 Step loop

The `GenesisEnv.step()` method follows this pipeline:

```
step(actions)
│
├── 1. _set_actions(actions)        # Apply actions (e.g., set DOF targets)
├── 2. _scene.step()                # Physics simulation
├── 3. get_states()                 # Query robot state
├── 4. compute_reward(states)       # Scalar reward per env
├── 5. compute_termination(states)  # Boolean done mask
├── 6. _post_physics_step()         # Post-step logic (e.g., update commands)
├── 7. reset(done_env_ids)          # Auto-reset if auto_reset=True (default)
├── 8. compute_observations(states) # Build observation dict
│
└── return (observations, rewards, terminated, truncated, infos)
```

NaN states are automatically detected and treated as terminations.

### 1.5 Additional Notes

- **`get_states()` / `set_states()`**: These are the most critical functions to implement correctly. State must include **everything** that affects reward and observation computation — not just robot DOFs (`joint_q`, `joint_qd`), but also commands, targets, progress buffers, and any other internal variables. `set_states(get_states())` must reproduce the exact same state. This is essential for AFRL's auxiliary environment rollouts.

- **Action range**: AFRL outputs actions in **(-1, 1)** . Scale them to your actuator range inside `_set_actions()`.

- **`compute_observations()`**: Called independently at each short-horizon rollout. It must be **deterministic** given the current state — calling it twice without an intervening `set_states()` or `step()` should return identical values. Be careful with history-dependent observations (e.g., stacked frames, velocity filters): any such history must be part of the state so it is properly saved/restored.

- **Logging custom metrics**: Reward alone may not reflect task progress. You can log additional metrics to TensorBoard/WandB by populating `self._infos` (a dict). All keys in `_infos` are automatically logged by the agent. See `allegro_hand.py` for an example (`self._infos["angle_diff"] = ...`).

### 1.6 Debug tools

Standalone scripts under `envs/genesis_env/scripts/` for debugging environments:

| Script | Purpose |
|---|---|
| `test_genesis_env.py` | Verify `get_states()`/`set_states()` consistency and action determinism across grouped envs. Change `env_name` at the top to target your env. |
| `adjust_render.py` | Tune camera settings (`pos`, `lookat`, `fov`, lighting). Pass `--traj_path path/to/trajectory.pt` to replay a saved trajectory under the current camera config. |

## 2. Tips for tuning AFRL agents

- Start with **state-based** observations to tune reward shaping, action scaling, and AFRL hyperparameters.
- Once state-based training works well, switching to **vision-based** is usually straightforward — add a visual encoder and increase the number of epochs.
- To quickly sanity-check a new environment (reward scale, action range, termination logic), run a PPO baseline with `rl_games` (requires `externals/rl_games`).
- In our experience, `horizon_length` and the **exploration strategy** are the most sensitive hyperparameters. For exploration, start with `fixed_std: true` (constant noise via `log_std_init`) to establish a baseline. Then try adaptive std (`fixed_std: false`). If `policy_std` decays too quickly, enable entropy regularization in the agent config. A good default is `state_dependent_std: false`, `actor_regularization: true`, and `soft_critic: false`.
- For **learning rate** and **network size**, use `genesis_hopper.yaml` as a minimal starting point for easy task. 
For harder tasks, refer to `genesis_humanoid.yaml`, `genesis_allegro_hand.yaml`, `genesis_go2_terrain`.
