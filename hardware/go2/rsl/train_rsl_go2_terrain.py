# from typing import Any, Dict, Tuple

import genesis as gs
import torch
import wandb
from rsl_rl.runners import TSRunner

from envs.genesis_env.go2_terrain_rsl import Go2Terrain

env_name = "go2"
num_envs = 4096
device = "cuda"
sim_options = gs.options.SimOptions(dt=0.02, substeps=4)
env_kwargs = {"show_viewer": False, "randomize_init": True}

train_cfg = {
    "algorithm": {
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.001,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    "init_member_classes": {},
    "policy": {
        "activation": "elu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 1.0,
    },
    "runner": {
        "algorithm_class_name": "PPO_TS",
        "checkpoint": -1,
        "experiment_name": "go2_rough",
        "load_run": -1,
        "log_interval": 1,
        "max_iterations": 5000,
        "num_steps_per_env": 24,
        "policy_class_name": "ActorCriticTS",
        "record_interval": 50,
        "resume": False,
        "resume_path": None,
        "run_name": "go2_rsl",
        "runner_class_name": "runner_class_name",
        "save_interval": 100,
        'record_interval': 50,
    },
    "runner_class_name": "TSRunner",
    "seed": 1,
}

domain_rand_options = {
    "randomize_friction": True,
    "friction_range": [0.2, 1.5],
    "randomize_base_mass": True,
    "added_mass_range": [-1.0, 3.0],
    "push_robot": True,
    "push_interval_s": 10,
    "max_push_vel_xy": 1.0,
    "randomize_com_displacement": True,
    "com_displacement_range": [-0.01, 0.01],
    "randomize_motor_strength": False,
    "motor_strength_range": [0.9, 1.1],
    "randomize_motor_offset": True,
    "motor_offset_range": [-0.02, 0.02],
    "randomize_kp_scale": True,
    "kp_scale_range": [0.8, 1.2],
    "randomize_kd_scale": True,
    "kd_scale_range": [0.8, 1.2],
    "use_terrain": True,
    "obtain_terrain_info_around_feet": True,
    "obtain_link_contact_states": True,
    "contact_state_link_names": ["thigh", "calf", "foot", "base", "hip"],
    "penalize_contacts_on": ["thigh", "calf", "base", "head", "hip"],
    "terrain_cfg": {
        "mesh_type": "heightfield",
        "curriculum": True,
        "selected": False,
        "obtain_terrain_info_around_feet": True,
        "measure_heights": True,
        "measured_points_x": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],  # 9x9=81
        "measured_points_y": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        "border_size": 20.0,
        "border_height": 1.0,
        "terrain_length": 8.0,
        "terrain_width": 8.0,
        "platform_size": 4.0,
        "num_rows": 10,  # number of terrain rows (levels)
        "num_cols": 10,  # number of terrain cols (types)
        "num_subterrains": 100,
        "horizontal_scale": 0.1,  # [m] distance between height samples in x and y direction
        "vertical_scale": 0.005,  # [m] distance between height samples in z direction
        "static_friction": 1.0,  # coefficient of static friction of the terrain
        "dynamic_friction": 1.0,  # coefficient of dynamic friction of the terrain
        "restitution": 0.0,  # coefficient of restitution of the terrain
        "max_init_terrain_level": 1,  # starting curriculum level
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, hurtle, stepping stones, gap, pit]
        "terrain_proportions": [0.2, 0.1, 0.25, 0.25, 0.2],
    },
}

log_dir = "logs/genesis_go2_rsl"


class RSL_Wrapper(Go2Terrain):
    """Go2 with step() adapted for rsl_rl: returns (obs, privileged_obs, rewards, dones, infos)."""

    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        obs_dict, rewards, terminated, truncated, info = super().step(actions, auto_reset)
        dones = terminated | truncated
        infos = {**info, "time_outs": truncated.float()}
        return (
            obs_dict["observations"],
            obs_dict["privileged_observations"],
            obs_dict["observations_history"],
            obs_dict["critic_observations"],
            rewards,
            dones.float(),
            infos,
        )
    
    def get_observations(self):
        return self._obs, self._privileged_obs_buf, self._obs_history_buf, self._critic_obs_history_buf

def main():
    env = RSL_Wrapper(
        num_envs=num_envs,
        device=device,
        seed=0,
        sim_options=sim_options,
        domain_rand_options=domain_rand_options,
        **env_kwargs,
    )

    runner = TSRunner(env, train_cfg, log_dir, device=gs.device)

    wandb.init(project="genesis", name="genesis_go2_rsl", dir=log_dir, mode="online")

    runner.learn(num_learning_iterations=5000, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()