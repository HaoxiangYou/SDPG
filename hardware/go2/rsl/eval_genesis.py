import copy
import os

# from typing import Any, Dict, Tuple
import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from envs.genesis_env.go2 import Go2

env_name = "go2"
num_envs = 20
device = "cuda"
sim_options = gs.options.SimOptions(dt=0.02, substeps=4)
env_kwargs = {"show_viewer": True, "randomize_init": True}

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
        "algorithm_class_name": "PPO",
        "checkpoint": -1,
        "experiment_name": "test_genesis_env_rsl",
        "load_run": -1,
        "log_interval": 1,
        "max_iterations": 1,
        "num_steps_per_env": 24,
        "policy_class_name": "ActorCritic",
        "record_interval": 50,
        "resume": False,
        "resume_path": None,
        "run_name": "",
        "runner_class_name": "runner_class_name",
        "save_interval": 100,
    },
    "runner_class_name": "OnPolicyRunner",
    "seed": 1,
}

domain_rand_options = {
    "randomize_friction": False,
    "friction_range": [0.2, 1.5],
    "randomize_base_mass": False,
    "added_mass_range": [-1.0, 3.0],
    "randomize_com_displacement": False,
    "com_displacement_range": [-0.01, 0.01],
    "randomize_motor_strength": False,
    "motor_strength_range": [0.9, 1.1],
    "randomize_motor_offset": False,
    "motor_offset_range": [-0.02, 0.02],
    "randomize_kp_scale": False,
    "kp_scale_range": [0.8, 1.2],
    "randomize_kd_scale": False,
    "kd_scale_range": [0.8, 1.2],
    "use_terrain": True,
    "terrain_cfg": {
        "mesh_type": "heightfield",
        "curriculum": True,
        "selected": False,
        "obtain_terrain_info_around_feet": True,
        "measure_heights": True,
        "measured_points_x": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],  # 9x9=81
        "measured_points_y": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        "border_size": 5.0,
        "border_height": 0.5,
        "terrain_length": 8.0,
        "terrain_width": 2.0,
        "platform_size": 1.0,
        "num_rows": 5,  # number of terrain rows (levels)
        "num_cols": 1,  # number of terrain cols (types)
        "num_subterrains": 5,
        "horizontal_scale": 0.05,  # [m] distance between height samples in x and y direction
        "vertical_scale": 0.005,  # [m] distance between height samples in z direction
        "static_friction": 1.0,  # coefficient of static friction of the terrain
        "dynamic_friction": 1.0,  # coefficient of dynamic friction of the terrain
        "restitution": 0.0,  # coefficient of restitution of the terrain
        "max_init_terrain_level": 1,  # starting curriculum level
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, hurtle, stepping stones, gap, pit]
        "terrain_proportions": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    },
}


class RSL_Wrapper(Go2):
    """Go2 with step() adapted for rsl_rl: returns (obs, privileged_obs, rewards, dones, infos)."""

    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        obs_dict, rewards, terminated, truncated, info = super().step(actions, auto_reset)
        dones = terminated | truncated
        infos = {**info, "time_outs": truncated.float()}
        return (
            obs_dict["observations"],
            obs_dict["privileged_observations"],
            rewards,
            dones.float(),
            infos,
        )


def export_policy_as_jit(actor_critic, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{name}.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)

def main():
    env = RSL_Wrapper(
        num_envs=num_envs,
        device=device,
        seed=0,
        sim_options=sim_options,
        domain_rand_options=domain_rand_options,
        **env_kwargs,
    )

    log_dir = "logs/genesis_go2_rsl"
    checkpooint = 1000
    jit_ckpt_path = os.path.join(log_dir, "exported", "jit_model.pt")
    if os.path.exists(jit_ckpt_path):
        policy = torch.jit.load(jit_ckpt_path)
        policy.to(device="cuda:0")
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

        resume_path = os.path.join(log_dir, f"model_{checkpooint}.pt")
        runner.load(resume_path)
        path = os.path.join(log_dir, "exported")
        export_policy_as_jit(runner.alg.actor_critic, path, "jit_model")
        policy = torch.jit.load(jit_ckpt_path)
        policy.to(device="cuda:0")

    env.reset()
    obs = env.get_observations()

    with torch.no_grad():
        stop = False
        n_frames = 0
        env.start_recording(record_internal=False)
        while not stop:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            n_frames += 1
            if n_frames == 1000:
                env.stop_recording(os.path.join(os.getcwd(), "video.mp4"))
                exit()


if __name__ == "__main__":
    main()
