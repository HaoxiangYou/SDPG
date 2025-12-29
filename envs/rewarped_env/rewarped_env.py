import importlib
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import warp as wp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rewarped.warp_env import WarpEnv

from envs.base_env import BaseEnv
from utils.common_utils import snakecase_to_pascalcase


class RewarpedEnv(BaseEnv):
    """Environment wrapper for the rewarped simulator."""

    def __init__(self, _wrapped_env: WarpEnv) -> None:
        super().__init__()
        self._wrapped_env = _wrapped_env
        self.state_tensors_names = self._wrapped_env.state_tensors_names

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs = self._wrapped_env.reset(env_ids)

        extra_info = {}
        return obs, extra_info

    def step(
        self, actions: torch.Tensor, auto_reset: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Rewrite the step function in self._wrapped_env to support non auto-reset."""

        with wp.ScopedTimer("simulate", active=False, detailed=False):
            self._wrapped_env.pre_physics_step(actions)
            self._wrapped_env.do_physics_step()

        self._wrapped_env.progress_buf += 1
        self._wrapped_env.num_frames += 1
        self._wrapped_env.reset_buf = torch.zeros_like(self._wrapped_env.reset_buf)
        # post_physics_step()
        self._wrapped_env.compute_observations()
        self._wrapped_env.compute_reward()
        extras = {
            "obs_before_reset": None,
        }

        env_ids = self._wrapped_env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if auto_reset and len(env_ids) > 0:
            if isinstance(self._wrapped_env.obs_buf, dict):
                obs_buf_before_reset = {k: v.clone() for k, v in self._wrapped_env.obs_buf.items()}
            else:
                obs_buf_before_reset = self._wrapped_env.obs_buf.clone()
            extras["obs_before_reset"] = obs_buf_before_reset

            with wp.ScopedTimer("reset", active=False, detailed=False):
                self._wrapped_env.reset(env_ids)

        # NOTE: this occurs post reset, so will render initial state (not terminal state)
        with wp.ScopedTimer("render", active=False, detailed=False):
            self._wrapped_env.render()

        return (
            self._wrapped_env.obs_buf,
            self._wrapped_env.rew_buf,
            self._wrapped_env.terminated_buf,
            self._wrapped_env.truncated_buf,
            extras,
        )

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        return {
            "robot_states": {
                name: wp.to_torch(getattr(self._wrapped_env.state_0, name)).view(self.num_envs, -1)[env_ids]
                for name in self.state_tensors_names
            },
            "progress_buf": self._wrapped_env.progress_buf[env_ids],
        }

    def set_states(self, states: Dict[str, Any], env_ids: Optional[Sequence[int]] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # set states
        for name in self.state_tensors_names:
            # TODO not test if this is correct
            wp.to_torch(self._wrapped_env.state_0, name).view(self.num_envs, -1)[env_ids] = wp.from_torch(
                states["robot_states"][name]
            )
            raise NotImplementedError("Set states for the rewarped env is not tested yet!")

        # set progress_buf
        self._wrapped_env.progress_buf[env_ids] = states["progress_buf"].clone()

    def render(self) -> None:
        return self._wrapped_env.render()

    def initialize_trajectory(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Cut off the computation graph to the previous states.
        """
        obs = self._wrapped_env.initialize_trajectory()
        extra_info = {}
        return obs, extra_info

    def save_video(self) -> None:
        self._wrapped_env.renderer.save()

    @property
    def requires_grad(self) -> bool:
        return self._wrapped_env.requires_grad

    @property
    def num_observations(self) -> int:
        return self._wrapped_env.num_observations

    @property
    def num_actions(self) -> int:
        return self._wrapped_env.num_actions

    @property
    def num_envs(self) -> int:
        return self._wrapped_env.num_envs

    @property
    def device(self) -> Any:
        return self._wrapped_env.device

    @property
    def renderer(self) -> Any:
        return self._wrapped_env.renderer

    @property
    def action_space(self) -> Any:
        return self._wrapped_env.action_space

    @property
    def observation_space(self) -> Any:
        return self._wrapped_env.observation_space


def make_envs(config: DictConfig) -> RewarpedEnv:
    env_kwargs = OmegaConf.to_container(config.task.config, resolve=True)
    env_name = config.task.name
    num_envs = env_kwargs.pop("num_envs")
    suite = env_kwargs.pop("suite")

    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg is not None:
            render_dir = hydra_cfg.run.dir
            os.environ["WARP_RENDER_DIR"] = render_dir
    except (RuntimeError, AttributeError):
        pass

    ENV = importlib.import_module(f"rewarped.envs.{suite}.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env = env_fn(num_envs=num_envs, device=config.device, seed=config.seed, **env_kwargs)

    return RewarpedEnv(env)
