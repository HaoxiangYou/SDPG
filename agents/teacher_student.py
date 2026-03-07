"""
Teacher–student DAgger agent: train a student policy on visual observations to mimic
a state-based teacher. Teacher is PPO (agents/rl_games.py); student uses
models.actor.build_actor from config. If teacher_checkpoint is null we train the
teacher first (writes to run_dir/teacher/ with .hydra, training_logs/nn, summaries);
otherwise teacher_checkpoint is an rl_games-style folder path and we load from it
(and copy that folder into run_dir/teacher/ when external).
"""

import copy
import os
import shutil
import time
from pathlib import Path

import gym
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import models
from utils.common_utils import TimeReport, make_envs, print_info
from utils.statistic_utils import AverageMeter

# Reuse PPO training from rl_games when train_teacher is True
from agents import rl_games as rl_games_agent


class TeacherPolicy(torch.nn.Module):
    """Wraps rl_games PpoPlayerContinuous (PPO teacher) for state-only expert actions."""

    def __init__(self, player, device="cuda:0"):
        super().__init__()
        self.player = player
        self.device = device

    @torch.no_grad()
    def forward(self, obs_dict, deterministic=True):
        """obs_dict: dict with 'privileged_observations'. Returns mean action (B, A)."""
        state = obs_dict["privileged_observations"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        elif state.device != self.player.device:
            state = state.to(self.player.device)
        action = self.player.get_action(state, is_deterministic=True)
        if isinstance(action, torch.Tensor):
            return action
        return torch.as_tensor(action, device=self.device, dtype=torch.float32)


class ExpertReplayBuffer:
    """Simple replay buffer for (vis_obs, expert_action) with max_size."""

    def __init__(self, vis_obs_shape, action_shape, max_size, device="cuda:0"):
        self.max_size = int(max_size)
        self.vis_obs = np.zeros((self.max_size,) + tuple(vis_obs_shape), dtype=np.uint8)
        self.actions = np.zeros((self.max_size,) + tuple(action_shape), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.device = device

    def append(self, vis_obs, actions):
        """vis_obs: (N, ...), actions: (N, num_actions)."""
        n = vis_obs.shape[0]
        if self.ptr + n <= self.max_size:
            self.vis_obs[self.ptr : self.ptr + n] = vis_obs
            self.actions[self.ptr : self.ptr + n] = actions
            self.ptr = (self.ptr + n) % self.max_size
            self.size = min(self.size + n, self.max_size)
        else:
            for i in range(n):
                self.vis_obs[self.ptr] = vis_obs[i]
                self.actions[self.ptr] = actions[i]
                self.ptr = (self.ptr + 1) % self.max_size
            self.size = self.max_size

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return (
            torch.from_numpy(self.vis_obs[indices]).to(self.device),
            torch.from_numpy(self.actions[indices]).to(self.device),
        )


class TeacherStudentRunner:
    """DAgger runner: teacher from PPO (rl_games) checkpoint, student from config; train with DAgger, eval student.
    Env and models are created only when needed (teacher training uses rl_games' env; student/DAgger creates env with vis_obs=True).
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.seed = config.seed
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.agent_config = config.agent.config
        # Support nested teacher/student config or flat config
        self._teacher_cfg = getattr(self.agent_config, "teacher", self.agent_config)
        self._student_cfg = getattr(self.agent_config, "student", self.agent_config)
        self.log_dir = config.log_dir
        self._env_and_models_initialized = False
        self.teacher_step_offset = 0
        self.teacher_time_offset = 0.0

    def _ensure_env_and_models(self):
        """Create env (vis_obs=True) and teacher+student models when running DAgger or eval. Idempotent."""
        if self._env_and_models_initialized:
            return
        OmegaConf.set_struct(self.config, False)
        if not self.config.task.config.get("vis_obs", False):
            self.config.task.config.vis_obs = True
        OmegaConf.set_struct(self.config, True)
        self.make_envs()
        self.num_actions = self.env.num_actions
        self.num_envs = self._student_cfg.num_envs
        self.make_models()
        self._env_and_models_initialized = True

    def _init_wandb(self, config: DictConfig) -> bool:
        if not getattr(config, "wandb", None) or not config.wandb.get("enable", False):
            return False
        wandb_config = config.wandb
        wandb_kwargs = {
            "project": wandb_config.get("project", "teacher_student"),
            "entity": wandb_config.get("entity"),
            "group": wandb_config.get("group"),
            "job_type": wandb_config.get("job_type"),
            "name": wandb_config.get("name"),
            "tags": wandb_config.get("tags", []),
            "notes": wandb_config.get("notes"),
        }
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        wandb.init(**wandb_kwargs)
        if wandb_config.get("log_config", True):
            config_dict = OmegaConf.to_container(config, resolve=True)
            wandb.config.update(config_dict)
        print_info("Wandb logging enabled")
        return True

    def make_envs(self):
        self.env = make_envs(self.config)
        obs_spaces = getattr(self.env.observation_space, "spaces", {})
        if "RGB" not in obs_spaces:
            raise ValueError("Teacher-student requires task with vis_obs=True (observation_space must contain 'RGB').")

    def _load_ppo_agent_config(self):
        """Load PPO agent config from cfgs/agent/{teacher_ppo_agent}.yaml (used for training and loading teacher)."""
        teacher_ppo_agent = self._teacher_cfg.get("teacher_ppo_agent", "ppo/genesis_hopper")
        parts = teacher_ppo_agent.strip().split("/")
        if len(parts) != 2:
            raise ValueError(f"teacher_ppo_agent must be like 'ppo/genesis_hopper', got: {teacher_ppo_agent}")
        cfgs_dir = Path(__file__).resolve().parent.parent / "cfgs"
        yaml_path = cfgs_dir / "agent" / parts[0] / f"{parts[1]}.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Teacher PPO config not found: {yaml_path}")
        return OmegaConf.load(yaml_path)

    @staticmethod
    def _resolve_teacher_ckpt_from_folder(folder_path: Path, rl_games_name: str) -> str:
        """Resolve .pth path from an rl_games-style folder (training_logs/nn or nn; best then last_*)."""
        nn_dir = folder_path / "training_logs" / "nn"
        if not nn_dir.is_dir():
            nn_dir = folder_path / "nn"
        if not nn_dir.is_dir():
            raise FileNotFoundError(
                f"Teacher folder has no nn dir: expected {folder_path / 'training_logs/nn'} or {folder_path / 'nn'}"
            )
        best_ckpt = nn_dir / f"{rl_games_name}.pth"
        if best_ckpt.is_file():
            return str(best_ckpt)
        candidates = list(nn_dir.glob(f"last_{rl_games_name}*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No teacher checkpoint in {nn_dir} (expected {rl_games_name}.pth or last_{rl_games_name}*.pth)"
            )
        return str(max(candidates, key=lambda p: p.stat().st_mtime))

    @staticmethod
    def _copy_teacher_folder_to_run(src_folder: Path, run_teacher_dir: Path) -> None:
        """Copy rl_games teacher run (e.g. .hydra, training_logs/nn, training_logs/summaries) into run_dir/teacher/."""
        run_teacher_dir.mkdir(parents=True, exist_ok=True)
        for item in src_folder.iterdir():
            dst = run_teacher_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

    @staticmethod
    def _load_teacher_step_time_from_tb(teacher_dir: Path) -> tuple[int, float]:
        """Read last step and total wall time from teacher's TensorBoard summaries. Returns (step_offset, time_offset_sec)."""
        summaries_dir = teacher_dir / "training_logs" / "summaries"
        if not summaries_dir.is_dir():
            summaries_dir = teacher_dir / "summaries"
        if not summaries_dir.is_dir():
            return 0, 0.0
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return 0, 0.0
        max_step = 0
        wall_times = []
        acc = EventAccumulator(str(summaries_dir), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for event in acc.Scalars(tag):
                max_step = max(max_step, event.step)
                wall_times.append(event.wall_time)
        if not wall_times:
            return max_step, 0.0
        wall_times.sort()
        total_time_sec = wall_times[-1] - wall_times[0]
        return max_step, max(0.0, total_time_sec)

    def _load_teacher(self):
        path_raw = self._teacher_cfg.get("teacher_checkpoint")
        if not path_raw or (isinstance(path_raw, str) and not path_raw.strip()):
            raise FileNotFoundError(
                "Teacher checkpoint not set (null). Train teacher first or set teacher_checkpoint to an rl_games-style folder path."
            )
        path = Path(path_raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint path does not exist: {path}")
        ppo_agent_cfg = self._load_ppo_agent_config()
        rl_games_name = ppo_agent_cfg.config.rl_games.config.get("name", "policy")
        if path.is_file():
            ckpt_path = str(path)
        else:
            ckpt_path = self._resolve_teacher_ckpt_from_folder(path, rl_games_name)
        rl_games_cfg = OmegaConf.to_container(ppo_agent_cfg.config.rl_games, resolve=True)
        # Build env_info from our env (state = privileged_observations for PPO actor)
        obs_space = self.env.observation_space.spaces["privileged_observations"]
        env_info = {
            "action_space": self.env.action_space,
            "observation_space": obs_space,
            "value_size": 1,
        }
        config = {
            "env_info": env_info,
            "env_name": "afrl_env",
            "num_actors": self.num_envs,
            "normalize_input": rl_games_cfg.get("config", {}).get("normalize_input", True),
            "normalize_value": rl_games_cfg.get("config", {}).get("normalize_value", True),
            "player": {"num_actors": self.num_envs, "games_num": self.num_envs},
            "reward_shaper": {"scale_value": 0.01},
        }
        params = {
            "config": config,
            "model": rl_games_cfg.get("model", {"name": "continuous_a2c_logstd"}),
            "network": rl_games_cfg.get("network"),
        }
        if params["network"] is None:
            raise ValueError(f"{self._teacher_cfg.teacher_ppo_agent} config must have rl_games.network")
        from rl_games.algos_torch import players
        from rl_games.common import tr_helpers
        config["reward_shaper"] = tr_helpers.DefaultRewardsShaper(**config["reward_shaper"])
        # No-op torch.compile while building player so teacher runs eager (rl_games compiles forward in __init__;
        # compiled path then fails with dynamo symbolic shapes even when input is e.g. (65536, 11))
        _compile = getattr(torch, "compile", None)
        if _compile is not None:
            def _noop_compile(fn=None, *a, **kw):
                if fn is not None:
                    return fn  # torch.compile(fn, ...)
                return lambda f: f  # torch.compile(mode=...)(fn) two-step call
            torch.compile = _noop_compile
        try:
            player = players.PpoPlayerContinuous(params)
            player.restore(ckpt_path)
        finally:
            if _compile is not None:
                torch.compile = _compile
        player.model.to(self.device)
        player.device = self.device
        # We pass (batch, obs_dim) e.g. (65536, 11); avoid unsqueeze_obs turning it into (1, 65536, 11) -> (1, 720896)
        player.has_batch_dimension = True
        return TeacherPolicy(player, device=self.device)

    def make_models(self):
        self.teacher = self._load_teacher()

        student_actor_config = self._student_cfg.actor
        self.student_input_keys = [input.name for input in student_actor_config.inputs]
        inputs_dim = {
            key: self.env.observation_space[key].shape
            for key in self.student_input_keys
        }
        self.student = models.actor.build_actor(
            actor_config=student_actor_config,
            inputs_dim=inputs_dim,
            num_actions=self.num_actions,
            device=self.device,
        )

    def _train_init(self):
        self.train_dir = os.path.join(self.log_dir, "train")
        self.nn_dir = os.path.join(self.train_dir, "nn")
        self.summary_dir = os.path.join(self.train_dir, "summary")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(self.summary_dir)
        self.use_wandb = self._init_wandb(self.config)

        self.max_epochs = self._student_cfg.max_epochs
        self.steps_per_epoch = self._student_cfg.steps_per_epoch
        self.batch_size = self._student_cfg.batch_size
        self.learning_starts = self._student_cfg.learning_starts
        self.num_update_per_epoch = self._student_cfg.num_update_per_epoch
        self.supervised_loss_threshold = self._student_cfg.get("supervised_loss_threshold", 1e-4)
        self.actor_lr = float(self._student_cfg.actor_lr)
        self.lr_schedule = self._student_cfg.get("lr_schedule", "constant")

        vis_shape = self.env.observation_space["RGB"].shape
        action_shape = (self.num_actions,)
        self.replay_buffer = ExpertReplayBuffer(
            vis_obs_shape=vis_shape,
            action_shape=action_shape,
            max_size=self._student_cfg.get("max_replay_buffer_size", 500_000),
            device=self.device,
        )
        self.student_optimizer = torch.optim.Adam(self.student.parameters(), lr=self.actor_lr, betas=self._student_cfg.get("betas", [0.9, 0.999]))

        self.time_report = TimeReport()
        self.time_report.add_timer("rollout")
        self.time_report.add_timer("expert_correction")
        self.time_report.add_timer("supervised_learning")
        self.time_report.add_timer("evaluation")

        self.step_count = 0
        self.supervised_iter_count = 0
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.save_frequency = self._student_cfg.get("save_frequency", 100)
        self._student_start_time = None  # set at start of train loop for time offset

    @torch.no_grad()
    def _sample_trajectories(self):
        """Roll out with student (vis_obs -> action), collect state_obs and vis_obs for expert labeling."""
        obs, _ = self.env.reset()
        vis_list, state_list = [], []
        episode_reward = torch.zeros(self.num_envs, device=self.device)
        episode_length = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        for _ in range(self.steps_per_epoch):
            vis = obs["RGB"]
            state = obs["privileged_observations"]
            vis_list.append(vis.cpu().numpy() if vis.is_cuda else vis.numpy())
            state_list.append(state)

            student_obs = {"RGB": vis}
            actions = self.student(student_obs)["mean"]
            obs, rewards, terminated, truncated, _ = self.env.step(torch.tanh(actions), auto_reset=True)
            dones = terminated | truncated
            episode_length += 1
            episode_reward += rewards
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                self.episode_reward_meter.update(episode_reward[done_ids])
                self.episode_length_meter.update(episode_length[done_ids].float())
                episode_length[done_ids] = 0
                episode_reward[done_ids] = 0.0

        self.step_count += self.steps_per_epoch * self.num_envs
        vis_arr = np.concatenate(vis_list, axis=0)
        state_tensors = {"privileged_observations": torch.cat(state_list, dim=0)}
        return vis_arr, state_tensors

    def _set_teacher_checkpoint_config(self, value: str) -> None:
        OmegaConf.set_struct(self.config, False)
        if hasattr(self.config.agent.config, "teacher"):
            self.config.agent.config.teacher.teacher_checkpoint = value
        else:
            self.config.agent.config.teacher_checkpoint = value
        OmegaConf.set_struct(self.config, True)
        self.agent_config = self.config.agent.config
        self._teacher_cfg = getattr(self.agent_config, "teacher", self.agent_config)

    def train(self, args=None):
        """If teacher_checkpoint is null: train teacher first (writes to run_dir/teacher/). Else: load from folder (copy to run_dir/teacher/ if external). Then DAgger."""
        args = args or {}
        teacher_dir = Path(self.log_dir) / "teacher"
        tc = self._teacher_cfg.get("teacher_checkpoint")
        tc_empty = tc is None or (isinstance(tc, str) and not tc.strip())
        if tc_empty:
            self._run_train_teacher(args)
            self._set_teacher_checkpoint_config(str(teacher_dir))
        else:
            tc_path = Path(tc).expanduser().resolve()
            if tc_path.is_dir() and tc_path != teacher_dir:
                self._copy_teacher_folder_to_run(tc_path, teacher_dir)
                self._set_teacher_checkpoint_config(str(teacher_dir))
        self.teacher_step_offset, self.teacher_time_offset = self._load_teacher_step_time_from_tb(teacher_dir)
        self._ensure_env_and_models()
        if args.get("checkpoint") and args["checkpoint"]:
            self.load(args["checkpoint"])
        self._train_init()
        self.save(filename="initial_policy")
        best_policy_reward = -float("inf")
        start_time = time.time()
        self._student_start_time = start_time
        print_info("========== Student (DAgger) training start ==========")

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()
            self.time_report.start_timer("rollout")
            vis_obs_np, state_obs = self._sample_trajectories()
            self.time_report.end_timer("rollout")

            self.time_report.start_timer("expert_correction")
            with torch.no_grad():
                expert_actions = self.teacher(state_obs)
                expert_actions = expert_actions.cpu().numpy()
            self.time_report.end_timer("expert_correction")

            self.replay_buffer.append(vis_obs_np, expert_actions)

            supervised_loss = float("inf")
            if len(self.replay_buffer) >= self.learning_starts:
                self.time_report.start_timer("supervised_learning")
                for _ in range(self.num_update_per_epoch):
                    if self.lr_schedule == "linear":
                        t = self.supervised_iter_count / max(1, self.max_epochs * self.num_update_per_epoch)
                        lr = 1e-5 + (self.actor_lr - 1e-5) * max(0, 1 - t)
                        for g in self.student_optimizer.param_groups:
                            g["lr"] = lr
                    b_vis, b_act = self.replay_buffer.sample(self.batch_size)
                    student_out = self.student({"RGB": b_vis})
                    loss = torch.nn.functional.mse_loss(student_out["mean"], b_act)
                    self.student_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self._student_cfg.get("grad_norm", 1.0))
                    self.student_optimizer.step()
                    supervised_loss = loss.item()
                    global_step = getattr(self, "teacher_step_offset", 0) + self.step_count
                    self.summary_writer.add_scalar("supervised_loss/iter", supervised_loss, self.supervised_iter_count)
                    if self.use_wandb:
                        wandb.log({"supervised_loss": supervised_loss, "env_step": global_step}, step=global_step)
                    self.supervised_iter_count += 1
                    if supervised_loss < self.supervised_loss_threshold:
                        break
                self.time_report.end_timer("supervised_learning")

            time_elapse = time.time() - start_time
            global_step = getattr(self, "teacher_step_offset", 0) + self.step_count
            global_time_sec = getattr(self, "teacher_time_offset", 0.0) + time_elapse
            policy_reward = self.episode_reward_meter.get_mean().item() if self.episode_reward_meter.current_size > 0 else float("-inf")
            episode_lengths = self.episode_length_meter.get_mean().item() if self.episode_length_meter.current_size > 0 else 0.0
            current_best = None
            if self.episode_reward_meter.current_size > 0 and policy_reward > best_policy_reward:
                best_policy_reward = policy_reward
                current_best = best_policy_reward
                print_info("Save best policy with reward: {:.2f}".format(best_policy_reward))
                self.save(filename="best_policy")

            self.summary_writer.add_scalar("rewards/iter", policy_reward, epoch)
            self.summary_writer.add_scalar("rewards/step", policy_reward, global_step)
            self.summary_writer.add_scalar("episode_lengths/iter", episode_lengths, epoch)
            self.summary_writer.add_scalar("time_sec", global_time_sec, global_step)
            if self.use_wandb:
                wandb.log({
                    "rewards": policy_reward,
                    "episode_lengths": episode_lengths,
                    "supervised_loss": supervised_loss,
                    "env_step": global_step,
                    "time_sec": global_time_sec,
                    "best_policy": current_best or best_policy_reward,
                }, step=global_step)

            print(
                "iter {}: ep reward {:.2f}, ep len {:.1f}, sup loss {:.4f}, fps {:.1f}".format(
                    epoch, policy_reward, episode_lengths, supervised_loss, (self.steps_per_epoch * self.num_envs) / (time.time() - time_start_epoch)
                )
            )
            if epoch % self.save_frequency == 0 or epoch == self.max_epochs - 1:
                self.save(filename="iter_{}_reward_{:.2f}".format(epoch, policy_reward))

        print_info("========== Student (DAgger) training done ==========")
        self.time_report.report()
        if self.use_wandb:
            wandb.finish()

    @torch.no_grad()
    def evaluate_policy(self, maximum_trajectory_length=None, save_trajectory=False):
        if maximum_trajectory_length is None:
            maximum_trajectory_length = getattr(self.env, "episode_length", 1000)
        episode_length = torch.zeros(self.num_envs, device=self.device)
        episode_reward = torch.zeros(self.num_envs, device=self.device)
        obs, _ = self.env.reset()
        for _ in range(maximum_trajectory_length):
            actions = self.student({"RGB": obs["RGB"]})["mean"]
            obs, rewards, terminated, truncated, _ = self.env.step(torch.tanh(actions), auto_reset=True)
            dones = terminated | truncated
            episode_length += 1
            episode_reward += rewards
            episode_length[dones] = 0
            episode_reward[dones] = 0
        mean_len = episode_length.float().mean().item()
        mean_rew = episode_reward.mean().item()
        if hasattr(self, "episode_length_meter") and self.episode_length_meter.current_size > 0:
            mean_len = self.episode_length_meter.get_mean().item()
            mean_rew = self.episode_reward_meter.get_mean().item()
        print_info("Eval episode length: {:.1f}, reward: {:.2f}".format(mean_len, mean_rew))

    def play(self):
        self.evaluate_policy()

    def run(self, args):
        if args.get("train", False):
            self.train(args)
        elif args.get("play", False):
            self._ensure_env_and_models()
            if args.get("checkpoint") and args["checkpoint"]:
                self.load(args["checkpoint"])
            self.play()

    def _run_train_teacher(self, args) -> None:
        """Train PPO teacher (state-based; vis_obs=False). Writes to run_dir/teacher/ (.hydra, training_logs/nn, training_logs/summaries)."""
        print_info("========== Teacher (PPO) training start ==========")
        ppo_agent_cfg = self._load_ppo_agent_config()
        cfg_teacher = copy.deepcopy(self.config)
        cfg_teacher.agent = copy.deepcopy(ppo_agent_cfg)
        # teacher.overrides (same structure as ppo yaml config) are merged into the loaded PPO config
        overrides = self._teacher_cfg.get("overrides")
        if overrides is not None:
            teacher_overrides = OmegaConf.to_container(overrides, resolve=True)
            if teacher_overrides:
                OmegaConf.set_struct(cfg_teacher.agent, False)
                cfg_teacher.agent.config = OmegaConf.merge(cfg_teacher.agent.config, OmegaConf.create(teacher_overrides))
                OmegaConf.set_struct(cfg_teacher.agent, True)
        # Teacher env must use PPO's num_envs (make_envs reads from task.config.num_envs)
        OmegaConf.set_struct(cfg_teacher, False)
        cfg_teacher.task.config.num_envs = cfg_teacher.agent.config.num_envs
        cfg_teacher.task.config.vis_obs = False  # Teacher: state only
        OmegaConf.set_struct(cfg_teacher, True)
        OmegaConf.resolve(cfg_teacher)
        # Teacher training uses its own subdirectory (rl_games uses config.log_dir when set)
        teacher_dir = Path(self.log_dir) / "teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.set_struct(cfg_teacher, False)
        cfg_teacher.log_dir = str(teacher_dir)
        OmegaConf.set_struct(cfg_teacher, True)
        # Save config used for teacher training into .hydra for reproducibility
        save_path = teacher_dir / ".hydra" / "teacher_train_config.yaml"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            OmegaConf.save(config=cfg_teacher, f=f)
        # Teacher uses its own env (state-only, vis_obs=False)
        teacher_env = make_envs(cfg_teacher)
        try:
            runner = rl_games_agent.make_runner(cfg_teacher, env=teacher_env)
            checkpoint = args.get("checkpoint") or None
            runner.run({"train": True, "play": False, "checkpoint": checkpoint})
        finally:
            # Close teacher env so resources are released; student env will be created from original task in _ensure_env_and_models()
            if getattr(teacher_env, "close", None) is not None:
                teacher_env.close()
        print_info("========== Teacher (PPO) training done ==========")

    def save(self, filename=None, save_dir=None):
        save_dir = save_dir or self.nn_dir
        filename = filename or "best_policy"
        path = os.path.join(save_dir, "{}.pt".format(filename))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.student.state_dict(), path)

    def load(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint not found: {}".format(path))
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.student.load_state_dict(state, strict=True)


def make_runner(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        output_dir = hydra_cfg.runtime.output_dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)
    return TeacherStudentRunner(config)
