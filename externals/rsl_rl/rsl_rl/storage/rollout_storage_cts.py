# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

from .rollout_storage import RolloutStorage


class RolloutStorageCTS(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.privileged_observations = None
            self.observation_histories = None

    def __init__(
        self,
        num_envs,
        num_teacher,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        critic_obs_shape,
        actions_shape,
        device="cpu",
    ):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)

        self.obs_history_shape = obs_history_shape
        self.critic_obs_shape = critic_obs_shape
        self.num_teacher = num_teacher

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # privileged observations are necessary
        if self.privileged_observations is None:
            raise ValueError("Privileged observations are required for RolloutStorageTS")
        self.observation_histories = torch.zeros(
            num_transitions_per_env, num_envs, *obs_history_shape, device=self.device
        )
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)

        # For PPO
        self.teacher_advantages = torch.zeros(num_transitions_per_env, num_teacher, 1, device=self.device)
        self.student_advantages = torch.zeros(num_transitions_per_env, num_envs - num_teacher, 1, device=self.device)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        # compute teacher advantages and returns, using indices from [0, num_teacher]
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values[0 : self.num_teacher]
            else:
                next_values = self.values[step + 1, 0 : self.num_teacher]
            next_is_not_terminal = 1.0 - self.dones[step, 0 : self.num_teacher].float()
            delta = (
                self.rewards[step, 0 : self.num_teacher]
                + next_is_not_terminal * gamma * next_values
                - self.values[step, 0 : self.num_teacher]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step, 0 : self.num_teacher] = advantage + self.values[step, 0 : self.num_teacher]

        ## compute and normalize the advantages
        self.teacher_advantages = self.returns[:, 0 : self.num_teacher] - self.values[:, 0 : self.num_teacher]
        self.teacher_advantages = (self.teacher_advantages - self.teacher_advantages.mean()) / (
            self.teacher_advantages.std() + 1e-8
        )

        # compute student advantages and returns, using indices from [num_teacher, num_envs]
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values[self.num_teacher :]
            else:
                next_values = self.values[step + 1, self.num_teacher :]
            next_is_not_terminal = 1.0 - self.dones[step, self.num_teacher :].float()
            delta = (
                self.rewards[step, self.num_teacher :]
                + next_is_not_terminal * gamma * next_values
                - self.values[step, self.num_teacher :]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step, self.num_teacher :] = advantage + self.values[step, self.num_teacher :]

        ## compute and normalize the advantages
        self.student_advantages = self.returns[:, self.num_teacher :] - self.values[:, self.num_teacher :]
        self.student_advantages = (self.student_advantages - self.student_advantages.mean()) / (
            self.student_advantages.std() + 1e-8
        )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Use constant teacher environment indices
        teacher_mini_batch_size = self.num_teacher * self.num_transitions_per_env // num_mini_batches
        student_mini_batch_size = (self.num_envs - self.num_teacher) * self.num_transitions_per_env // num_mini_batches
        teacher_indices = torch.randperm(
            num_mini_batches * teacher_mini_batch_size, requires_grad=False, device=self.device
        )
        student_indices = torch.randperm(
            num_mini_batches * student_mini_batch_size, requires_grad=False, device=self.device
        )
        total_mini_batch_size = teacher_mini_batch_size + student_mini_batch_size
        total_indices = torch.randperm(
            num_mini_batches * total_mini_batch_size, requires_grad=False, device=self.device
        )

        # Split data into teacher group and student group
        teacher_observations = self.observations[:, 0 : self.num_teacher].flatten(0, 1)
        teacher_privileged_observations = self.privileged_observations[:, 0 : self.num_teacher].flatten(0, 1)
        # teacher_obs_histories = self.observation_histories[:, self.teacher_env_ids].flatten(0, 1)
        teacher_actions = self.actions[:, 0 : self.num_teacher].flatten(0, 1)
        teacher_old_actions_log_prob = self.actions_log_prob[:, 0 : self.num_teacher].flatten(0, 1)
        teacher_advantages = self.teacher_advantages.flatten(0, 1)
        teacher_old_mu = self.mu[:, 0 : self.num_teacher].flatten(0, 1)
        teacher_old_sigma = self.sigma[:, 0 : self.num_teacher].flatten(0, 1)

        student_observations = self.observations[:, self.num_teacher :].flatten(0, 1)
        student_privileged_observations = self.privileged_observations[:, self.num_teacher :].flatten(0, 1)
        student_obs_histories = self.observation_histories[:, self.num_teacher :].flatten(0, 1)
        student_actions = self.actions[:, self.num_teacher :].flatten(0, 1)
        student_old_actions_log_prob = self.actions_log_prob[:, self.num_teacher :].flatten(0, 1)
        student_advantages = self.student_advantages.flatten(0, 1)
        # Values do not need to split
        critic_observations = self.critic_observations.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Generate batch indices
                start_teacher = i * teacher_mini_batch_size
                end_teacher = (i + 1) * teacher_mini_batch_size
                teacher_batch_idx = teacher_indices[start_teacher:end_teacher]
                start_student = i * student_mini_batch_size
                end_student = (i + 1) * student_mini_batch_size
                student_batch_idx = student_indices[start_student:end_student]
                start_total = i * total_mini_batch_size
                end_total = (i + 1) * total_mini_batch_size
                total_batch_idx = total_indices[start_total:end_total]

                teacher_obs_batch = teacher_observations[teacher_batch_idx]
                teacher_privileged_obs_batch = teacher_privileged_observations[teacher_batch_idx]
                teacher_actions_batch = teacher_actions[teacher_batch_idx]
                teacher_old_actions_log_prob_batch = teacher_old_actions_log_prob[teacher_batch_idx]
                teacher_advantages_batch = teacher_advantages[teacher_batch_idx]
                teacher_old_mu_batch = teacher_old_mu[teacher_batch_idx]
                teacher_old_sigma_batch = teacher_old_sigma[teacher_batch_idx]

                student_obs_batch = student_observations[student_batch_idx]
                student_privileged_obs_batch = student_privileged_observations[student_batch_idx]
                student_obs_histories_batch = student_obs_histories[student_batch_idx]
                student_actions_batch = student_actions[student_batch_idx]
                student_old_actions_log_prob_batch = student_old_actions_log_prob[student_batch_idx]
                student_advantages_batch = student_advantages[student_batch_idx]

                critic_obs_batch = critic_observations[total_batch_idx]
                values_batch = values[total_batch_idx]
                returns_batch = returns[total_batch_idx]

                yield (
                    teacher_obs_batch,
                    teacher_privileged_obs_batch,
                    teacher_actions_batch,
                    teacher_old_actions_log_prob_batch,
                    teacher_advantages_batch,
                    teacher_old_mu_batch,
                    teacher_old_sigma_batch,
                    student_obs_batch,
                    student_privileged_obs_batch,
                    student_obs_histories_batch,
                    student_actions_batch,
                    student_old_actions_log_prob_batch,
                    student_advantages_batch,
                    critic_obs_batch,
                    values_batch,
                    returns_batch,
                    (None, None),
                    None,
                )
