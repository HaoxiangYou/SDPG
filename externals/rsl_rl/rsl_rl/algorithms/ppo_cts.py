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
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticCTS
from rsl_rl.storage import RolloutStorageCTS

"""
PPO with concurrent teacher-student architecture, refer to https://clearlab-sustech.github.io/concurrentTS/
"""


class PPO_CTS(PPO):
    actor_critic: ActorCriticCTS

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        use_spo=False,
        device="cpu",
        encoder_lr=1e-3,  # learning rate for history encoder
        num_encoder_epochs=1,  # number of epochs for history encoder via supervised learning
        num_teacher=1,
    ):
        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device,
        )

        self.encoder_lr = encoder_lr
        self.num_encoder_epochs = num_encoder_epochs
        self.num_teacher = num_teacher

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.rl_params = (
            list(self.actor_critic.actor.parameters())
            + list(self.actor_critic.critic.parameters())
            + list(self.actor_critic.privilege_encoder.parameters())
            + [self.actor_critic.std]
        )
        self.optimizer = optim.Adam(
            self.rl_params, lr=learning_rate
        )  # do not consider paramters of student encoder during RL update
        self.history_encoder_optimizer = optim.Adam(
            self.actor_critic.history_encoder.parameters(), lr=encoder_lr
        )  # for history encoder supervised learning update
        self.transition = RolloutStorageCTS.Transition()

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorageCTS(
            num_envs,
            self.num_teacher,
            num_transitions_per_env,
            actor_obs_shape,
            privileged_obs_shape,
            obs_history_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def act(self, obs, privileged_obs, obs_history, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # In storage, first num_teacher indices are teacher envs, the rest are student envs
        # Compute the actions and values
        teacher_actions = self.actor_critic.act(
            obs[: self.num_teacher], None, privileged_obs[: self.num_teacher], act_type="teacher"
        ).detach()
        teacher_actions_log_prob = self.actor_critic.get_actions_log_prob(teacher_actions).detach()
        teacher_action_mean = self.actor_critic.action_mean.detach()
        teacher_action_sigma = self.actor_critic.action_std.detach()
        student_actions = self.actor_critic.act(
            obs[self.num_teacher :], obs_history[self.num_teacher :], None, act_type="student"
        ).detach()
        student_actions_log_prob = self.actor_critic.get_actions_log_prob(student_actions).detach()
        student_action_mean = self.actor_critic.action_mean.detach()
        student_action_sigma = self.actor_critic.action_std.detach()
        # store the actions and log probs
        self.transition.actions = torch.cat((teacher_actions, student_actions), dim=0)
        self.transition.actions_log_prob = torch.cat((teacher_actions_log_prob, student_actions_log_prob), dim=0)
        self.transition.action_mean = torch.cat((teacher_action_mean, student_action_mean), dim=0)
        self.transition.action_sigma = torch.cat((teacher_action_sigma, student_action_sigma), dim=0)
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = privileged_obs.detach()
        self.transition.observation_histories = obs_history.detach()
        self.transition.critic_observations = critic_obs.detach()
        self.transition.values = self.actor_critic.evaluate(self.transition.critic_observations).detach()
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_teacher_surrogate_loss = 0
        mean_student_surrogate_loss = 0
        mean_reconstruction_loss = 0
        generator = self._get_data_generator()
        for (
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
            target_values_batch,
            returns_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            loss, teacher_surrogate_loss, student_surrogate_loss, value_loss = self._compute_rl_loss(
                teacher_obs_batch,
                teacher_privileged_obs_batch,
                teacher_actions_batch,
                teacher_old_actions_log_prob_batch,
                teacher_advantages_batch,
                teacher_old_mu_batch,
                teacher_old_sigma_batch,
                student_obs_batch,
                student_obs_histories_batch,
                student_actions_batch,
                student_old_actions_log_prob_batch,
                student_advantages_batch,
                critic_obs_batch,
                target_values_batch,
                returns_batch,
                hid_states_batch,
                masks_batch,
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
            self.optimizer.step()

        generator = self._get_data_generator()
        for (
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
            target_values_batch,
            returns_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            # Reconstruction gradient step
            for _ in range(self.num_encoder_epochs):
                reconstruction_loss = self._compute_encoder_loss(
                    student_obs_histories_batch, student_privileged_obs_batch
                )
                self.history_encoder_optimizer.zero_grad()
                reconstruction_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                self.history_encoder_optimizer.step()

                mean_reconstruction_loss += reconstruction_loss.item()

            mean_value_loss += value_loss.item()
            mean_teacher_surrogate_loss += teacher_surrogate_loss.item()
            mean_student_surrogate_loss += student_surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_teacher_surrogate_loss /= num_updates
        mean_student_surrogate_loss /= num_updates
        mean_reconstruction_loss /= num_updates * self.num_encoder_epochs
        self.storage.clear()

        return mean_value_loss, mean_teacher_surrogate_loss, mean_student_surrogate_loss, mean_reconstruction_loss

    def _compute_rl_loss(
        self,
        teacher_obs_batch,
        teacher_privileged_obs_batch,
        teacher_actions_batch,
        teacher_old_actions_log_prob_batch,
        teacher_advantages_batch,
        teacher_old_mu_batch,
        teacher_old_sigma_batch,
        student_obs_batch,
        student_obs_histories_batch,
        student_actions_batch,
        student_old_actions_log_prob_batch,
        student_advantages_batch,
        critic_obs_batch,
        target_values_batch,
        returns_batch,
        hid_states_batch,
        masks_batch,
    ):
        # Teacher update
        self.actor_critic.act(
            teacher_obs_batch,
            None,
            teacher_privileged_obs_batch,
            act_type="teacher",
            masks=masks_batch,
            hidden_states=hid_states_batch[0],
        )
        teacher_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(teacher_actions_batch)
        teacher_entropy_batch = self.actor_critic.entropy
        teacher_mu_batch = self.actor_critic.action_mean
        teacher_sigma_batch = self.actor_critic.action_std

        ## Teacher KL, adapt learning rate
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(teacher_sigma_batch / teacher_old_sigma_batch + 1.0e-5)
                    + (torch.square(teacher_old_sigma_batch) + torch.square(teacher_old_mu_batch - teacher_mu_batch))
                    / (2.0 * torch.square(teacher_sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        ## Surrogate loss
        ratio = torch.exp(teacher_actions_log_prob_batch - torch.squeeze(teacher_old_actions_log_prob_batch))
        surrogate = -torch.squeeze(teacher_advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(teacher_advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        teacher_surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Student update
        self.actor_critic.act(
            student_obs_batch,
            student_obs_histories_batch,
            None,
            act_type="student",
            masks=masks_batch,
            hidden_states=hid_states_batch[0],
        )
        student_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(student_actions_batch)
        student_entropy_batch = self.actor_critic.entropy

        ## Surrogate loss
        ratio = torch.exp(student_actions_log_prob_batch - torch.squeeze(student_old_actions_log_prob_batch))
        surrogate = -torch.squeeze(student_advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(student_advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        student_surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        total_entropy_batch = torch.cat((teacher_entropy_batch, student_entropy_batch), dim=0)

        loss = (
            self.value_loss_coef * value_loss
            + teacher_surrogate_loss
            + student_surrogate_loss
            - self.entropy_coef * (total_entropy_batch.mean())
        )

        return loss, teacher_surrogate_loss, student_surrogate_loss, value_loss

    def _compute_encoder_loss(self, student_obs_histories_batch, student_privileged_obs_batch):
        encoder_predictions = self.actor_critic.history_encoder(student_obs_histories_batch)

        with torch.no_grad():  # don't backpropagate through the encoder targets
            encoder_targets = self.actor_critic.privilege_encoder(student_privileged_obs_batch)

        reconstruction_loss = nn.functional.mse_loss(  # use mse loss
            encoder_predictions, encoder_targets
        )

        return reconstruction_loss
