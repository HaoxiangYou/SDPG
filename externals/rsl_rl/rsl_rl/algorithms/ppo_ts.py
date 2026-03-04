import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticTS
from rsl_rl.storage import RolloutStorageTS

"""
PPO with teacher-student architecture, refer to https://github.com/Improbable-AI/rapid-locomotion-rl
"""


class PPO_TS(PPO):
    actor_critic: ActorCriticTS

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

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Only include parameter of actor, critic, privilege encoder and std
        self.rl_parameters = (
            list(self.actor_critic.actor.parameters())
            + list(self.actor_critic.critic.parameters())
            + list(self.actor_critic.privilege_encoder.parameters())
            + [self.actor_critic.std]
        )
        self.optimizer = optim.Adam(self.rl_parameters, lr=learning_rate)
        self.history_encoder_optimizer = optim.Adam(self.actor_critic.history_encoder.parameters(), lr=encoder_lr)
        self.transition = RolloutStorageTS.Transition()

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
        self.storage = RolloutStorageTS(
            num_envs,
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
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_encoder_loss = 0
        generator = self._get_data_generator()
        for (
            obs_batch,
            privileged_obs_batch,
            obs_histories_batch,
            critic_obs_batch,
            terminated_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            loss, surrogate_loss, value_loss = self._compute_rl_loss(
                obs_batch,
                privileged_obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        # encoder update
        generator = self._get_data_generator()
        for (
            obs_batch,
            privileged_obs_batch,
            obs_histories_batch,
            critic_obs_batch,
            terminated_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            # history encoder gradient step
            for _ in range(self.num_encoder_epochs):
                encoder_loss = self._compute_encoder_loss(obs_histories_batch, privileged_obs_batch, terminated_batch)
                self.history_encoder_optimizer.zero_grad()
                encoder_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                self.history_encoder_optimizer.step()
                mean_encoder_loss += encoder_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_encoder_loss /= num_updates * self.num_encoder_epochs
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_encoder_loss

    def _compute_rl_loss(
        self,
        obs_batch,
        privileged_obs_batch,
        critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
    ):
        self.actor_critic.act(obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        self._adjust_learning_rate(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate_loss = self._compute_surrogate_loss(ratio, advantages_batch)

        # Value function loss
        value_loss = self._compute_value_function_loss(value_batch, returns_batch, target_values_batch)

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

        return loss, surrogate_loss, value_loss

    def _compute_encoder_loss(self, obs_histories_batch, privileged_obs_batch, terminated_batch):
        if self.actor_critic.history_encoder_type == "TCN":
            if obs_histories_batch.dim() == 2:
                # input shape (batch_size, obs_history_len) -> (batch_size, 1, obs_history_len)
                obs_histories_batch = obs_histories_batch.unsqueeze(1)
        encoder_predictions = self.actor_critic.history_encoder(obs_histories_batch)

        with torch.no_grad():  # don't backpropagate through the encoder targets
            encoder_targets = self.actor_critic.privilege_encoder(privileged_obs_batch)

        encoder_loss = nn.functional.mse_loss(  # use mse loss
            encoder_predictions * terminated_batch, encoder_targets * terminated_batch
        )

        return encoder_loss
