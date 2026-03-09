import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticEE
from rsl_rl.storage import RolloutStorageEE

"""
PPO with explicit estimator, refer to https://arxiv.org/abs/2202.05481
"""


class PPO_EE(PPO):
    actor_critic: ActorCriticEE

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
        estimator_lr=1e-3,  # learning rate for estimator
        num_estimator_epochs=1,  # number of epochs for estimator via supervised learning
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
        self.estimator_lr = estimator_lr
        self.num_estimator_epochs = num_estimator_epochs

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        # Only include parameters of actor, critic and std
        self.rl_parameters = (
            list(self.actor_critic.actor.parameters())
            + list(self.actor_critic.critic.parameters())
            + [self.actor_critic.std]
        )
        self.optimizer = optim.Adam(self.rl_parameters, lr=learning_rate)
        self.estimator_optimizer = optim.Adam(self.actor_critic.estimator.parameters(), lr=estimator_lr)
        self.transition = RolloutStorageEE.Transition()

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        privileged_obs_shape,
        estimator_feature_shape,
        estimator_label_shape,
        action_shape,
    ):
        self.storage = RolloutStorageEE(
            num_envs,
            num_transitions_per_env,
            privileged_obs_shape,
            estimator_feature_shape,
            estimator_label_shape,
            action_shape,
            self.device,
        )

    def act(self, estimator_features, critic_obs, estimator_labels):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(estimator_features).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.critic_observations = critic_obs
        self.transition.estimator_features = estimator_features
        self.transition.estimator_labels = estimator_labels
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        generator = self._get_data_generator()
        for (
            critic_obs_batch,
            estimator_features_batch,
            estimator_labels_batch,
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
                critic_obs_batch,
                estimator_features_batch,
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

        generator = self._get_data_generator()
        for (
            critic_obs_batch,
            estimator_features_batch,
            estimator_labels_batch,
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
            # estimator gradient step
            for _ in range(self.num_estimator_epochs):
                estimator_loss = self._compute_estimator_loss(
                    estimator_features_batch, estimator_labels_batch, terminated_batch
                )
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()
                mean_estimator_loss += estimator_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates * self.num_estimator_epochs
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss

    def _compute_rl_loss(
        self,
        critic_obs_batch,
        estimator_features_batch,
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
        self.actor_critic.act(estimator_features_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
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

    def _compute_estimator_loss(self, estimator_features_batch, estimator_labels_batch, terminated_batch):
        estimator_predictions = self.actor_critic.estimator(estimator_features_batch)
        estimator_loss = nn.functional.mse_loss(
            estimator_predictions * terminated_batch, estimator_labels_batch * terminated_batch
        )
        return estimator_loss
