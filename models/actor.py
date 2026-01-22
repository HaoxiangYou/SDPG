import numpy as np
import torch
import torch.nn as nn

from models.encoder import build_encoder
from utils.model_utils import get_activation_func, init_module


class PolicyHeadBase(nn.Module):
    """Base class for policy heads.

    Policy heads take encoded features and output actions.
    """

    def __init__(self, input_dim, num_actions, device="cuda:0"):
        """
        Args:
            input_dim: Dimension of input features (after encoding and concatenation)
            num_actions: Number of action dimensions
            device: Device to place model on
        """
        super(PolicyHeadBase, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = device

    def forward(self, features) -> torch.Tensor:
        """
        Args:
            features: Encoded and concatenated features of shape (batch, input_dim)

        Returns:
            Actions tensor of shape (batch, num_actions)
        """
        raise NotImplementedError("Subclasses must implement forward")


class DeterministicMLPPolicyHead(PolicyHeadBase):
    """Deterministic MLP policy head."""

    def __init__(self, input_dim, num_actions, cfg_network, device="cuda:0"):
        """
        Args:
            input_dim: Dimension of input features
            num_actions: Number of action dimensions
            cfg_network: Network config with units and activation
            device: Device to place model on
        """
        super(DeterministicMLPPolicyHead, self).__init__(input_dim, num_actions, device)

        units = list(cfg_network["units"])
        activation = cfg_network["activation"]
        layer_dims = [input_dim] + units + [num_actions]
        modules = []
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            init_module(linear_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
            modules.append(linear_layer)

            if i < len(layer_dims) - 2:
                modules.append(get_activation_func(activation))
                modules.append(nn.LayerNorm(layer_dims[i + 1]))

        self.policy_head = nn.Sequential(*modules).to(device)

    def forward(self, features):
        """Forward pass."""
        return self.policy_head(features)


def build_policy_head(policy_type, input_dim, num_actions, cfg_network, device="cuda:0") -> PolicyHeadBase:
    """Factory function to build policy heads.

    Args:

        input_dim: Dimension of input features
        num_actions: Number of action dimensions
        cfg_network: Network config dict
        device: Device to place model on

    Returns:
        PolicyHeadBase instance
    """
    policy_type = policy_type.lower()

    if policy_type == "deterministic_mlp":
        network_cfg = cfg_network.get("network", {})
        return DeterministicMLPPolicyHead(input_dim, num_actions, network_cfg, device=device)
    else:
        raise NotImplementedError(f"Policy head type {policy_type} not implemented")


class Actor(nn.Module):
    """Base actor class with encoder and policy_head trunks.

    The encoder trunk encodes all input observations and concatenates them.
    The policy_head trunk processes the concatenated features to generate actions.
    """

    def __init__(self, actor_config, inputs_dim, num_actions, device="cuda:0", obs_rms=None):
        """
        Args:
            actor_config: Full actor config containing inputs and policy_head
            inputs_dim: Dict mapping input keys to their shapes (from environment)
            num_actions: Number of action dimensions
            device: Device to place model on
            obs_rms: Optional dict of RunningMeanStd modules for each input key
        """
        super(Actor, self).__init__()

        self.device = device
        self.num_actions = num_actions

        # Extract input keys and configs from actor_config
        self.input_keys = [input.name for input in actor_config.inputs]
        inputs_config = {input.name: input for input in actor_config.inputs}

        # Build encoder trunk: encode each input
        self.encoders = nn.ModuleDict()
        encoder_info = []

        for key in self.input_keys:
            input_cfg = inputs_config[key]
            input_shape = inputs_dim[key]
            encoder_cfg = input_cfg.get("encoder")

            encoder = build_encoder(key, input_shape, encoder_cfg, device=device)
            self.encoders[key] = encoder
            encoder_type = encoder.__class__.__name__
            encoder_info.append(f"{key}: {encoder_type}({encoder.output_dim})")

        # Total encoded feature dimension after concatenation
        encoder_output_dims = [encoder.output_dim for encoder in self.encoders.values()]
        self.encoder_output_dim = int(sum(encoder_output_dims))

        # Extract policy_head config from actor_config
        policy_head_cfg = actor_config.policy_head
        policy_type = policy_head_cfg["type"]
        self.policy_head = build_policy_head(
            policy_type=policy_type,
            input_dim=self.encoder_output_dim,
            num_actions=num_actions,
            cfg_network=policy_head_cfg,
            device=device,
        )

        print("Actor:")
        print(f"  Input keys: {self.input_keys}")
        print(f"  Encoder output dims: {encoder_info}")
        print(f"  Total encoder dim: {self.encoder_output_dim}")
        print(f"  Policy head: {self.policy_head}")

    def encode_observations(self, observations):
        """Encode all observations and concatenate them.

        Args:
            observations: Dict of observations, e.g., {"privileged_observations": tensor, "RGB": tensor}

        Returns:
            Concatenated encoded features of shape (..., encoder_output_dim)
        """
        encoded_features = []

        for key in self.input_keys:
            obs = observations[key]

            # Encode
            encoded = self.encoders[key](obs)
            encoded_features.append(encoded)

        # Concatenate all encoded features
        features = torch.cat(encoded_features, dim=-1)

        return features

    def forward(self, observations, deterministic=False):
        """Forward pass: encode observations, then pass through policy head.

        Args:
            observations: Dict of observations
            deterministic: Whether to use deterministic policy (for stochastic policies)

        Returns:
            Actions tensor of shape (batch, num_actions)
        """
        # Encode and concatenate all observations
        encoded_features = self.encode_observations(observations)

        # Pass through policy head
        actions = self.policy_head(encoded_features)

        return actions

    def get_logstd(self):
        """Returns log std from policy head (None for deterministic policy, tensor for stochastic)."""
        return self.policy_head.get_logstd()


def build_actor(actor_config, inputs_dim, num_actions, device="cuda:0", obs_rms=None):
    """Factory function to build actors.

    Args:
        actor_config: Full actor config containing inputs and policy_head
        inputs_dim: Dict mapping input keys to their shapes (from environment)
        num_actions: Number of action dimensions
        device: Device to place model on
        obs_rms: Optional dict of RunningMeanStd modules for each input key

    Returns:
        ActorBase instance
    """

    return Actor(
        actor_config=actor_config, inputs_dim=inputs_dim, num_actions=num_actions, device=device, obs_rms=obs_rms
    )
