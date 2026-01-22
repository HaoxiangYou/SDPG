import numpy as np
import torch
import torch.nn as nn

from models.encoder import build_encoder
from utils.model_utils import get_activation_func, init_module


class Critic(nn.Module):
    """Critic with encoder and value_head trunks.

    The encoder trunk encodes all input observations and concatenates them.
    The value_head trunk processes the concatenated features to output a scalar value.
    """

    def __init__(self, critic_config, inputs_dim, device="cuda:0"):
        """
        Args:
            critic_config: Full critic config containing inputs and value_head
            inputs_dim: Dict mapping input keys to their shapes (from environment)
            device: Device to place model on
        """
        super(Critic, self).__init__()

        self.device = device

        # Extract input keys and configs from critic_config
        self.input_keys = [input.name for input in critic_config.inputs]
        inputs_config = {input.name: input for input in critic_config.inputs}

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

        # Extract value_head config from critic_config
        value_head_cfg = critic_config.value_head
        value_type = value_head_cfg["type"]

        # Build value head
        if value_type == "mlp":
            network_cfg = value_head_cfg["network"]
            units = list(network_cfg["units"])
            activation = network_cfg["activation"]

            layer_dims = [self.encoder_output_dim] + units + [1]
            modules = []
            for i in range(len(layer_dims) - 1):
                linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
                init_module(linear_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
                modules.append(linear_layer)

                if i < len(layer_dims) - 2:
                    modules.append(get_activation_func(activation))
                    modules.append(nn.LayerNorm(layer_dims[i + 1]))

            self.value_head = nn.Sequential(*modules).to(device)
        else:
            raise NotImplementedError(f"Value head type {value_type} not implemented")

        print("Critic:")
        print(f"  Input keys: {self.input_keys}")
        print(f"  Encoder output dims: {encoder_info}")
        print(f"  Total encoder dim: {self.encoder_output_dim}")
        print(f"  Value head: {self.value_head}")

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

    def forward(self, observations) -> torch.Tensor:
        """Forward pass: encode observations, then pass through value head.

        Args:
            observations: Dict of observations

        Returns:
            Value tensor of shape (batch, 1)
        """
        # Encode and concatenate all observations
        encoded_features = self.encode_observations(observations)

        # Pass through value head
        values = self.value_head(encoded_features)

        return values


def build_critic(critic_config, inputs_dim, device="cuda:0"):
    """Factory function to build critics.

    Args:
        critic_config: Full critic config containing inputs and value_head
        inputs_dim: Dict mapping input keys to their shapes (from environment)
        device: Device to place model on

    Returns:
        Critic instance
    """
    return Critic(
        critic_config=critic_config,
        inputs_dim=inputs_dim,
        device=device,
    )
