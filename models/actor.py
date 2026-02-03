import numpy as np
import torch
import torch.nn as nn

from models.encoder import build_encoder
from utils.model_utils import get_activation_func, init_module


class PolicyHeadBase(nn.Module):
    """Base for policy heads: encoded features -> dict with "mean" and "std"."""

    def __init__(self, input_dim, num_actions, device="cuda:0"):
        super(PolicyHeadBase, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = device

    def forward(self, features):
        """Returns dict of tensors (batch, num_actions): "mean", "std"."""
        raise NotImplementedError


class MLPPolicyHead(PolicyHeadBase):
    """MLP policy head: forward returns {"mean", "log_std"}.

    fixed_std True: use log_std_init as constant log_std (non-learnable).
    fixed_std False: state_dependent_std -> 2*num_actions split, else nn.Parameter log_std (init log_std_init).
    """

    def __init__(self, input_dim, num_actions, cfg_network, device="cuda:0"):
        super(MLPPolicyHead, self).__init__(input_dim, num_actions, device)

        fixed_std = cfg_network.get("fixed_std")
        learn = not fixed_std
        dep = cfg_network.get("state_dependent_std", False)
        init_val = cfg_network.get("log_std_init", 0.0)
        bounds = cfg_network.get("log_std_bounds", None)

        nc = cfg_network.get("network", cfg_network)
        units = list(nc["units"])
        act = nc["activation"]
        out_dim = units[-1] if units else input_dim

        if learn and dep:
            dims = [input_dim] + units
            mods = []
            for i in range(len(dims) - 1):
                lin = nn.Linear(dims[i], dims[i + 1])
                init_module(lin, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
                mods.append(lin)
                if i < len(dims) - 2:
                    mods.append(get_activation_func(act))
                    mods.append(nn.LayerNorm(dims[i + 1]))
            self.trunk = nn.Sequential(*mods).to(device) if mods else nn.Identity()
            self.out = nn.Linear(out_dim, 2 * num_actions).to(device)
            init_module(self.out, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
            nn.init.constant_(self.out.bias[num_actions:], init_val)
            self.mlp = None
            self.log_std = None
        else:
            dims = [input_dim] + units + [num_actions]
            mods = []
            for i in range(len(dims) - 1):
                lin = nn.Linear(dims[i], dims[i + 1])
                init_module(lin, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
                mods.append(lin)
                if i < len(dims) - 2:
                    mods.append(get_activation_func(act))
                    mods.append(nn.LayerNorm(dims[i + 1]))
            self.mlp = nn.Sequential(*mods).to(device)
            self.trunk = None
            self.out = None
            if learn:
                self.log_std = nn.Parameter(torch.full((num_actions,), init_val, dtype=torch.float32, device=device))
            else:
                self.log_std = None

        self._learn_std = learn
        self._state_dependent_std = dep
        self._log_std_bounds = bounds
        self._fixed_log_std = float(init_val)

    def forward(self, x):
        """Returns {"mean": (B, A), "log_std": (B, A)}."""
        if self._learn_std and self._state_dependent_std:
            h = self.trunk(x)
            o = self.out(h)
            m, l = o.chunk(2, dim=-1)
            if self._log_std_bounds is not None:
                l = torch.clamp(l, self._log_std_bounds[0], self._log_std_bounds[1])
            return {"mean": m, "log_std": l}
        m = self.mlp(x)
        if self._learn_std:
            l = self.log_std
            if self._log_std_bounds is not None:
                l = torch.clamp(l, self._log_std_bounds[0], self._log_std_bounds[1])
            l = l.unsqueeze(0).expand_as(m)
        else:
            l = torch.full_like(m, self._fixed_log_std, device=m.device, dtype=m.dtype)
        return {"mean": m, "log_std": l}


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

    if policy_type == "mlp":
        return MLPPolicyHead(input_dim, num_actions, cfg_network, device=device)
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

    def forward(self, observations):
        """Encode observations and policy head forward. Returns dict with "mean" and "std" (batch, num_actions)."""
        f = self.encode_observations(observations)
        return self.policy_head(f)


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
