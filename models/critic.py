import numpy as np
import torch
import torch.nn as nn

from utils.model_utils import get_activation_func, init_module


class CriticMLP(nn.Module):
    def __init__(self, num_observations, cfg_network, device="cuda:0"):
        super(CriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [num_observations] + cfg_network["mlp"]["units"] + [1]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            # Initialize linear layer with orthogonal weights and zero bias
            linear_layer = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            init_module(linear_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
            modules.append(linear_layer)

            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(cfg_network["mlp"]["activation"]))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)

        self.obs_dim = num_observations

        print("Critic: ", self.critic)

    def forward(self, observations):
        return self.critic(observations)
