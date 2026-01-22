import numpy as np
import torch
import torch.nn as nn

from utils.model_utils import get_activation_func, init_module


class IdentityEncoder(nn.Module):
    """Pass-through encoder (no encoding)."""

    def __init__(self, input_shape, device="cuda:0"):
        super(IdentityEncoder, self).__init__()
        self.input_shape = input_shape
        if isinstance(input_shape, tuple):
            self.output_dim = int(input_shape[-1])
        else:
            self.output_dim = int(input_shape)
        self.device = device

    def forward(self, x):
        return x


class MLPEncoder(nn.Module):
    """MLP encoder for vector observations."""

    def __init__(self, input_shape, encoder_cfg, device="cuda:0"):
        super(MLPEncoder, self).__init__()
        self.device = device

        # Calculate input dimension
        if isinstance(input_shape, tuple):
            input_dim = np.prod(input_shape)
        else:
            input_dim = input_shape

        # Get encoder config
        units = encoder_cfg.get("units", [])
        activation = encoder_cfg.get("activation", "elu")
        output_dim = encoder_cfg.get("output_dim", units[-1] if units else input_dim)

        # Build MLP
        layer_dims = [input_dim] + units
        modules = []
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            init_module(linear_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
            modules.append(linear_layer)

            if i < len(layer_dims) - 1:
                modules.append(get_activation_func(activation))
                modules.append(nn.LayerNorm(layer_dims[i + 1]))

        # Output projection if needed
        if output_dim != layer_dims[-1]:
            output_layer = nn.Linear(layer_dims[-1], output_dim)
            init_module(output_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
            modules.append(output_layer)

        self.encoder = nn.Sequential(*modules).to(device)
        self.output_dim = output_dim

    def forward(self, x):
        # Flatten spatial dimensions if needed
        if len(x.shape) > 2:
            x = x.flatten(start_dim=1)
        return self.encoder(x)


class CNNEncoder(nn.Module):
    """CNN encoder for image observations."""

    def __init__(self, input_shape, encoder_cfg, device="cuda:0"):
        super(CNNEncoder, self).__init__()
        self.device = device

        # Get encoder config
        channels = encoder_cfg.get("channels", [32, 64, 64])
        kernels = encoder_cfg.get("kernels", [3, 3, 3])
        strides = encoder_cfg.get("strides", [2, 2, 2])
        activation = encoder_cfg.get("activation", "relu")
        output_dim = encoder_cfg.get("output_dim", 256)

        # Input shape: (C, H, W) or (batch, C, H, W)
        if isinstance(input_shape, tuple):
            in_channels = input_shape[0]
        else:
            in_channels = input_shape

        # Build CNN
        modules = []
        current_channels = in_channels
        for out_channels, kernel, stride in zip(channels, kernels, strides, strict=False):
            conv = nn.Conv2d(current_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
            init_module(conv, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
            modules.append(conv)
            modules.append(get_activation_func(activation))
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*modules).to(device)

        # Calculate flattened size (approximate, will be computed dynamically)
        # For now, use a dummy forward to compute output size
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                in_channels,
                input_shape[1] if len(input_shape) > 1 else 64,
                input_shape[2] if len(input_shape) > 2 else 64,
            )
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.flatten(start_dim=1).shape[1]

        # Output projection
        self.fc = nn.Linear(flattened_size, output_dim).to(device)
        init_module(self.fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
        self.output_dim = output_dim

    def forward(self, x):
        # x shape: (batch, C, H, W)
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def build_encoder(input_name, input_shape, encoder_cfg, device="cuda:0"):
    """Build an encoder based on config.

    Args:
        input_name: Name of the input (for debugging)
        input_shape: Shape of the input (tuple or int)
        encoder_cfg: Encoder config (None for identity, or dict with type and params)
        device: Device to place encoder on

    Returns:
        Encoder module
    """
    if encoder_cfg is None:
        return IdentityEncoder(input_shape, device=device)

    encoder_type = encoder_cfg.get("type", "mlp").lower()

    if encoder_type == "identity":
        return IdentityEncoder(input_shape, device=device)
    elif encoder_type == "mlp":
        return MLPEncoder(input_shape, encoder_cfg, device=device)
    elif encoder_type == "cnn":
        return CNNEncoder(input_shape, encoder_cfg, device=device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type} for input {input_name}")
