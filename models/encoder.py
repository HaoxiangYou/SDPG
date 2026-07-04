import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

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

        # Calculate input dimension (use Python int for OmegaConf compatibility)
        if isinstance(input_shape, tuple):
            input_dim = int(np.prod(input_shape))
        else:
            input_dim = int(input_shape)

        # Get encoder config (convert to Python ints so OmegaConf never sees numpy/torch scalars)
        units = [int(u) for u in encoder_cfg.get("units", [])]
        activation = encoder_cfg.get("activation", "elu")
        output_dim = int(encoder_cfg.get("output_dim", units[-1] if units else input_dim))

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
    """CNN encoder for image observations with flexible architecture.

    Supports configurable:
    - Conv layers (channels, kernels, strides, padding, initialization)
    - Output projection (Linear or Sequential with LayerNorm/Tanh)
    - Input normalization
    """

    def __init__(self, input_shape, encoder_cfg, device="cuda:0"):
        super(CNNEncoder, self).__init__()
        self.device = device

        # Get encoder config
        output_dim = encoder_cfg.get("output_dim", 256)
        channels = encoder_cfg.get("channels", [32, 64, 64])
        kernels = encoder_cfg.get("kernels", [3, 3, 3])
        strides = encoder_cfg.get("strides", [2, 2, 2])
        paddings = encoder_cfg.get("paddings", None)  # If None, use kernel // 2
        activation = encoder_cfg.get("activation", "relu")
        conv_init_gain = encoder_cfg.get("conv_init_gain", np.sqrt(2))

        # Output projection config
        projection_type = encoder_cfg.get("projection_type", "linear")  # "linear" or "sequential"
        projection_init_gain = encoder_cfg.get("projection_init_gain", np.sqrt(2))
        use_layernorm = encoder_cfg.get("use_layernorm", False)
        use_tanh = encoder_cfg.get("use_tanh", False)

        # Input shape: (C, H, W) or (batch, C, H, W)
        if isinstance(input_shape, tuple):
            in_channels = input_shape[0]
        else:
            in_channels = input_shape

        # Build CNN
        modules = []
        current_channels = in_channels
        for i, (out_channels, kernel, stride) in enumerate(zip(channels, kernels, strides, strict=False)):
            # Determine padding
            if paddings is not None and i < len(paddings):
                padding = paddings[i]
            else:
                padding = kernel // 2

            conv = nn.Conv2d(current_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
            init_module(conv, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=conv_init_gain)
            modules.append(conv)
            modules.append(get_activation_func(activation))
            current_channels = out_channels

        self.convnet = nn.Sequential(*modules).to(device)

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                in_channels,
                input_shape[1] if len(input_shape) > 1 else 64,
                input_shape[2] if len(input_shape) > 2 else 64,
                device=device,
            )
            dummy_output = self.convnet(dummy_input)
            repr_dim = dummy_output.flatten(start_dim=1).shape[1]

        self.repr_dim = repr_dim

        # Build output projection
        if projection_type == "sequential" or use_layernorm or use_tanh:
            projection_modules = []
            linear = nn.Linear(repr_dim, output_dim)
            init_module(linear, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=projection_init_gain)
            projection_modules.append(linear)

            if use_layernorm:
                projection_modules.append(nn.LayerNorm(output_dim))
            if use_tanh:
                projection_modules.append(nn.Tanh())

            self.output_projection = nn.Sequential(*projection_modules).to(device)
        else:
            # Simple linear projection
            self.output_projection = nn.Linear(repr_dim, output_dim).to(device)
            init_module(
                self.output_projection,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                gain=projection_init_gain,
            )

        self.output_dim = output_dim

    def forward(self, x):
        # Normalize input if the input is uint8
        if x.dtype == torch.uint8:
            x = x / 255.0 - 0.5

        # Forward through convnet
        h = self.convnet(x)

        h = h.flatten(start_dim=-3)

        # Forward through output projection
        h = self.output_projection(h)

        return h


class Drqv2Encoder(CNNEncoder):
    """Drqv2-style encoder that inherits from CNNEncoder.

    Architecture:
    - Conv2d(in_channels, 32, 3, stride=2, padding=0) -> ReLU
    - Conv2d(32, 32, 3, stride=1, padding=1) -> ReLU (x3)
    - Projection: Linear(repr_dim, output_dim) -> LayerNorm -> Tanh
    - Input normalization: x / 255.0 - 0.5
    """

    def __init__(self, input_shape, encoder_cfg, device="cuda:0"):
        # Configure Drqv2-specific parameters
        # Convert OmegaConf DictConfig to regular dict if needed
        if encoder_cfg is None:
            drqv2_cfg = {}
        elif isinstance(encoder_cfg, dict) and not isinstance(encoder_cfg, DictConfig):
            drqv2_cfg = encoder_cfg.copy()
        else:
            # Convert OmegaConf DictConfig to regular dict
            drqv2_cfg = OmegaConf.to_container(encoder_cfg, resolve=True)
            if drqv2_cfg is None:
                drqv2_cfg = {}

        # Drqv2 architecture: all convs use padding=0 (default)
        # Conv1: stride=2, padding=0 → (84-3)/2+1 = 41
        # Conv2-4: stride=1, padding=0 → reduces by 2 each: 41→39→37→35
        drqv2_cfg["channels"] = [32, 32, 32, 32]
        drqv2_cfg["kernels"] = [3, 3, 3, 3]
        drqv2_cfg["strides"] = [2, 1, 1, 1]
        drqv2_cfg["paddings"] = [0, 0, 0, 0]  # All convs use padding=0 (default)
        drqv2_cfg["activation"] = "relu"
        drqv2_cfg["conv_init_gain"] = np.sqrt(2)

        # Output projection: Linear -> LayerNorm -> Tanh
        drqv2_cfg["projection_type"] = "sequential"
        drqv2_cfg["projection_init_gain"] = 1.0  # Drqv2 uses gain=1.0 for linear
        drqv2_cfg["use_layernorm"] = True
        drqv2_cfg["use_tanh"] = True

        # Call parent constructor with Drqv2 configuration
        super(Drqv2Encoder, self).__init__(input_shape, drqv2_cfg, device=device)


class DepthEncoder(CNNEncoder):
    """Extreme-parkour-style depth encoder (DepthOnlyFCBackbone58x87): 1×58×87 → latent.

    Inherits ``forward`` / projection layout from :class:`CNNEncoder`; fixed conv + pool backbone matches
    ``externals/extreme-parkour``. Expects ``x`` of shape (batch, 1, 58, 87) or (batch, 58, 87) with
    preprocessed depth in ~[-0.5, 0.5] (e.g. ``go2_terrain`` pipeline).

    Config (same style as other encoders): optional nested ``network``; ``output_dim`` overrides latent size
    (default 32). Other CNNEncoder conv/projection keys are ignored.
    """

    def __init__(self, input_shape, encoder_cfg, device="cuda:0"):
        nn.Module.__init__(self)
        self.device = device
        if encoder_cfg is None:
            net: dict = {}
        elif isinstance(encoder_cfg, dict):
            net = dict(encoder_cfg)
        elif isinstance(encoder_cfg, DictConfig):
            net = OmegaConf.to_container(encoder_cfg, resolve=True) or {}
        else:
            net = {}

        if "network" in net:
            net = net.get("network", {}) or {}

        output_dim = int(net.get("output_dim", 32))

        if isinstance(input_shape, tuple) and len(input_shape) == 3:
            in_ch = int(input_shape[0])
        else:
            in_ch = 1

        # Match externals/extreme-parkour DepthOnlyFCBackbone58x87 (no padding on convs).
        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
        ).to(device)

        repr_dim = 64 * 25 * 39
        self.repr_dim = repr_dim
        self.output_projection = nn.Sequential(
            nn.Linear(repr_dim, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        ).to(device)

        for module in (self.convnet, self.output_projection):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init_module(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
                elif isinstance(m, nn.Linear):
                    init_module(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"DepthEncoder expects (B,H,W) or (B,C,H,W); got shape {tuple(x.shape)}")
        return CNNEncoder.forward(self, x)

def build_encoder(input_name, input_shape, encoder_cfg, device="cuda:0"):
    """Build an encoder based on config.

    Args:
        input_name: Name of the input (for debugging)
        input_shape: Shape of the input (tuple or int)
        encoder_cfg: Encoder config (None for identity, or dict with type and network/params)
                     If network key exists, encoder-specific params are under network
        device: Device to place encoder on

    Returns:
        Encoder module
    """
    if encoder_cfg is None:
        return IdentityEncoder(input_shape, device=device)

    encoder_type = encoder_cfg.get("type", "mlp").lower()

    # Extract network config if it exists, otherwise use encoder_cfg directly (backward compatibility)
    if "network" in encoder_cfg:
        network_cfg = encoder_cfg.get("network", {})
    else:
        network_cfg = encoder_cfg

    if encoder_type == "identity":
        return IdentityEncoder(input_shape, device=device)
    elif encoder_type == "mlp":
        return MLPEncoder(input_shape, network_cfg, device=device)
    elif encoder_type == "cnn":
        return CNNEncoder(input_shape, network_cfg, device=device)
    elif encoder_type == "drqv2":
        return Drqv2Encoder(input_shape, network_cfg, device=device)
    elif encoder_type == "depth":
        return DepthEncoder(input_shape, network_cfg, device=device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type} for input {input_name}")
