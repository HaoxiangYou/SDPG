import torch.nn as nn


def init_module(module, weight_init, bias_init, gain=1):
    """
    Initialize a module's weights and bias (if present).

    Args:
        module: The module to initialize (e.g., nn.Linear)
        weight_init: Function to initialize weights (e.g., nn.init.orthogonal_)
        bias_init: Function to initialize bias (e.g., lambda x: nn.init.constant_(x, 0))
        gain: Gain factor for weight initialization

    Returns:
        The initialized module
    """
    # Initialize weights if the module has them
    if hasattr(module, "weight") and module.weight is not None:
        weight_init(module.weight.data, gain=gain)

    # Initialize bias if the module has it
    if hasattr(module, "bias") and module.bias is not None:
        bias_init(module.bias.data)

    return module


def get_activation_func(activation_name):
    if activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "identity":
        return nn.Identity()
    else:
        raise NotImplementedError("Actication func {} not defined".format(activation_name))
