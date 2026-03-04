import os

import hydra
import onnx
import rl_games.algos_torch.flatten as flatten
import torch
from omegaconf import DictConfig, OmegaConf

from agents.rl_games import make_runner


class ModelWrapper(torch.nn.Module):
    """
    Main idea is to ignore outputs which we don't need from model
    """

    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model

    def forward(self, input_dict):
        input_dict["obs"] = self._model.norm_obs(input_dict["obs"])
        """
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        """
        return self._model.a2c_network(input_dict)


@hydra.main(version_base=None, config_path="../../../cfgs", config_name="config")
def main(config: DictConfig):
    # Resolve all interpolations in the config (resolves ${...} references)
    OmegaConf.resolve(config)

    runner = make_runner(config)

    agent = runner.create_player()
    agent.restore(config["checkpoint"])
    # agent.model.eval()  # Avoid RunningMeanStd updating buffers during trace
    inputs = {
        "obs": torch.zeros((1,) + agent.obs_shape).to(agent.device),
        "rnn_states": agent.states,
    }

    with torch.no_grad():
        adapter = flatten.TracingAdapter(ModelWrapper(agent.model), inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)

    # Get the directory where this script is located (go2_hardware/rlgames)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, "go2_walking.onnx")

    # Use legacy TorchScript exporter: TS2EPConverter fails on prim::If with prim::SetAttr
    torch.onnx.export(
        traced,
        adapter.flattened_inputs,
        onnx_path,
        verbose=True,
        input_names=["obs"],
        output_names=["mu", "log_std", "value"],
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_path)

    # Check that the model is well formed
    onnx.checker.check_model(onnx_model)

    print(f"🚀 ONNX model saved to {onnx_path}")

    return


if __name__ == "__main__":
    main()
