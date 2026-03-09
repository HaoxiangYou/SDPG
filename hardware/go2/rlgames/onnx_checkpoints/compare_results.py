import os

import hydra
import numpy as np
import onnxruntime as ort
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
    # Resolve interpolations, build runner and agent
    OmegaConf.resolve(config)
    runner = make_runner(config)
    agent = runner.create_player()
    agent.restore(config["checkpoint"])

    device = agent.device

    # Build a deterministic test observation (you can change this)
    torch.manual_seed(0)
    obs = torch.randn((1,) + agent.obs_shape, device=device)
    rnn_states = agent.states  # same initial RNN state as used for export

    # ----- PyTorch / rl_games output -----
    inputs = {
        "obs": obs.clone(),  # clone so we don't accidentally modify it
        "rnn_states": rnn_states,
    }

    with torch.no_grad():
        wrapped_model = ModelWrapper(agent.model).to(device)
        outputs_pt = wrapped_model(inputs)

        # a2c_network may return multiple tensors; action mean is the first
        if isinstance(outputs_pt, (tuple, list)):
            mu_pt = outputs_pt[0]
        elif isinstance(outputs_pt, dict) and "mu" in outputs_pt:
            mu_pt = outputs_pt["mu"]
        else:
            mu_pt = outputs_pt

        mu_pt = mu_pt.detach().cpu().numpy()

    # ----- ONNX output -----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, "go2_walking.onnx")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = sess.get_inputs()[0].name  # "obs" from export

    onnx_outputs = sess.run(
        None,
        {onnx_input_name: obs.detach().cpu().numpy()},
    )
    mu_onnx = onnx_outputs[0]

    # ----- Compare -----
    def compare(name, a, b, rtol=1e-4, atol=1e-5):
        same = np.allclose(a, b, rtol=rtol, atol=atol)
        diff = np.max(np.abs(a - b))
        print(f"{name}: allclose={same}, max_abs_diff={diff}")
        return same

    ok_mu = compare("mu", mu_pt, mu_onnx)

    if ok_mu:
        print("✅ ONNX policy action output matches rl_games model on this test input.")
    else:
        print("❌ Mismatch between ONNX and rl_games action outputs.")


if __name__ == "__main__":
    main()
