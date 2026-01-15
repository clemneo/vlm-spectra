import torch


def test_model_not_on_cpu(model):
    param_device = next(model.model.parameters()).device
    print(f"Model parameters are on {param_device}")
    assert (
        param_device.type != "cpu"
    ), f"Model parameters are on CPU: {param_device}"
