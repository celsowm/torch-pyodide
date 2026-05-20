import importlib


def test_public_imports_and_entrypoints():
    torch = importlib.import_module("torch")
    assert callable(torch.add)
    assert hasattr(torch, "Tensor")

    nn_functional = importlib.import_module("torch.nn.functional")
    assert callable(nn_functional.relu)

    autograd = importlib.import_module("torch.autograd")
    assert callable(autograd.grad)
