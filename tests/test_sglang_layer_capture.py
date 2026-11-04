from types import SimpleNamespace

import torch

from inferkernellab.sglang_layer_capture import make_layer_capture_hook


def test_layer_capture_hook_is_opt_in_and_combines_residual_streams(tmp_path):
    hook = make_layer_capture_hook({"directory": str(tmp_path)})
    attention = SimpleNamespace(attn=SimpleNamespace(layer_id=0))
    mlp = SimpleNamespace()
    block = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(layer_id=0)), mlp=mlp
    )
    vector = torch.tensor([[1.0, 2.0]])

    hook(attention, (), vector)
    assert not list(tmp_path.glob("*.pt"))

    (tmp_path / "capture.active").touch()
    hook(attention, (), vector)
    hook(mlp, (), vector * 2)
    hook(block, (), (vector, vector * 3))

    captured = {
        path.name: torch.load(path, map_location="cpu", weights_only=True)
        for path in tmp_path.glob("*.pt")
    }
    assert torch.equal(captured["pass000_layer000_attention.pt"], vector[0])
    assert torch.equal(captured["pass000_layer000_mlp.pt"], (vector * 2)[0])
    assert torch.equal(captured["pass000_layer000_block.pt"], (vector * 4)[0])
