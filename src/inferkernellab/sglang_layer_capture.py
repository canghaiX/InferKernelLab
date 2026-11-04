"""Forward hook for collecting one-token SGLang decoder diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def make_layer_capture_hook(config: dict[str, Any]):
    """Build a SGLang hook that writes one final-token vector per module call.

    The caller creates ``capture.active`` before one diagnostic request and
    removes it after the request. This keeps normal serving traffic out of the
    diagnostic files.
    """
    root = Path(config["directory"])
    root.mkdir(parents=True, exist_ok=True)
    state = {"pass": -1, "layer": 0, "norm_calls": 0}
    state["capture_id"] = None
    forced_stage = config.get("stage")
    num_layers = int(config.get("num_layers", 28))

    def hook(module, _inputs, output):
        marker = root / "capture.active"
        if not marker.exists():
            return
        capture_id = marker.read_text(encoding="utf-8")
        if capture_id != state["capture_id"]:
            state.update(
                {"pass": -1, "layer": 0, "norm_calls": 0, "capture_id": capture_id}
            )

        if forced_stage in {
            "input_norm",
            "post_attention_norm",
            "gate_up_projection",
            "down_projection",
        }:
            call = state["norm_calls"]
            state["norm_calls"] += 1
            layer = call % num_layers
            pass_id = call // num_layers
            stage = forced_stage
        elif hasattr(module, "attn") and hasattr(module.attn, "layer_id"):
            layer = int(module.attn.layer_id)
            stage = "attention"
            state["layer"] = layer
            if layer == 0:
                state["pass"] += 1
            pass_id = state["pass"]
        elif hasattr(module, "self_attn") and hasattr(module, "mlp"):
            layer = int(module.self_attn.attn.layer_id)
            stage = "block"
            pass_id = state["pass"]
        else:
            layer = int(state["layer"])
            stage = "mlp"
            pass_id = state["pass"]

        if isinstance(output, (tuple, list)):
            tensors = [item for item in output if isinstance(item, torch.Tensor)]
            if stage == "block" and len(tensors) > 1:
                value = tensors[0].float() + tensors[1].float()
                for stream_name, stream in zip(
                    ("block_hidden", "block_residual"), tensors[:2]
                ):
                    _save_vector(root, pass_id, layer, stream_name, stream)
            elif tensors:
                value = tensors[0]
            else:
                return
        elif isinstance(output, torch.Tensor):
            value = output
        else:
            return

        _save_vector(root, pass_id, layer, stage, value)

    return hook


def _save_vector(root: Path, pass_id: int, layer: int, stage: str, value: torch.Tensor):
    if value.ndim < 2 or value.numel() == 0:
        return
    vector = value.reshape(-1, value.shape[-1])[-1].detach().float().cpu()
    name = f"pass{pass_id:03d}_layer{layer:03d}_{stage}.pt"
    temporary = root / f".{name}.{os.getpid()}.tmp"
    torch.save(vector, temporary)
    temporary.replace(root / name)
