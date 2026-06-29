"""Select the vLLM architecture name for residual injection from HF config."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_hf_config(model: str) -> dict[str, Any] | None:
    path = Path(model)
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                return json.load(f)
    return None


def resolve_residual_architecture(model: str) -> str:
    """
    Return the class name to write into hf_overrides["architectures"].

    Currently supported:
      - Llama family -> LlamaForResidualInjection
      - Qwen3        -> Qwen3ForResidualInjection
    """
    config = _load_hf_config(model)
    if config is None:
        return "LlamaForResidualInjection"

    model_type = config.get("model_type", "")
    archs = config.get("architectures") or []
    arch_str = " ".join(archs)

    if model_type == "qwen3" or "Qwen3" in arch_str:
        return "Qwen3ForResidualInjection"
    return "LlamaForResidualInjection"


def hf_overrides_for_model(model: str, *, arch: str | None = None) -> dict[str, list[str]]:
    name = arch or resolve_residual_architecture(model)
    return {"architectures": [name]}
