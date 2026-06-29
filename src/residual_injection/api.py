"""Single-process entry: call once before importing vllm / constructing LLM."""
from __future__ import annotations

import os

from .config import set_config


def enable_residual_injection(alpha: float = 1.0, top_k: int = 8,
                              force_single_process: bool = True) -> None:
    """
    Simplest way to enable (no pip install of this package required).

    force_single_process=True sets VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs
    in the same process as the caller, ensuring registration and monkeypatch take effect.
    If the package is pip-installed (plugin entry), skip this function and use env vars.
    """
    set_config(alpha=alpha, top_k=top_k, enabled=True)

    if force_single_process:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from .model import register_residual_models
    from .patch import patch_runner

    register_residual_models()
    patch_runner()
