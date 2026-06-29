"""vLLM plugin entry point (multiprocess mode).

Registered via pyproject.toml [project.entry-points."vllm.general_plugins"].
vLLM calls register() in *every* process (including engine worker subprocesses) at
startup, so this works even with V1 multiprocessing enabled. Configuration is passed
via environment variables (see config.py).

register() must be reentrant (may be called multiple times).
"""
from __future__ import annotations


def register() -> None:
    from .config import load_from_env
    from .model import register_residual_models
    from .patch import patch_runner

    load_from_env()
    register_residual_models()
    patch_runner()
