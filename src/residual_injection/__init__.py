"""Residual Injection for vLLM (V1)."""
from .api import enable_residual_injection
from .arch import hf_overrides_for_model, resolve_residual_architecture
from .config import set_config, CONFIG
from .plugin import register

__all__ = [
    "enable_residual_injection",
    "register",
    "set_config",
    "CONFIG",
    "resolve_residual_architecture",
    "hf_overrides_for_model",
]
__version__ = "0.1.0"
