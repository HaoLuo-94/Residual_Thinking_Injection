from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class InjectionConfig:
    alpha: float = 1.0      # Global base injection strength coefficient α
    top_k: int = 8          # Top-K (legacy e_soft Top-K; used when align_sampler=False)
    enabled: bool = True
    inject_phase: str = "all"            # "all"=entire generation; "think"=thinking phase only
    think_start_id: int | None = None    # token id for <think> (think mode)
    think_end_id: int | None = None      # token id for </think> (think mode)
    inject_layer: int = -1               # Injection layer: -1=all; 0=first; k=layer k (0-indexed)

    # ---- e_soft aligned with sampler (when align_sampler=True; else legacy top_k path above) ----
    align_sampler: bool = False          # True: e_soft mirrors temperature->top_k->top_p
    soft_temperature: float = 1.0        # e_soft temperature under align (should match Soft Thinking baseline)
    soft_top_p: float = 1.0              # 1.0=disabled
    soft_top_k: int = 0                  # <=0=disabled
    soft_pool_k: int = 1024              # Candidate pool size (must be >= actual survivor set; sparse gather only)

    # ---- e_hard anchor ----
    hard_anchor: str = "committed"          # "argmax"=greedy token; "committed"=actually sampled token


# Global singleton config (model is built inside vLLM and cannot receive args directly)
CONFIG = InjectionConfig()


def load_from_env() -> None:
    """Update CONFIG from environment variables (only overrides explicitly set items)."""
    if "RESIDUAL_INJECTION_ALPHA" in os.environ:
        CONFIG.alpha = float(os.environ["RESIDUAL_INJECTION_ALPHA"])
    if "RESIDUAL_INJECTION_TOP_K" in os.environ:
        CONFIG.top_k = int(os.environ["RESIDUAL_INJECTION_TOP_K"])
    if "RESIDUAL_INJECTION_ENABLED" in os.environ:
        CONFIG.enabled = os.environ["RESIDUAL_INJECTION_ENABLED"] not in ("0", "false", "False")
    if "RESIDUAL_INJECTION_PHASE" in os.environ:
        CONFIG.inject_phase = os.environ["RESIDUAL_INJECTION_PHASE"]
    if "RESIDUAL_INJECTION_THINK_START_ID" in os.environ:
        CONFIG.think_start_id = int(os.environ["RESIDUAL_INJECTION_THINK_START_ID"])
    if "RESIDUAL_INJECTION_THINK_END_ID" in os.environ:
        CONFIG.think_end_id = int(os.environ["RESIDUAL_INJECTION_THINK_END_ID"])
    if "RESIDUAL_INJECTION_LAYER" in os.environ:
        CONFIG.inject_layer = int(os.environ["RESIDUAL_INJECTION_LAYER"])
    # ---- align_sampler related ----
    if "RESIDUAL_INJECTION_ALIGN_SAMPLER" in os.environ:
        CONFIG.align_sampler = os.environ["RESIDUAL_INJECTION_ALIGN_SAMPLER"] not in ("0", "false", "False")
    if "RESIDUAL_INJECTION_SOFT_TEMPERATURE" in os.environ:
        CONFIG.soft_temperature = float(os.environ["RESIDUAL_INJECTION_SOFT_TEMPERATURE"])
    if "RESIDUAL_INJECTION_SOFT_TOP_P" in os.environ:
        CONFIG.soft_top_p = float(os.environ["RESIDUAL_INJECTION_SOFT_TOP_P"])
    if "RESIDUAL_INJECTION_SOFT_TOP_K" in os.environ:
        CONFIG.soft_top_k = int(os.environ["RESIDUAL_INJECTION_SOFT_TOP_K"])
    if "RESIDUAL_INJECTION_SOFT_POOL_K" in os.environ:
        CONFIG.soft_pool_k = int(os.environ["RESIDUAL_INJECTION_SOFT_POOL_K"])
    # ---- hard_anchor ----
    if "RESIDUAL_INJECTION_HARD_ANCHOR" in os.environ:
        CONFIG.hard_anchor = os.environ["RESIDUAL_INJECTION_HARD_ANCHOR"]


def set_config(alpha: float | None = None, top_k: int | None = None,
               enabled: bool | None = None, inject_phase: str | None = None,
               think_start_id: int | None = None,
               think_end_id: int | None = None,
               inject_layer: int | None = None,
               align_sampler: bool | None = None,
               soft_temperature: float | None = None,
               soft_top_p: float | None = None,
               soft_top_k: int | None = None,
               soft_pool_k: int | None = None,
               hard_anchor: str | None = None) -> None:
    if alpha is not None:
        CONFIG.alpha = alpha
    if top_k is not None:
        CONFIG.top_k = top_k
    if enabled is not None:
        CONFIG.enabled = enabled
    if inject_phase is not None:
        CONFIG.inject_phase = inject_phase
    if think_start_id is not None:
        CONFIG.think_start_id = think_start_id
    if think_end_id is not None:
        CONFIG.think_end_id = think_end_id
    if inject_layer is not None:
        CONFIG.inject_layer = inject_layer
    if align_sampler is not None:
        CONFIG.align_sampler = align_sampler
    if soft_temperature is not None:
        CONFIG.soft_temperature = soft_temperature
    if soft_top_p is not None:
        CONFIG.soft_top_p = soft_top_p
    if soft_top_k is not None:
        CONFIG.soft_top_k = soft_top_k
    if soft_pool_k is not None:
        CONFIG.soft_pool_k = soft_pool_k
    if hard_anchor is not None:
        CONFIG.hard_anchor = hard_anchor
