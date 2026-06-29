"""Resolve evaluators in utils by DATASET_NAME."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from utils.base_evaluator import BaseCodeEvaluator, BaseMathEvaluator

# alias -> evaluator class (lazy-loaded to avoid unused dependencies slowing startup)
_EVALUATOR_REGISTRY: Dict[str, str] = {
    "aime": "utils.aime.eval:Aime23_25Evaluator",
    "aime2025": "utils.aime.eval:Aime23_25Evaluator",
    "aime2325": "utils.aime.eval:Aime23_25Evaluator",
    "aime23_25": "utils.aime.eval:Aime23_25Evaluator",
    "aime23-25": "utils.aime.eval:Aime23_25Evaluator",
    "math500": "utils.math500.eval:Math500Evaluator",
    "minerva": "utils.minerva.eval:MinervaEvaluator",
    "mbpp": "utils.mbpp.eval:MBPPEvaluator",
    "humaneval": "utils.humaneval.eval:HumanEvalEvaluator",
    "livecodebench": "utils.livecodebench.eval:LiveCodeBenchEvaluator",
    # --- IFEval (instruction following) ---
    # IFEval reuses BaseCodeEvaluator's generate/eval loop, so is_code_evaluator()
    # returns True for it; use is_ifeval_evaluator() below to distinguish.
    "ifeval": "utils.ifeval.eval:IFEvalEvaluator",
    "if_eval": "utils.ifeval.eval:IFEvalEvaluator",
    "if-eval": "utils.ifeval.eval:IFEvalEvaluator",
    "instruction_following": "utils.ifeval.eval:IFEvalEvaluator",
}


def _import_class(dotted: str) -> Type[Any]:
    module_path, _, class_name = dotted.partition(":")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def resolve_evaluator_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    return key if key in _EVALUATOR_REGISTRY else None


def get_evaluator(dataset_name: Optional[str]) -> Optional[Any]:
    """Return an evaluator instance for the dataset name; None for unknown names."""
    key = resolve_evaluator_name(dataset_name)
    if key is None:
        return None
    cls = _import_class(_EVALUATOR_REGISTRY[key])
    return cls()


def is_math_evaluator(evaluator: Any) -> bool:
    return isinstance(evaluator, BaseMathEvaluator)


def is_code_evaluator(evaluator: Any) -> bool:
    return isinstance(evaluator, BaseCodeEvaluator)


def is_ifeval_evaluator(evaluator: Any) -> bool:
    """IFEval-specific check: subclass of BaseCodeEvaluator but semantically instruction following.

    Uses eval_name to avoid importing IFEvalEvaluator (keeps lazy loading).
    For strict/loose x prompt/instruction four metrics, call with return_details=True
    and then evaluator.summarize(details) when is_ifeval_evaluator is True.
    """
    return getattr(evaluator, "eval_name", None) == "IFEval"
