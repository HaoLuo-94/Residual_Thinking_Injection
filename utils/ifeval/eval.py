#!/usr/bin/env python3
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.base_evaluator import BaseMathEvaluator

# Verification module from the first file. Save it under sources/
# and adjust the import path here to match the actual filename.
from .ifeval_instructions import verify_all_instructions

# Official IFEval practice: no system prompt; feed the prompt directly as a user message.
# You may set one if needed, but note that a system prompt may conflict with sample
# instructions (e.g. "write in all caps").
IFEVAL_SYSTEM_PROMPT: Optional[str] = None


class IFEvalEvaluator(BaseMathEvaluator):
    """
    IFEval: each sample has a prompt and a set of verifiable instructions
    (instruction_id_list + kwargs). is_correct is defined as
    "all instructions satisfied", so evaluate() returns
    prompt-level strict accuracy (IFEval's primary metric).
    """

    eval_name = "IFEval"
    # Instruction checks are regex/count-based and lightweight; 10s is enough.
    per_sample_timeout: float = 10.0

    system_prompt: Optional[str] = IFEVAL_SYSTEM_PROMPT

    def get_question_key(self) -> str:
        return "prompt"

    def get_ground_truth_key(self) -> str:
        # IFEval has no "ground truth" string; this points to the instruction list
        # only for display in detail records, not for scoring.
        return "instruction_id_list"

    def build_prompt(
        self, tokenizer, question: str, *, enable_thinking: bool = False
    ) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            rendered = tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        return rendered if isinstance(rendered, str) else str(rendered)

    @staticmethod
    def _strip_thinking(completion: str) -> str:
        """IFEval constrains the response body (format, word count, etc.); strip chain-of-thought before verification."""
        if not completion:
            return ""
        if "</think>" in completion:
            return completion.split("</think>")[-1].strip()
        if "<think>" in completion:
            return completion.split("<think>")[0].strip()
        return completion.strip()

    def extract_answer(self, completion: str) -> Optional[str]:
        if completion is None:
            return None
        return self._strip_thinking(completion)

    def extract_ground_truth(self, sample: Dict[str, Any]) -> Optional[str]:
        # Not used for scoring; instruction info is taken directly from sample.
        return None

    def _instructions(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        inst_ids = sample.get("instruction_id_list") or []
        kwargs_list = sample.get("kwargs") or [{} for _ in inst_ids]
        # Normalize None entries and length mismatches to dicts to avoid .items() errors in verify_all_instructions.
        kwargs_list = [(kw or {}) for kw in kwargs_list]
        if len(kwargs_list) < len(inst_ids):
            kwargs_list = kwargs_list + [{}] * (len(inst_ids) - len(kwargs_list))
        return inst_ids, kwargs_list

    def is_correct(
        self,
        prediction: Optional[str],
        ground_truth: Optional[str],
        sample: Dict[str, Any],
    ) -> bool:
        if prediction is None:
            return False
        inst_ids, kwargs_list = self._instructions(sample)
        if not inst_ids:
            return False
        all_satisfied, _ = verify_all_instructions(prediction, inst_ids, kwargs_list)
        return all_satisfied

    def get_detail_record(
        self,
        sample: Dict[str, Any],
        completion: str,
        prediction: Optional[str],
        ground_truth: Optional[str],
        is_correct: bool,
    ) -> Dict[str, Any]:
        inst_ids, kwargs_list = self._instructions(sample)
        if prediction is None:
            per_instruction = [False] * len(inst_ids)
        else:
            _, per_instruction = verify_all_instructions(
                prediction, inst_ids, kwargs_list
            )
        return {
            "prompt": self.get_question(sample),
            "instruction_id_list": inst_ids,
            "completion": completion,
            # Per-instruction satisfaction flags for offline instruction-level / loose metrics
            "follow_instruction_list": per_instruction,
            "prompt_strict_correct": is_correct,
            "instruction_num": len(inst_ids),
            "instruction_correct_num": int(sum(per_instruction)),
        }


def evaluate_ifeval_accuracy(
    model,
    tokenizer,
    test_path: str,
    device: Optional[torch.device],
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    return_details: bool = False,
    save_results_path: Optional[str] = None,
):
    """Convenience entry point: run IFEval via the base evaluator.

    Returns prompt-level strict accuracy (or (accuracy, details)).
    """
    return IFEvalEvaluator().evaluate(
        model=model,
        tokenizer=tokenizer,
        test_path=test_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_details=return_details,
        save_results_path=save_results_path,
    )
