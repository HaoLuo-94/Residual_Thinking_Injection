#!/usr/bin/env python3
from typing import Any, Dict, Optional

import torch

from utils.base_evaluator import BaseMathEvaluator
from utils.math_utils import extract_answer as extract_answer_from_box, grade_answer

MATH500_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

class Math500Evaluator(BaseMathEvaluator):
    """MATH500: problem/answer jsonl, \\boxed extraction, grade_answer for LaTeX equivalence."""

    eval_name = "MATH500"

    def get_question_key(self) -> str:
        return "problem"

    def get_ground_truth_key(self) -> str:
        return "answer"

    def build_prompt(self, tokenizer, question: str, *, enable_thinking: bool = False) -> str:
        messages = [
            {"role": "system", "content": MATH500_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return rendered if isinstance(rendered, str) else str(rendered)

    def extract_answer(self, completion: str) -> Optional[str]:
        if not completion:
            return None
        raw = extract_answer_from_box(completion, use_last_number=True)
        if not raw or not raw.strip():
            return None
        return raw.strip()

    def extract_ground_truth(self, sample: Dict[str, Any]) -> Optional[str]:
        raw = self.get_ground_truth_raw(sample)
        if raw is None:
            return None
        s = str(raw).strip()
        if s == "":
            return None
        return s

    def is_correct(
        self,
        prediction: Optional[str],
        ground_truth: Optional[str],
        sample: Dict[str, Any],
    ) -> bool:
        if prediction is None or ground_truth is None:
            return False
        return grade_answer(prediction, ground_truth)


def evaluate_math500_accuracy(
    model,
    tokenizer,
    test_path: str,
    device: Optional[torch.device],
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    return_details: bool = False,
    save_results_path: Optional[str] = None,
):
    """Convenience entry point: run MATH500 evaluation via the base evaluator."""
    return Math500Evaluator().evaluate(
        model=model,
        tokenizer=tokenizer,
        test_path=test_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_details=return_details,
        save_results_path=save_results_path,
    )
