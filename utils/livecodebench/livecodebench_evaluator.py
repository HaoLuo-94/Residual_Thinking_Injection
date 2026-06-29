"""LiveCodeBench evaluator that follows the OFFICIAL evaluation method.

Grading, pass@k and dataset decoding are delegated to the vendored official
modules in `lcb_official/` (copied unchanged from
https://github.com/LiveCodeBench/LiveCodeBench). This adapter only wires them
into the `BaseCodeEvaluator` interface used by the MBPP evaluator, reusing the
official prompt template and the official `extract_code` logic.

Two usage modes:
  1. Drop-in, same as MBPP -> single-sample pass@1, graded by the official
     `check_correctness`:
         evaluate_livecodebench_accuracy(model, tokenizer, test_path, device, ...)
  2. Official pass@k over n samples per problem (you generate the samples):
         score_with_codegen_metrics(items, generations_list, k_list=[1, 5])
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import pyarrow.parquet as pq

from sources.base_evaluator import BaseCodeEvaluator

# ---- Official, vendored core (DO NOT reimplement). ------------------------- #
from lcb_official import (
    CodeGenerationProblem,
    check_correctness,
    codegen_metrics,
    estimate_pass_at_k,
)

DEFAULT_TIMEOUT = 6  # official default (seconds per test)


# Official generic system message (lcb_runner/prompts/code_generation.py).
SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def official_question_template(question_content: str, starter_code: str) -> str:
    """Exact port of get_generic_question_template_answer (official)."""
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMATTING_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def official_extract_code(model_output: str) -> str:
    """Exact port of extract_code (official, generic chat-model branch):
    take the content between the LAST pair of ``` fences."""
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


class LiveCodeBenchEvaluator(BaseCodeEvaluator):
    eval_name = "LiveCodeBench"

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        self.SYSTEM_PROMPT = SYSTEM_MESSAGE_GENERIC

    # ------------------------------------------------------------------ #
    def get_sync_dummy_item(self) -> Dict[str, Any]:
        return {
            "question_content": "Read an integer n from stdin and print it.",
            "starter_code": "",
            "difficulty": "easy",
            "platform": "atcoder",
            "question_id": "dummy",
            "contest_date": "2023-05-13T00:00:00",
            "eval_sample": {
                "input_output": json.dumps(
                    {"inputs": ["1\n"], "outputs": ["1\n"], "fn_name": None}
                )
            },
        }

    # ------------------------------------------------------------------ #
    # Dataset loading. Uses the official decoding (CodeGenerationProblem),
    # so plain-JSON and base64+zlib+pickle private test cases are handled
    # identically to upstream. Each returned item is a lightweight dict that
    # carries the precomputed official `input_output` eval sample.
    #   - local .parquet path -> read it (must have the upstream columns)
    #   - otherwise -> pull livecodebench/code_generation_lite from the hub
    #     (version via LCB_VERSION, date window via LCB_START_DATE/LCB_END_DATE)
    # ------------------------------------------------------------------ #
    def load_dataset(self, test_path: str) -> List[Dict[str, Any]]:
        problems: List[CodeGenerationProblem] = []

        if test_path and Path(test_path).exists():
            rows = pq.read_table(str(test_path)).to_pylist()
            for row in rows:
                problems.append(CodeGenerationProblem(**row))
        else:
            from lcb_official import load_code_generation_dataset

            problems = load_code_generation_dataset(
                release_version=os.environ.get("LCB_VERSION", "release_latest"),
                start_date=os.environ.get("LCB_START_DATE"),
                end_date=os.environ.get("LCB_END_DATE"),
            )

        items: List[Dict[str, Any]] = []
        for p in problems:
            items.append(
                {
                    "question_content": p.question_content,
                    "starter_code": p.starter_code,
                    "difficulty": p.difficulty.value,
                    "platform": p.platform.value,
                    "question_id": p.question_id,
                    "contest_date": p.contest_date.isoformat(),
                    "eval_sample": p.get_evaluation_sample(),
                }
            )
        return items

    # ------------------------------------------------------------------ #
    def build_prompt(self, tokenizer, item: Dict[str, Any]) -> str:
        question = str(item.get("question_content") or "")
        starter = str(item.get("starter_code") or "")
        user_prompt = official_question_template(question, starter)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return rendered if isinstance(rendered, str) else str(rendered)

    # ------------------------------------------------------------------ #
    def extract_code_block(self, text: str) -> str:
        return official_extract_code(text)

    # ------------------------------------------------------------------ #
    # Grade ONE generation against ONE problem with the official checker.
    # A generation passes iff every per-test result is strictly positive
    # (this is exactly how the official `compute_metrics_from_results`
    # decides correctness: `np.all(gen > 0)`).
    # ------------------------------------------------------------------ #
    def is_correct(self, generated_code: Optional[str], item: Dict[str, Any]) -> bool:
        if not generated_code:
            return False
        sample = item["eval_sample"]
        try:
            result, _metadata = check_correctness(
                sample, generated_code, timeout=self.timeout, debug=self.debug
            )
        except Exception:
            return False
        try:
            return bool(np.all(np.asarray(result) > 0))
        except Exception:
            return False


# ============================================================================= #
# Official pass@k over multiple samples per problem.
#   items            : list from LiveCodeBenchEvaluator.load_dataset(...)
#   generations_list : list (aligned with items) of lists of code strings,
#                      i.e. generations_list[i] = [code_1, ..., code_n] for
#                      problem items[i]. Pass ALREADY-EXTRACTED code.
# Returns the official metrics dict, e.g. {"pass@1": ..., "pass@5": ..., ...}.
# ============================================================================= #
def score_with_codegen_metrics(
    items: List[Dict[str, Any]],
    generations_list: List[List[str]],
    k_list: Optional[List[int]] = None,
    num_process_evaluate: int = 16,
    timeout: int = DEFAULT_TIMEOUT,
    debug: bool = False,
):
    if k_list is None:
        k_list = [1, 5]
    samples_list = [it["eval_sample"] for it in items]
    metrics, results, metadatas = codegen_metrics(
        samples_list,
        generations_list,
        k_list=k_list,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
        debug=debug,
    )
    return metrics, results, metadatas


def grouped_pass_at_1(items, results, key: str = "difficulty") -> Dict[str, float]:
    """Break pass@1 down by `difficulty` or `platform`, using the per-instance
    grades produced by `codegen_metrics` (results[idx] = list over n samples,
    each a per-test result list)."""
    buckets: Dict[str, list] = {}
    for idx, item in enumerate(items):
        gens = results.get(idx, []) if isinstance(results, dict) else results[idx]
        correct = [1 if np.all(np.asarray(g) > 0) else 0 for g in gens]
        n, c = len(correct), sum(correct)
        if n == 0:
            continue
        p1 = float(estimate_pass_at_k([n], [c], 1)[0])
        buckets.setdefault(str(item.get(key)), []).append(p1)
    return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}


# ============================================================================= #
# Drop-in entry point, mirroring evaluate_mbpp_accuracy. Single sample per
# problem -> official-faithful pass@1.
# ============================================================================= #
def evaluate_livecodebench_accuracy(
    model,
    tokenizer,
    test_path: str,
    device: Optional[torch.device],
    batch_size: int = 1,
    max_new_tokens: int = 4096,
    return_details: bool = False,
    save_results_path: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
):
    return LiveCodeBenchEvaluator(timeout=timeout).evaluate(
        model=model,
        tokenizer=tokenizer,
        test_path=test_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_details=return_details,
        save_results_path=save_results_path,
    )
