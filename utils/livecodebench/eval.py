"""LiveCodeBench evaluator for run_batch / utils.registry."""
from __future__ import annotations
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from utils.base_evaluator import BaseCodeEvaluator
from utils.livecodebench.code_generation_benchmark import CodeGenerationProblem
from utils.livecodebench.compute_code_generation_metrics import check_correctness
DEFAULT_TIMEOUT = 6
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
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
def _normalize_contest_date(value: Any) -> str:
    if isinstance(value, (int, float)):
        sec = float(value) / 1000.0 if float(value) > 1e12 else float(value)
        return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
def _normalize_lcb_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "contest_date" in out:
        out["contest_date"] = _normalize_contest_date(out["contest_date"])
    meta = out.get("metadata")
    if isinstance(meta, dict):
        out["metadata"] = json.dumps(meta, ensure_ascii=False)
    elif meta is None:
        out["metadata"] = "{}"
    return out
def _problem_to_item(problem: CodeGenerationProblem) -> Dict[str, Any]:
    return {
        "question_content": problem.question_content,
        "starter_code": problem.starter_code,
        "difficulty": problem.difficulty.value,
        "platform": problem.platform.value,
        "question_id": problem.question_id,
        "contest_date": problem.contest_date.isoformat(),
        "eval_sample": problem.get_evaluation_sample(),
    }
class LiveCodeBenchEvaluator(BaseCodeEvaluator):
    eval_name = "LiveCodeBench"
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        self.SYSTEM_PROMPT = SYSTEM_MESSAGE_GENERIC

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
    def load_dataset(self, test_path: str) -> List[Dict[str, Any]]:
        path = Path(test_path)
        if not path.is_file():
            raise FileNotFoundError(f"LiveCodeBench data file not found: {test_path}")
        problems: List[CodeGenerationProblem] = []
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = _normalize_lcb_row(json.loads(line))
                    problems.append(CodeGenerationProblem(**row))
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ImportError(
                    "Reading .parquet requires pyarrow; run: pip install pyarrow"
                ) from e
            for row in pq.read_table(str(path)).to_pylist():
                problems.append(CodeGenerationProblem(**_normalize_lcb_row(row)))
        else:
            raise ValueError(
                f"Unsupported LiveCodeBench format {suffix!r}. Use .jsonl or .parquet"
            )
        return [_problem_to_item(p) for p in problems]
    def build_prompt(
        self, tokenizer, item: Dict[str, Any], *, enable_thinking: bool = False
    ) -> str:
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
            enable_thinking=enable_thinking,
        )
        return rendered if isinstance(rendered, str) else str(rendered)
    def extract_code_block(self, text: str) -> str:
        code = official_extract_code(text)
        if code.strip():
            return code
        matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        match = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            return re.sub(r"```$", "", code).strip()
        return ""
    def build_test_code(self, item: Dict[str, Any], generated_code: str):
        return item["eval_sample"], generated_code
    def run_test(self, test_data) -> Dict[str, Any]:
        sample, generated_code = test_data
        try:
            result, _metadata = check_correctness(
                sample, generated_code, timeout=self.timeout, debug=self.debug
            )
            passed = bool(np.all(np.asarray(result) > 0))
            return {"pass": passed, "error": None}
        except Exception:
            return {"pass": False, "error": traceback.format_exc()}
    def is_correct(self, generated_code: Optional[str], item: Dict[str, Any]) -> bool:
        if not generated_code:
            return False
        # check_correctness already runs in a subprocess; do not wrap with run_with_process_timeout,
        # otherwise nested fork + multiprocessing yields no child result and always scores wrong.
        result = self.run_test((item["eval_sample"], generated_code))
        return bool(result.get("pass"))