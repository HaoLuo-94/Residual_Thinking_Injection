import traceback
from typing import Any, Dict, Optional
import re
import sys
from io import StringIO

import torch

from utils.base_evaluator import BaseCodeEvaluator


class HumanEvalEvaluator(BaseCodeEvaluator):
    eval_name = "HumanEval"

    def __init__(self):
        self.SYSTEM_PROMPT = (
            "You are a helpful assistant that solves Python coding problems. "
            "You are given a function signature and you need to complete the function. "
            "Only provide code without explanations unless asked."
        )

    def get_sync_dummy_item(self) -> Dict[str, Any]:
        return {
            "prompt": "def _sync_placeholder():\n    return 0\n",
            "test": "",
            "entry_point": "_sync_placeholder",
            "canonical_solution": "",
        }

    def build_prompt(self, tokenizer, item: Dict[str, Any], *, enable_thinking: bool = False) -> str:
        question = item.get("prompt", "")
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return rendered if isinstance(rendered, str) else str(rendered)

    def strip_thinking(self, text: str) -> str:
        return re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    def extract_code_block(self, text: str) -> str:
        text = self.strip_thinking(text).strip()
        matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        match = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"```$", "", code).strip()
            return code
        return text.strip()

    def build_test_code(self, item: Dict[str, Any], generated_code: str):
        namespace = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "re": __import__("re"),
            "sys": __import__("sys"),
            "bisect": __import__("bisect"),
            "heapq": __import__("heapq"),
            "typing": __import__("typing"),
            "print": lambda *args, **kwargs: None,
        }

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            exec(generated_code, namespace, namespace)
        finally:
            sys.stdout = old_stdout

        funcs = [
            (k, v)
            for k, v in namespace.items()
            if callable(v) and not k.startswith("__")
        ]

        if len(funcs) == 0:
            raise ValueError("No function found in generated code")

        entry_point = item.get("entry_point")
        candidate = None
        if entry_point and entry_point in namespace and callable(namespace[entry_point]):
            candidate = namespace[entry_point]
        else:
            _, candidate = funcs[-1]

        test_code = (
            "from typing import List, Optional, Dict, Tuple, Union, Any\n"
            + (item.get("test") or "")
        )

        def test_func():
            local_ns = {
                "__builtins__": __builtins__,
                "candidate": candidate,
                "math": namespace.get("math"),
                "collections": namespace.get("collections"),
                "itertools": namespace.get("itertools"),
                "re": namespace.get("re"),
                "sys": namespace.get("sys"),
                "bisect": namespace.get("bisect"),
                "heapq": namespace.get("heapq"),
                "typing": namespace.get("typing"),
                "print": lambda *args, **kwargs: None,
            }
            old_stdout2 = sys.stdout
            sys.stdout = StringIO()
            try:
                exec(test_code, local_ns, local_ns)

                if "check" not in local_ns:
                    raise ValueError("No check function in test code")

                local_ns["check"](candidate)
            finally:
                sys.stdout = old_stdout2

        return test_func

    def is_correct(self, generated_code: Optional[str], item: Dict[str, Any]) -> bool:
        if not generated_code:
            return False
        try:
            test_code = self.build_test_code(item, generated_code)
        except Exception:
            return False
        result = self.run_test(test_code)
        return result["pass"]

    def run_test(self, test_func) -> Dict[str, Any]:
        try:
            test_func()
            return {"pass": True, "error": None}
        except BaseException:
            tb = traceback.format_exc()
            return {"pass": False, "error": tb}


def evaluate_humaneval_accuracy(
    model,
    tokenizer,
    test_path: str,
    device: Optional[torch.device],
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    return_details: bool = False,
    save_results_path: Optional[str] = None,
):
    return HumanEvalEvaluator().evaluate(
        model=model,
        tokenizer=tokenizer,
        test_path=test_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_details=return_details,
        save_results_path=save_results_path,
    )
