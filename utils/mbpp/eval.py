import ast
import json
import re
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.base_evaluator import BaseCodeEvaluator


class MBPPEvaluator(BaseCodeEvaluator):
    eval_name = "MBPP"

    def __init__(self):
        self.SYSTEM_PROMPT = (
            "You are a helpful assistant that solves Python coding problems. "
            "You are given a programming task and need to write Python code that passes all tests. "
            "Only provide code without explanations unless asked."
        )

    def get_sync_dummy_item(self) -> Dict[str, Any]:
        return {
            "prompt": "pass",
            "text": "",
            "test_list": ["assert True"],
            "test_imports": [],
        }

    def load_dataset(self, test_path: str) -> List[Dict[str, Any]]:
        test_file = Path(test_path)
        if test_file.is_file():
            suffix = test_file.suffix.lower()
            if suffix == ".jsonl":
                samples: List[Dict[str, Any]] = []
                with test_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
                return samples
            if suffix == ".json":
                with test_file.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, dict):
                    for key in ("data", "examples", "records", "samples"):
                        if isinstance(raw.get(key), list):
                            return raw[key]
                raise ValueError(f"JSON file must be a list or dict with a list field: {test_file}")
            if suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq
                except ImportError as e:
                    raise ImportError(
                        "Reading .parquet requires pyarrow; run: pip install pyarrow"
                    ) from e
                table = pq.read_table(str(test_file))
                return table.to_pylist()
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "No local data file found and huggingface `datasets` is not installed. "
                "Provide a .jsonl/.json/.parquet path, or run: pip install datasets"
            ) from e
        ds = load_dataset("mbpp", "sanitized", split="test")
        if hasattr(ds, "to_list"):
            return ds.to_list()
        return [ds[i] for i in range(len(ds))]

    def build_prompt(self, tokenizer, item: Dict[str, Any], *, enable_thinking: bool = False) -> str:
        text = str(item.get("prompt") or item.get("text") or "").strip()
        test_list = self._parse_test_list(item.get("test_list"))
        tests = "\n".join(test_list)
        user_prompt = (
            f"# Problem:\n{text}\n\n"
            f"# Tests:\n{tests}\n\n"
            "# Solution:\n"
        )
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
        text = text.strip()
        matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        match = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"```$", "", code).strip()
            return code
        return text.strip()

    def _parse_test_list(self, raw_tests: Any) -> List[str]:
        if raw_tests is None:
            return []
        if isinstance(raw_tests, list):
            return [str(t).strip() for t in raw_tests if str(t).strip()]
        if isinstance(raw_tests, str):
            text = raw_tests.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(t).strip() for t in parsed if str(t).strip()]
            except Exception:
                pass
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [str(raw_tests).strip()]

    def _parse_test_imports(self, raw_imports: Any) -> List[str]:
        if raw_imports is None:
            return []
        if isinstance(raw_imports, list):
            return [str(x).strip() for x in raw_imports if str(x).strip()]
        if isinstance(raw_imports, str):
            text = raw_imports.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [str(raw_imports).strip()]

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
        test_imports = self._parse_test_imports(item.get("test_imports"))
        for import_stmt in test_imports:
            exec(import_stmt, namespace, namespace)
        test_list = self._parse_test_list(item.get("test_list"))
        return namespace, test_list

    def run_test(self, test_data) -> Dict[str, Any]:
        namespace, test_list = test_data
        try:
            for test_expr in test_list:
                exec(test_expr, namespace, namespace)
            return {"pass": True, "error": None}
        except BaseException:
            return {"pass": False, "error": traceback.format_exc()}

    def is_correct(self, generated_code: Optional[str], item: Dict[str, Any]) -> bool:
        if not generated_code:
            return False
        try:
            test_data = self.build_test_code(item, generated_code)
        except Exception:
            return False
        result = self.run_test(test_data)
        return result["pass"]


def evaluate_mbpp_accuracy(
    model,
    tokenizer,
    test_path: str,
    device: Optional[torch.device],
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    return_details: bool = False,
    save_results_path: Optional[str] = None,
):
    return MBPPEvaluator().evaluate(
        model=model,
        tokenizer=tokenizer,
        test_path=test_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_details=return_details,
        save_results_path=save_results_path,
    )
