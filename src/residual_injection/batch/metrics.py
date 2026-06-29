"""Evaluation metrics (adapted from code-v7/utils/metrics.py; core accuracy logic retained)."""
from __future__ import annotations

import re
from typing import Optional

PUNCTUATION = [
    ".</s>", ".\n", ".", ";", "!", ",", "?", "\n",
    "</s>", "<pad>", " ", ":", '"', "'", "(", ")",
    "[", "]", "{", "}", "-", "_", "/", "\\", "*",
    "&", "^", "%", "$", "#", "@", "~", "`", "|",
    "<", ">", "=", "+",
]


def _normalize_boxed_answer(s: str) -> str:
    s = s.replace("\\dfrac", "\\frac")
    s = re.sub(r"\\text\{\(([A-F])\)\}", r"\1", s)
    s = re.sub(r"x\s*(?:\\in|∈)\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None

    boxed_pat = re.compile(r"\\?boxed\b", re.IGNORECASE)

    def parse_after_keyword(start: int) -> Optional[str]:
        n = len(text)
        j = start
        while j < n and text[j].isspace():
            j += 1
        if j >= n:
            return None
        if text[j] == ":":
            j += 1
            while j < n and text[j].isspace():
                j += 1
        if j < n and text[j] == "{":
            depth = 0
            open_idx = j
            i = j
            while i < n:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        inner = text[open_idx + 1 : i].strip()
                        return inner if inner else None
                i += 1
            return None
        if j >= n or text[j] in "}\n\r":
            return None
        k = j
        while k < n and text[k] not in "\n\r":
            k += 1
        inner = text[j:k].strip()
        return inner if inner else None

    last: Optional[str] = None
    for m in boxed_pat.finditer(text):
        content = parse_after_keyword(m.end())
        if content is not None:
            last = _normalize_boxed_answer(content)
    return last


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    ans = str(answer).strip().lower().split("\n")[0]
    for p in PUNCTUATION:
        ans = ans.replace(p.lower(), "")
    return ans.strip()


def compute_accuracy(prediction: str, reference: str) -> bool:
    norm_ref = normalize_answer(reference)
    if not norm_ref:
        return False

    boxed_answer = extract_boxed_answer(prediction)
    if boxed_answer is not None:
        if normalize_answer(boxed_answer) == norm_ref:
            return True

    norm_pred = normalize_answer(prediction)
    if norm_pred == norm_ref:
        return True
    if len(norm_ref) <= 2 and norm_pred and norm_pred[0] == norm_ref[0]:
        return True
    return False


def extract_final_result(output: str) -> str:
    if not output:
        return ""
    boxed = extract_boxed_answer(output)
    if boxed is None:
        return ""
    if boxed.startswith("{") and boxed.endswith("}") and len(boxed) >= 2:
        boxed = boxed[1:-1].strip()
    return boxed
