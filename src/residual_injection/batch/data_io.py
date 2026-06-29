from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def write_jsonl_record(path: str | Path, record: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_prompt(
    record: Dict,
    prompt_field: Optional[str],
    prompt_template: Optional[str],
    instruction_field: str = "instruction",
    input_field: str = "input",
    fallback_prompt_field: str = "prompt",
) -> str:
    if prompt_template:
        try:
            return prompt_template.format(**record)
        except KeyError as exc:
            raise KeyError(f"Prompt template missing field: {exc}") from exc

    if prompt_field and prompt_field in record:
        return str(record[prompt_field])

    if fallback_prompt_field in record:
        return str(record[fallback_prompt_field])

    for key in ("question", "problem", "query", "input"):
        if key in record and record[key] is not None:
            return str(record[key])

    if instruction_field in record:
        instruction = str(record[instruction_field]).strip()
        extra_input = str(record.get(input_field, "")).strip()
        if extra_input:
            return f"{instruction}\n\n{extra_input}"
        return instruction

    raise KeyError(
        "Could not build prompt. Provide prompt in record or use --prompt-field / --prompt-template."
    )


def slice_records(records: List[Dict], skip: int = 0, limit: Optional[int] = None) -> List[Dict]:
    sliced = records[skip:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def chunk_records(records: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
