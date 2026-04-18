from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_records(path: str) -> List[Dict]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects.")
        return data

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def count_existing_records(path: str) -> int:
    output_path = Path(path)
    if not output_path.exists():
        return 0
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("Resume mode currently only supports jsonl output.")

    with output_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_jsonl_record(path: str, record: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_prompt(
    record: Dict,
    prompt_field: Optional[str],
    prompt_template: Optional[str],
    instruction_field: str,
    input_field: str,
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

    if "question" in record:
        return str(record["question"])

    if instruction_field in record:
        instruction = str(record[instruction_field]).strip()
        extra_input = str(record.get(input_field, "")).strip()
        if extra_input:
            return f"{instruction}\n\n{extra_input}"
        return instruction

    raise KeyError(
        "Could not build prompt. Please provide --prompt-field or --prompt-template."
    )


def slice_records(records: List[Dict], skip: int = 0, limit: Optional[int] = None) -> List[Dict]:
    sliced = records[skip:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def chunk_records(records: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
