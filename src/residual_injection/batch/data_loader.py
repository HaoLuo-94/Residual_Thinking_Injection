from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]

PROMPT_KEYS = ("prompt", "question", "problem", "query", "input", "instruction")
ANSWER_KEYS = ("answer", "target", "output", "final_answer", "label", "solution")


def _resolve_data_file(
    data: PathLike,
    data_root: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """Resolve the data file path.

    If ``data`` is already an existing file, return it; otherwise look under data_root
    by dataset_name.
    """
    path = Path(data)
    if path.is_file():
        return path

    if not data_root:
        raise FileNotFoundError(f"Data file not found: {data}")

    candidates = [
        path,
        Path(data_root) / path.name,
    ]
    if dataset_name:
        candidates.insert(1, Path(data_root) / dataset_name / path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"Cannot find data file {data!r}. Searched:\n{searched}")


def _load_raw_rows(dataset_file: Path) -> List[Dict]:
    suffix = dataset_file.suffix.lower()
    if suffix == ".jsonl":
        with dataset_file.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with dataset_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for key in ("data", "examples", "records", "samples"):
                if isinstance(raw.get(key), list):
                    return raw[key]
        if isinstance(raw, list):
            return raw
        raise ValueError(f"JSON file must be a list or dict with a list field: {dataset_file}")
    raise ValueError(f"Unsupported format {suffix}. Use .jsonl or .json")


def normalize_record(
    item: Dict,
    idx: int,
    *,
    dataset_name: str = "custom",
    dataset_file: Optional[str] = None,
) -> Dict:
    if not isinstance(item, dict):
        raise ValueError(f"Record #{idx} is not a json object")

    row = dict(item)

    if "prompt" not in row:
        for key in PROMPT_KEYS:
            if key in row and row[key] is not None and key != "prompt":
                row["prompt"] = str(row[key])
                break

    if "answer" not in row:
        for key in ANSWER_KEYS:
            if key in row and row[key] is not None:
                row["answer"] = str(row[key])
                break

    if "answer" not in row:
        raise ValueError(
            f"Record #{idx} has no answer field. Expected one of: {', '.join(ANSWER_KEYS)}"
        )

    row["answer"] = str(row["answer"])
    row.setdefault("dataset_name", dataset_name)
    if dataset_file:
        row["dataset_file"] = dataset_file
        row.setdefault("dataset_split", Path(dataset_file).name)
    return row


def resolve_data_file(
    data: PathLike,
    *,
    data_root: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """Resolve the data file path (shared by run_batch and similar scripts)."""
    return _resolve_data_file(data, data_root=data_root, dataset_name=dataset_name)


def load_records_from_file(
    data: PathLike,
    *,
    data_root: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> List[Dict]:
    """Load from a JSON/JSONL file and normalize to a unified record format."""
    dataset_file = _resolve_data_file(data, data_root=data_root, dataset_name=dataset_name)
    raw_rows = _load_raw_rows(dataset_file)
    if not isinstance(raw_rows, list):
        raise ValueError(f"Dataset must be a list of records: {dataset_file}")

    name = dataset_name or dataset_file.parent.name
    return [
        normalize_record(item, idx, dataset_name=name, dataset_file=str(dataset_file))
        for idx, item in enumerate(raw_rows)
    ]


def assign_original_indices(records: List[Dict], skip: int = 0) -> List[Dict]:
    for i, record in enumerate(records):
        record["__original_idx"] = skip + i
    return records
