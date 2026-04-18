from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from batch_runner.data_io import (
    chunk_records,
    count_existing_records,
    render_prompt,
    slice_records,
    write_jsonl_record,
)
from batch_runner.model_wrapper import (
    BatchModelRunner,
    GenerationConfig,
    ResidualGenerationConfig,
)
from data import DatasetLoader

from utils.metrics import compute_accuracy, extract_boxed_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch inference with an unchanged HuggingFace causal LM."
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name handled by code/data/loader.py, e.g. gsm8k, mmlu, boolq.",
    )
    parser.add_argument("--output-file", required=True, help="Output file, recommend .jsonl.")
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Dataset root used with --dataset-name.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train", "test", "both"],
        default="test",
        help="Dataset split used with --dataset-name.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=["generation", "chat"],
        default="generation",
        help="Whether to send raw text or render via chat template.",
    )
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt.")
    parser.add_argument("--prompt-field", default=None, help="Field used as the prompt.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Python format template, e.g. 'Question: {question}\\nAnswer:'",
    )
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="model_output")
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional summary json path. Defaults to <output-file stem>.summary.json.",
    )
    parser.add_argument(
        "--final-result-field",
        default="final_result",
        help="Field used to store the extracted final answer from the model output.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--enable-residual",
        action="store_true",
        help="Enable residual injection decoding defined in code/model/residual.py.",
    )
    parser.add_argument("--residual-alpha", type=float, default=0.001)
    parser.add_argument("--residual-layer-start", type=int, default=0)
    parser.add_argument("--residual-layer-end", type=int, default=-1)
    parser.add_argument("--residual-entropy-threshold", type=float, default=0.01)
    parser.add_argument("--residual-low-entropy-patience", type=int, default=5)
    parser.add_argument("--residual-think-end-token", default="</think>")
    parser.add_argument("--residual-topk", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing jsonl output by skipping already written rows.",
    )
    return parser


def build_records(
    samples: List[Tuple[str, str, str]],
    dataset_name: str,
    dataset_split: str,
) -> List[Dict]:
    return [
        {
            "prompt": prompt,
            "answer": correct_answer,
            "answer_wrong": wrong_answer,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
        }
        for prompt, correct_answer, wrong_answer in samples
    ]


def extract_final_result(output: Optional[str]) -> str:
    if not output:
        return ""

    boxed_answer = extract_boxed_answer(output)
    if boxed_answer is not None:
        return boxed_answer

    return output.strip()


def enrich_records(
    records: List[Dict],
    outputs: List[str],
    output_field: str,
    final_result_field: str,
) -> List[Dict]:
    enriched: List[Dict] = []
    for record, output in zip(records, outputs):
        row = dict(record)
        row[output_field] = output
        row[final_result_field] = extract_final_result(output)
        enriched.append(row)
    return enriched


def resolve_summary_path(output_path: Path, summary_file: Optional[str]) -> Path:
    if summary_file:
        return Path(summary_file)
    return output_path.with_suffix(".summary.json")


def write_summary_file(
    summary_path: Path,
    *,
    model: str,
    dataset_name: str,
    dataset_split: str,
    output_file: str,
    processed: int,
    correct: int,
    accuracy: float,
    elapsed_seconds: float,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "output_file": output_file,
        "processed": processed,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed_seconds,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_file)
    summary_path = resolve_summary_path(output_path, args.summary_file)

    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("This script currently writes jsonl output only.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output file already exists: {args.output_file}. Use --resume or change the path."
        )

    dataset_loader = DatasetLoader(data_root=args.data_root, model_name=args.model)
    train_data, test_data = dataset_loader.load(args.dataset_name, split=args.dataset_split)

    all_records: List[Dict] = []
    if args.dataset_split in {"train", "both"}:
        all_records.extend(build_records(train_data, args.dataset_name, "train"))
    if args.dataset_split in {"test", "both"}:
        all_records.extend(build_records(test_data, args.dataset_name, "test"))

    resume_skip = count_existing_records(args.output_file) if args.resume else 0
    effective_skip = args.skip + resume_skip
    records = slice_records(all_records, skip=effective_skip, limit=args.limit)

    if not records:
        print("No records to process.")
        return

    prompts = [
        render_prompt(
            record=record,
            prompt_field=args.prompt_field,
            prompt_template=args.prompt_template,
            instruction_field=args.instruction_field,
            input_field=args.input_field,
        )
        for record in records
    ]

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
    )
    residual_config = ResidualGenerationConfig(
        enabled=args.enable_residual,
        alpha=args.residual_alpha,
        layer_start=args.residual_layer_start,
        layer_end=args.residual_layer_end,
        entropy_threshold=args.residual_entropy_threshold,
        low_entropy_patience=args.residual_low_entropy_patience,
        think_end_token=args.residual_think_end_token,
        topk=args.residual_topk,
    )
    runner = BatchModelRunner(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        residual_config=residual_config,
    )

    start_time = time.time()
    processed = 0
    correct = 0
    total_records = len(records)
    total_batches = (total_records + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(
        chunk_records(records, args.batch_size),
        total=total_batches,
        desc=f"{args.dataset_name}:{args.dataset_split}",
        unit="batch",
    )
    for batch_index, batch_records in enumerate(progress_bar, start=1):
        prompt_start = (batch_index - 1) * args.batch_size
        prompt_end = prompt_start + len(batch_records)
        batch_prompts = prompts[prompt_start:prompt_end]

        outputs = runner.generate_batch(
            prompts=batch_prompts,
            generation_config=generation_config,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )

        for row in enrich_records(
            batch_records,
            outputs,
            args.output_field,
            args.final_result_field,
        ):
            if compute_accuracy(row[args.final_result_field], row["answer"]):
                correct += 1
            write_jsonl_record(args.output_file, row)
            processed += 1
        accuracy = correct / processed if processed else 0.0
        progress_bar.set_postfix(
            batch=f"{batch_index}/{total_batches}",
            records=f"{processed}/{total_records}",
            batch_size=len(batch_records),
            accuracy=f"{accuracy:.2%}",
        )

    elapsed = time.time() - start_time
    write_summary_file(
        summary_path,
        model=args.model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        output_file=args.output_file,
        processed=processed,
        correct=correct,
        accuracy=accuracy,
        elapsed_seconds=elapsed,
    )
    print(f"Processed {processed} records in {elapsed:.2f}s")
    print(f"Accuracy: {correct}/{processed} = {accuracy:.2%}")
    print(f"Output written to {args.output_file}")
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()
