from __future__ import annotations

import argparse
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
from batch_runner.model_wrapper import BatchModelRunner, GenerationConfig
from data import DatasetLoader

from utils.metrics import extract_boxed_answer


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


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_file)

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
    runner = BatchModelRunner(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    start_time = time.time()
    processed = 0
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
            write_jsonl_record(args.output_file, row)
            processed += 1
        progress_bar.set_postfix(
            batch=f"{batch_index}/{total_batches}",
            records=f"{processed}/{total_records}",
            batch_size=len(batch_records),
        )

    elapsed = time.time() - start_time
    print(f"Processed {processed} records in {elapsed:.2f}s")
    print(f"Output written to {args.output_file}")


if __name__ == "__main__":
    # 与 start_batch.sh 中默认 CMD 对齐（不含可选的 --system-prompt / --do-sample / --resume）
    sys.argv = [
        sys.argv[0],
        "--model", "/kpfs-llm-text/models/Qwen3-4B",
        "--dataset-name", "gsm8k",
        "--data-root", "/kpfs-llm-text/hao.luo/project/Residual/data",
        "--dataset-split", "test",
        "--output-file", "/kpfs-llm-text/hao.luo/project/Residual/code/output/batch_output_debug.jsonl",
        "--prompt-field", "prompt",
        "--prompt-format", "chat",
        "--output-field", "/kpfs-llm-text/hao.luo/project/Residual/code/llm_output",
        "--batch-size", "8",
        "--max-new-tokens", "1024",
        "--dtype", "bfloat16",
        "--device-map", "auto",
        "--temperature", "0.0",
        "--top-p", "1.0",
        "--repetition-penalty", "1.0",
    ]
    main()
