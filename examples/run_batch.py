from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from residual_injection.batch import (
    assign_original_indices,
    chunk_records,
    compute_accuracy,
    extract_final_result,
    load_records_from_file,
    render_prompt,
    resolve_data_file,
    slice_records,
    write_jsonl_record,
)
from utils.registry import get_evaluator, is_code_evaluator, is_math_evaluator, resolve_evaluator_name
from utils.timeout import time_limit, EvalTimeout


# Mirrors eval_checkpoints.py MAX_NEW_TOKENS: override max_tokens by dataset label.
# Label is the data file stem (without extension). Unlisted datasets fall back to --max-tokens.
DATASET_MAX_TOKENS: Dict[str, int] = {
    # "aime2325": 32768,
    # "math500": 2048,
    # "minerva": 8192,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vLLM residual injection batch inference and evaluation")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="One or more data file paths (.jsonl / .json), or filenames relative to data-root",
    )
    p.add_argument("--output-file", required=True, help="Output JSONL path (for multiple datasets, uses the parent directory)")
    p.add_argument(
        "--data-root",
        default=None,
        help="Data root directory; used with --dataset-name to resolve relative --data paths",
    )
    p.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name: resolves <data-root>/<name>/<data>, and enables matching prompt/evaluator in utils (single dataset only)",
    )
    p.add_argument("--summary-file", default=None, help="Summary JSON; default <output>.summary.json")
    p.add_argument("--prompt-field", default=None, help="Field name used as the prompt")
    p.add_argument(
        "--prompt-template",
        default=None,
        help="Python format template, e.g. 'Question: {problem}\\nAnswer:'",
    )
    p.add_argument(
        "--prompt-format",
        choices=["generation", "chat"],
        default="generation",
        help="generation=raw text; chat=apply_chat_template",
    )
    p.add_argument("--system-prompt", default=None, help="System prompt in chat mode")
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in chat template (generates ... block)",
    )
    p.add_argument("--output-field", default="model_output", help="Field name for model output")
    p.add_argument("--final-result-field", default="final_result", help="Field name for extracted answer")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512, help="Maximum generation tokens (may be overridden by DATASET_MAX_TOKENS)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--alpha", type=float, default=1.0, help="Injection strength α")
    p.add_argument(
        "--residual-top-k",
        type=int,
        default=8,
        dest="residual_top_k",
        help="Residual injection Top-K soft embeddings (unrelated to sampling top-k; only when align_sampler is off)",
    )
    p.add_argument("--plugin-mode", action="store_true", help="vLLM plugin mode")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Run native baseline model: no residual injection, no patching, no architecture override",
    )
    p.add_argument(
        "--inject-layer",
        type=int,
        default=-1,
        dest="inject_layer",
        help="Injection layer: -1=all layers (default); 0=first layer; k=layer k (0-indexed)",
    )
    p.add_argument(
        "--inject-phase",
        choices=["all", "think"],
        default="all",
        dest="inject_phase",
        help="Injection phase: all=entire generation; think=only <think>...</think> thinking phase",
    )
    p.add_argument("--think-start-token", default="<think>", dest="think_start_token")
    p.add_argument("--think-end-token", default="</think>", dest="think_end_token")
    p.add_argument("--think-start-id", type=int, default=None, dest="think_start_id")
    p.add_argument("--think-end-id", type=int, default=None, dest="think_end_id")
    # --- e_soft / e_hard alignment (reuse sampling params below; no extra soft_* flags) ---
    p.add_argument(
        "--align-sampler",
        action="store_true",
        dest="align_sampler",
        help="e_soft mirrors sampling temperature->top_k->top_p (reuses --temperature/--top-p/--top-k directly); "
             "when off, uses legacy path (temperature 1.0 + --residual-top-k + no top_p)",
    )
    p.add_argument(
        "--hard-anchor",
        choices=["argmax", "committed"],
        default="argmax",
        dest="hard_anchor",
        help="e_hard anchor: argmax=greedy token (default); committed=token actually sampled in the previous step",
    )
    p.add_argument("--no-eval", action="store_true", help="Skip accuracy evaluation (when no answer field)")
    # sampling (vLLM SamplingParams)
    p.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding")
    p.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    p.add_argument("--top-k", type=int, default=-1, dest="top_k", help="vLLM sampling top-k; -1 means unlimited")
    p.add_argument("--presence-penalty", type=float, default=0.0, dest="presence_penalty")
    p.add_argument("--repetition-penalty", type=float, default=1.0, dest="repetition_penalty")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility for engine layer only (LLM init); sampling layer always uses seed=None (natural randomness).",
    )
    # --- pass@k (ported from eval_checkpoints.py) ---
    p.add_argument(
        "--num-samples",
        type=int,
        default=1,
        dest="num_samples",
        help="pass@k: samples per question. 1=greedy pass@1 (default); >1=pass@k, pass if any sample is correct.",
    )
    p.add_argument(
        "--passk-temperature",
        type=float,
        default=0.7,
        dest="passk_temperature",
        help="Default temperature for pass@k (num-samples>1); only used when --temperature is not explicitly set.",
    )
    p.add_argument(
        "--passk-top-p",
        type=float,
        default=0.95,
        dest="passk_top_p",
        help="Default top-p for pass@k (num-samples>1); only used when --top-p is not explicitly set.",
    )
    p.add_argument(
        "--residual-arch",
        default=None,
        help="Override auto-detected residual architecture name (LlamaForResidualInjection / Qwen3ForResidualInjection)",
    )
    # --- Engine management (ported from run_evals) ---
    p.add_argument(
        "--rebuild-engine-per-dataset",
        action="store_true",
        default=True,
        help="Rebuild a fresh engine per dataset (isolated/reproducible, but reloads weights for each dataset). Enabled by default.",
    )
    p.add_argument(
        "--reuse-engine",
        dest="rebuild_engine_per_dataset",
        action="store_false",
        help="Reuse the same engine for all datasets (avoids reloading weights; recommended for multi-dataset runs with the same model).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable CUDA deterministic algorithms + cublas workspace config (reproducible, slightly slower).",
    )
    p.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1. Improves determinism further but significantly slows inference; off by default.",
    )
    p.add_argument(
        "--per-sample-timeout",
        type=float,
        default=30.0,
        dest="per_sample_timeout",
        help="Per-sample evaluation timeout (seconds); timeout counts as wrong and skips; <=0 disables.",
    )
    return p


# ---------------------------------------------------------------------------
# Determinism (ported from run_evals top-level env setup)
# ---------------------------------------------------------------------------
def setup_determinism(deterministic: bool, cuda_launch_blocking: bool) -> None:
    if not deterministic and not cuda_launch_blocking:
        return
    import torch

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True: vLLM falls back instead of erroring on kernels that lack deterministic support
        torch.use_deterministic_algorithms(True, warn_only=True)
    if cuda_launch_blocking:
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def resolve_token_id(tokenizer: Any, token: str) -> Optional[int]:
    """Best-effort parse a token string into a single token id; returns None on failure (unknown or multi-token)."""
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    except Exception:
        pass
    try:
        ids = tokenizer.encode(token, add_special_tokens=False)
    except Exception:
        return None
    return ids[0] if len(ids) == 1 else None


def setup_residual_injection(
    alpha: float,
    top_k: int,
    plugin_mode: bool,
    *,
    inject_phase: str = "all",
    think_start_id: Optional[int] = None,
    think_end_id: Optional[int] = None,
    inject_layer: int = -1,
    align_sampler: bool = False,
    soft_temperature: float = 1.0,
    soft_top_p: float = 1.0,
    soft_top_k: int = 0,
    soft_pool_k: int = 1024,
    hard_anchor: str = "argmax",
) -> None:
    if plugin_mode:
        os.environ["RESIDUAL_INJECTION_ALPHA"] = str(alpha)
        os.environ["RESIDUAL_INJECTION_TOP_K"] = str(top_k)
        os.environ["RESIDUAL_INJECTION_ENABLED"] = "1"
        os.environ["RESIDUAL_INJECTION_PHASE"] = inject_phase
        os.environ["RESIDUAL_INJECTION_LAYER"] = str(inject_layer)
        if think_start_id is not None:
            os.environ["RESIDUAL_INJECTION_THINK_START_ID"] = str(think_start_id)
        if think_end_id is not None:
            os.environ["RESIDUAL_INJECTION_THINK_END_ID"] = str(think_end_id)
        # align_sampler / hard_anchor (read by load_from_env during subprocess engine init)
        os.environ["RESIDUAL_INJECTION_ALIGN_SAMPLER"] = "1" if align_sampler else "0"
        os.environ["RESIDUAL_INJECTION_SOFT_TEMPERATURE"] = str(soft_temperature)
        os.environ["RESIDUAL_INJECTION_SOFT_TOP_P"] = str(soft_top_p)
        os.environ["RESIDUAL_INJECTION_SOFT_TOP_K"] = str(soft_top_k)
        os.environ["RESIDUAL_INJECTION_SOFT_POOL_K"] = str(soft_pool_k)
        os.environ["RESIDUAL_INJECTION_HARD_ANCHOR"] = hard_anchor
    else:
        from residual_injection import enable_residual_injection
        from residual_injection.config import set_config

        enable_residual_injection(alpha=alpha, top_k=top_k)
        set_config(
            inject_phase=inject_phase,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            inject_layer=inject_layer,
            align_sampler=align_sampler,
            soft_temperature=soft_temperature,
            soft_top_p=soft_top_p,
            soft_top_k=soft_top_k,
            soft_pool_k=soft_pool_k,
            hard_anchor=hard_anchor,
        )


# ---------------------------------------------------------------------------
# Engine construction: residual injection setup + LLM init + think id backfill
# Core of "engine management" — every rebuild runs the full pipeline.
# ---------------------------------------------------------------------------
def build_llm(args: argparse.Namespace) -> Tuple[Any, Any]:
    from vllm import LLM

    if args.baseline:
        print("[baseline] residual injection DISABLED — using native vLLM model.")
        hf_overrides: dict = {}
    else:
        # soft_* reuses sampling params: when align_sampler is on, e_soft matches the actual sampling distribution.
        # vLLM top_k=-1 (unlimited) is normalized to 0 (off) to avoid confusing -1 in summaries.
        soft_top_k = args.top_k if (args.top_k and args.top_k > 0) else 0
        setup_residual_injection(
            args.alpha,
            args.residual_top_k,
            args.plugin_mode,
            inject_phase=args.inject_phase,
            think_start_id=args.think_start_id,
            think_end_id=args.think_end_id,
            inject_layer=args.inject_layer,
            align_sampler=args.align_sampler,
            soft_temperature=args.temperature,   # reuse sampling temperature
            soft_top_p=args.top_p,               # reuse sampling top_p
            soft_top_k=soft_top_k,               # reuse sampling top_k (-1 -> 0)
            hard_anchor=args.hard_anchor,        # soft_pool_k uses default 1024
        )
        print(
            f"[residual] alpha={args.alpha}, residual_top_k={args.residual_top_k}, "
            f"align_sampler={args.align_sampler}, hard_anchor={args.hard_anchor}"
        )
        if args.align_sampler and args.temperature == 0.0 and args.hard_anchor == "argmax":
            print("[residual] [warn] sampling temperature=0 -> soft temperature also 0, Δ≈0, injection is effectively disabled.")
        from residual_injection import hf_overrides_for_model

        hf_overrides = hf_overrides_for_model(args.model, arch=args.residual_arch)
        print(f"Residual architecture: {hf_overrides['architectures'][0]}")

    llm_kwargs = dict(
        model=args.model,
        enforce_eager=True,            # residual injection requires python forward; cannot disable
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_overrides=hf_overrides,
    )
    # if not args.baseline:
    #     llm_kwargs["enforce_eager"] = True
    #     llm_kwargs["enable_chunked_prefill"] = False
    # else:
    #     # baseline uses vLLM default optimized path
    #     pass
    # Engine-layer seed: --seed applies here only (LLM init reproducibility).
    # vLLM >=0.17: seed must be int; cannot pass None explicitly
    llm_kwargs["trust_remote_code"] = True
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed
        print(f"[engine] seed={args.seed} (engine layer only; sampling layer always random)")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # think mode (single process): tokenizer resolves boundary token ids, then backfills global CONFIG.
    if not args.baseline and args.inject_phase == "think" and not args.plugin_mode:
        from residual_injection.config import set_config

        start_id = args.think_start_id
        end_id = args.think_end_id
        if start_id is None:
            start_id = resolve_token_id(tokenizer, args.think_start_token)
        if end_id is None:
            end_id = resolve_token_id(tokenizer, args.think_end_token)
        if start_id is None or end_id is None:
            raise ValueError(
                f"Failed to resolve think boundary token ids "
                f"(start={args.think_start_token!r}->{start_id}, "
                f"end={args.think_end_token!r}->{end_id}). "
                f"Specify --think-start-id/--think-end-id."
            )
        set_config(inject_phase="think", think_start_id=start_id, think_end_id=end_id)
        print(f"[residual] think-only injection: start_id={start_id}, end_id={end_id}")

    return llm, tokenizer


def cleanup_llm(llm: Any) -> None:
    """Best-effort release GPU memory held by the engine (ported from run_evals del + empty_cache)."""
    import torch

    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_prompts(
    prompts: List[str],
    *,
    prompt_format: str,
    tokenizer: Any,
    system_prompt: Optional[str],
    enable_thinking: bool = False,
) -> List[str]:
    if prompt_format == "generation":
        return prompts

    formatted: List[str] = []
    for i, text in enumerate(prompts):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking:
            kwargs["enable_thinking"] = True
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        formatted.append(rendered if isinstance(rendered, str) else str(rendered))
    return formatted


def write_summary(summary_path: Path, **fields: Any) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_dataset_evaluator(
    dataset_name: Optional[str],
    label: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """Look up a utils evaluator by dataset_name or file stem (label)."""
    for name in (dataset_name, label):
        if name and resolve_evaluator_name(name):
            ev = get_evaluator(name)
            if ev is not None:
                return ev, name
    return None, None


def load_dataset_records(
    data: str,
    *,
    data_root: Optional[str],
    dataset_name: Optional[str],
    evaluator: Optional[Any],
    skip: int,
    limit: Optional[int],
) -> List[Dict]:
    if evaluator is not None:
        path = resolve_data_file(data, data_root=data_root, dataset_name=dataset_name)
        records = evaluator.load_dataset(str(path))
    else:
        records = load_records_from_file(data, data_root=data_root, dataset_name=dataset_name)
    records = slice_records(records, skip=skip, limit=limit)
    return assign_original_indices(records, skip=skip)


def build_evaluator_prompts(
    records: List[Dict],
    evaluator: Any,
    tokenizer: Any,
    *,
    enable_thinking: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for i, r in enumerate(records):
        if is_math_evaluator(evaluator):
            prompt = evaluator.build_prompt(
                tokenizer, evaluator.get_question(r), enable_thinking=enable_thinking
            )
        else:
            prompt = evaluator.build_prompt(tokenizer, r, enable_thinking=enable_thinking)
        if i == 0:
            print(
                f"[build_evaluator_prompts] evaluator={evaluator.eval_name} "
                f"enable_thinking={enable_thinking} (first sample only)"
            )
            print(f"[build_evaluator_prompts] sample=0 rendered=\n{prompt}\n{'=' * 60}")
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# pass@k scoring: any-correct evaluation over k samples for one record
# (matches inner loop + score_with_evaluator in eval_with_vllm)
# ---------------------------------------------------------------------------
def _is_correct_with_timeout(fn, *fn_args, timeout: float) -> bool:
    """Call scoring function; on timeout, mark as wrong (False) without blocking the batch."""
    try:
        with time_limit(timeout):
            return bool(fn(*fn_args))
    except EvalTimeout:
        return False

def score_samples(
    evaluator: Optional[Any],
    record: Dict,
    completions: List[Any],
    *,
    output_field: str,
    final_result_field: str,
    no_eval: bool,
    timeout: float = 20.0, 
) -> Tuple[Dict, bool]:
    """Score k samples for one question.

    Returns (row, any_correct). Primary row fields follow the *first* sample (consistent with
    eval_checkpoints.py); full k-sample details are stored in row["pass_at_k_samples"].
    """
    texts = [c.text for c in completions]
    per_sample: List[Dict[str, Any]] = []
    any_correct = False

    if evaluator is not None and not no_eval:
        if is_math_evaluator(evaluator):
            ground_truth = evaluator.extract_ground_truth(record)
            for text in texts:
                prediction = evaluator.extract_answer(text)
                correct = _is_correct_with_timeout(
                    evaluator.is_correct, prediction, ground_truth, record, timeout=timeout
                )
                any_correct = any_correct or correct
                per_sample.append(
                    {"completion": text, "prediction": prediction, "is_correct": correct}
                )
            ground_truth_extracted = ground_truth
        else:
            ground_truth_extracted = None
            for text in texts:
                extracted = evaluator.extract_code_block(text)
                correct = _is_correct_with_timeout(
                    evaluator.is_correct, extracted, record, timeout=timeout
                )
                any_correct = any_correct or correct
                per_sample.append(
                    {"completion": text, "prediction": extracted, "is_correct": correct}
                )
    elif not no_eval and "answer" in record:
        for text in texts:
            pred = extract_final_result(text) or text
            correct = _is_correct_with_timeout(
                compute_accuracy, text, str(record["answer"]), timeout=timeout
            )
            any_correct = any_correct or correct
            per_sample.append(
                {"completion": text, "prediction": extract_final_result(text), "is_correct": correct}
            )
        ground_truth_extracted = None
    else:
        # No evaluation: record text only
        for text in texts:
            per_sample.append(
                {"completion": text, "prediction": extract_final_result(text), "is_correct": False}
            )
        ground_truth_extracted = None

    # Primary fields follow the first sample
    row = dict(record)
    first_text = texts[0]
    row[output_field] = first_text
    if evaluator is not None and not no_eval and is_math_evaluator(evaluator):
        row["prediction_extracted"] = per_sample[0]["prediction"]
        row["ground_truth_extracted"] = ground_truth_extracted
        row[final_result_field] = per_sample[0]["prediction"] or ""
    elif evaluator is not None and not no_eval:
        row[final_result_field] = per_sample[0]["prediction"] or ""
    else:
        row[final_result_field] = extract_final_result(first_text)

    if not no_eval:
        row["is_correct"] = any_correct

    # Token counts follow the first sample
    token_ids = getattr(completions[0], "token_ids", None)
    if token_ids is not None:
        row[f"{output_field}_token_count"] = len(token_ids)

    # When k>1, store all samples (omit when k==1 to match original single-sample output)
    if len(completions) > 1:
        row["pass_at_k_samples"] = per_sample

    return row, any_correct


# ---------------------------------------------------------------------------
# Single-dataset evaluation (matches eval_with_vllm in eval_checkpoints.py)
# ---------------------------------------------------------------------------
def run_one_dataset(
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    data: str,
    label: str,
    output_path: Path,
    summary_path: Path,
    dataset_name: Optional[str],
) -> Dict[str, Any]:
    from vllm import SamplingParams

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}. Choose another path.")

    evaluator, eval_key = resolve_dataset_evaluator(dataset_name, label)
    if evaluator is not None:
        print(f"[{label}] Using utils evaluator: {evaluator.eval_name} (dataset={eval_key})")
        if args.prompt_template or args.prompt_field:
            print(f"[{label}] [warn] utils evaluator enabled; ignoring --prompt-field/--prompt-template")
    elif dataset_name and resolve_evaluator_name(dataset_name) is None:
        print(f"[{label}] [warn] unknown DATASET_NAME={dataset_name!r}, falling back to generic prompt/evaluation")

    records = load_dataset_records(
        data,
        data_root=args.data_root,
        dataset_name=dataset_name,
        evaluator=evaluator,
        skip=args.skip,
        limit=args.limit,
    )
    if not records:
        print(f"[{label}] No records to process.")
        return {"processed": 0, "correct": 0, "accuracy": 0.0, "elapsed_seconds": 0.0}

    max_tokens = DATASET_MAX_TOKENS.get(label, args.max_tokens)
    if evaluator is not None and eval_key:
        max_tokens = DATASET_MAX_TOKENS.get(eval_key, max_tokens)

    if evaluator is not None:
        prompts = build_evaluator_prompts(
            records, evaluator, tokenizer, enable_thinking=args.enable_thinking
        )
        use_chat_format = False
    else:
        prompts = [
            render_prompt(r, prompt_field=args.prompt_field, prompt_template=args.prompt_template)
            for r in records
        ]
        use_chat_format = args.prompt_format == "chat"
    # Sort by prompt length to improve batch padding efficiency
    if len(records) > 1:
        order = sorted(range(len(records)), key=lambda i: len(prompts[i]))
        records = [records[i] for i in order]
        prompts = [prompts[i] for i in order]

    # --- pass@k sampling config (ported from eval_with_vllm) ---
    use_passk = args.num_samples > 1
    # When --temperature/--top-p not explicitly set, pass@k uses dedicated defaults; otherwise respect user values.
    temperature = args.temperature
    top_p = args.top_p
    if use_passk:
        if args.temperature == 0.0:   # user left default greedy temperature -> use pass@k temperature
            temperature = args.passk_temperature
        if args.top_p == 1.0:         # user left default top-p -> use pass@k top-p
            top_p = args.passk_top_p
        print(
            f"[{label}] pass@{args.num_samples}: temperature={temperature}, "
            f"top_p={top_p}, sampling_seed=None (always random)"
        )

    # Sampling layer always random: SamplingParams.seed is always None.
    # (--seed applies to engine layer only; see build_llm; no per-prompt seed.)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=max_tokens,
        n=args.num_samples,
        seed=None,
        # seed=args.seed,
    )

    start_time = time.time()
    processed = correct = out_tokens = 0
    total = len(records)
    total_batches = (total + args.batch_size - 1) // args.batch_size

    progress = tqdm(
        chunk_records(records, args.batch_size),
        total=total_batches,
        desc=label,
        unit="batch",
    )

    batch_offset = 0
    for batch_records in progress:
        batch_prompts = prompts[batch_offset : batch_offset + len(batch_records)]
        batch_offset += len(batch_records)

        if use_chat_format:
            formatted = format_prompts(
                batch_prompts,
                prompt_format=args.prompt_format,
                tokenizer=tokenizer,
                system_prompt=args.system_prompt,
                enable_thinking=args.enable_thinking,
            )
        else:
            formatted = batch_prompts

        outputs = llm.generate(formatted, sampling)

        for record, req_output in zip(batch_records, outputs):
            row, any_correct = score_samples(
                evaluator,
                record,
                req_output.outputs,
                output_field=args.output_field,
                final_result_field=args.final_result_field,
                no_eval=args.no_eval,
                timeout=args.per_sample_timeout,
            )
            if not args.no_eval and any_correct:
                correct += 1
            for completion in req_output.outputs:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids is not None:
                    out_tokens += len(token_ids)
            write_jsonl_record(output_path, row)
            processed += 1

        acc = correct / processed if processed and not args.no_eval else 0.0
        progress.set_postfix(
            records=f"{processed}/{total}",
            accuracy=f"{acc:.2%}" if not args.no_eval else "n/a",
        )

    elapsed = time.time() - start_time
    accuracy = correct / processed if processed and not args.no_eval else 0.0

    # When align_sampler is on, soft_* reuses sampling params (set via args.* in build_llm); record effective values for reproducibility.
    soft_top_k_eff = (args.top_k if (args.top_k and args.top_k > 0) else 0)
    summary_fields: Dict[str, Any] = {
        "model": args.model,
        "data": data,
        "dataset": label,
        "output_file": str(output_path),
        "processed": processed,
        "correct": correct,
        "accuracy": accuracy,
        "pass_at_k": args.num_samples,
        "elapsed_seconds": elapsed,
        "out_tokens": out_tokens,
        "tokens_per_s": round(out_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "s_per_sample": round(elapsed / processed, 4) if processed else 0.0,
        "max_tokens": max_tokens,
        "residual_alpha": args.alpha,
        "residual_top_k": args.residual_top_k,
        "baseline": args.baseline,
        "inject_phase": args.inject_phase,
        "enable_thinking": args.enable_thinking,
        "inject_layer": args.inject_layer,
        "align_sampler": args.align_sampler,
        "hard_anchor": args.hard_anchor,
        "soft_temperature": args.temperature if args.align_sampler else None,
        "soft_top_p": args.top_p if args.align_sampler else None,
        "soft_top_k": soft_top_k_eff if args.align_sampler else None,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "engine_seed": args.seed,      # --seed applies to engine layer only
        "sampling_seed": None,         # sampling layer always random
    }
    if evaluator is not None:
        summary_fields["evaluator"] = evaluator.eval_name
        summary_fields["dataset_name"] = eval_key
    write_summary(summary_path, **summary_fields)

    print(f"[{label}] processed {processed} in {elapsed:.2f}s", end="")
    if out_tokens:
        print(f" | out_tok={out_tokens} ({out_tokens / elapsed:.0f} tok/s)", end="")
    if not args.no_eval:
        print(f" | pass@{args.num_samples} {correct}/{processed} = {accuracy:.2%}", end="")
    print(f" | output {output_path}")

    return {
        "processed": processed,
        "correct": correct,
        "accuracy": accuracy,
        "pass_at_k": args.num_samples,
        "elapsed_seconds": elapsed,
        "output_file": str(output_path),
    }


def derive_paths(output_file: str, label: str, multi: bool) -> Tuple[Path, Path]:
    """Single dataset uses --output-file; multiple datasets write separate files by label in the same directory."""
    base = Path(output_file)
    if base.suffix.lower() != ".jsonl":
        raise ValueError("Currently only .jsonl output is supported.")
    if not multi:
        return base, base.with_suffix(".summary.json")
    out = base.parent / f"{label}.jsonl"
    return out, out.with_suffix(".summary.json")


# ---------------------------------------------------------------------------
# Main loop (ported from run_evals: multi-dataset + per-dataset engine + aggregate summary)
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    if (
        not args.baseline
        and args.plugin_mode
        and args.inject_phase == "think"
        and (args.think_start_id is None or args.think_end_id is None)
    ):
        raise ValueError(
            "plugin mode cannot resolve think boundaries from tokenizer; "
            "provide --think-start-id and --think-end-id explicitly."
        )

    setup_determinism(args.deterministic, args.cuda_launch_blocking)

    datasets = args.data
    multi = len(datasets) > 1
    if multi and args.dataset_name:
        print("[warn] --dataset-name applies to all input files in multi-dataset mode.")

    all_results: Dict[str, Dict[str, Any]] = {}
    llm = tokenizer = None

    try:
        for data in datasets:
            label = Path(data).stem
            dataset_name = args.dataset_name
            output_path, summary_path = derive_paths(args.output_file, label, multi)
            if not multi and args.summary_file:   # single dataset: honor explicit --summary-file
                summary_path = Path(args.summary_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Engine management: rebuild or reuse
            if llm is None or args.rebuild_engine_per_dataset:
                if llm is not None:
                    cleanup_llm(llm)
                    llm = None
                print(f"=== [{label}] building vLLM engine ===")
                llm, tokenizer = build_llm(args)

            all_results[label] = run_one_dataset(
                llm,
                tokenizer,
                args,
                data=data,
                label=label,
                output_path=output_path,
                summary_path=summary_path,
                dataset_name=dataset_name,
            )
    finally:
        if llm is not None:
            cleanup_llm(llm)

    # Write aggregate summary for multi-dataset runs
    if multi:
        agg_path = Path(args.output_file).parent / "summary_all.json"
        write_summary(agg_path, model=args.model, datasets=all_results)
        print(f"\n=== Summary ({len(all_results)} datasets, pass@{args.num_samples}) ===")
        for label, res in all_results.items():
            acc = res.get("accuracy", 0.0)
            print(f"  {label}: acc={acc:.2%}  processed={res.get('processed', 0)}")
        print(f"Aggregate summary -> {agg_path}")


if __name__ == "__main__":
    main()