# LiveCodeBench evaluator (official-faithful)

Grading, pass@k, and dataset decoding are delegated to the **vendored official
code** in `lcb_official/` (copied unchanged from
https://github.com/LiveCodeBench/LiveCodeBench, `lcb_runner/`). Only two
intra-package import lines were rewritten to be relative. The adapter
`livecodebench_evaluator.py` wires that core into the same `BaseCodeEvaluator`
interface your MBPP evaluator uses, and reuses the official prompt template and
`extract_code` logic.

## Layout
    livecodebench_evaluator.py     # adapter -> your sources.base_evaluator.BaseCodeEvaluator
    lcb_official/
        testing_util.py                    # official run_test  (UNCHANGED)
        compute_code_generation_metrics.py # official check_correctness / codegen_metrics
        pass_k_utils.py                    # official estimate_pass_at_k (UNCHANGED)
        code_generation_benchmark.py       # official CodeGenerationProblem / loader (UNCHANGED)

Put `livecodebench_evaluator.py` next to your `mbpp` evaluator and `lcb_official/`
on the import path (same dir is fine).

## Mode 1 — drop-in pass@1 (same call shape as MBPP)
    from livecodebench_evaluator import evaluate_livecodebench_accuracy
    evaluate_livecodebench_accuracy(model, tokenizer, test_path="", device=device,
                                    max_new_tokens=4096)
`test_path=""` pulls `livecodebench/code_generation_lite` from the hub.
Control the slice with env vars: LCB_VERSION (e.g. release_v6, default
release_latest), LCB_START_DATE / LCB_END_DATE (YYYY-MM-DD).

## Mode 2 — official pass@k over n samples you generate
    from livecodebench_evaluator import (
        LiveCodeBenchEvaluator, score_with_codegen_metrics, grouped_pass_at_1)
    ev = LiveCodeBenchEvaluator()
    items = ev.load_dataset("")                       # or a local .parquet
    # generations_list[i] = [extracted_code_1, ..., extracted_code_n] for items[i]
    metrics, results, _ = score_with_codegen_metrics(items, generations_list, k_list=[1,5])
    print(metrics["pass@1"], metrics["pass@5"])
    print(grouped_pass_at_1(items, results, "difficulty"))

## Notes
- `is_correct` returns True iff every per-test result is strictly positive,
  exactly how the official `compute_metrics_from_results` decides correctness.
- Grading always runs in a separate process with the official per-test +
  global timeouts. Run untrusted code in a container regardless.
- To refresh the vendored core, re-copy the four files from upstream and
  re-apply the two relative-import edits in compute_code_generation_metrics.py.
