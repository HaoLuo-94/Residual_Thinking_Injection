"""Vendored, UNCHANGED evaluation core from the official LiveCodeBench repo
(https://github.com/LiveCodeBench/LiveCodeBench, lcb_runner/), so that grading,
pass@k and dataset decoding match the official results exactly.

Only the two intra-package import lines in compute_code_generation_metrics.py
were rewritten to be relative. Do not edit these files; update them by
re-copying from the upstream repo.
"""
from .testing_util import run_test
from .pass_k_utils import (
    estimate_pass_at_k,
    compute_metrics_from_results,
    extract_instance_results,
)
from .compute_code_generation_metrics import (
    check_correctness,
    codegen_metrics,
    evaluate_generations,
)
from .code_generation_benchmark import (
    CodeGenerationProblem,
    Test,
    TestType,
    Platform,
    Difficulty,
    load_code_generation_dataset,
)
