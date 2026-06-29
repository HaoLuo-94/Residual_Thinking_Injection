"""GPUModelRunner patch: publish batch layout before each forward step.

VERSION SEAM
============
This is the only file coupled to vLLM internal APIs. After upgrading vLLM, if you
see AttributeError / ImportError, usually only these three places need changes:
  1) GPUModelRunner import path
  2) How self.input_batch.req_ids / .num_reqs are accessed
  3) How scheduler_output.num_scheduled_tokens is accessed
Target: vLLM V1 engine (approx. 0.8 ~ 0.11).
"""
from __future__ import annotations

from .runtime import BATCH_CTX, BatchLayout

_PATCHED = False


def patch_runner() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # SEAM 1

    orig_execute = GPUModelRunner.execute_model

    def patched_execute(self, scheduler_output, *args, **kwargs):
        try:
            num_reqs = self.input_batch.num_reqs                 # SEAM 2
            raw_ids = self.input_batch.req_ids[:num_reqs]
            req_ids = [r for r in raw_ids if r is not None]
            nst = scheduler_output.num_scheduled_tokens          # SEAM 3
            num_scheduled = [int(nst[r]) for r in req_ids]
            # print("[patch] req_ids=", req_ids, "num_scheduled=", num_scheduled)  # add this line
            BATCH_CTX.set(BatchLayout(req_ids=req_ids, num_scheduled=num_scheduled))
        except Exception:
            BATCH_CTX.set(None)   # On failure, fall back to no injection; inference unaffected
        try:
            return orig_execute(self, scheduler_output, *args, **kwargs)
        finally:
            BATCH_CTX.set(None)

    GPUModelRunner.execute_model = patched_execute
    _PATCHED = True
