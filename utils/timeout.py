"""Evaluation timeout utilities.

- ``time_limit`` / ``EvalTimeout``: SIGALRM-based, for lightweight logic on the main thread (math eval, etc.).
- ``run_with_process_timeout``: runs in a subprocess; terminate/kill on timeout, for HumanEval/MBPP ``exec`` scenarios.
"""
from __future__ import annotations

import multiprocessing as mp
import signal
import sys
from contextlib import contextmanager
from typing import Any, Callable, Optional, Tuple

DEFAULT_CODE_EVAL_TIMEOUT = 10.0


class EvalTimeout(Exception):
    """Per-sample evaluation timeout."""


@contextmanager
def time_limit(seconds: float):
    """Raise EvalTimeout if the with block exceeds seconds; seconds<=0 means no limit."""
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise EvalTimeout(f"evaluation exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _mp_context() -> mp.context.BaseContext:
    if sys.platform != "win32" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _process_target(
    queue: mp.Queue,
    target: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> None:
    try:
        queue.put(("ok", target(*args, **kwargs)))
    except BaseException as exc:
        queue.put(("err", repr(exc)))


def run_with_process_timeout(
    target: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = DEFAULT_CODE_EVAL_TIMEOUT,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Run target in a subprocess; terminate the child on timeout.

    Returns:
        (completed, result, error_message)
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    kwargs = kwargs or {}
    ctx = _mp_context()
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_process_target,
        args=(queue, target, args, kwargs),
        daemon=True,
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, None, f"execution timed out after {timeout:g}s"

    if queue.empty():
        code = proc.exitcode
        if code not in (0, None):
            return False, None, f"worker exited with code {code}"
        return False, None, "worker produced no result"

    status, payload = queue.get()
    if status == "ok":
        return True, payload, None
    return False, None, str(payload)
