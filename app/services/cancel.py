"""Cooperative job cancellation + timeout.

No queue can preempt the running native ML code (torch/onnxruntime/optuna run in
C++ and release the GIL), so cancellation is cooperative: the engine polls a
`should_cancel()` predicate at every phase/trial checkpoint. This module backs
that predicate two ways:

* in-process set — works when the worker runs on a thread in the API process
  (inline mode / single box), and is instant.
* Redis flag — works across processes, so an API replica can cancel a job
  running in a separate worker container.

`make_should_cancel` also folds in a wall-clock deadline so a job that overruns
its timeout fails cleanly through the same PipelineCancelled path.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from app.config import get_settings

_local_cancelled: set[str] = set()
_redis_client = None
_redis_failed = False


def _redis():
    global _redis_client, _redis_failed
    if _redis_failed:
        return None
    if _redis_client is None:
        try:
            import redis

            _redis_client = redis.Redis.from_url(
                get_settings().redis_url, socket_timeout=2, socket_connect_timeout=2,
            )
            _redis_client.ping()
        except Exception:  # noqa: BLE001 — Redis is optional (inline mode)
            _redis_failed = True
            _redis_client = None
    return _redis_client


def _key(run_id: str) -> str:
    return f"astra:cancel:{run_id}"


def request_cancel(run_id: str) -> None:
    """Signal a job to stop at its next checkpoint (idempotent)."""
    _local_cancelled.add(run_id)
    r = _redis()
    if r is not None:
        try:
            r.set(_key(run_id), "1", ex=24 * 3600)
        except Exception:  # noqa: BLE001 — best-effort
            pass


def is_cancelled(run_id: str) -> bool:
    if run_id in _local_cancelled:
        return True
    r = _redis()
    if r is not None:
        try:
            return bool(r.exists(_key(run_id)))
        except Exception:  # noqa: BLE001
            return False
    return False


def clear_cancel(run_id: str) -> None:
    _local_cancelled.discard(run_id)
    r = _redis()
    if r is not None:
        try:
            r.delete(_key(run_id))
        except Exception:  # noqa: BLE001
            pass


def make_should_cancel(run_id: str, deadline_ts: float | None = None) -> Callable[[], bool]:
    """Build the predicate the engine polls: cancelled OR past the deadline."""
    def _check() -> bool:
        if deadline_ts is not None and time.time() > deadline_ts:
            return True
        return is_cancelled(run_id)

    return _check


def reset_for_tests() -> None:
    _local_cancelled.clear()
