"""Real model serving — load a compressed ONNX artifact and run inference.

This is the missing half of the loop: the optimization pipeline only ran models
during its post-compression benchmark (`engine.adapter._benchmark_model`). Here
the same onnxruntime mechanics are factored into a reusable, cached serving path
that the public `/api/v1/infer` endpoint and the traffic simulator both drive.

Sessions are expensive to build (deserialize + graph optimize), so they are kept
in a small bounded LRU keyed by the artifact storage key. Weights-only (.npz)
checkpoints have no executable graph and are rejected with a structured error.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from time import perf_counter
from typing import Any

from app.config import get_settings
from app.services.storage import get_storage

# ── structured errors (mapped to HTTP by the router) ─────────────────────────


class InferenceError(Exception):
    """A client-facing inference failure with a stable `code` + HTTP `status`."""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ── ORT tensor-type → numpy dtype ────────────────────────────────────────────

_ORT_DTYPE = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def _np_dtype(ort_type: str):
    import numpy as np

    return np.dtype(_ORT_DTYPE.get(ort_type, "float32"))


def is_executable(artifact_key: str | None) -> bool:
    """True only for ONNX artifacts — weights-only .npz can't be served."""
    return bool(artifact_key) and artifact_key.endswith(".onnx")


# ── bounded session cache ────────────────────────────────────────────────────

_session_cache: "OrderedDict[str, Any]" = OrderedDict()
_cache_lock = threading.Lock()


def reset_inference_cache() -> None:
    """Drop all cached sessions (tests swap storage/DB between runs)."""
    with _cache_lock:
        _session_cache.clear()


def _load_session(artifact_key: str):
    if not is_executable(artifact_key):
        raise InferenceError(
            "weights_only_checkpoint",
            "This deployment's artifact is weights-only (state_dict) and has no "
            "executable graph — it cannot be served. Deploy an ONNX model instead.",
            status=422,
        )
    storage = get_storage()
    if not storage.exists(artifact_key):
        raise InferenceError("no_artifact", "Compressed artifact is missing.", status=404)
    import onnxruntime as ort

    data = storage.read_bytes(artifact_key)
    try:
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        raise InferenceError("bad_artifact", f"Failed to load model: {exc}", status=500) from exc
    # Warm the graph once so the first served request isn't a cold outlier.
    try:
        session.run(None, _build_feed(session, None, batch=1))
    except Exception:  # noqa: BLE001 — warmup is best-effort
        pass
    return session


def get_session(artifact_key: str):
    """Return a cached (warm) ORT session for the artifact, building on miss."""
    with _cache_lock:
        cached = _session_cache.get(artifact_key)
        if cached is not None:
            _session_cache.move_to_end(artifact_key)
            return cached
    session = _load_session(artifact_key)  # slow — built outside the lock
    with _cache_lock:
        _session_cache[artifact_key] = session
        _session_cache.move_to_end(artifact_key)
        cap = max(1, get_settings().inference_cache_size)
        while len(_session_cache) > cap:
            _session_cache.popitem(last=False)
    return session


# ── shape / feed helpers ─────────────────────────────────────────────────────


def _concrete_shape(shape: list, batch: int) -> list[int]:
    """Resolve dynamic dims: the leading dynamic dim is the batch, the rest 1."""
    out: list[int] = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        else:
            out.append(max(1, batch) if idx == 0 else 1)
    return out


def _build_feed(session, inputs: dict | None, batch: int) -> dict:
    import numpy as np

    specs = session.get_inputs()
    feed: dict = {}
    if inputs is not None:
        expected = {s.name for s in specs}
        missing = expected - set(inputs)
        if missing:
            raise InferenceError(
                "bad_input", f"missing required inputs: {sorted(missing)}", status=422,
            )
        for s in specs:
            try:
                feed[s.name] = np.asarray(inputs[s.name], dtype=_np_dtype(s.type))
            except (ValueError, TypeError) as exc:
                raise InferenceError(
                    "bad_input", f"input '{s.name}' is not a valid tensor: {exc}", status=422,
                ) from exc
        return feed
    # No inputs provided → synthesize a valid random batch (sim / smoke tests).
    cap = max(1, get_settings().max_infer_batch)
    batch = min(max(1, batch), cap)
    for s in specs:
        shape = _concrete_shape(list(s.shape), batch)
        dt = _np_dtype(s.type)
        if np.issubdtype(dt, np.floating):
            feed[s.name] = np.random.rand(*shape).astype(dt)
        elif np.issubdtype(dt, np.bool_):
            feed[s.name] = (np.random.rand(*shape) > 0.5)
        else:
            feed[s.name] = (np.random.rand(*shape) * 10).astype(dt)
    return feed


def _summarize_outputs(session, outs: list) -> list[dict]:
    """Return output metadata; include values only when small enough to be cheap."""
    import numpy as np

    names = [o.name for o in session.get_outputs()]
    summary: list[dict] = []
    for name, arr in zip(names, outs):
        arr = np.asarray(arr)
        item: dict = {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.size <= 256:
            item["data"] = arr.tolist()
        summary.append(item)
    return summary


def input_spec(artifact_key: str) -> list[dict]:
    """Concrete input spec (name/shape/dtype) for SDK snippets + UI. Batch=1."""
    session = get_session(artifact_key)
    return [
        {"name": s.name, "shape": _concrete_shape(list(s.shape), 1),
         "dtype": str(_np_dtype(s.type))}
        for s in session.get_inputs()
    ]


def run_inference(
    artifact_key: str, inputs: dict | None = None, *, batch: int = 1,
) -> tuple[list[dict], float]:
    """Run one inference. Returns (output summaries, latency_ms).

    Raises InferenceError on bad input / unservable artifact; lets unexpected
    runtime errors propagate so the caller records a failed event.
    """
    session = get_session(artifact_key)
    feed = _build_feed(session, inputs, batch)
    t0 = perf_counter()
    outs = session.run(None, feed)
    latency_ms = (perf_counter() - t0) * 1000.0
    return _summarize_outputs(session, outs), latency_ms


def record_event(
    db,
    *,
    user_id: str,
    model_id: str,
    deployment_id: str,
    latency_ms: float,
    success: bool,
    error_code: str | None = None,
    batch_size: int = 1,
    region: str = "",
    ts: str | None = None,
) -> None:
    """Persist one inference as a raw telemetry fact. `ts` lets the simulator
    backfill historical events; defaults to now."""
    from datetime import datetime, timezone

    from app.config import iso
    from app.dbmodels import InferenceEventRow

    db.add(InferenceEventRow(
        user_id=user_id,
        model_id=model_id,
        deployment_id=deployment_id,
        ts=ts or iso(datetime.now(timezone.utc)),
        latency_ms=round(float(latency_ms), 3),
        success=success,
        error_code=error_code,
        batch_size=batch_size,
        region=region,
    ))
    db.commit()
