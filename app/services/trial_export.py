"""Materialize the ONNX artifact for ANY Pareto trial on demand.

The pipeline persists (a) the post-ingestion ONNX and (b) every trial's
compression config. Exporting trial N = re-applying config N to that exact
graph with the same engine primitives the pipeline used — deterministic, so
the artifact is cached in storage and re-served on subsequent requests.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from fastapi import HTTPException
from sqlmodel import Session

from app.config import get_settings
from app.repositories import get_cached_result
from app.services.storage import StorageError, get_storage, ingested_key, trial_artifact_key

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _model_lock(model_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(model_id, threading.Lock())


def _trial_config(session: Session, model_id: str, user_id: str, trial_number: int) -> dict:
    cached = get_cached_result(session, model_id, "pareto_configs", user_id=user_id)
    if not cached or not (cached.get("trials") or {}):
        raise HTTPException(status_code=409, detail={
            "code": "trial_export_unavailable",
            "message": "Per-trial export needs the trial configs recorded during "
                       "optimization — this model predates that. Re-run "
                       "optimization (re-import the model) to enable it.",
        })
    config = (cached["trials"] or {}).get(str(trial_number))
    if config is None:
        raise HTTPException(status_code=404, detail={
            "code": "trial_not_found",
            "message": f"Trial {trial_number} does not exist in this experiment.",
        })
    return config


def export_trial(
    session: Session, model_id: str, user_id: str, trial_number: int,
) -> dict:
    """Build (or reuse) the artifact for one trial. Returns download metadata."""
    settings = get_settings()
    storage = get_storage()
    key = trial_artifact_key(model_id, trial_number)

    config = _trial_config(session, model_id, user_id, trial_number)

    if not storage.exists(key):
        src_key = ingested_key(model_id)
        if not storage.exists(src_key):
            raise HTTPException(status_code=409, detail={
                "code": "trial_export_unavailable",
                "message": "The ingested source graph for this model is not "
                           "stored — re-run optimization to enable per-trial "
                           "export.",
            })
        with _model_lock(model_id):
            if not storage.exists(key):  # double-checked after the lock
                _materialize(storage, src_key, key, config, settings.trial_export_max_mb)

    data = storage.read_bytes(key)
    return {
        "status": "ready",
        "trialNumber": trial_number,
        "fileName": Path(key).name,
        "sizeBytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "downloadPath": f"/api/models/{model_id}/pareto/trials/{trial_number}/artifact",
    }


def _materialize(
    storage, src_key: str, dst_key: str, config: dict, max_mb: int,
) -> None:
    import tempfile

    import onnx
    from peops.core.compression_actions import ActionTranslator
    from peops.graph.onnx_analyzer import OnnxAnalyzer
    from peops.graph.onnx_transformer import OnnxTransformer
    from peops.sdk import _reconstruct_model

    raw = storage.read_bytes(src_key)
    if len(raw) > max_mb * 1e6:
        raise HTTPException(status_code=413, detail={
            "code": "model_too_large",
            "message": f"Source model exceeds the {max_mb} MB per-trial export limit.",
        })

    model = onnx.load_from_string(raw)
    graph_info = OnnxAnalyzer().analyze(model)  # deterministic re-derivation
    compressed = _reconstruct_model(
        model, graph_info, config, ActionTranslator(), OnnxTransformer(),
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        onnx.save(compressed, tmp.name)
        storage.upload_file(tmp.name, dst_key)


def stream_trial_artifact(session: Session, model_id: str, user_id: str, trial_number: int):
    """(iterator, size, filename) for the cached trial artifact; 404 if absent."""
    # Validate ownership of the trial before serving bytes.
    _trial_config(session, model_id, user_id, trial_number)
    storage = get_storage()
    key = trial_artifact_key(model_id, trial_number)
    try:
        stream, size = storage.open_stream(key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "trial_artifact_missing",
            "message": "This trial's artifact has not been exported yet — call "
                       "the export endpoint first.",
        }) from None
    return stream, size, Path(key).name
