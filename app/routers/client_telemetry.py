"""Public API-key-authed endpoints for the astra-ai-sdk pip package.

  POST /api/v1/telemetry/{deployment_id}/batch   — ship telemetry batches
  GET  /api/v1/artifacts/{deployment_id}         — pull the deployed artifact
  GET  /api/v1/artifacts/{deployment_id}/info    — artifact metadata (ETag basis)

Like /api/v1/infer these sit OUTSIDE the session-cookie gate: callers are
processes holding a deployment API key, not browsers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from sqlmodel import Session

from app.config import get_settings
from app.db import get_session
from app.routers.infer import _resolve
from app.schemas.client_telemetry import BatchAccepted, TelemetryBatch
from app.services import apikeys
from app.services.client_telemetry import ingest_batch
from app.services.limits import limiter
from app.services.storage import StorageError, get_storage

router = APIRouter(prefix="/v1", tags=["client-telemetry"])


@router.post("/telemetry/{deployment_id}/batch")
@limiter.limit(get_settings().rate_limit_telemetry)
def telemetry_batch(
    deployment_id: str,
    body: TelemetryBatch,
    request: Request,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> BatchAccepted:
    key, dep, model = _resolve(deployment_id, authorization, session)
    apikeys.touch_key(session, key)
    try:
        accepted, dropped = ingest_batch(session, dep, model, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "batch_too_large", "message": str(exc),
        }) from exc
    return BatchAccepted(accepted=accepted, dropped=dropped)


def _artifact_bytes(model) -> tuple[bytes, str]:
    key = model.artifact_key
    if not key:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        })
    try:
        data = get_storage().read_bytes(key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        }) from None
    return data, key


@router.get("/artifacts/{deployment_id}/info")
def artifact_info(
    deployment_id: str,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> dict:
    _key, _dep, model = _resolve(deployment_id, authorization, session)
    data, key = _artifact_bytes(model)
    return {
        "fileName": Path(key).name,
        "sizeBytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "kind": "onnx" if key.endswith(".onnx") else "npz",
    }


@router.get("/artifacts/{deployment_id}")
def artifact_download(
    deployment_id: str,
    authorization: str | None = Header(default=None),
    if_none_match: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> Response:
    """Stream the deployed artifact to the SDK. ETag = sha256 so clients cache
    on disk and re-download only when the artifact actually changed."""
    _key, _dep, model = _resolve(deployment_id, authorization, session)
    data, key = _artifact_bytes(model)
    etag = f'"{hashlib.sha256(data).hexdigest()}"'
    if if_none_match and if_none_match.strip() == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(key).name}"',
            "Content-Length": str(len(data)),
            "ETag": etag,
        },
    )
