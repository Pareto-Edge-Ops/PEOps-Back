"""POST /api/v1/infer/{deployment_id} — the public, API-key-authed serving path.

This is where REAL user traffic flows. Unlike every other router it is NOT
behind the session-cookie gate (real callers are apps/devices, not browsers);
it authenticates with `Authorization: Bearer <astra_sk_…>` instead. Every call —
success or failure — is recorded as an InferenceEventRow, which is exactly the
raw data the Telemetry Dashboard aggregates.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
from app.db import get_session
from app.dbmodels import ApiKeyRow, DeploymentRow, ModelRow
from app.services import apikeys
from app.services.inference import InferenceError, record_event, run_inference
from app.services.limits import limiter

router = APIRouter(prefix="/v1/infer", tags=["inference"])


class InferRequest(BaseModel):
    inputs: dict | None = None      # {input_name: nested-list}; null → random valid input
    region: str | None = None       # optional caller-supplied region tag
    batch: int | None = None        # used only when inputs is null (synthesized)


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def _resolve(
    deployment_id: str, authorization: str | None, session: Session,
) -> tuple[ApiKeyRow, DeploymentRow, ModelRow]:
    key = apikeys.resolve_key(session, _bearer(authorization))
    if key is None:
        raise HTTPException(status_code=401, detail={
            "code": "invalid_api_key", "message": "Missing or invalid API key.",
        })
    # 404 (not 403) on mismatch — never confirm a deployment the key can't reach.
    dep = session.exec(
        select(DeploymentRow).where(DeploymentRow.id == deployment_id)
    ).first()
    if dep is None or key.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail={
            "code": "deployment_not_found", "message": "Deployment not found.",
        })
    if dep.status == "paused":
        raise HTTPException(status_code=409, detail={
            "code": "deployment_paused", "message": "This deployment is paused.",
        })
    model = session.get(ModelRow, dep.model_id)
    if model is None or not model.artifact_key:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        })
    return key, dep, model


@router.post("/{deployment_id}")
@limiter.limit(get_settings().rate_limit_infer)
def infer(
    deployment_id: str,
    body: InferRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> dict:
    key, dep, model = _resolve(deployment_id, authorization, session)
    apikeys.touch_key(session, key)
    region = (body.region or dep.region or "").strip()
    batch = body.batch or 1

    try:
        outputs, latency_ms = run_inference(
            model.artifact_key, body.inputs, batch=batch,
        )
    except InferenceError as exc:
        # A bad request / unservable artifact is still a real, recorded event.
        record_event(
            session, user_id=model.user_id, model_id=model.id,
            deployment_id=dep.id, latency_ms=0.0, success=False,
            error_code=exc.code, batch_size=batch, region=region,
        )
        raise HTTPException(status_code=exc.status, detail={
            "code": exc.code, "message": exc.message,
        }) from exc
    except Exception as exc:  # noqa: BLE001 — unexpected runtime failure
        record_event(
            session, user_id=model.user_id, model_id=model.id,
            deployment_id=dep.id, latency_ms=0.0, success=False,
            error_code="inference_error", batch_size=batch, region=region,
        )
        raise HTTPException(status_code=500, detail={
            "code": "inference_error", "message": str(exc),
        }) from exc

    record_event(
        session, user_id=model.user_id, model_id=model.id,
        deployment_id=dep.id, latency_ms=latency_ms, success=True,
        batch_size=batch, region=region,
    )
    return {
        "requestId": uuid.uuid4().hex,
        "deploymentId": dep.id,
        "modelId": model.id,
        "latencyMs": round(latency_ms, 3),
        "outputs": outputs,
    }
