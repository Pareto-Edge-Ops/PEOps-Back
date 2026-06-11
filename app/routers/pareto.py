"""GET /api/models/:id/pareto (+ /pareto/scene).

Serves ONLY real Optuna trial results cached at job completion. There is no
generated fallback. Weights-only checkpoints (raw state_dict) cannot be
executed, so latency/accuracy are unmeasurable — they get a structured 404
the SPA turns into an explanatory empty state."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from app.auth.dependencies import CurrentUser
from app.db import get_session
from app.repositories import get_cached_result, owned_model
from app.services.trial_export import export_trial, stream_trial_artifact
from app.services.viz.pareto_scene import build_pareto_scene

router = APIRouter(tags=["pareto"])


def get_pareto_payload(session: Session, model_id: str, user_id: str) -> dict:
    model = owned_model(session, model_id, user_id)

    cached = get_cached_result(session, model_id, "pareto", user_id=user_id)
    if cached:
        return cached

    if model.weights_only:
        raise HTTPException(status_code=404, detail={
            "code": "weights_only_checkpoint",
            "message": "This checkpoint is weights-only (state_dict) — it has no "
                       "executable graph, so latency and accuracy cannot be measured "
                       "and Pareto search is not possible. Export the full model "
                       "(torch.save(model, path)) or ONNX to enable it.",
        })
    raise HTTPException(status_code=404, detail={
        "code": "not_analyzed",
        "message": "No Pareto results for this model yet — analysis is incomplete "
                   "or failed.",
    })


@router.get("/models/{model_id}/pareto")
def pareto(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> JSONResponse:
    return JSONResponse(get_pareto_payload(session, model_id, current_user.id))


@router.post("/models/{model_id}/pareto/trials/{trial_number}/export")
def trial_export(
    model_id: str,
    trial_number: int,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> dict:
    """Materialize (or reuse) the ONNX artifact for one Pareto trial.

    Synchronous by design: re-applying a trial config is graph surgery on an
    already-ingested ONNX — seconds at the <=24-compressible-op scale this
    pipeline caps at. Idempotent: the artifact is cached in storage."""
    owned_model(session, model_id, current_user.id)
    return export_trial(session, model_id, current_user.id, trial_number)


@router.get("/models/{model_id}/pareto/trials/{trial_number}/artifact")
def trial_artifact(
    model_id: str,
    trial_number: int,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    owned_model(session, model_id, current_user.id)
    stream, size, name = stream_trial_artifact(
        session, model_id, current_user.id, trial_number)
    return StreamingResponse(
        stream,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Content-Length": str(size),
        },
    )


@router.get("/models/{model_id}/pareto/scene")
def pareto_scene(
    model_id: str,
    current_user: CurrentUser,
    maxLatency: float | None = Query(default=None, gt=0),
    maxAccuracyDrop: float | None = Query(default=None, ge=0),
    maxSize: float | None = Query(default=None, gt=0),
    session: Session = Depends(get_session),
) -> JSONResponse:
    exp = get_pareto_payload(session, model_id, current_user.id)
    scene = build_pareto_scene(
        exp,
        max_latency=maxLatency,
        max_accuracy_drop=maxAccuracyDrop,
        max_size=maxSize,
    )
    return JSONResponse(scene)
