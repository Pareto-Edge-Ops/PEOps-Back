"""GET /api/models/:id/pareto (+ /pareto/scene).

Serves ONLY real Optuna trial results cached at job completion. There is no
generated fallback. Weights-only checkpoints (raw state_dict) cannot be
executed, so latency/accuracy are unmeasurable — they get a structured 404
the SPA turns into an explanatory empty state."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.auth.dependencies import CurrentUser
from app.db import get_session
from app.repositories import get_cached_result, owned_model
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
