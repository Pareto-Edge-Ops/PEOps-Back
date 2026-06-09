"""GET /api/models/:id/architecture (+ /architecture/scene).

Serves ONLY the cached mapping of the model's actual analysis — the real ONNX
graph + UOSA sensitivities for executable models, or the real tensor inventory
+ SQNR sensitivities for weights-only checkpoints. There is no generated
fallback: a model that was never analyzed gets a structured 404.

`/scene` adds every value the SPA's 3D renderer would otherwise compute
client-side (perceptron positions, edge geometry, colors, camera, layer
descriptions)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.auth.dependencies import CurrentUser
from app.db import get_session
from app.repositories import get_cached_result, owned_model
from app.services.viz.arch_scene import build_architecture_scene

router = APIRouter(tags=["architecture"])


def get_architecture_payload(session: Session, model_id: str, user_id: str) -> dict:
    owned_model(session, model_id, user_id)

    cached = get_cached_result(session, model_id, "architecture", user_id=user_id)
    if cached:
        return cached

    raise HTTPException(status_code=404, detail={
        "code": "not_analyzed",
        "message": "No architecture analysis for this model yet — it is produced "
                   "by the ingestion pipeline when a model is uploaded.",
    })


@router.get("/models/{model_id}/architecture")
def architecture(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> JSONResponse:
    return JSONResponse(get_architecture_payload(session, model_id, current_user.id))


@router.get("/models/{model_id}/architecture/scene")
def architecture_scene(
    model_id: str,
    current_user: CurrentUser,
    segments: str | None = Query(default=None, description="1 to inline explicit line segments"),
    session: Session = Depends(get_session),
) -> JSONResponse:
    arch = get_architecture_payload(session, model_id, current_user.id)
    scene = build_architecture_scene(arch, include_segments=segments == "1")
    return JSONResponse(scene)
