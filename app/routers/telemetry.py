"""GET /api/models/:id/telemetry/* — REAL live data, benchmark as cold-start.

Once a deployed model has served real inference (inference_events exist), every
endpoint aggregates that live traffic — KPIs carry a true window-over-window
delta, charts are time-bucketed, the deployments table shows monitor-maintained
live metrics, and alerts are real drift events. Before any traffic exists the
same endpoints fall back to the post-compression benchmark, byte-identical to
before (so a not-yet-deployed model — and the contract tests — are unchanged).
A model with neither benchmark nor traffic still gets the honest structured 404.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.auth.dependencies import CurrentUser, current_user_id_for_stream
from app.config import get_settings, iso
from app.db import get_session, open_session
from app.dbmodels import AlertRow, DeploymentRow, ModelRow
from app.repositories import get_cached_result, get_model, owned_model
from app.schemas.telemetry import (
    Alert,
    Deployment,
    Percentiles,
    TelemetryKpi,
    TelemetryPoint,
)
from app.services import telemetry_agg

router = APIRouter(prefix="/models/{model_id}/telemetry", tags=["telemetry"])

_RANGES = {"1h", "6h", "24h", "7d", "30d"}


def _model(session: Session, model_id: str, user_id: str) -> ModelRow:
    return owned_model(session, model_id, user_id)


def _benchmark(session: Session, model: ModelRow) -> dict:
    bench = get_cached_result(session, model.id, "benchmark", user_id=model.user_id)
    if bench:
        return bench
    if model.weights_only:
        raise HTTPException(status_code=404, detail={
            "code": "weights_only_checkpoint",
            "message": "This checkpoint is weights-only (state_dict) — it cannot be "
                       "executed, so no benchmark exists. Export the full model or "
                       "ONNX to enable benchmarking.",
        })
    raise HTTPException(status_code=404, detail={
        "code": "no_benchmark",
        "message": "No benchmark measurements for this model yet — the benchmark "
                   "runs automatically when compression completes.",
    })


def _range(value: str) -> str:
    return value if value in _RANGES else "24h"


@router.get("/kpi")
def kpi(
    model_id: str,
    current_user: CurrentUser,
    range: str = Query(default="24h"),  # noqa: A002 — public URL param name
    session: Session = Depends(get_session),
) -> TelemetryKpi:
    model = _model(session, model_id, current_user.id)
    if telemetry_agg.has_any_events(session, model_id):
        bench = get_cached_result(session, model_id, "benchmark", user_id=model.user_id)
        return telemetry_agg.kpi_live(session, model_id, bench, _range(range))
    return telemetry_agg.kpi_from_benchmark(_benchmark(session, model))


@router.get("/series")
def series(
    model_id: str,
    current_user: CurrentUser,
    range: str = Query(default="24h"),  # noqa: A002
    session: Session = Depends(get_session),
) -> list[TelemetryPoint]:
    model = _model(session, model_id, current_user.id)
    if telemetry_agg.has_any_events(session, model_id):
        return telemetry_agg.series_live(session, model_id, _range(range))
    return telemetry_agg.series_from_benchmark(_benchmark(session, model))


@router.get("/percentiles")
def percentiles(
    model_id: str,
    current_user: CurrentUser,
    range: str = Query(default="24h"),  # noqa: A002
    session: Session = Depends(get_session),
) -> Percentiles:
    model = _model(session, model_id, current_user.id)
    if telemetry_agg.has_any_events(session, model_id):
        return telemetry_agg.percentiles_live(session, model_id, _range(range))
    return telemetry_agg.percentiles_from_benchmark(_benchmark(session, model))


@router.get("/meta")
def meta(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> dict:
    """Lets the SPA label KPIs (live vs benchmark) and show a liveness dot."""
    _model(session, model_id, current_user.id)
    live = telemetry_agg.has_any_events(session, model_id)
    deps = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    return {
        "source": "live" if live else "benchmark",
        "deployments": len(deps),
        "liveDeployments": sum(1 for d in deps if d.status != "paused"),
    }


@router.get("/deployments")
def deployments(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> list[Deployment]:
    _model(session, model_id, current_user.id)
    rows = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    return [
        Deployment(
            endpoint=d.endpoint, region=d.region, qps=d.qps, p95=d.p95,
            errorsPct=d.errors_pct, status=d.status,  # type: ignore[arg-type]
        )
        for d in rows
    ]


@router.get("/alerts")
def alerts(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> list[Alert]:
    _model(session, model_id, current_user.id)
    rows = session.exec(
        select(AlertRow).where(AlertRow.model_id == model_id)
    ).all()
    rows.sort(key=lambda r: r.at, reverse=True)
    return [
        Alert(id=a.id, level=a.level, title=a.title, body=a.body, at=a.at)  # type: ignore[arg-type]
        for a in rows
    ]


class SimulateRequest(BaseModel):
    count: int = 240
    hours: int = 6
    incidents: bool = True


@router.post("/simulate")
def simulate(
    model_id: str,
    current_user: CurrentUser,
    body: SimulateRequest | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Demo affordance (gated): generate realistic traffic for a deployment, then
    run a monitor pass so live metrics + drift alerts update immediately."""
    if not get_settings().telemetry_sim_enabled:
        raise HTTPException(status_code=403, detail={
            "code": "sim_disabled",
            "message": "Traffic simulation is disabled. Set "
                       "PEOPS_TELEMETRY_SIM_ENABLED=1 to enable it.",
        })
    from app.services import traffic_sim
    from app.services.drift_monitor import drift_monitor_pass

    model = _model(session, model_id, current_user.id)
    dep = traffic_sim.first_live_deployment(session, model_id)
    if dep is None:
        raise HTTPException(status_code=409, detail={
            "code": "no_deployment",
            "message": "Deploy this model first to start collecting telemetry.",
        })
    req = body or SimulateRequest()
    summary = traffic_sim.simulate_burst(
        session, dep, model,
        count=max(1, min(req.count, 2000)),
        hours=max(1, min(req.hours, 168)),
        incidents=req.incidents,
    )
    summary["monitor"] = drift_monitor_pass(session)
    return summary


def _stream_snapshot(session: Session, model_id: str) -> dict:
    live = telemetry_agg.has_any_events(session, model_id)
    deps = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    open_alerts = session.exec(
        select(AlertRow).where(AlertRow.model_id == model_id)
    ).all()
    return {
        "source": "live" if live else "benchmark",
        "deployments": len(deps),
        "liveDeployments": sum(1 for d in deps if d.status != "paused"),
        "openAlerts": len(open_alerts),
        "at": iso(datetime.now(timezone.utc)),
    }


@router.get("/stream")
async def stream(
    model_id: str,
    request: Request,
    user_id: str = Depends(current_user_id_for_stream),
) -> EventSourceResponse:
    """SSE liveness/refresh tick. Each `snapshot` event tells the SPA to refetch
    its telemetry queries (with whatever range is selected) and updates the live
    badge counts — so the dashboard breathes without polling on a fixed timer."""
    with open_session() as s:
        if get_model(s, model_id, user_id) is None:
            raise HTTPException(status_code=404, detail="model not found")

    async def event_source():
        while True:
            if await request.is_disconnected():
                return
            with open_session() as s:
                if get_model(s, model_id, user_id) is None:
                    return
                payload = _stream_snapshot(s, model_id)
            yield {"event": "snapshot", "data": json.dumps(payload)}
            await asyncio.sleep(4.0)

    return EventSourceResponse(event_source())
