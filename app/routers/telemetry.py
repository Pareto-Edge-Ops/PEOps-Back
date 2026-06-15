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
from app.dbmodels import (
    AlertRow,
    DeploymentRow,
    InferenceEventRow,
    ModelRow,
    TelemetrySnapshotRow,
    TelemetryWindowStatsRow,
)
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
    """Lets the SPA label KPIs (live vs benchmark) and show a liveness dot.

    `available`/`reason` are the SPA's single gate for whether kpi/series/
    percentiles will return data: those endpoints serve live events when any
    exist, else fall back to the benchmark, else raise the structured 404. The
    SPA reads this here and simply does not fire (or retry) the doomed requests —
    so a weights-only / not-yet-benchmarked model shows its terminal state
    instead of looping 404s.
    """
    model_obj = _model(session, model_id, current_user.id)
    live = telemetry_agg.has_any_events(session, model_id)
    bench = get_cached_result(
        session, model_id, "benchmark", user_id=model_obj.user_id
    )
    available = live or bool(bench)
    reason = None if available else (
        "weights_only_checkpoint" if model_obj.weights_only else "no_benchmark"
    )
    deps = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    from app.services.client_telemetry import source_counts

    snap_ts = session.exec(
        select(TelemetrySnapshotRow.ts)
        .where(TelemetrySnapshotRow.model_id == model_id)
        .order_by(TelemetrySnapshotRow.ts.desc())  # type: ignore[attr-defined]
        .limit(1)
    ).first()
    return {
        "source": "live" if live else "benchmark",
        # Whether the data endpoints (kpi/series/percentiles) have anything to
        # return; `reason` mirrors the structured 404 code they would otherwise
        # raise (weights_only_checkpoint | no_benchmark) or null when available.
        "available": available,
        "reason": reason,
        "deployments": len(deps),
        "liveDeployments": sum(1 for d in deps if d.status != "paused"),
        # Honest labeling of where live data comes from: hosted /v1/infer
        # ("server") vs peops-sdk local serving ("client").
        "sources": source_counts(session, model_id),
        "lastSnapshotAt": snap_ts,
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


_RANGE_DELTAS = {
    "1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7, "30d": 24 * 30,  # hours
}


def _range_start(range_key: str) -> str:
    from datetime import timedelta

    hours = _RANGE_DELTAS.get(range_key, 24)
    return iso(datetime.now(timezone.utc) - timedelta(hours=hours))


@router.get("/clients")
def clients(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> list[dict]:
    """SDK client hosts serving this model locally — latest snapshot per client
    (empty list when no peops-sdk traffic exists; the SPA hides the panel)."""
    _model(session, model_id, current_user.id)
    rows = session.exec(
        select(TelemetrySnapshotRow)
        .where(TelemetrySnapshotRow.model_id == model_id)
        .order_by(TelemetrySnapshotRow.ts.desc())  # type: ignore[attr-defined]
        .limit(500)
    ).all()
    latest: dict[str, TelemetrySnapshotRow] = {}
    for r in rows:
        if r.client_id not in latest:
            latest[r.client_id] = r
    out = []
    for r in latest.values():
        try:
            runtime = json.loads(r.runtime_json)
        except ValueError:
            runtime = {}
        out.append({
            "clientId": r.client_id,
            "host": runtime.get("host", ""),
            "os": runtime.get("os", ""),
            "arch": runtime.get("arch", ""),
            "provider": runtime.get("provider", ""),
            "sdkVersion": r.sdk_version,
            "ortVersion": runtime.get("ort", ""),
            "lastSeen": r.ts,
            "reqPerMin": r.throughput_rpm,
            "cpuPct": r.cpu_pct,
            "memMb": r.rss_mb,
            "droppedEvents": r.dropped_events,
        })
    out.sort(key=lambda c: c["lastSeen"], reverse=True)
    return out


@router.get("/breakdown")
def breakdown(
    model_id: str,
    current_user: CurrentUser,
    # Aliased (not `range` like the sibling routes): this handler needs the
    # builtin `range()` for bucketing, which the param name would shadow.
    range_: str = Query(default="24h", alias="range"),
    session: Session = Depends(get_session),
) -> dict:
    """Client-side latency breakdown (preprocess / inference / postprocess) —
    only SDK events carry the split, so this is empty without client traffic."""
    _model(session, model_id, current_user.id)
    start = _range_start(_range(range_))
    events = session.exec(
        select(InferenceEventRow)
        .where(
            InferenceEventRow.model_id == model_id,
            InferenceEventRow.source == "client",
            InferenceEventRow.ts >= start,
            InferenceEventRow.latency_pre_ms.is_not(None),  # type: ignore[union-attr]
        )
        .order_by(InferenceEventRow.ts)  # type: ignore[arg-type]
    ).all()
    if not events:
        return {"points": []}
    n_buckets = min(48, len(events))
    size = max(1, len(events) // n_buckets)
    points = []
    for i in range(0, len(events), size):
        chunk = events[i:i + size]
        m = len(chunk)
        points.append({
            "t": chunk[0].ts,
            "preprocessMs": round(sum(e.latency_pre_ms or 0 for e in chunk) / m, 3),
            "inferenceMs": round(sum(e.latency_ms for e in chunk) / m, 3),
            "postprocessMs": round(sum(e.latency_post_ms or 0 for e in chunk) / m, 3),
        })
    return {"points": points}


@router.get("/output-stats")
def output_stats(
    model_id: str,
    current_user: CurrentUser,
    range: str = Query(default="24h"),  # noqa: A002
    session: Session = Depends(get_session),
) -> dict:
    """Aggregated output distribution from SDK window stats over the range."""
    _model(session, model_id, current_user.id)
    start = _range_start(_range(range))
    wins = session.exec(
        select(TelemetryWindowStatsRow)
        .where(
            TelemetryWindowStatsRow.model_id == model_id,
            TelemetryWindowStatsRow.window_start >= start,
        )
        .order_by(TelemetryWindowStatsRow.window_start)  # type: ignore[arg-type]
    ).all()
    if not wins:
        return {"bins": [], "meanConfidence": None, "meanEntropy": None,
                "classDist": [], "windows": 0}

    hist_acc: list[float] = []
    conf_num = ent_num = total_n = 0.0
    class_acc: dict[str, float] = {}
    for w in wins:
        try:
            out = json.loads(w.output_json)
        except ValueError:
            continue
        n = max(1, w.n)
        hist = out.get("hist") or []
        if hist:
            if not hist_acc:
                hist_acc = [0.0] * len(hist)
            if len(hist) == len(hist_acc):
                hist_acc = [a + float(h) for a, h in zip(hist_acc, hist)]
        if out.get("top1ConfMean") is not None:
            conf_num += float(out["top1ConfMean"]) * n
        if out.get("entropyMean") is not None:
            ent_num += float(out["entropyMean"]) * n
        for cls, frac in (out.get("classDist") or {}).items():
            class_acc[str(cls)] = class_acc.get(str(cls), 0.0) + float(frac) * n
        total_n += n

    top_classes = sorted(class_acc.items(), key=lambda kv: -kv[1])[:10]
    return {
        "bins": [
            {"label": f"{i / max(1, len(hist_acc)):.2f}", "count": round(v, 1)}
            for i, v in enumerate(hist_acc)
        ],
        "meanConfidence": round(conf_num / total_n, 4) if total_n and conf_num else None,
        "meanEntropy": round(ent_num / total_n, 4) if total_n and ent_num else None,
        "classDist": [
            {"classIndex": cls, "share": round(v / total_n, 4)}
            for cls, v in top_classes
        ] if total_n else [],
        "windows": len(wins),
    }


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
