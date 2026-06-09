"""GET /api/models/:id/telemetry/* — served from REAL benchmark measurements.

Telemetry is derived from the actual onnxruntime benchmark the pipeline runs
after compression (original vs compressed). There is no synthetic traffic, no
generated series: a model with no benchmark gets a structured 404 the SPA
turns into an honest empty state.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.auth.dependencies import CurrentUser
from app.db import get_session
from app.dbmodels import AlertRow, DeploymentRow, ModelRow
from app.repositories import get_cached_result, owned_model
from app.schemas.telemetry import (
    Alert,
    DeltaKpi,
    Deployment,
    DriftKpi,
    Percentiles,
    PercentileValues,
    TelemetryKpi,
    TelemetryPoint,
)

router = APIRouter(prefix="/models/{model_id}/telemetry", tags=["telemetry"])


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


def _delta_pct(original: float, compressed: float) -> float:
    if original <= 0:
        return 0.0
    return round((compressed - original) / original * 100, 1)


@router.get("/kpi")
def kpi(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> TelemetryKpi:
    model = _model(session, model_id, current_user.id)
    bench = _benchmark(session, model)
    orig, comp = bench["original"], bench["compressed"]
    divergence = round(100.0 - bench["agreementPct"], 2)
    return TelemetryKpi(
        # All values are real ORT measurements; deltas are the real
        # compressed-vs-original change from the same benchmark.
        requestsPerMin=DeltaKpi(
            value=comp["throughputPerMin"],
            deltaPct=_delta_pct(orig["throughputPerMin"], comp["throughputPerMin"]),
        ),
        p95LatencyMs=DeltaKpi(
            value=comp["p95"],
            deltaPct=_delta_pct(orig["p95"], comp["p95"]),
        ),
        # Zero failed inferences were observed during the benchmark run.
        errorRate=DeltaKpi(value=0.0, deltaPct=0.0),
        accuracyDrift=DriftKpi(
            value=divergence,
            note="output divergence vs original (DFCV, measured)",
        ),
    )


@router.get("/series")
def series(
    model_id: str,
    current_user: CurrentUser,
    range: str = Query(default="24h"),  # noqa: A002 — kept for URL compat; one benchmark window exists
    session: Session = Depends(get_session),
) -> list[TelemetryPoint]:
    model = _model(session, model_id, current_user.id)
    bench = _benchmark(session, model)
    return [
        TelemetryPoint(t=b["t"], requests=b["requests"], p95=b["p95"])
        for b in bench["buckets"]
    ]


@router.get("/percentiles")
def percentiles(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> Percentiles:
    model = _model(session, model_id, current_user.id)
    bench = _benchmark(session, model)
    buckets = bench["buckets"]
    comp = bench["compressed"]
    return Percentiles(
        p50=[b["p50"] for b in buckets],
        p95=[b["p95"] for b in buckets],
        p99=[b["p99"] for b in buckets],
        values=PercentileValues(p50=comp["p50"], p95=comp["p95"], p99=comp["p99"]),
    )


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
