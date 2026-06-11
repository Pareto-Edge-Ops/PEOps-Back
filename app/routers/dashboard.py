"""GET /api/dashboard/* — every number aggregated from real DB state.

Sparks are real daily event counts, deltas are real week-over-week
comparisons, compute hours are real measured run durations, and the Pareto
snapshot is the latest model with actual cached trial results. Nothing is
generated; an empty workspace honestly reports zeros / structured 404s.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.auth.dependencies import CurrentUser
from app.config import get_settings
from app.db import get_session
from app.dbmodels import (
    ActivityRow,
    DeploymentRow,
    IngestionRunRow,
    ModelRow,
    ResultCacheRow,
    RunRow,
)
from app.repositories import get_cached_result
from app.schemas.common import Spark
from app.schemas.dashboard import (
    ActivityEvent,
    ComputeCost,
    ComputeUsed,
    CostSegment,
    DashboardRun,
    KpiBlock,
    KpiSummary,
    ParetoSnapshot,
    ParetoSnapshotPoint,
    TopModel,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_SEGMENT_COLORS = ["#ADB4F3", "#6976EB", "#483EB7", "#5E69D1", "#D7DAF3", "#40BF6B"]


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _daily_spark(timestamps: list[str], days: int = 16) -> list[Spark]:
    """Real daily event counts for the trailing `days` days (zeros included)."""
    now = datetime.now(timezone.utc)
    counts = [0] * days
    for ts in timestamps:
        dt = _parse_iso(ts)
        if dt is None:
            continue
        age = (now.date() - dt.date()).days
        if 0 <= age < days:
            counts[days - 1 - age] += 1
    start = now.date() - timedelta(days=days - 1)
    return [
        Spark(t=(start + timedelta(days=i)).isoformat(), value=float(c))
        for i, c in enumerate(counts)
    ]


_RANGE_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}


def _range_delta(timestamps: list[str], days: int, label: str) -> str:
    """Real current-window vs previous-window event-count comparison."""
    now = datetime.now(timezone.utc)
    current = previous = 0
    for ts in timestamps:
        dt = _parse_iso(ts)
        if dt is None:
            continue
        age_days = (now - dt).total_seconds() / 86400
        if age_days < days:
            current += 1
        elif age_days < 2 * days:
            previous += 1
    diff = current - previous
    return f"{diff:+d} vs previous {label}"


def _week_delta(timestamps: list[str]) -> str:
    return _range_delta(timestamps, 7, "7d")


def _run_started_timestamps(session: Session, user_id: str) -> list[str]:
    rows = session.exec(select(RunRow).where(RunRow.user_id == user_id)).all()
    return [r.created_at for r in rows if r.created_at]


def _run_completed_timestamps(session: Session, user_id: str) -> list[str]:
    rows = session.exec(
        select(ActivityRow).where(
            ActivityRow.kind == "run_completed", ActivityRow.user_id == user_id,
        )
    ).all()
    return [r.timestamp for r in rows]


def _real_compute_hours(
    session: Session, user_id: str, since: datetime | None = None,
) -> float:
    """Sum of real wall-clock durations of finished ingestion runs
    (optionally only runs that started after `since`)."""
    rows = session.exec(
        select(IngestionRunRow).where(IngestionRunRow.user_id == user_id)
    ).all()
    total_sec = 0.0
    for r in rows:
        if not r.finished_at:
            continue
        start, end = _parse_iso(r.started_at), _parse_iso(r.finished_at)
        if start and end and end > start:
            if since is not None and start < since:
                continue
            total_sec += (end - start).total_seconds()
    return round(total_sec / 3600, 6)


@router.get("/summary")
def summary(
    current_user: CurrentUser,
    range: str = Query(default="7d"),  # noqa: A002 — public URL param name
    session: Session = Depends(get_session),
) -> KpiSummary:
    uid = current_user.id
    days = _RANGE_DAYS.get(range, 7)
    label = range if range in _RANGE_DAYS else "7d"

    runs = session.exec(select(RunRow).where(RunRow.user_id == uid)).all()
    deployments = session.exec(
        select(DeploymentRow).where(DeploymentRow.user_id == uid)
    ).all()
    active = sum(1 for r in runs if r.status in ("running", "queued"))
    live = sum(1 for d in deployments if d.status != "paused")

    now = datetime.now(timezone.utc)
    started = _run_started_timestamps(session, uid)
    completed = _run_completed_timestamps(session, uid)
    done_in_range = sum(
        1 for ts in completed
        if (dt := _parse_iso(ts)) and (now - dt).total_seconds() < days * 86400
    )

    # Sparks stay a fixed 16-day daily trend (the SPA's sparkline contract);
    # the range scopes counts and deltas, not the spark window.
    spark_days = 16
    deployment_created = [d.created_at for d in deployments if d.created_at]

    used = _real_compute_hours(session, uid)
    quota = get_settings().compute_quota_h
    if quota > 0:
        progress_note = f"{used / quota * 100:.1f}% of quota"
    else:
        progress_note = "no quota configured"
    return KpiSummary(
        activeRuns=KpiBlock(
            value=active, deltaText=_range_delta(started, days, label),
            spark=_daily_spark(started, days=spark_days),
        ),
        completedThisWeek=KpiBlock(
            value=done_in_range, deltaText=_range_delta(completed, days, label),
            spark=_daily_spark(completed, days=spark_days),
        ),
        liveDeployments=KpiBlock(
            value=live, deltaText=_range_delta(deployment_created, days, label),
            spark=_daily_spark(deployment_created, days=spark_days),
        ),
        computeUsed=ComputeUsed(
            used=used, quota=quota,
            label=f"{used:g} / {quota:g} compute·h" if quota > 0
                  else f"{used:g} compute·h",
            progressNote=progress_note,
        ),
        periodLabel=label,
    )


@router.get("/runs")
def runs(
    current_user: CurrentUser,
    status: str = Query(default="all"),
    session: Session = Depends(get_session),
) -> list[DashboardRun]:
    rows = session.exec(select(RunRow).where(RunRow.user_id == current_user.id)).all()
    if status and status != "all":
        rows = [r for r in rows if r.status == status]
    return [
        DashboardRun(
            id=r.id, modelId=r.model_id, name=r.name, status=r.status,  # type: ignore[arg-type]
            progressPct=r.progress_pct, iter=r.iter,
            bestAcc=r.best_acc, deltaAcc=r.delta_acc,
        )
        for r in rows
    ]


def _latest_pareto_model(session: Session, user_id: str) -> tuple[ModelRow, dict] | None:
    """Newest model (owned by the user) that has REAL cached Pareto trials."""
    cache_rows = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.kind == "pareto", ResultCacheRow.user_id == user_id,
        )
    ).all()
    candidates: list[tuple[ModelRow, dict]] = []
    for row in cache_rows:
        model = session.get(ModelRow, row.model_id)
        if model is not None and model.user_id == user_id:
            cached = get_cached_result(session, row.model_id, "pareto", user_id=user_id)
            if cached and cached.get("trials"):
                candidates.append((model, cached))
    if not candidates:
        return None
    candidates.sort(key=lambda mc: mc[0].last_optimized_at or mc[0].last_learned_at, reverse=True)
    return candidates[0]


@router.get("/pareto-snapshot", response_model_exclude_none=True)
def pareto_snapshot(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> ParetoSnapshot:
    latest = _latest_pareto_model(session, current_user.id)
    if latest is None:
        raise HTTPException(status_code=404, detail={
            "code": "no_completed_runs",
            "message": "No completed Pareto runs yet — upload a model to start one.",
        })
    model, cached = latest
    trials = cached["trials"]
    budget = cached["budget"]

    # 36-point snapshot; frontier re-marked with the dashboard's 2D rule
    # (best accuracy per latency band) over the REAL trials.
    pts = [
        ParetoSnapshotPoint(
            id=f"p_{i}", accuracy=t["accuracy"], latency=t["latency"],
            size=t["size"], onFrontier=False,
        )
        for i, t in enumerate(trials[:36])
    ]
    best = -math.inf
    for p in sorted(pts, key=lambda p: p.latency):
        if p.accuracy > best:
            p.onFrontier = True
            best = p.accuracy
    return ParetoSnapshot(
        modelId=model.id,
        modelName=f"{cached['modelName']} · pareto",
        subtitle=(
            f"Pareto search · budget latency≤{budget['maxLatency']:g}ms, "
            f"size≤{budget['maxSize']:g}MB"
        ),
        points=pts,
        bestAccuracy=max((t["accuracy"] for t in trials), default=None),
    )


@router.get("/top-models")
def top_models(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> list[TopModel]:
    rows = session.exec(
        select(ModelRow).where(ModelRow.user_id == current_user.id)
    ).all()
    scored = [r for r in rows if r.best_accuracy is not None]
    scored.sort(key=lambda r: r.best_accuracy or 0, reverse=True)
    out: list[TopModel] = []
    for rank, row in enumerate(scored[:5], start=1):
        cached = get_cached_result(session, row.id, "pareto", user_id=current_user.id)
        trials = (cached or {}).get("trials", [])
        # Real frontier share + real trial-accuracy progression.
        coverage = (
            round(100 * sum(1 for t in trials if t.get("onFrontier")) / len(trials), 1)
            if trials else 0.0
        )
        spark = [round(float(t["accuracy"]), 2) for t in trials[:16]]
        out.append(TopModel(
            rank=rank, modelId=row.id, name=row.name,
            bestAccuracy=row.best_accuracy or 0,
            paretoCoverage=coverage, spark=spark,
        ))
    return out


@router.get("/compute-cost", response_model_exclude_none=True)
def compute_cost(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> ComputeCost:
    uid = current_user.id
    # Scoped to the current calendar month — the card's "this month" label is
    # now backed by the actual measurement window.
    month_start = datetime.now(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0)
    used = _real_compute_hours(session, uid, since=month_start)
    quota = get_settings().compute_quota_h

    # Real per-phase wall time aggregated across all benchmark caches.
    phase_totals: dict[str, float] = {}
    bench_rows = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.kind == "benchmark", ResultCacheRow.user_id == uid,
        )
    ).all()
    for row in bench_rows:
        cached = get_cached_result(session, row.model_id, "benchmark", user_id=uid)
        for ph in (cached or {}).get("phases", []):
            phase_totals[ph["name"]] = phase_totals.get(ph["name"], 0.0) + ph["sec"]
    segments = [
        CostSegment(label=name, value=round(sec, 1), color=_SEGMENT_COLORS[i % len(_SEGMENT_COLORS)])
        for i, (name, sec) in enumerate(
            sorted(phase_totals.items(), key=lambda kv: kv[1], reverse=True)
        )
    ]
    return ComputeCost(
        usedGpuHours=used, quotaGpuHours=quota,
        periodLabel="this month",
        # No cloud billing exists for local compute — these stay absent
        # rather than invented (costUsd / region / resetDateText / noteText).
        segments=segments,
    )


@router.get("/activity")
def activity(
    current_user: CurrentUser,
    limit: int = Query(default=5, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[ActivityEvent]:
    rows = session.exec(
        select(ActivityRow).where(ActivityRow.user_id == current_user.id)
    ).all()
    rows.sort(key=lambda r: r.timestamp, reverse=True)
    return [
        ActivityEvent(id=r.id, kind=r.kind, text=r.text, timestamp=r.timestamp)  # type: ignore[arg-type]
        for r in rows[:limit]
    ]
