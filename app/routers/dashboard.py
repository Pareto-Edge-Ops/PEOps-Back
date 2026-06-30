"""GET /api/dashboard/* — every number aggregated from real DB state.

Sparks are real daily event counts, deltas are real week-over-week
comparisons, compute hours are real measured run durations, and the Pareto
snapshot is the latest model with actual cached trial results. Nothing is
generated; an empty workspace honestly reports zeros / structured 404s.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.auth.dependencies import CurrentUser
from app.db import get_session
from app.dbmodels import (
    ActivityRow,
    DeploymentRow,
    ModelRow,
    RunRow,
)
from app.repositories import get_cached_result, user_artifact_metas
from app.schemas.common import Spark
from app.schemas.cost import WorkspaceCostSavings
from app.schemas.dashboard import (
    ActivityEvent,
    CompressionBest,
    CompressionMap,
    CompressionPoint,
    DashboardRun,
    FleetHealth,
    GuaranteeCoverage,
    GuaranteeSegment,
    KpiBlock,
    KpiSummary,
    SizeReduced,
    TopModel,
)
from app.services.hardware import _CPU_X86_HOURLY, est_cost_per_million

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_SEGMENT_COLORS = ["#ADB4F3", "#6976EB", "#483EB7", "#5E69D1", "#D7DAF3", "#40BF6B"]

# Guarantee-rung → fixed colour. Certified rungs ride the lavender accent ramp;
# the uncertified fallback stays muted grey so the donut reads "covered vs not".
_RUNG_COLORS = {
    "PARETO_CERTIFIED": "#ADB4F3",
    "INT8_uosa_mixed": "#6976EB",
    "INT8_uniform": "#6976EB",
    "W8_weight_only": "#5E69D1",
    "FP16": "#483EB7",
    "fallback": "#939496",
    "ORIGINAL": "#939496",
}


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


def _cumulative_saved_spark(
    events: list[tuple[str, float]], days: int = 16,
) -> list[Spark]:
    """Cumulative bytes-saved over the trailing window. Savings older than the
    window seed the running total so the line reflects the true portfolio sum,
    not a window-local reset."""
    now = datetime.now(timezone.utc)
    daily = [0.0] * days
    seed = 0.0
    for ts, saved in events:
        dt = _parse_iso(ts)
        if dt is None:
            continue
        age = (now.date() - dt.date()).days
        if 0 <= age < days:
            daily[days - 1 - age] += saved
        elif age >= days:
            seed += saved
    start = now.date() - timedelta(days=days - 1)
    out: list[Spark] = []
    run = seed
    for i in range(days):
        run += daily[i]
        out.append(Spark(t=(start + timedelta(days=i)).isoformat(), value=round(run, 0)))
    return out


def _size_reduced(
    session: Session, user_id: str, days: int, label: str,
) -> SizeReduced:
    """Σ bytes saved + mean × smaller across models whose served artifact
    records a real compression ratio (source=='pareto')."""
    saved_total = 0.0
    ratios: list[float] = []
    events: list[tuple[str, float]] = []
    for model, meta in user_artifact_metas(session, user_id):
        ratio = meta.get("sizeRatio")
        size_bytes = meta.get("sizeBytes")
        if not ratio or not size_bytes or ratio <= 0:
            continue
        baseline = size_bytes / ratio
        saved = baseline - size_bytes
        if saved <= 0:
            continue
        saved_total += saved
        ratios.append(1.0 / ratio)
        ts = model.last_optimized_at or model.last_learned_at
        if ts:
            events.append((ts, saved))
    avg_x = round(sum(ratios) / len(ratios), 2) if ratios else 0.0
    return SizeReduced(
        bytesSaved=round(saved_total, 0),
        avgReductionX=avg_x,
        modelCount=len(ratios),
        deltaText=_range_delta([ts for ts, _ in events], days, label),
        spark=_cumulative_saved_spark(events),
    )


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
        sizeReduced=_size_reduced(session, uid, days, label),
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


@router.get("/compression-map", response_model_exclude_none=True)
def compression_map(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> CompressionMap:
    """Portfolio value map: each optimized model's served pick plotted as size
    reduction (×) vs accuracy retained (%). Pareto picks carry a measurable
    ratio+accuracy and become points; certifiedCount/modelCount cover the whole
    portfolio (ladder/fallback picks have no ratio to plot but still count)."""
    uid = current_user.id
    metas = user_artifact_metas(session, uid)
    certified = sum(1 for _, m in metas if m.get("source") in ("pareto", "ladder"))

    points: list[CompressionPoint] = []
    for model, meta in metas:
        ratio = meta.get("sizeRatio")
        acc = meta.get("accuracy")
        if not ratio or acc is None or ratio <= 0:
            continue  # ladder/fallback picks carry no ratio+accuracy to plot
        pareto = get_cached_result(session, model.id, "pareto", user_id=uid) or {}
        base_acc = pareto.get("baseAccuracy")
        max_drop = (pareto.get("budget") or {}).get("maxAccuracyDrop")
        drop = round(base_acc - acc, 2) if base_acc is not None else 0.0
        within = (
            drop <= max_drop if base_acc is not None and max_drop is not None else True
        )
        latency = next(
            (t.get("latency") for t in pareto.get("trials", [])
             if t.get("trialNumber") == meta.get("trialNumber")),
            None,
        )
        points.append(CompressionPoint(
            modelId=model.id, name=model.name,
            reductionX=round(1.0 / ratio, 2), sizeRatio=ratio,
            accuracyRetained=acc, accuracyDrop=drop, withinTolerance=within,
            certified=meta.get("source") in ("pareto", "ladder"),
            rung=meta.get("rung"), latencyMs=latency,
            # Single-stream $/1M on a reference x86 CPU — a $ chip on each point.
            estCostPer1M=(
                est_cost_per_million(latency, _CPU_X86_HOURLY) if latency else None
            ),
        ))

    pool = [p for p in points if p.withinTolerance] or points
    best = None
    if pool:
        b = max(pool, key=lambda p: p.reductionX)
        best = CompressionBest(
            modelId=b.modelId, reductionX=b.reductionX,
            accuracyRetained=b.accuracyRetained,
        )
    return CompressionMap(
        points=points, modelCount=len(metas), certifiedCount=certified, best=best,
    )


@router.get("/cost-savings", response_model_exclude_none=True)
def cost_savings(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> WorkspaceCostSavings:
    """Workspace $ rollup: monthly inference cost original vs compressed across
    deployments with measured traffic, plus the average % cheaper across all
    optimized models (honest even at zero live traffic)."""
    from app.services import cost as cost_svc

    return WorkspaceCostSavings(**cost_svc.workspace_cost_savings(session, current_user.id))


@router.get("/fleet-health", response_model_exclude_none=True)
def fleet_health(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> FleetHealth:
    """Workspace deployment health: live endpoints, deployments past their
    tolerance, and open alerts — derived from existing rows, no new measurement."""
    from app.services import fleet

    return FleetHealth(**fleet.workspace_fleet_health(session, current_user.id))


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


@router.get("/guarantee-coverage", response_model_exclude_none=True)
def guarantee_coverage(
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> GuaranteeCoverage:
    """How many optimized models carry a fidelity guarantee, plus the rung
    distribution. Certified = the served artifact cleared a guarantee gate
    (Pareto-certified or a ladder rung); a fallback pick is uncertified."""
    metas = user_artifact_metas(session, current_user.id)
    certified = 0
    buckets: dict[str, int] = {}
    fidelities: list[float] = []
    for _, meta in metas:
        source = meta.get("source")
        if source in ("pareto", "ladder"):
            certified += 1
            label = meta.get("rung") or source
        else:
            label = "fallback"
        buckets[label] = buckets.get(label, 0) + 1
        ofs = meta.get("ofs")
        if ofs is not None:
            fidelities.append(ofs)
    segments = [
        GuaranteeSegment(
            label=label, value=count,
            color=_RUNG_COLORS.get(label, _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]),
        )
        for i, (label, count) in enumerate(
            sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
        )
    ]
    avg_fid = round(sum(fidelities) / len(fidelities), 4) if fidelities else None
    return GuaranteeCoverage(
        certifiedCount=certified, totalModels=len(metas),
        avgFidelity=avg_fid, segments=segments,
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
