"""Drift monitor — the closed-loop half the README advertises.

Every pass, for each deployment, it reads a rolling window of inference_events
and (1) refreshes the deployment's live metrics (QPS / p95 / error% / accuracy
drift / last_event_at), (2) upserts per-minute TelemetryRollupRow rows the chart
uses for long ranges, and (3) raises real AlertRow + ActivityRow when p95 spikes
past the benchmark baseline or the error rate breaches its threshold (with a
cooldown so one incident isn't re-alerted every minute).

Scope (locked decision): detection + alerting + live metrics only — no automatic
re-optimization trigger. The accuracy_drift COLUMN carries the static benchmark
divergence (context) ONLY for deployments with real traffic; a freshly deployed,
untrafficked deployment reports 0 (no benchmark-derived value before any traffic).
Real distribution drift IS detected from SDK-shipped window stats: prediction
drift (PSI of the output class distribution vs the deployment's pinned reference)
and input drift (per-input mean shift in sigmas).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.config import get_settings, iso
from app.dbmodels import (
    ActivityRow,
    AlertRow,
    DeploymentRow,
    InferenceEventRow,
    ResultCacheRow,
    TelemetryRollupRow,
    TelemetryWindowStatsRow,
)

_ALERT_COOLDOWN_MIN = 15
_REF_WINDOWS = 5            # earliest windows that form the drift reference
_WINDOW_FRESH_MIN = 30      # ignore stale windows (client gone quiet)


def _pct(values: list[float], q: float) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _baseline_p95(session: Session, model_id: str) -> float | None:
    """The compressed-model p95 from the post-compression benchmark, if any."""
    import json

    row = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == model_id, ResultCacheRow.kind == "benchmark",
        )
    ).first()
    if not row:
        return None
    try:
        return float(json.loads(row.payload)["compressed"]["p95"])
    except (KeyError, ValueError, TypeError):
        return None


def _benchmark_divergence(session: Session, model_id: str) -> float:
    import json

    row = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == model_id, ResultCacheRow.kind == "benchmark",
        )
    ).first()
    if not row:
        return 0.0
    try:
        return round(100.0 - float(json.loads(row.payload)["agreementPct"]), 2)
    except (KeyError, ValueError, TypeError):
        return 0.0


def _events(
    session: Session, deployment_id: str, start: datetime, end: datetime,
) -> list[InferenceEventRow]:
    return list(session.exec(
        select(InferenceEventRow).where(
            InferenceEventRow.deployment_id == deployment_id,
            InferenceEventRow.ts >= iso(start),
            InferenceEventRow.ts < iso(end),
        )
    ).all())


def _recent_alert(session: Session, model_id: str, title: str, within_min: int) -> bool:
    cutoff = iso(datetime.now(timezone.utc) - timedelta(minutes=within_min))
    rows = session.exec(
        select(AlertRow).where(
            AlertRow.model_id == model_id, AlertRow.title == title,
        )
    ).all()
    return any(r.at >= cutoff for r in rows)


def _raise_alert(
    session: Session, dep: DeploymentRow, *, level: str, title: str, body: str,
) -> bool:
    """Create an alert + activity event unless one of the same kind is in cooldown.
    Returns True when an alert was actually raised."""
    if _recent_alert(session, dep.model_id, title, _ALERT_COOLDOWN_MIN):
        return False
    now = iso(datetime.now(timezone.utc))
    token = uuid.uuid4().hex[:8]
    session.add(AlertRow(
        id=f"al_{token}", user_id=dep.user_id, model_id=dep.model_id,
        level=level, title=title, body=body, at=now,
    ))
    session.add(ActivityRow(
        id=f"act_al_{token}", user_id=dep.user_id, kind="accuracy_drift",
        text=f"{title} — {dep.name}", timestamp=now,
    ))
    return True


def _update_live_metrics(
    session: Session, dep: DeploymentRow, events: list[InferenceEventRow],
    window: timedelta, now: datetime,
) -> tuple[float, float]:
    """Refresh the deployment row; returns (rolling_p95, rolling_error_pct)."""
    lats = [e.latency_ms for e in events if e.success]
    total = len(events)
    errors = sum(1 for e in events if not e.success)
    win_sec = window.total_seconds()
    rolling_p95 = round(_pct(lats, 95), 3)
    err_pct = round(100.0 * errors / total, 3) if total else 0.0
    dep.qps = round(total / win_sec, 3) if win_sec > 0 else 0.0
    dep.p95 = rolling_p95
    dep.errors_pct = err_pct
    # Only a deployment with real traffic carries a drift figure; a freshly
    # deployed, untrafficked deployment reports 0 (no benchmark-derived value).
    dep.accuracy_drift = _benchmark_divergence(session, dep.model_id) if total else 0.0
    last = max((e.ts for e in events), default=None)
    if last:
        dep.last_event_at = last
    session.add(dep)
    return rolling_p95, err_pct


def _upsert_rollups(session: Session, dep: DeploymentRow, events: list[InferenceEventRow]) -> None:
    """Recompute the per-minute rollups touched by the window (idempotent)."""
    by_minute: dict[str, list[InferenceEventRow]] = {}
    for e in events:
        dt = _parse(e.ts)
        if dt is None:
            continue
        key = iso(dt.replace(second=0, microsecond=0))
        by_minute.setdefault(key, []).append(e)
    for bucket_ts, evs in by_minute.items():
        lats = [e.latency_ms for e in evs if e.success]
        existing = session.exec(
            select(TelemetryRollupRow).where(
                TelemetryRollupRow.deployment_id == dep.id,
                TelemetryRollupRow.bucket_ts == bucket_ts,
            )
        ).first()
        row = existing or TelemetryRollupRow(deployment_id=dep.id, bucket_ts=bucket_ts)
        row.count = len(evs)
        row.errors = sum(1 for e in evs if not e.success)
        row.sum_latency = round(sum(lats), 3)
        row.p50 = round(_pct(lats, 50), 3)
        row.p95 = round(_pct(lats, 95), 3)
        row.p99 = round(_pct(lats, 99), 3)
        session.add(row)


def _psi(reference: dict[str, float], current: dict[str, float]) -> float:
    """Population Stability Index between two class distributions."""
    import math

    eps = 1e-4
    keys = set(reference) | set(current)
    psi = 0.0
    for k in keys:
        p = max(reference.get(k, 0.0), eps)
        q = max(current.get(k, 0.0), eps)
        psi += (q - p) * math.log(q / p)
    return psi


def _drift_reference(
    session: Session, dep: DeploymentRow, windows: list[TelemetryWindowStatsRow],
) -> dict | None:
    """The drift baseline: mean stats over the EARLIEST windows, cached once.

    Cached in ResultCacheRow(kind="drift_ref_{dep.id}") so the reference stays
    pinned to the deployment's initial traffic instead of sliding with drift."""
    import json

    kind = f"drift_ref_{dep.id}"
    row = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == dep.model_id, ResultCacheRow.kind == kind,
        )
    ).first()
    if row:
        try:
            return json.loads(row.payload)
        except ValueError:
            return None
    if len(windows) < _REF_WINDOWS:
        return None  # not enough history to pin a reference yet

    ref_windows = windows[:_REF_WINDOWS]
    class_acc: dict[str, list[float]] = {}
    input_acc: dict[str, dict[str, list[float]]] = {}
    for w in ref_windows:
        try:
            out = json.loads(w.output_json)
            ins = json.loads(w.input_stats_json)
        except ValueError:
            continue
        for cls, frac in (out.get("classDist") or {}).items():
            class_acc.setdefault(str(cls), []).append(float(frac))
        for name, stat in (ins or {}).items():
            acc = input_acc.setdefault(name, {"mean": [], "std": []})
            acc["mean"].append(float(stat.get("mean", 0.0)))
            acc["std"].append(float(stat.get("std", 0.0)))

    n_ref = max(1, len(ref_windows))
    ref = {
        "classDist": {k: sum(v) / n_ref for k, v in class_acc.items()},
        "inputs": {
            name: {
                "mean": sum(acc["mean"]) / max(1, len(acc["mean"])),
                "std": sum(acc["std"]) / max(1, len(acc["std"])),
            }
            for name, acc in input_acc.items()
        },
        "windows": len(ref_windows),
    }
    session.add(ResultCacheRow(
        user_id=dep.user_id, model_id=dep.model_id, kind=kind,
        payload=json.dumps(ref),
    ))
    return ref


def _client_drift_checks(
    session: Session, dep: DeploymentRow, now: datetime,
) -> int:
    """Prediction (PSI) + input-distribution drift from SDK window stats."""
    import json

    settings = get_settings()
    windows = list(session.exec(
        select(TelemetryWindowStatsRow)
        .where(TelemetryWindowStatsRow.deployment_id == dep.id)
        .order_by(TelemetryWindowStatsRow.window_start)  # type: ignore[arg-type]
    ).all())
    if len(windows) <= _REF_WINDOWS:
        return 0
    ref = _drift_reference(session, dep, windows)
    if not ref:
        return 0

    latest = windows[-1]
    started = _parse(latest.window_start)
    if started is None or now - started > timedelta(minutes=_WINDOW_FRESH_MIN):
        return 0  # client went quiet — nothing fresh to judge

    raised = 0
    try:
        out = json.loads(latest.output_json)
        ins = json.loads(latest.input_stats_json)
    except ValueError:
        return 0

    current_dist = {str(k): float(v) for k, v in (out.get("classDist") or {}).items()}
    ref_dist = ref.get("classDist") or {}
    if current_dist and ref_dist:
        psi = _psi(ref_dist, current_dist)
        if psi > settings.drift_psi:
            level = "danger" if psi > 2 * settings.drift_psi else "warning"
            if _raise_alert(
                session, dep, level=level, title="prediction drift",
                body=f"output class distribution PSI {psi:.3f} vs the deployment's "
                     f"reference window (threshold {settings.drift_psi}) — the model "
                     f"is predicting differently than when it was deployed.",
            ):
                raised += 1

    ref_inputs = ref.get("inputs") or {}
    for name, stat in (ins or {}).items():
        ref_stat = ref_inputs.get(name)
        if not ref_stat:
            continue
        ref_std = max(float(ref_stat.get("std", 0.0)), 1e-6)
        z = abs(float(stat.get("mean", 0.0)) - float(ref_stat.get("mean", 0.0))) / ref_std
        if z > settings.drift_input_z:
            if _raise_alert(
                session, dep, level="warning", title="input distribution shift",
                body=f"input '{name}' mean moved {z:.1f} sigma from the reference "
                     f"window (threshold {settings.drift_input_z} sigma) — incoming "
                     f"data no longer matches what the model was optimized on.",
            ):
                raised += 1
            break  # one input-shift alert per pass is enough signal
    return raised


def drift_monitor_pass(session: Session) -> dict:
    """Run one monitoring pass over all deployments. Returns a small summary."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    window = timedelta(minutes=settings.monitor_window_min)
    deps = session.exec(select(DeploymentRow)).all()
    alerts_raised = 0
    for dep in deps:
        events = _events(session, dep.id, now - window, now)
        rolling_p95, err_pct = _update_live_metrics(session, dep, events, window, now)
        _upsert_rollups(session, dep, events)
        if dep.status == "paused":
            continue
        # Prediction/input drift from SDK-shipped window stats — checked even
        # without request events in the window (a client can ship stats while
        # the request stream is bucketed into a different window).
        alerts_raised += _client_drift_checks(session, dep, now)
        if not events:
            continue
        # p95 spike vs the benchmark baseline.
        base = _baseline_p95(session, dep.model_id)
        if base and base > 0 and rolling_p95 > base * (1 + settings.drift_p95_pct / 100.0):
            rise = round((rolling_p95 / base - 1) * 100)
            if _raise_alert(
                session, dep, level="warning", title="p95 latency spike",
                body=f"p95 {rolling_p95:.1f}ms vs baseline {base:.1f}ms (+{rise}%) "
                     f"over the last {settings.monitor_window_min}m.",
            ):
                alerts_raised += 1
        # 5xx / error-rate spike.
        if err_pct > settings.drift_error_pct:
            if _raise_alert(
                session, dep, level="danger", title="5xx error spike",
                body=f"error rate {err_pct:.2f}% over the last "
                     f"{settings.monitor_window_min}m exceeds the "
                     f"{settings.drift_error_pct:.2f}% threshold.",
            ):
                alerts_raised += 1
    session.commit()
    return {"deployments": len(deps), "alertsRaised": alerts_raised}
