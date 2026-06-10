"""In-process traffic generator — make the dashboard live without real users yet.

A demo burst does two things:
  1. Fires a few REAL inferences through the genuine serving path (run_inference)
     so the end-to-end loop is actually exercised and produces real latencies.
  2. Synthesizes a believable recent history (a diurnal-ish request curve with
     lognormal latency jitter and the occasional incident — a latency spike or an
     error burst) so the charts, percentiles, and drift alerts have something
     real-shaped to show immediately.

Everything is written as ordinary InferenceEventRow rows — identical to what the
public /api/v1/infer endpoint records — so the aggregation + drift monitor treat
simulated and organic traffic exactly the same.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.config import iso
from app.dbmodels import DeploymentRow, InferenceEventRow, ModelRow, ResultCacheRow
from app.services.inference import run_inference


def _benchmark_p50(session: Session, model_id: str) -> float | None:
    row = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == model_id, ResultCacheRow.kind == "benchmark",
        )
    ).first()
    if not row:
        return None
    try:
        return float(json.loads(row.payload)["compressed"]["p50"])
    except (KeyError, ValueError, TypeError):
        return None


def _pick_incidents(now: datetime, hours: int) -> list[tuple[datetime, datetime, str]]:
    """0–2 short incident windows somewhere in the last `hours`."""
    windows: list[tuple[datetime, datetime, str]] = []
    for kind in random.sample(["latency", "errors"], k=random.randint(0, 2)):
        start_h = random.uniform(0.2, hours - 0.6)
        dur_min = random.uniform(12, 30)
        w0 = now - timedelta(hours=hours) + timedelta(hours=start_h)
        windows.append((w0, w0 + timedelta(minutes=dur_min), kind))
    return windows


def _diurnal_weight(dt: datetime) -> float:
    """A gentle day/night demand shape (busier midday, quieter at night)."""
    h = dt.hour + dt.minute / 60.0
    return 0.45 + 0.55 * (0.5 - 0.5 * math.cos((h - 3) / 24.0 * 2 * math.pi))


def simulate_burst(
    session: Session,
    dep: DeploymentRow,
    model: ModelRow,
    *,
    count: int = 240,
    hours: int = 6,
    incidents: bool = True,
) -> dict:
    """Generate `count` events for a deployment over the last `hours`. Returns a
    small summary. Commits once (bulk insert) for speed."""
    now = datetime.now(timezone.utc)
    rows: list[InferenceEventRow] = []

    # 1) A few genuinely-served inferences — proves the path and measures latency.
    base_latency: float | None = None
    real_calls = min(5, max(1, count))
    real_done = 0
    for _ in range(real_calls):
        try:
            _, lat = run_inference(model.artifact_key, None, batch=1)
        except Exception:  # noqa: BLE001 — fall back to synthesized latency
            break
        base_latency = lat if base_latency is None else (base_latency + lat) / 2
        rows.append(InferenceEventRow(
            user_id=model.user_id, model_id=model.id, deployment_id=dep.id,
            ts=iso(now), latency_ms=round(lat, 3), success=True, region=dep.region,
        ))
        real_done += 1

    if base_latency is None:
        base_latency = _benchmark_p50(session, model.id) or 5.0

    # 2) Synthesize the recent history with jitter + incidents.
    windows = _pick_incidents(now, hours) if incidents else []
    errors = 0
    for _ in range(max(0, count - real_done)):
        # Weight timestamps toward busier hours so the request curve looks real.
        frac = random.random()
        ts_dt = now - timedelta(hours=hours) + timedelta(hours=hours * frac)
        if random.random() > _diurnal_weight(ts_dt):
            ts_dt = now - timedelta(hours=hours) + timedelta(hours=hours * random.random())
        lat = base_latency * math.exp(random.gauss(0.0, 0.28))
        success, err_code = True, None
        for w0, w1, kind in windows:
            if w0 <= ts_dt < w1:
                if kind == "latency":
                    lat *= random.uniform(3.0, 6.0)
                elif kind == "errors" and random.random() < 0.3:
                    success, err_code, lat = False, "inference_error", 0.0
        if not success:
            errors += 1
        rows.append(InferenceEventRow(
            user_id=model.user_id, model_id=model.id, deployment_id=dep.id,
            ts=iso(ts_dt), latency_ms=round(lat, 3), success=success,
            error_code=err_code, region=dep.region,
        ))

    session.add_all(rows)
    session.commit()
    return {
        "deploymentId": dep.id,
        "events": len(rows),
        "realServed": real_done,
        "errors": errors,
        "incidents": [k for _, _, k in windows],
    }


def first_live_deployment(session: Session, model_id: str) -> DeploymentRow | None:
    rows = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    rows.sort(key=lambda d: (d.status == "paused", d.created_at))
    return rows[0] if rows else None
