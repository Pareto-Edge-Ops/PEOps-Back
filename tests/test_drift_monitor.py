"""Drift monitor — live metrics refresh + real alerts on threshold breaches."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _seed_breaching_events(mid: str, dep_id: str) -> float:
    """Insert a p95 spike + an error burst in the last few minutes. Returns the
    benchmark baseline p95 the monitor compares against."""
    from app.config import iso
    from app.db import open_session
    from app.dbmodels import InferenceEventRow, ModelRow
    from app.repositories import get_cached_result

    now = datetime.now(timezone.utc)
    with open_session() as s:
        owner = s.get(ModelRow, mid).user_id
        base = get_cached_result(s, mid, "benchmark", user_id=owner)["compressed"]["p95"]
        for _ in range(28):  # well above baseline → p95 spike
            s.add(InferenceEventRow(
                user_id=owner, model_id=mid, deployment_id=dep_id,
                ts=iso(now - timedelta(minutes=2)), latency_ms=base * 5 + 5,
                success=True, region="ap-northeast-2",
            ))
        for _ in range(4):   # 4/32 = 12.5% errors → 5xx spike
            s.add(InferenceEventRow(
                user_id=owner, model_id=mid, deployment_id=dep_id,
                ts=iso(now - timedelta(minutes=1)), latency_ms=0.0,
                success=False, error_code="inference_error", region="ap-northeast-2",
            ))
        s.commit()
    return base


def test_monitor_raises_alerts_and_updates_metrics(make_live_model, deploy_model, client):
    mid = make_live_model("drift-a.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    base = _seed_breaching_events(mid, dep_id)

    from app.db import open_session
    from app.services.drift_monitor import drift_monitor_pass

    with open_session() as s:
        summary = drift_monitor_pass(s)
    assert summary["alertsRaised"] >= 2

    alerts = client.get(f"/api/models/{mid}/telemetry/alerts").json()
    titles = {a["title"] for a in alerts}
    assert "p95 latency spike" in titles
    assert "5xx error spike" in titles
    assert any(a["level"] == "danger" for a in alerts)

    dep = next(d for d in client.get(f"/api/models/{mid}/deployments").json()
               if d["id"] == dep_id)
    assert dep["p95"] > base
    assert dep["qps"] > 0
    assert dep["errorsPct"] > 1.0
    assert dep["lastEventAt"] is not None


def test_alert_cooldown_prevents_duplicates(make_live_model, deploy_model, client):
    mid = make_live_model("drift-b.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    _seed_breaching_events(mid, dep_id)

    from app.db import open_session
    from app.services.drift_monitor import drift_monitor_pass

    with open_session() as s:
        drift_monitor_pass(s)
    first = len(client.get(f"/api/models/{mid}/telemetry/alerts").json())
    with open_session() as s:
        again = drift_monitor_pass(s)
    second = len(client.get(f"/api/models/{mid}/telemetry/alerts").json())
    assert again["alertsRaised"] == 0   # within cooldown
    assert second == first


def test_monitor_writes_rollups(make_live_model, deploy_model, client):
    mid = make_live_model("drift-c.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    _seed_breaching_events(mid, dep_id)

    from app.db import open_session
    from app.dbmodels import TelemetryRollupRow
    from app.services.drift_monitor import drift_monitor_pass
    from sqlmodel import select

    with open_session() as s:
        drift_monitor_pass(s)
        rollups = s.exec(
            select(TelemetryRollupRow).where(TelemetryRollupRow.deployment_id == dep_id)
        ).all()
    assert rollups, "monitor should upsert per-minute rollups"
    assert all(r.count > 0 for r in rollups)
