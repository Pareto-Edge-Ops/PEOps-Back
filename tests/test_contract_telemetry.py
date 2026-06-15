"""Telemetry endpoints — REAL benchmark-derived data + structured 404s."""

from __future__ import annotations


def test_kpi_real_benchmark(client, real_model):
    kpi = client.get(f"/api/models/{real_model['modelId']}/telemetry/kpi").json()
    assert set(kpi) == {"requestsPerMin", "p95LatencyMs", "errorRate", "accuracyDrift"}
    for key in ("requestsPerMin", "p95LatencyMs", "errorRate"):
        assert set(kpi[key]) == {"value", "deltaPct"}
    assert set(kpi["accuracyDrift"]) == {"value", "note"}  # NO deltaPct here
    # Real measurements: throughput and p95 must be positive numbers.
    assert kpi["requestsPerMin"]["value"] > 0
    assert kpi["p95LatencyMs"]["value"] > 0
    # Divergence is the real DFCV output-fidelity complement.
    assert 0 <= kpi["accuracyDrift"]["value"] <= 100
    assert "measured" in kpi["accuracyDrift"]["note"]


def test_kpi_matches_benchmark_cache(client, real_model):
    """KPI numbers must be the exact benchmark measurements, not derived fiction."""
    from app.db import open_session
    from app.dbmodels import ModelRow
    from app.repositories import get_cached_result

    with open_session() as s:
        owner = s.get(ModelRow, real_model["modelId"]).user_id
        bench = get_cached_result(s, real_model["modelId"], "benchmark", user_id=owner)
    assert bench, "benchmark cache must exist after the pipeline"
    kpi = client.get(f"/api/models/{real_model['modelId']}/telemetry/kpi").json()
    assert kpi["requestsPerMin"]["value"] == bench["compressed"]["throughputPerMin"]
    assert kpi["p95LatencyMs"]["value"] == bench["compressed"]["p95"]


def test_series_real_buckets(client, real_model):
    series = client.get(f"/api/models/{real_model['modelId']}/telemetry/series").json()
    assert series, "expected real benchmark buckets"
    for p in series:
        assert set(p) == {"t", "requests", "p95"}
        assert p["requests"] > 0
        assert p["p95"] > 0
    ts = [p["t"] for p in series]
    assert ts == sorted(ts)  # real wall-clock timestamps, ascending


def test_percentiles_real(client, real_model):
    pct = client.get(f"/api/models/{real_model['modelId']}/telemetry/percentiles").json()
    assert len(pct["p50"]) == len(pct["p95"]) == len(pct["p99"]) > 0
    assert set(pct["values"]) == {"p50", "p95", "p99"}
    v = pct["values"]
    assert v["p50"] <= v["p95"] <= v["p99"]  # real percentiles are ordered


def test_deployments_empty_until_real(client, real_model):
    deps = client.get(f"/api/models/{real_model['modelId']}/telemetry/deployments").json()
    assert deps == []  # no fake fixtures; nothing is actually deployed


def test_alerts_are_real_events_only(client, real_model, failed_model):
    # The failed upload produced a REAL danger alert, scoped to that model.
    failed_alerts = client.get(
        f"/api/models/{failed_model['modelId']}/telemetry/alerts"
    ).json()
    assert any(
        a["level"] == "danger" and "Pipeline failed" in a["title"]
        for a in failed_alerts
    )
    for a in failed_alerts:
        assert set(a) == {"id", "level", "title", "body", "at"}
    # The healthy model has no fabricated alerts.
    ok_alerts = client.get(
        f"/api/models/{real_model['modelId']}/telemetry/alerts"
    ).json()
    for a in ok_alerts:
        assert a["level"] in {"warning", "danger"}  # only real pipeline alerts


def test_statedict_telemetry_structured_404(client, statedict_model):
    for path in ("kpi", "series", "percentiles"):
        r = client.get(f"/api/models/{statedict_model['modelId']}/telemetry/{path}")
        assert r.status_code == 404, path
        assert r.json()["detail"]["code"] == "weights_only_checkpoint", path


def test_failed_model_telemetry_no_benchmark(client, failed_model):
    r = client.get(f"/api/models/{failed_model['modelId']}/telemetry/kpi")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_benchmark"


def test_telemetry_404_missing_model(client):
    assert client.get("/api/models/m_missing/telemetry/kpi").status_code == 404


def test_meta_availability_gate(client, real_model, statedict_model, failed_model):
    """meta.available/reason are the SPA's single gate and must mirror exactly
    what kpi/series/percentiles would do — data for a benchmarked model, the
    weights_only_checkpoint / no_benchmark 404 codes otherwise. This is what lets
    the SPA skip the doomed requests instead of looping 404s."""
    ok = client.get(f"/api/models/{real_model['modelId']}/telemetry/meta").json()
    assert ok["available"] is True
    assert ok["reason"] is None

    wo = client.get(
        f"/api/models/{statedict_model['modelId']}/telemetry/meta"
    ).json()
    assert wo["available"] is False
    assert wo["reason"] == "weights_only_checkpoint"

    nb = client.get(f"/api/models/{failed_model['modelId']}/telemetry/meta").json()
    assert nb["available"] is False
    assert nb["reason"] == "no_benchmark"
