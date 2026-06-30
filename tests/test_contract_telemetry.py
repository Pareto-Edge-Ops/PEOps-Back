"""Telemetry endpoints — empty until real traffic + structured gates.

A benchmarked-but-undeployed model (`real_model`) exposes NO benchmark-derived
telemetry: kpi/series/percentiles return empty shapes and /meta reports
not_deployed. A deployed-but-untrafficked model reports source "none" with empty
data (the SPA renders the "—" / waiting-for-traffic chrome). Real numbers appear
only once the model serves traffic (see test_telemetry_live).
"""

from __future__ import annotations


def test_kpi_empty_until_traffic(client, real_model):
    kpi = client.get(f"/api/models/{real_model['modelId']}/telemetry/kpi").json()
    assert set(kpi) == {"requestsPerMin", "p95LatencyMs", "errorRate", "accuracyDrift"}
    for key in ("requestsPerMin", "p95LatencyMs", "errorRate"):
        assert set(kpi[key]) == {"value", "deltaPct"}
        assert kpi[key]["value"] == 0.0      # no benchmark fallback — empty
        assert kpi[key]["deltaPct"] == 0.0
    assert set(kpi["accuracyDrift"]) == {"value", "note"}  # NO deltaPct here
    assert kpi["accuracyDrift"]["value"] == 0.0


def test_series_empty_until_traffic(client, real_model):
    series = client.get(f"/api/models/{real_model['modelId']}/telemetry/series").json()
    assert series == []   # empty list — not benchmark buckets


def test_percentiles_empty_until_traffic(client, real_model):
    pct = client.get(f"/api/models/{real_model['modelId']}/telemetry/percentiles").json()
    assert pct["p50"] == pct["p95"] == pct["p99"] == []
    assert pct["values"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_deployed_no_traffic_is_empty_not_benchmark(make_live_model, deploy_model, client):
    """A deployed model with zero traffic: the dashboard chrome IS available, but
    the data is empty (source "none") — never the benchmark."""
    mid = make_live_model("armed.onnx")["modelId"]
    deploy_model(mid)

    meta = client.get(f"/api/models/{mid}/telemetry/meta").json()
    assert meta["available"] is True
    assert meta["reason"] is None
    assert meta["source"] == "none"

    kpi = client.get(f"/api/models/{mid}/telemetry/kpi").json()
    assert kpi["requestsPerMin"]["value"] == 0.0
    assert kpi["p95LatencyMs"]["value"] == 0.0

    assert client.get(f"/api/models/{mid}/telemetry/series").json() == []
    pct = client.get(f"/api/models/{mid}/telemetry/percentiles").json()
    assert pct["p50"] == [] and pct["values"]["p95"] == 0.0

    c = client.get(f"/api/models/{mid}/telemetry/cost").json()
    assert c["source"] == "none"
    assert c["perHardware"] == []          # no "Benchmark reference · CPU" row


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


def test_statedict_telemetry_empty_200(client, statedict_model):
    """A weights-only checkpoint can't be served — the data endpoints just return
    empty (no benchmark, no 404). The weights_only gate lives in /meta and /cost."""
    mid = statedict_model["modelId"]
    for path in ("kpi", "series", "percentiles"):
        assert client.get(f"/api/models/{mid}/telemetry/{path}").status_code == 200, path
    assert client.get(f"/api/models/{mid}/telemetry/series").json() == []
    kpi = client.get(f"/api/models/{mid}/telemetry/kpi").json()
    assert kpi["requestsPerMin"]["value"] == 0.0


def test_failed_model_telemetry_empty(client, failed_model):
    """No benchmark + no deployment → empty data (200) and a not_deployed gate."""
    r = client.get(f"/api/models/{failed_model['modelId']}/telemetry/kpi")
    assert r.status_code == 200
    assert r.json()["requestsPerMin"]["value"] == 0.0
    meta = client.get(f"/api/models/{failed_model['modelId']}/telemetry/meta").json()
    assert meta["available"] is False
    assert meta["reason"] == "not_deployed"


def test_telemetry_404_missing_model(client):
    assert client.get("/api/models/m_missing/telemetry/kpi").status_code == 404


def test_meta_availability_gate(
    client, real_model, statedict_model, failed_model, make_live_model, deploy_model
):
    """meta.available/reason is the SPA's gate: a live OR deployed model → the
    dashboard chrome (available); a weights-only checkpoint or an undeployed model
    → the matching full-page terminal. No benchmark-derived availability."""
    nd = client.get(f"/api/models/{real_model['modelId']}/telemetry/meta").json()
    assert nd["available"] is False
    assert nd["reason"] == "not_deployed"

    wo = client.get(f"/api/models/{statedict_model['modelId']}/telemetry/meta").json()
    assert wo["available"] is False
    assert wo["reason"] == "weights_only_checkpoint"

    fb = client.get(f"/api/models/{failed_model['modelId']}/telemetry/meta").json()
    assert fb["available"] is False
    assert fb["reason"] == "not_deployed"

    mid = make_live_model("gate-armed.onnx")["modelId"]
    deploy_model(mid)
    armed = client.get(f"/api/models/{mid}/telemetry/meta").json()
    assert armed["available"] is True
    assert armed["reason"] is None
    assert armed["source"] == "none"
